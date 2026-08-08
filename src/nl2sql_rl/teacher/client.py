"""带限流、重试、费用闸门的 OpenAI-compatible Action 客户端。"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from dataclasses import dataclass
from typing import Any, Literal

import httpx
from pydantic import Field

from nl2sql_rl.agent.parser import ActionParseError, parse_action
from nl2sql_rl.io_utils import stable_json
from nl2sql_rl.models import AgentAction, StrictRecord

ACTION_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "list_tables",
            "description": "列出数据库中的用户表和视图",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "describe_schema",
            "description": "查看指定表的 SQLite DDL",
            "parameters": {
                "type": "object",
                "properties": {
                    "tables": {"type": "array", "items": {"type": "string"}, "minItems": 1}
                },
                "required": ["tables"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_values",
            "description": "在一个表列中搜索候选值",
            "parameters": {
                "type": "object",
                "properties": {
                    "table": {"type": "string"},
                    "column": {"type": "string"},
                    "query": {"type": "string"},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 20},
                },
                "required": ["table", "column", "query"],
                "additionalProperties": False,
            },
        },
    },
    *[
        {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": {
                    "type": "object",
                    "properties": {"sql": {"type": "string"}},
                    "required": ["sql"],
                    "additionalProperties": False,
                },
            },
        }
        for name, description in (
            ("execute_sql", "只读执行候选 SQL 并查看受限结果"),
            ("submit_sql", "提交最终 SQL 并结束 episode"),
        )
    ],
]


class LLMClientConfig(StrictRecord):
    endpoint: str
    model: str
    concurrency: int = Field(default=4, ge=1, le=64)
    max_retries: int = Field(default=4, ge=0, le=10)
    retry_base_seconds: float = Field(default=0.5, ge=0)
    timeout_seconds: float = Field(default=60.0, gt=0)
    max_completion_tokens: int = Field(default=512, ge=1)
    normalized_action_token_limit: int = Field(default=512, ge=16)
    temperature: float = Field(default=0.0, ge=0)
    seed: int = 42
    enable_thinking: bool = False
    reasoning_effort: Literal["low", "medium", "high"] | None = None
    input_price_per_million: float = Field(default=0.0, ge=0)
    output_price_per_million: float = Field(default=0.0, ge=0)
    max_request_cost_usd: float = Field(default=1.0, gt=0)
    real_api: bool = True

    def fingerprint(self) -> str:
        encoded = stable_json(self.model_dump(mode="json")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def behavior_fingerprint(self) -> str:
        """只哈希会改变模型输出分布或模型可见协议的配置。"""
        payload = {
            "endpoint": self.endpoint.rstrip("/"),
            "model": self.model,
            "max_completion_tokens": self.max_completion_tokens,
            "normalized_action_token_limit": self.normalized_action_token_limit,
            "temperature": self.temperature,
            "seed": self.seed,
            "enable_thinking": self.enable_thinking,
            "reasoning_effort": self.reasoning_effort,
        }
        return hashlib.sha256(stable_json(payload).encode("utf-8")).hexdigest()


class TeacherAPIError(RuntimeError):
    """Teacher endpoint 重试耗尽或响应不合法。"""


class CostLimitExceeded(RuntimeError):
    """下一次请求的预留费用会超过显式上限。"""


@dataclass(frozen=True)
class LLMCompletion:
    action: AgentAction
    response_id: str | None
    input_tokens: int
    output_tokens: int
    cost_usd: float
    reasoning_tokens: int | None = None
    action_tokens: int | None = None
    cached_input_tokens: int = 0
    latency_ms: float = 0.0
    finish_reason: str | None = None


class CostLedger:
    def __init__(
        self,
        cap_usd: float,
        reservation_usd: float,
        *,
        initial_spent_usd: float = 0.0,
    ) -> None:
        if cap_usd <= 0:
            raise ValueError("Teacher 费用上限必须大于 0")
        if initial_spent_usd < 0 or initial_spent_usd > cap_usd + 1e-12:
            raise ValueError("Teacher 历史费用必须位于 0 到费用上限之间")
        self.cap_usd = cap_usd
        self.reservation_usd = reservation_usd
        self.spent_usd = initial_spent_usd
        self.reserved_usd = 0.0
        self._lock = asyncio.Lock()

    async def reserve(self) -> None:
        async with self._lock:
            projected = self.spent_usd + self.reserved_usd + self.reservation_usd
            if projected > self.cap_usd + 1e-12:
                raise CostLimitExceeded(
                    f"费用预留将超过上限：{projected:.6f} > {self.cap_usd:.6f} USD"
                )
            self.reserved_usd += self.reservation_usd

    async def settle(self, actual_usd: float) -> None:
        async with self._lock:
            self.reserved_usd = max(0.0, self.reserved_usd - self.reservation_usd)
            self.spent_usd += actual_usd
            if self.spent_usd > self.cap_usd + 1e-12:
                raise CostLimitExceeded(
                    f"实际费用超过上限：{self.spent_usd:.6f} > {self.cap_usd:.6f} USD"
                )

    async def release(self) -> None:
        async with self._lock:
            self.reserved_usd = max(0.0, self.reserved_usd - self.reservation_usd)


def api_compatible_messages(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
    """把内部 tool observation 转为无需 tool_call_id 的 Qwen user 消息。"""
    converted: list[dict[str, str]] = []
    for message in messages:
        role = str(message.get("role", ""))
        content = str(message.get("content", ""))
        if role == "tool":
            converted.append(
                {
                    "role": "user",
                    "content": f"<tool_response>\n{content}\n</tool_response>",
                }
            )
        else:
            converted.append({"role": role, "content": content})
    return converted


def _normalize_choice(message: dict[str, Any]) -> AgentAction:
    tool_calls = message.get("tool_calls")
    if isinstance(tool_calls, list) and tool_calls:
        if len(tool_calls) != 1:
            raise ActionParseError("每轮只允许一个原生 tool_call")
        call = tool_calls[0]
        if not isinstance(call, dict) or not isinstance(call.get("function"), dict):
            raise ActionParseError("原生 tool_call 结构不合法")
        function = call["function"]
        arguments = function.get("arguments", {})
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError as exc:
                raise ActionParseError("tool_call arguments 不是合法 JSON") from exc
        return AgentAction.model_validate(
            {"action": function.get("name"), "arguments": arguments}
        )
    content = message.get("content")
    if not isinstance(content, str):
        raise ActionParseError("Teacher 响应既无 tool_call，也无文本 JSON")
    return parse_action(content)


class LLMClient:
    def __init__(
        self,
        config: LLMClientConfig,
        *,
        api_key: str,
        cost_limit_usd: float,
        initial_spent_usd: float = 0.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.config = config
        self._semaphore = asyncio.Semaphore(config.concurrency)
        self._ledger = CostLedger(
            cost_limit_usd,
            config.max_request_cost_usd,
            initial_spent_usd=initial_spent_usd,
        )
        self._client = httpx.AsyncClient(
            base_url=config.endpoint.rstrip("/") + "/",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=config.timeout_seconds,
            transport=transport,
        )

    @property
    def config_hash(self) -> str:
        return self.config.fingerprint()

    @property
    def behavior_config_hash(self) -> str:
        return self.config.behavior_fingerprint()

    @property
    def real_api(self) -> bool:
        return self.config.real_api

    @property
    def spent_usd(self) -> float:
        return self._ledger.spent_usd

    async def close(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> LLMClient:
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.close()

    def _cost(self, input_tokens: int, output_tokens: int) -> float:
        return (
            input_tokens * self.config.input_price_per_million
            + output_tokens * self.config.output_price_per_million
        ) / 1_000_000

    async def complete_action(
        self,
        messages: list[dict[str, Any]],
        *,
        max_tokens: int | None = None,
    ) -> LLMCompletion:
        await self._ledger.reserve()
        settled = False
        started = time.monotonic()
        try:
            async with self._semaphore:
                payload = {
                    "model": self.config.model,
                    "messages": api_compatible_messages(messages),
                    "tools": ACTION_TOOLS,
                    "tool_choice": "auto",
                    "temperature": self.config.temperature,
                    "seed": self.config.seed,
                    "max_tokens": max_tokens or self.config.max_completion_tokens,
                }
                if self.config.enable_thinking:
                    payload["enable_thinking"] = True
                if self.config.reasoning_effort is not None:
                    payload["reasoning_effort"] = self.config.reasoning_effort
                response: httpx.Response | None = None
                for attempt in range(self.config.max_retries + 1):
                    try:
                        response = await self._client.post("chat/completions", json=payload)
                        if response.status_code != 429 and response.status_code < 500:
                            response.raise_for_status()
                            break
                    except (httpx.TimeoutException, httpx.TransportError):
                        if attempt >= self.config.max_retries:
                            raise
                    if attempt >= self.config.max_retries:
                        status = response.status_code if response is not None else "transport"
                        raise TeacherAPIError(f"Teacher 请求重试耗尽：{status}")
                    retry_after = (
                        float(response.headers.get("Retry-After", "0"))
                        if response is not None
                        else 0.0
                    )
                    delay = max(retry_after, self.config.retry_base_seconds * (2**attempt))
                    if delay:
                        await asyncio.sleep(delay)
                if response is None:
                    raise TeacherAPIError("Teacher 未返回响应")
                raw: Any = response.json()
                if not isinstance(raw, dict):
                    raise TeacherAPIError("Teacher 响应顶层不是 JSON 对象")
                choices = raw.get("choices")
                if not isinstance(choices, list) or not choices:
                    raise TeacherAPIError("Teacher 响应缺少 choices")
                first = choices[0]
                if not isinstance(first, dict) or not isinstance(first.get("message"), dict):
                    raise TeacherAPIError("Teacher choice.message 不合法")
                message = first["message"]
                action = _normalize_choice(message)
                # reasoning_content 可能很长；只读取存在性，绝不写入返回值或后续消息。
                has_reasoning = bool(message.get("reasoning_content"))
                usage_value = raw.get("usage")
                usage: dict[str, Any] = usage_value if isinstance(usage_value, dict) else {}
                input_tokens = int(usage.get("prompt_tokens", 0))
                output_tokens = int(usage.get("completion_tokens", 0))
                completion_details_value = usage.get(
                    "completion_tokens_details", usage.get("output_tokens_details")
                )
                completion_details = (
                    completion_details_value
                    if isinstance(completion_details_value, dict)
                    else {}
                )
                prompt_details_value = usage.get(
                    "prompt_tokens_details", usage.get("input_tokens_details")
                )
                prompt_details = (
                    prompt_details_value if isinstance(prompt_details_value, dict) else {}
                )
                reasoning_value = completion_details.get("reasoning_tokens")
                reasoning_tokens = (
                    int(reasoning_value) if reasoning_value is not None else None
                )
                if reasoning_tokens is not None and reasoning_tokens > output_tokens:
                    raise TeacherAPIError("reasoning_tokens 大于 completion_tokens")
                action_tokens = (
                    output_tokens - reasoning_tokens
                    if reasoning_tokens is not None
                    else (output_tokens if not has_reasoning else None)
                )
                cached_input_tokens = int(prompt_details.get("cached_tokens", 0))
                cost = self._cost(input_tokens, output_tokens)
                await self._ledger.settle(cost)
                settled = True
                return LLMCompletion(
                    action=action,
                    response_id=str(raw["id"]) if raw.get("id") is not None else None,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cost_usd=cost,
                    reasoning_tokens=reasoning_tokens,
                    action_tokens=action_tokens,
                    cached_input_tokens=cached_input_tokens,
                    latency_ms=(time.monotonic() - started) * 1000,
                    finish_reason=(
                        str(first["finish_reason"])
                        if first.get("finish_reason") is not None
                        else None
                    ),
                )
        finally:
            if not settled:
                await self._ledger.release()
