"""带限流、重试、费用闸门的 OpenAI-compatible Action 客户端。"""

from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import dataclass
from typing import Any

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
    temperature: float = Field(default=0.0, ge=0)
    input_price_per_million: float = Field(default=0.0, ge=0)
    output_price_per_million: float = Field(default=0.0, ge=0)
    max_request_cost_usd: float = Field(default=1.0, gt=0)
    real_api: bool = True

    def fingerprint(self) -> str:
        encoded = stable_json(self.model_dump(mode="json")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


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


class CostLedger:
    def __init__(self, cap_usd: float, reservation_usd: float) -> None:
        if cap_usd <= 0:
            raise ValueError("Teacher 费用上限必须大于 0")
        self.cap_usd = cap_usd
        self.reservation_usd = reservation_usd
        self.spent_usd = 0.0
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
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.config = config
        self._semaphore = asyncio.Semaphore(config.concurrency)
        self._ledger = CostLedger(cost_limit_usd, config.max_request_cost_usd)
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
        try:
            async with self._semaphore:
                payload = {
                    "model": self.config.model,
                    "messages": messages,
                    "tools": ACTION_TOOLS,
                    "tool_choice": "auto",
                    "temperature": self.config.temperature,
                    "max_tokens": max_tokens or self.config.max_completion_tokens,
                }
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
                action = _normalize_choice(first["message"])
                usage_value = raw.get("usage")
                usage: dict[str, Any] = usage_value if isinstance(usage_value, dict) else {}
                input_tokens = int(usage.get("prompt_tokens", 0))
                output_tokens = int(usage.get("completion_tokens", 0))
                cost = self._cost(input_tokens, output_tokens)
                await self._ledger.settle(cost)
                settled = True
                return LLMCompletion(
                    action=action,
                    response_id=str(raw["id"]) if raw.get("id") is not None else None,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cost_usd=cost,
                )
        finally:
            if not settled:
                await self._ledger.release()
