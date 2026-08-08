"""带限流、重试、费用与 Token 闸门的 OpenAI-compatible Action 客户端。"""

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
from nl2sql_rl.agent.spec import ACTION_TOOLS
from nl2sql_rl.io_utils import stable_json
from nl2sql_rl.models import AgentAction, StrictRecord


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
    parallel_tool_calls: Literal[False] = False
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
            "parallel_tool_calls": self.parallel_tool_calls,
        }
        return hashlib.sha256(stable_json(payload).encode("utf-8")).hexdigest()

    def pricing_fingerprint(self) -> str:
        """冻结美元折算口径；Token 总配额由 Campaign 账本单独冻结。"""
        has_usd_pricing = (
            self.input_price_per_million > 0
            and self.output_price_per_million > 0
        )
        payload = {
            "mode": "usd_pricing" if has_usd_pricing else "no_usd_conversion",
            "input_price_per_million": self.input_price_per_million,
            "output_price_per_million": self.output_price_per_million,
        }
        return hashlib.sha256(stable_json(payload).encode("utf-8")).hexdigest()


class TeacherAPIError(RuntimeError):
    """Teacher endpoint 重试耗尽或响应不合法。"""

    def __init__(
        self,
        message: str,
        *,
        error_code: str = "teacher_api_error",
        request_sent: bool = True,
        cost_usd: float = 0.0,
        input_tokens: int = 0,
        output_tokens: int = 0,
        diagnostic_detail: dict[str, str | int | bool] | None = None,
    ) -> None:
        super().__init__(message)
        self.error_code = error_code
        self.request_sent = request_sent
        self.cost_usd = cost_usd
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.diagnostic_detail = diagnostic_detail or {}


def _provider_error_detail(response: httpx.Response) -> dict[str, str | int | bool]:
    """只保留状态和稳定错误类型，避免把请求内容或服务端消息落盘。"""
    detail: dict[str, str | int | bool] = {"status_code": response.status_code}
    try:
        payload = response.json()
    except ValueError:
        return detail
    if not isinstance(payload, dict):
        return detail
    error = payload.get("error")
    if not isinstance(error, dict):
        return detail
    for source, target in (("code", "provider_code"), ("type", "provider_type")):
        value = error.get(source)
        if isinstance(value, (str, int, bool)):
            detail[target] = value
    return detail


class CostLimitExceeded(RuntimeError):
    """下一次请求的预留费用会超过显式上限。"""

    error_code = "cost_limit_exceeded"

    def __init__(
        self,
        message: str,
        *,
        request_sent: bool = False,
        cost_usd: float = 0.0,
        input_tokens: int = 0,
        output_tokens: int = 0,
    ) -> None:
        super().__init__(message)
        self.request_sent = request_sent
        self.cost_usd = cost_usd
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class TokenLimitExceeded(RuntimeError):
    """下一次请求的预留 Token 或实际用量超过累计上限。"""

    error_code = "token_limit_exceeded"

    def __init__(
        self,
        message: str,
        *,
        request_sent: bool = False,
        input_tokens: int = 0,
        output_tokens: int = 0,
    ) -> None:
        super().__init__(message)
        self.request_sent = request_sent
        self.cost_usd = 0.0
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


@dataclass(frozen=True)
class LLMCompletion:
    action: AgentAction | None
    response_id: str | None
    input_tokens: int
    output_tokens: int
    cost_usd: float
    action_text: str | None = None
    reasoning_tokens: int | None = None
    action_tokens: int | None = None
    cached_input_tokens: int = 0
    latency_ms: float = 0.0
    finish_reason: str | None = None
    response_format: Literal["native_tool_call", "text_json"] = "text_json"
    normalization_error: str | None = None
    reasoning_present: bool = False


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
            if actual_usd > self.reservation_usd + 1e-12:
                raise CostLimitExceeded(
                    "单次请求实际费用超过保守预留："
                    f"{actual_usd:.6f} > {self.reservation_usd:.6f} USD",
                    request_sent=True,
                    cost_usd=actual_usd,
                )
            if self.spent_usd > self.cap_usd + 1e-12:
                raise CostLimitExceeded(
                    f"实际费用超过上限：{self.spent_usd:.6f} > {self.cap_usd:.6f} USD",
                    request_sent=True,
                    cost_usd=actual_usd,
                )

    async def release(self) -> None:
        async with self._lock:
            self.reserved_usd = max(0.0, self.reserved_usd - self.reservation_usd)


class TokenLedger:
    """按接口返回的输入与输出 Token 做并发安全的累计记账。"""

    def __init__(
        self,
        cap_tokens: int | None,
        reservation_tokens: int,
        *,
        initial_used_tokens: int = 0,
    ) -> None:
        if cap_tokens is not None and cap_tokens <= 0:
            raise ValueError("Teacher Token 上限必须大于 0")
        if cap_tokens is not None and reservation_tokens <= 0:
            raise ValueError("Teacher 单次请求 Token 预留必须大于 0")
        if initial_used_tokens < 0 or (
            cap_tokens is not None and initial_used_tokens > cap_tokens
        ):
            raise ValueError("Teacher 历史 Token 必须位于 0 到 Token 上限之间")
        self.cap_tokens = cap_tokens
        self.reservation_tokens = reservation_tokens
        self.used_tokens = initial_used_tokens
        self.reserved_tokens = 0
        self._lock = asyncio.Lock()

    async def reserve(self) -> None:
        async with self._lock:
            if self.cap_tokens is None:
                return
            projected = (
                self.used_tokens + self.reserved_tokens + self.reservation_tokens
            )
            if projected > self.cap_tokens:
                raise TokenLimitExceeded(
                    f"Token 预留将超过上限：{projected} > {self.cap_tokens}"
                )
            self.reserved_tokens += self.reservation_tokens

    async def settle(self, input_tokens: int, output_tokens: int) -> None:
        actual_tokens = input_tokens + output_tokens
        async with self._lock:
            self.reserved_tokens = max(
                0, self.reserved_tokens - self.reservation_tokens
            )
            self.used_tokens += actual_tokens
            if self.cap_tokens is None:
                return
            if actual_tokens > self.reservation_tokens:
                raise TokenLimitExceeded(
                    "单次请求实际 Token 超过保守预留："
                    f"{actual_tokens} > {self.reservation_tokens}",
                    request_sent=True,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                )
            if self.used_tokens > self.cap_tokens:
                raise TokenLimitExceeded(
                    f"实际 Token 超过上限：{self.used_tokens} > {self.cap_tokens}",
                    request_sent=True,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                )

    async def release(self) -> None:
        async with self._lock:
            if self.cap_tokens is None:
                return
            self.reserved_tokens = max(
                0, self.reserved_tokens - self.reservation_tokens
            )


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


def _normalize_choice(
    message: dict[str, Any],
) -> tuple[AgentAction | None, str, str | None, Literal["native_tool_call", "text_json"]]:
    tool_calls = message.get("tool_calls")
    if isinstance(tool_calls, list) and tool_calls:
        if len(tool_calls) != 1:
            error = "每轮只允许一个原生 tool_call"
            text = stable_json({"native_tool_call_count": len(tool_calls)})
            return None, text, error, "native_tool_call"
        call = tool_calls[0]
        if not isinstance(call, dict) or not isinstance(call.get("function"), dict):
            error = "原生 tool_call 结构不合法"
            text = stable_json({"invalid_native_tool_call": True})
            return None, text, error, "native_tool_call"
        function = call["function"]
        arguments = function.get("arguments", {})
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = function.get("arguments", "")
        action_text = stable_json(
            {"action": function.get("name"), "arguments": arguments}
        )
        try:
            action = parse_action(action_text)
        except ActionParseError as exc:
            return None, action_text, str(exc), "native_tool_call"
        return action, action_text, None, "native_tool_call"
    content = message.get("content")
    if not isinstance(content, str):
        error = "Teacher 响应既无 tool_call，也无文本 JSON"
        return None, "", error, "text_json"
    try:
        action = parse_action(content)
    except ActionParseError as exc:
        return None, content, str(exc), "text_json"
    return action, stable_json(action.model_dump(mode="json")), None, "text_json"


class LLMClient:
    def __init__(
        self,
        config: LLMClientConfig,
        *,
        api_key: str,
        cost_limit_usd: float | None = None,
        initial_spent_usd: float = 0.0,
        token_limit: int | None = None,
        initial_used_tokens: int = 0,
        max_request_tokens: int | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.config = config
        self._semaphore = asyncio.Semaphore(config.concurrency)
        has_usd_pricing = (
            config.input_price_per_million > 0
            and config.output_price_per_million > 0
        )
        if has_usd_pricing and cost_limit_usd is None:
            raise ValueError("设置美元 Token 单价时必须同时设置累计费用上限")
        if config.real_api and cost_limit_usd is None and token_limit is None:
            raise ValueError("真实 Teacher API 必须设置费用上限或 Token 上限")
        if token_limit is not None and max_request_tokens is None:
            raise ValueError("设置 Token 上限时必须提供单次请求的保守 Token 上界")
        self._ledger = (
            CostLedger(
                cost_limit_usd,
                config.max_request_cost_usd,
                initial_spent_usd=initial_spent_usd,
            )
            if cost_limit_usd is not None
            else None
        )
        if token_limit is not None:
            assert max_request_tokens is not None
            self._token_ledger = TokenLedger(
                token_limit,
                max_request_tokens,
                initial_used_tokens=initial_used_tokens,
            )
        else:
            self._token_ledger = TokenLedger(
                None,
                0,
                initial_used_tokens=initial_used_tokens,
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
    def pricing_config_hash(self) -> str:
        return self.config.pricing_fingerprint()

    @property
    def real_api(self) -> bool:
        return self.config.real_api

    @property
    def spent_usd(self) -> float:
        return self._ledger.spent_usd if self._ledger is not None else 0.0

    @property
    def used_tokens(self) -> int:
        return self._token_ledger.used_tokens

    @property
    def token_limit(self) -> int | None:
        return self._token_ledger.cap_tokens

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
        cost_reserved = False
        tokens_reserved = False
        started = time.monotonic()
        try:
            if self._ledger is not None:
                await self._ledger.reserve()
                cost_reserved = True
            await self._token_ledger.reserve()
            tokens_reserved = True
            async with self._semaphore:
                payload = {
                    "model": self.config.model,
                    "messages": api_compatible_messages(messages),
                    "tools": ACTION_TOOLS,
                    "tool_choice": "auto",
                    "parallel_tool_calls": self.config.parallel_tool_calls,
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
                        if response.status_code < 400:
                            break
                        if response.status_code not in {429} and response.status_code < 500:
                            if response.status_code in {401, 403}:
                                error_code = "authentication_error"
                            elif response.status_code in {400, 404, 422}:
                                error_code = "model_or_request_error"
                            else:
                                error_code = "api_client_error"
                            raise TeacherAPIError(
                                f"Teacher 返回不可重试状态：{response.status_code}",
                                error_code=error_code,
                                diagnostic_detail=_provider_error_detail(response),
                            )
                    except (httpx.TimeoutException, httpx.TransportError):
                        if attempt >= self.config.max_retries:
                            raise TeacherAPIError(
                                "Teacher 网络请求重试耗尽",
                                error_code="api_transport_error",
                            ) from None
                    if attempt >= self.config.max_retries:
                        status = response.status_code if response is not None else "transport"
                        raise TeacherAPIError(
                            f"Teacher 请求重试耗尽：{status}",
                            error_code="api_retry_exhausted",
                            diagnostic_detail=(
                                _provider_error_detail(response)
                                if response is not None
                                else {}
                            ),
                        )
                    retry_after = (
                        float(response.headers.get("Retry-After", "0"))
                        if response is not None
                        else 0.0
                    )
                    delay = max(retry_after, self.config.retry_base_seconds * (2**attempt))
                    if delay:
                        await asyncio.sleep(delay)
                if response is None:
                    raise TeacherAPIError(
                        "Teacher 未返回响应", error_code="api_empty_response"
                    )
                try:
                    raw: Any = response.json()
                except ValueError as exc:
                    raise TeacherAPIError(
                        "Teacher 响应不是合法 JSON",
                        error_code="api_invalid_response",
                    ) from exc
                if not isinstance(raw, dict):
                    raise TeacherAPIError(
                        "Teacher 响应顶层不是 JSON 对象",
                        error_code="api_invalid_response",
                    )
                choices = raw.get("choices")
                if not isinstance(choices, list) or not choices:
                    raise TeacherAPIError(
                        "Teacher 响应缺少 choices", error_code="api_invalid_response"
                    )
                first = choices[0]
                if not isinstance(first, dict) or not isinstance(first.get("message"), dict):
                    raise TeacherAPIError(
                        "Teacher choice.message 不合法",
                        error_code="api_invalid_response",
                    )
                message = first["message"]
                action, action_text, normalization_error, response_format = (
                    _normalize_choice(message)
                )
                # reasoning_content 可能很长；只读取存在性，绝不写入返回值或后续消息。
                has_reasoning = bool(message.get("reasoning_content"))
                usage_value = raw.get("usage")
                if self.config.real_api and not isinstance(usage_value, dict):
                    raise TeacherAPIError(
                        "Teacher 真实响应缺少 usage",
                        error_code="api_invalid_response",
                    )
                usage: dict[str, Any] = usage_value if isinstance(usage_value, dict) else {}
                if self.config.real_api and not {
                    "prompt_tokens",
                    "completion_tokens",
                }.issubset(usage):
                    raise TeacherAPIError(
                        "Teacher 真实响应缺少 prompt/completion token 用量",
                        error_code="api_invalid_response",
                    )
                try:
                    input_tokens = int(usage.get("prompt_tokens", 0))
                    output_tokens = int(usage.get("completion_tokens", 0))
                except (TypeError, ValueError) as exc:
                    raise TeacherAPIError(
                        "Teacher token 用量不是整数",
                        error_code="api_invalid_response",
                    ) from exc
                if input_tokens < 0 or output_tokens < 0:
                    raise TeacherAPIError(
                        "Teacher token 用量不能为负数",
                        error_code="api_invalid_response",
                    )
                cost = self._cost(input_tokens, output_tokens)
                budget_error: RuntimeError | None = None
                if self._ledger is not None:
                    try:
                        await self._ledger.settle(cost)
                    except CostLimitExceeded as exc:
                        exc.input_tokens = input_tokens
                        exc.output_tokens = output_tokens
                        budget_error = exc
                    finally:
                        cost_reserved = False
                try:
                    await self._token_ledger.settle(input_tokens, output_tokens)
                except TokenLimitExceeded as exc:
                    if budget_error is None:
                        budget_error = exc
                finally:
                    tokens_reserved = False
                if budget_error is not None:
                    raise budget_error
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
                try:
                    reasoning_tokens = (
                        int(reasoning_value) if reasoning_value is not None else None
                    )
                    cached_input_tokens = int(prompt_details.get("cached_tokens", 0))
                except (TypeError, ValueError) as exc:
                    raise TeacherAPIError(
                        "Teacher token 明细不是整数",
                        error_code="api_invalid_response",
                        cost_usd=cost,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                    ) from exc
                if reasoning_tokens is not None and not 0 <= reasoning_tokens <= output_tokens:
                    raise TeacherAPIError(
                        "reasoning_tokens 不在 completion_tokens 范围内",
                        error_code="api_invalid_response",
                        cost_usd=cost,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                    )
                action_tokens = (
                    output_tokens - reasoning_tokens
                    if reasoning_tokens is not None
                    else (output_tokens if not has_reasoning else None)
                )
                if cached_input_tokens < 0 or cached_input_tokens > input_tokens:
                    raise TeacherAPIError(
                        "cached_tokens 不在 prompt_tokens 范围内",
                        error_code="api_invalid_response",
                        cost_usd=cost,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                    )
                return LLMCompletion(
                    action=action,
                    action_text=action_text,
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
                    response_format=response_format,
                    normalization_error=normalization_error,
                    reasoning_present=has_reasoning,
                )
        finally:
            if cost_reserved and self._ledger is not None:
                await self._ledger.release()
            if tokens_reserved:
                await self._token_ledger.release()
