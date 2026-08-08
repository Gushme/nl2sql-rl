from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx
import pytest

from nl2sql_rl.models import AgentAction
from nl2sql_rl.teacher.client import (
    CostLimitExceeded,
    LLMClient,
    LLMClientConfig,
    LLMCompletion,
    TeacherAPIError,
    api_compatible_messages,
)
from nl2sql_rl.teacher.probe import run_function_call_probe


def _response(
    request: httpx.Request,
    *,
    content: str | None = None,
    tool_call: bool = False,
    status: int = 200,
) -> httpx.Response:
    if status != 200:
        return httpx.Response(status, request=request)
    message: dict[str, Any] = {"content": content}
    if tool_call:
        message["tool_calls"] = [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "list_tables", "arguments": "{}"},
            }
        ]
    return httpx.Response(
        200,
        request=request,
        json={
            "id": "response_1",
            "choices": [{"message": message}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 20},
        },
    )


def _config(**updates: Any) -> LLMClientConfig:
    values: dict[str, Any] = {
        "endpoint": "https://mock.local/v1",
        "model": "mock-teacher",
        "concurrency": 2,
        "max_retries": 3,
        "retry_base_seconds": 0,
        "input_price_per_million": 1.0,
        "output_price_per_million": 2.0,
        "max_request_cost_usd": 0.1,
        "real_api": False,
    }
    values.update(updates)
    return LLMClientConfig.model_validate(values)


@pytest.mark.asyncio
async def test_client_normalizes_native_tool_call_and_text_json() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        assert request.read()
        assert json.loads(request.content)["seed"] == 42
        if calls == 1:
            return _response(request, tool_call=True)
        return _response(
            request,
            content='{"action":"submit_sql","arguments":{"sql":"SELECT 1"}}',
        )

    async with LLMClient(
        _config(),
        api_key="mock",
        cost_limit_usd=1.0,
        transport=httpx.MockTransport(handler),
    ) as client:
        first = await client.complete_action([])
        second = await client.complete_action([])
        assert first.action.action == "list_tables"
        assert second.action.action == "submit_sql"
        assert first.cost_usd == pytest.approx(0.00014)
        assert client.spent_usd == pytest.approx(0.00028)


@pytest.mark.asyncio
async def test_thinking_is_requested_accounted_and_not_returned() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert payload["enable_thinking"] is True
        assert payload["reasoning_effort"] == "high"
        assert payload["max_tokens"] == 8_192
        return httpx.Response(
            200,
            request=request,
            json={
                "id": "thinking_1",
                "choices": [
                    {
                        "finish_reason": "tool_calls",
                        "message": {
                            "content": None,
                            "reasoning_content": "不会进入轨迹",
                            "tool_calls": [
                                {
                                    "type": "function",
                                    "function": {
                                        "name": "list_tables",
                                        "arguments": "{}",
                                    },
                                }
                            ],
                        },
                    }
                ],
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 80,
                    "prompt_tokens_details": {"cached_tokens": 20},
                    "completion_tokens_details": {"reasoning_tokens": 70},
                },
            },
        )

    config = _config(
        enable_thinking=True,
        reasoning_effort="high",
        max_completion_tokens=8_192,
    )
    async with LLMClient(
        config,
        api_key="mock",
        cost_limit_usd=1.0,
        initial_spent_usd=0.25,
        transport=httpx.MockTransport(handler),
    ) as client:
        completion = await client.complete_action([])
        assert completion.reasoning_tokens == 70
        assert completion.action_tokens == 10
        assert completion.cached_input_tokens == 20
        assert completion.finish_reason == "tool_calls"
        assert not hasattr(completion, "reasoning_content")
        assert client.spent_usd == pytest.approx(0.25026)


@pytest.mark.asyncio
async def test_client_retries_429_5xx_and_timeout() -> None:
    outcomes: list[int | str] = [429, 500, "timeout", 200]

    def handler(request: httpx.Request) -> httpx.Response:
        outcome = outcomes.pop(0)
        if outcome == "timeout":
            raise httpx.ReadTimeout("mock timeout", request=request)
        assert isinstance(outcome, int)
        return _response(
            request,
            content='{"action":"list_tables","arguments":{}}',
            status=outcome,
        )

    async with LLMClient(
        _config(max_retries=4),
        api_key="mock",
        cost_limit_usd=1.0,
        transport=httpx.MockTransport(handler),
    ) as client:
        completion = await client.complete_action([])
    assert completion.action.action == "list_tables"
    assert outcomes == []


@pytest.mark.asyncio
async def test_invalid_native_tool_call_becomes_protocol_input_not_api_failure() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            request=request,
            json={
                "id": "invalid_1",
                "choices": [
                    {
                        "message": {
                            "content": None,
                            "tool_calls": [
                                {
                                    "type": "function",
                                    "function": {
                                        "name": "unknown_tool",
                                        "arguments": "{}",
                                    },
                                }
                            ],
                        }
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            },
        )

    async with LLMClient(
        _config(),
        api_key="mock",
        cost_limit_usd=1.0,
        transport=httpx.MockTransport(handler),
    ) as client:
        completion = await client.complete_action([])
    assert completion.action is None
    assert completion.action_text is not None
    assert "unknown_tool" in completion.action_text
    assert completion.normalization_error is not None
    assert completion.response_format == "native_tool_call"


@pytest.mark.asyncio
async def test_authentication_error_is_fatal_and_classified() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, request=request)

    async with LLMClient(
        _config(),
        api_key="mock",
        cost_limit_usd=1.0,
        transport=httpx.MockTransport(handler),
    ) as client:
        with pytest.raises(TeacherAPIError) as captured:
            await client.complete_action([])
    assert captured.value.error_code == "authentication_error"
    assert captured.value.request_sent is True


@pytest.mark.asyncio
async def test_real_response_requires_token_usage() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            request=request,
            json={
                "id": "missing_usage",
                "choices": [
                    {
                        "message": {
                            "content": '{"action":"list_tables","arguments":{}}'
                        }
                    }
                ],
            },
        )

    async with LLMClient(
        _config(real_api=True),
        api_key="mock",
        cost_limit_usd=1.0,
        transport=httpx.MockTransport(handler),
    ) as client:
        with pytest.raises(TeacherAPIError) as captured:
            await client.complete_action([])
    assert captured.value.error_code == "api_invalid_response"


@pytest.mark.asyncio
async def test_function_probe_requires_native_call_and_reasoning_breakdown() -> None:
    class ProbeClient:
        config_hash = "probe"

        def __init__(self, completion: LLMCompletion) -> None:
            self.completion = completion

        async def complete_action(
            self, messages: list[dict[str, Any]], *, max_tokens: int | None = None
        ) -> LLMCompletion:
            assert messages and max_tokens is None
            return self.completion

    action = AgentAction(action="list_tables", arguments={})
    text_report = await run_function_call_probe(
        ProbeClient(
            LLMCompletion(
                action=action,
                response_id="text",
                input_tokens=1,
                output_tokens=1,
                cost_usd=0.0,
                response_format="text_json",
            )
        )
    )
    assert text_report["ok"] is False
    assert text_report["error_code"] == "native_tool_call_not_used"

    missing_breakdown = await run_function_call_probe(
        ProbeClient(
            LLMCompletion(
                action=action,
                response_id="thinking",
                input_tokens=1,
                output_tokens=2,
                cost_usd=0.0,
                response_format="native_tool_call",
                reasoning_present=True,
            )
        )
    )
    assert missing_breakdown["ok"] is False
    assert missing_breakdown["error_code"] == "reasoning_token_breakdown_missing"


@pytest.mark.asyncio
async def test_client_enforces_concurrency_limit() -> None:
    active = 0
    maximum = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal active, maximum
        active += 1
        maximum = max(maximum, active)
        await asyncio.sleep(0.01)
        active -= 1
        return _response(request, content='{"action":"list_tables","arguments":{}}')

    async with LLMClient(
        _config(concurrency=2),
        api_key="mock",
        cost_limit_usd=1.0,
        transport=httpx.MockTransport(handler),
    ) as client:
        await asyncio.gather(*(client.complete_action([]) for _ in range(4)))
    assert maximum == 2


@pytest.mark.asyncio
async def test_cost_cap_blocks_request_before_external_call() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return _response(request, content='{"action":"list_tables","arguments":{}}')

    async with LLMClient(
        _config(max_request_cost_usd=0.1),
        api_key="mock",
        cost_limit_usd=0.05,
        transport=httpx.MockTransport(handler),
    ) as client:
        with pytest.raises(CostLimitExceeded):
            await client.complete_action([])
    assert calls == 0


@pytest.mark.asyncio
async def test_actual_request_cost_cannot_exceed_reservation() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return _response(request, content='{"action":"list_tables","arguments":{}}')

    async with LLMClient(
        _config(max_request_cost_usd=0.0001),
        api_key="mock",
        cost_limit_usd=1.0,
        transport=httpx.MockTransport(handler),
    ) as client:
        with pytest.raises(CostLimitExceeded) as captured:
            await client.complete_action([])
    assert captured.value.request_sent is True
    assert captured.value.cost_usd == pytest.approx(0.00014)
    assert client.spent_usd == pytest.approx(0.00014)


def test_internal_tool_observation_becomes_qwen_compatible_user_message() -> None:
    converted = api_compatible_messages(
        [
            {"role": "assistant", "content": '{"action":"list_tables"}'},
            {
                "role": "tool",
                "name": "list_tables",
                "event_id": "event_1",
                "content": '{"ok":true}',
            },
        ]
    )
    assert converted[1] == {
        "role": "user",
        "content": '<tool_response>\n{"ok":true}\n</tool_response>',
    }
