from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx
import pytest
import typer

from nl2sql_rl.cli import _validate_teacher_budget
from nl2sql_rl.models import AgentAction
from nl2sql_rl.teacher.client import (
    CostLimitExceeded,
    LLMClient,
    LLMClientConfig,
    LLMCompletion,
    TeacherAPIError,
    TokenLimitExceeded,
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


def test_teacher_budget_accepts_token_only_mode_without_fake_usd_price() -> None:
    request_tokens = _validate_teacher_budget(
        input_price_per_million=0.0,
        output_price_per_million=0.0,
        max_context_tokens=16_384,
        max_completion_tokens=8_192,
        max_request_cost_usd=0.01,
        cost_limit_usd=None,
        token_limit=1_000_000,
    )
    assert request_tokens == 28_672


def test_teacher_budget_rejects_missing_or_incomplete_guards() -> None:
    common = {
        "max_context_tokens": 16_384,
        "max_completion_tokens": 8_192,
        "max_request_cost_usd": 0.01,
        "cost_limit_usd": None,
    }
    with pytest.raises(typer.BadParameter, match="Token 上限或完整费用闸门"):
        _validate_teacher_budget(
            input_price_per_million=0.0,
            output_price_per_million=0.0,
            token_limit=None,
            **common,
        )
    with pytest.raises(typer.BadParameter, match="同时设置"):
        _validate_teacher_budget(
            input_price_per_million=1.0,
            output_price_per_million=0.0,
            token_limit=1_000_000,
            **common,
        )
    with pytest.raises(typer.BadParameter, match="保守 Token 上界"):
        _validate_teacher_budget(
            input_price_per_million=0.0,
            output_price_per_million=0.0,
            token_limit=28_671,
            **common,
        )


@pytest.mark.asyncio
async def test_client_normalizes_native_tool_call_and_text_json() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        assert request.read()
        payload = json.loads(request.content)
        assert payload["seed"] == 42
        assert payload["parallel_tool_calls"] is False
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
        assert client.used_tokens == 240


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
async def test_multiple_native_tool_calls_are_rejected_as_one_step_protocol_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert json.loads(request.content)["parallel_tool_calls"] is False
        call = {
            "type": "function",
            "function": {"name": "list_tables", "arguments": "{}"},
        }
        return httpx.Response(
            200,
            request=request,
            json={
                "id": "parallel_1",
                "choices": [
                    {"message": {"content": None, "tool_calls": [call, call]}}
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
    assert completion.action_text == '{"native_tool_call_count":2}'
    assert completion.normalization_error == "每轮只允许一个原生 tool_call"


@pytest.mark.asyncio
async def test_authentication_error_is_fatal_and_classified() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            401,
            request=request,
            json={
                "error": {
                    "code": "invalid_api_key",
                    "type": "authentication_error",
                    "message": "该内容不得写入诊断",
                }
            },
        )

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
    assert captured.value.diagnostic_detail == {
        "status_code": 401,
        "provider_code": "invalid_api_key",
        "provider_type": "authentication_error",
    }
    assert "message" not in captured.value.diagnostic_detail


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
async def test_function_probe_accepts_both_transports_and_requires_reasoning_breakdown() -> None:
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
    assert text_report["ok"] is True
    assert text_report["error_code"] is None
    assert text_report["native_tool_call_used"] is False
    assert text_report["compatible_action_transport"] is True

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


@pytest.mark.asyncio
async def test_token_quota_tracks_exact_returned_usage() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return _response(request, content='{"action":"list_tables","arguments":{}}')

    async with LLMClient(
        _config(
            input_price_per_million=0.0,
            output_price_per_million=0.0,
        ),
        api_key="mock",
        token_limit=1_000,
        max_request_tokens=200,
        transport=httpx.MockTransport(handler),
    ) as client:
        await client.complete_action([])
        await client.complete_action([])
        assert client.used_tokens == 240
        assert client.token_limit == 1_000
        assert client.spent_usd == 0.0


@pytest.mark.asyncio
async def test_token_quota_blocks_before_external_call() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return _response(request, content='{"action":"list_tables","arguments":{}}')

    async with LLMClient(
        _config(
            input_price_per_million=0.0,
            output_price_per_million=0.0,
        ),
        api_key="mock",
        token_limit=100,
        max_request_tokens=200,
        transport=httpx.MockTransport(handler),
    ) as client:
        with pytest.raises(TokenLimitExceeded) as captured:
            await client.complete_action([])
    assert captured.value.request_sent is False
    assert calls == 0


@pytest.mark.asyncio
async def test_actual_tokens_cannot_exceed_conservative_reservation() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return _response(request, content='{"action":"list_tables","arguments":{}}')

    async with LLMClient(
        _config(
            input_price_per_million=0.0,
            output_price_per_million=0.0,
        ),
        api_key="mock",
        token_limit=1_000,
        max_request_tokens=100,
        transport=httpx.MockTransport(handler),
    ) as client:
        with pytest.raises(TokenLimitExceeded) as captured:
            await client.complete_action([])
        assert client.used_tokens == 120
    assert captured.value.request_sent is True
    assert captured.value.input_tokens == 100
    assert captured.value.output_tokens == 20


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
