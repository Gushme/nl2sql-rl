from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx
import pytest

from nl2sql_rl.teacher.client import (
    CostLimitExceeded,
    LLMClient,
    LLMClientConfig,
    api_compatible_messages,
)


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
