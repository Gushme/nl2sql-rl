"""不执行数据库操作的最小原生 tool call / 文本 JSON 兼容性探针。"""

from __future__ import annotations

from typing import Any, Protocol

from nl2sql_rl.agent.spec import SYSTEM_PROMPT
from nl2sql_rl.io_utils import stable_json
from nl2sql_rl.teacher.client import LLMCompletion


class ProbeClient(Protocol):
    @property
    def config_hash(self) -> str: ...

    async def complete_action(
        self, messages: list[dict[str, Any]], *, max_tokens: int | None = None
    ) -> LLMCompletion: ...


async def run_function_call_probe(client: ProbeClient) -> dict[str, Any]:
    completion = await client.complete_action(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": stable_json(
                    {
                        "probe": True,
                        "instruction": "只调用 list_tables，arguments 必须为空对象",
                    }
                ),
            },
        ]
    )
    action = completion.action
    valid_action = (
        action is not None and action.action == "list_tables" and not action.arguments
    )
    error_code: str | None = None
    if not valid_action:
        error_code = "invalid_probe_action"
    elif completion.reasoning_present and completion.reasoning_tokens is None:
        error_code = "reasoning_token_breakdown_missing"
    ok = error_code is None
    return {
        "schema_version": 1,
        "ok": ok,
        "error_code": error_code,
        "response_id": completion.response_id,
        "action": action.model_dump(mode="json") if action is not None else None,
        "response_format": completion.response_format,
        "native_tool_call_used": completion.response_format == "native_tool_call",
        "compatible_action_transport": completion.response_format
        in {"native_tool_call", "text_json"},
        "normalization_error": completion.normalization_error,
        "usage": {
            "input_tokens": completion.input_tokens,
            "output_tokens": completion.output_tokens,
            "total_tokens": completion.input_tokens + completion.output_tokens,
            "reasoning_tokens": completion.reasoning_tokens,
            "reasoning_present": completion.reasoning_present,
            "action_tokens": completion.action_tokens,
            "cached_input_tokens": completion.cached_input_tokens,
            "latency_ms": completion.latency_ms,
            "cost_usd": completion.cost_usd,
        },
        "client_config_hash": client.config_hash,
        "reasoning_content_saved": False,
    }
