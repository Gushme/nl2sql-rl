"""生成覆盖全部行为语义的 Harness 配置哈希。"""

from __future__ import annotations

import hashlib
from typing import Any

from nl2sql_rl.agent.spec import (
    ACCEPTANCE_VERSION,
    ACTION_TOOLS,
    HARNESS_VERSION,
    SYSTEM_PROMPT,
    TOOL_OBSERVATION_VERSION,
)
from nl2sql_rl.config import RuntimeConfig
from nl2sql_rl.io_utils import stable_json
from nl2sql_rl.models import AgentAction, ToolObservation
from nl2sql_rl.sqlite_exec import SQL_GUARD_VERSION


def visible_protocol_payload() -> dict[str, Any]:
    return {
        "system_prompt": SYSTEM_PROMPT,
        "tools": ACTION_TOOLS,
        "action_schema": AgentAction.model_json_schema(),
    }


def visible_protocol_hash() -> str:
    return hashlib.sha256(stable_json(visible_protocol_payload()).encode("utf-8")).hexdigest()


def harness_payload(runtime: RuntimeConfig) -> dict[str, Any]:
    """包含 Prompt、工具、observation、限制、Guard 与接受标准。"""
    return {
        "harness_version": HARNESS_VERSION,
        "visible_protocol": visible_protocol_payload(),
        "observation_schema": ToolObservation.model_json_schema(),
        "observation_version": TOOL_OBSERVATION_VERSION,
        "runtime": runtime.model_dump(mode="json"),
        "sql_guard_version": SQL_GUARD_VERSION,
        "acceptance_version": ACCEPTANCE_VERSION,
        "submission_rules": {
            "requires_describe_schema": True,
            "requires_execute_sql": True,
            "submit_matches_last_execute": True,
            "all_physical_tables_described": True,
            "all_observations_successful": True,
        },
    }


def harness_config_hash(runtime: RuntimeConfig) -> str:
    return hashlib.sha256(stable_json(harness_payload(runtime)).encode("utf-8")).hexdigest()
