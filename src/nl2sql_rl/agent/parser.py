"""严格 JSON Action 解析与规范化。"""

from __future__ import annotations

import json

from pydantic import ValidationError

from nl2sql_rl.io_utils import stable_json
from nl2sql_rl.models import AgentAction


class ActionParseError(ValueError):
    """模型输出不满足严格 Action 协议。"""


def parse_action(content: str) -> AgentAction:
    if not content.strip():
        raise ActionParseError("Action 不能为空")
    try:
        raw = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ActionParseError(f"Action 不是合法 JSON：{exc.msg}") from exc
    if not isinstance(raw, dict):
        raise ActionParseError("Action 顶层必须是 JSON 对象")
    try:
        return AgentAction.model_validate(raw)
    except ValidationError as exc:
        raise ActionParseError(f"Action schema 不合法：{exc.errors(include_url=False)}") from exc


def normalized_action(action: AgentAction) -> str:
    return stable_json(action.model_dump(mode="json"))
