"""严格 Action 协议驱动的多步 Agent 循环。"""

from __future__ import annotations

import hashlib
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from nl2sql_rl.agent.parser import ActionParseError, normalized_action, parse_action
from nl2sql_rl.agent.reward import score_terminal
from nl2sql_rl.agent.tools import SQLiteToolbox
from nl2sql_rl.config import RuntimeConfig
from nl2sql_rl.io_utils import stable_json
from nl2sql_rl.models import (
    AgentAction,
    AuditStatus,
    EpisodeResult,
    HiddenAnswer,
    TaskView,
    TerminalReason,
    ToolObservation,
    TrajectoryEvent,
)
from nl2sql_rl.sqlite_exec import execute_read_only

SYSTEM_PROMPT = """你是一个 SQLite NL2SQL Agent。每一轮只输出一个 JSON 对象：
{"action":"工具名","arguments":{...}}
工具仅允许 list_tables、describe_schema、search_values、execute_sql、submit_sql。
先检查 schema，必要时执行候选 SQL；确认后用 submit_sql 结束。不要输出 Markdown 或解释。"""


@dataclass(frozen=True)
class ModelResponse:
    content: str
    usage: dict[str, int] = field(default_factory=dict)


class ActionPolicy(Protocol):
    def generate(
        self, messages: list[dict[str, Any]], *, max_tokens: int
    ) -> ModelResponse: ...


class ScriptedPolicy:
    """用于测试、回放和 CPU 演练的确定性 Action 提供器。"""

    def __init__(self, responses: Sequence[str | AgentAction]) -> None:
        self._responses = list(responses)
        self._index = 0

    def generate(self, messages: list[dict[str, Any]], *, max_tokens: int) -> ModelResponse:
        del messages, max_tokens
        if self._index >= len(self._responses):
            return ModelResponse(content="")
        value = self._responses[self._index]
        self._index += 1
        content = (
            stable_json(value.model_dump(mode="json"))
            if isinstance(value, AgentAction)
            else value
        )
        return ModelResponse(content=content, usage={"output_tokens": 1})


def build_actor_messages(task: TaskView) -> list[dict[str, Any]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": stable_json(task.actor_payload())},
    ]


def _merge_usage(total: dict[str, int], update: dict[str, int]) -> None:
    for key, value in update.items():
        total[key] = total.get(key, 0) + int(value)


def _protocol_event(event_id: str, step: int, message: str) -> TrajectoryEvent:
    observation = ToolObservation(
        event_id=event_id,
        tool="protocol",
        ok=False,
        payload={"message": message[:500]},
        error_code="invalid_action",
    )
    return TrajectoryEvent(event_id=event_id, step=step, action=None, observation=observation)


def _append_actor_event(
    messages: list[dict[str, Any]], event: TrajectoryEvent, action: AgentAction | None
) -> None:
    if action is not None:
        messages.append(
            {"role": "assistant", "content": stable_json(action.model_dump(mode="json"))}
        )
    messages.append(
        {
            "role": "tool",
            "event_id": event.event_id,
            "name": event.observation.tool,
            "content": stable_json(event.observation.model_dump(mode="json")),
        }
    )


def run_episode(
    task: TaskView,
    answer: HiddenAnswer,
    db_path: Path,
    policy: ActionPolicy,
    *,
    runtime: RuntimeConfig,
    config_hash: str,
    episode_id: str | None = None,
) -> EpisodeResult:
    if task.task_id != answer.task_id:
        raise ValueError("TaskView 与 HiddenAnswer 的 task_id 不一致")
    resolved_episode_id = episode_id or f"episode_{uuid.uuid4().hex}"
    messages = build_actor_messages(task)
    events: list[TrajectoryEvent] = []
    usage: dict[str, int] = {}
    terminal_reason = TerminalReason.MAX_ACTIONS
    submitted_sql: str | None = None
    toolbox = SQLiteToolbox(
        db_path,
        exploration_timeout_seconds=runtime.exploration_timeout_seconds,
        submission_timeout_seconds=runtime.gold_timeout_seconds,
        max_observation_bytes=runtime.max_observation_bytes,
    )
    repeated_key: str | None = None
    repeated_count = 0

    if not db_path.is_file():
        return EpisodeResult(
            episode_id=resolved_episode_id,
            task_id=task.task_id,
            terminal_reason=TerminalReason.INFRASTRUCTURE_ERROR,
            reward=None,
            valid_for_training=False,
            config_hash=config_hash,
        )

    for step in range(runtime.max_episode_actions):
        event_id = f"{resolved_episode_id}:{step:02d}"
        try:
            response = policy.generate(messages, max_tokens=runtime.max_action_tokens)
        except Exception:
            terminal_reason = TerminalReason.INFRASTRUCTURE_ERROR
            break
        _merge_usage(usage, response.usage)
        if response.usage.get("output_tokens", 0) > runtime.max_action_tokens:
            event = _protocol_event(event_id, step, "Action 超过 token 上限")
            events.append(event)
            _append_actor_event(messages, event, None)
            continue
        if response.usage.get("context_tokens", 0) > runtime.max_context_tokens:
            terminal_reason = TerminalReason.INFRASTRUCTURE_ERROR
            break
        try:
            action = parse_action(response.content)
        except ActionParseError as exc:
            event = _protocol_event(event_id, step, str(exc))
            events.append(event)
            _append_actor_event(messages, event, None)
            key = "invalid:" + hashlib.sha256(response.content.encode()).hexdigest()
            repeated_count = repeated_count + 1 if key == repeated_key else 1
            repeated_key = key
            if repeated_count >= 3:
                terminal_reason = TerminalReason.LOOP
                break
            continue

        key = normalized_action(action)
        repeated_count = repeated_count + 1 if key == repeated_key else 1
        repeated_key = key
        if repeated_count >= 3:
            observation = ToolObservation(
                event_id=event_id,
                tool=action.action,
                ok=False,
                payload={"message": "连续三次相同 Action，已终止循环"},
                error_code="loop_detected",
            )
            event = TrajectoryEvent(
                event_id=event_id, step=step, action=action, observation=observation
            )
            events.append(event)
            _append_actor_event(messages, event, action)
            terminal_reason = TerminalReason.LOOP
            break

        observation = toolbox.call(action, event_id=event_id)
        event = TrajectoryEvent(
            event_id=event_id,
            step=step,
            action=action,
            observation=observation,
        )
        events.append(event)
        _append_actor_event(messages, event, action)
        if observation.error_code == AuditStatus.UNSAFE_SQL.value:
            terminal_reason = TerminalReason.UNSAFE_SQL
            break
        if action.action == "submit_sql":
            sql_value = action.arguments.get("sql")
            submitted_sql = sql_value if isinstance(sql_value, str) else None
            terminal_reason = TerminalReason.SUBMITTED
            break

    if terminal_reason is TerminalReason.SUBMITTED:
        gold_execution = execute_read_only(
            answer.gold_sql,
            db_path,
            timeout_seconds=runtime.gold_timeout_seconds,
        )
        decision = score_terminal(
            terminal_reason,
            prediction=toolbox.last_submission,
            gold=gold_execution,
        )
    else:
        decision = score_terminal(terminal_reason)
    return EpisodeResult(
        episode_id=resolved_episode_id,
        task_id=task.task_id,
        terminal_reason=terminal_reason,
        events=events,
        submitted_sql=submitted_sql,
        reward=decision.reward,
        valid_for_training=decision.valid_for_training,
        usage=usage,
        config_hash=config_hash,
    )
