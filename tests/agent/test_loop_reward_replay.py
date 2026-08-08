from __future__ import annotations

import json
from pathlib import Path
from typing import Any, ClassVar

import pytest

from nl2sql_rl.agent.fingerprint import harness_config_hash, harness_payload
from nl2sql_rl.agent.loop import (
    ModelResponse,
    ScriptedPolicy,
    build_actor_messages,
    run_episode,
)
from nl2sql_rl.agent.replay import replay_episode
from nl2sql_rl.agent.reward import score_terminal
from nl2sql_rl.config import RuntimeConfig
from nl2sql_rl.models import (
    AgentAction,
    AuditStatus,
    HiddenAnswer,
    TaskView,
    TerminalReason,
)
from nl2sql_rl.sqlite_exec import QueryExecution


def _task() -> TaskView:
    return TaskView(
        task_id="fixture_001",
        split="train",
        db_id="fixture",
        question="员工总数是多少？",
        evidence="",
        db_ref="agent.sqlite",
    )


def _answer() -> HiddenAnswer:
    return HiddenAnswer(
        task_id="fixture_001",
        gold_sql="SELECT COUNT(*) FROM employees",
        audit_status=AuditStatus.PASSED,
    )


def _execution(status: AuditStatus, digest: str | None = "digest") -> QueryExecution:
    return QueryExecution(
        status=status,
        digest=digest,
        row_count=1,
        unique_row_count=1,
        result_bytes=1,
        elapsed_ms=1,
        empty_result=False,
        result_too_large=False,
    )


def test_agent_can_recover_from_tool_error_and_submit_correct_sql(agent_db: Path) -> None:
    policy = ScriptedPolicy(
        [
            AgentAction(action="describe_schema", arguments={"tables": ["employees"]}),
            AgentAction(action="execute_sql", arguments={"sql": "SELECT missing FROM employees"}),
            AgentAction(
                action="execute_sql", arguments={"sql": "SELECT COUNT(*) FROM employees"}
            ),
            AgentAction(action="submit_sql", arguments={"sql": "SELECT COUNT(*) FROM employees"}),
        ]
    )
    episode = run_episode(
        _task(),
        _answer(),
        agent_db,
        policy,
        runtime=RuntimeConfig(),
        config_hash="fixture",
        episode_id="episode_fixture",
    )
    assert episode.terminal_reason is TerminalReason.SUBMITTED
    assert episode.reward == 1.0
    assert episode.events[1].observation.error_code == "missing_table_or_column"
    assert episode.events[2].observation.ok
    assert episode.events[3].observation.ok

    replay = replay_episode(
        episode, _task(), _answer(), agent_db, runtime=RuntimeConfig()
    )
    assert replay.exact_terminal_outcome
    assert replay.exact_event_replay


def test_three_identical_actions_terminate_as_loop(agent_db: Path) -> None:
    action = AgentAction(action="list_tables", arguments={})
    episode = run_episode(
        _task(),
        _answer(),
        agent_db,
        ScriptedPolicy([action, action, action]),
        runtime=RuntimeConfig(),
        config_hash="fixture",
    )
    assert episode.terminal_reason is TerminalReason.LOOP
    assert episode.reward == -0.4
    assert episode.events[-1].observation.error_code == "loop_detected"


def test_unsafe_action_terminates_immediately(agent_db: Path) -> None:
    episode = run_episode(
        _task(),
        _answer(),
        agent_db,
        ScriptedPolicy(
            [AgentAction(action="submit_sql", arguments={"sql": "DROP TABLE employees"})]
        ),
        runtime=RuntimeConfig(),
        config_hash="fixture",
    )
    assert episode.terminal_reason is TerminalReason.UNSAFE_SQL
    assert episode.reward == -1.0


def test_actor_context_never_contains_hidden_answer_or_reward(agent_db: Path) -> None:
    messages = build_actor_messages(_task())
    serialized = json.dumps(messages, ensure_ascii=False)
    assert _answer().gold_sql not in serialized
    assert "gold_sql" not in serialized
    assert "reward" not in serialized

    episode = run_episode(
        _task(),
        _answer(),
        agent_db,
        ScriptedPolicy([AgentAction(action="list_tables", arguments={})]),
        runtime=RuntimeConfig(max_episode_actions=1),
        config_hash="fixture",
    )
    rollout = episode.model_dump_json()
    assert _answer().gold_sql not in rollout
    assert "gold_sql" not in rollout


def test_policy_error_keeps_only_sanitized_diagnostic_detail(agent_db: Path) -> None:
    class DiagnosticError(RuntimeError):
        error_code = "model_or_request_error"
        request_sent = True
        diagnostic_detail: ClassVar[dict[str, str | int]] = {
            "status_code": 400,
            "provider_code": "invalid_parameter",
        }

    class FailingPolicy:
        def generate(
            self, messages: list[dict[str, Any]], *, max_tokens: int
        ) -> ModelResponse:
            del messages, max_tokens
            raise DiagnosticError("不得落盘的服务端消息")

    episode = run_episode(
        _task(),
        _answer(),
        agent_db,
        FailingPolicy(),
        runtime=RuntimeConfig(),
        config_hash="fixture",
    )
    assert episode.infrastructure_error_detail == {
        "status_code": 400,
        "provider_code": "invalid_parameter",
    }
    assert "不得落盘" not in episode.model_dump_json()


def test_harness_hash_covers_runtime_protocol_guard_and_acceptance() -> None:
    first = RuntimeConfig(exploration_timeout_seconds=5.0)
    second = RuntimeConfig(exploration_timeout_seconds=6.0)
    assert harness_config_hash(first) != harness_config_hash(second)
    payload = harness_payload(first)
    assert payload["visible_protocol"]["system_prompt"]
    assert payload["visible_protocol"]["tools"]
    assert payload["observation_schema"]
    assert payload["sql_guard_version"]
    assert payload["acceptance_version"] == 2


def test_full_event_replay_detects_observation_tampering(agent_db: Path) -> None:
    actions = [
        AgentAction(action="describe_schema", arguments={"tables": ["employees"]}),
        AgentAction(
            action="execute_sql", arguments={"sql": "SELECT COUNT(*) FROM employees"}
        ),
        AgentAction(action="submit_sql", arguments={"sql": "SELECT COUNT(*) FROM employees"}),
    ]
    episode = run_episode(
        _task(),
        _answer(),
        agent_db,
        ScriptedPolicy(actions),
        runtime=RuntimeConfig(),
        config_hash="fixture",
        episode_id="tamper",
    )
    first_event = episode.events[0]
    tampered_observation = first_event.observation.model_copy(
        update={"payload": {"schemas": []}}
    )
    tampered = episode.model_copy(
        update={
            "events": [
                first_event.model_copy(update={"observation": tampered_observation}),
                *episode.events[1:],
            ]
        }
    )
    replay = replay_episode(
        tampered,
        _task(),
        _answer(),
        agent_db,
        runtime=RuntimeConfig(),
    )
    assert replay.exact_terminal_outcome
    assert not replay.events_match
    assert replay.event_mismatches == (0,)
    assert not replay.exact_event_replay


@pytest.mark.parametrize(
    ("terminal", "prediction", "gold", "reward", "valid"),
    [
        (
            TerminalReason.SUBMITTED,
            _execution(AuditStatus.PASSED),
            _execution(AuditStatus.PASSED),
            1.0,
            True,
        ),
        (
            TerminalReason.SUBMITTED,
            _execution(AuditStatus.PASSED, "wrong"),
            _execution(AuditStatus.PASSED),
            0.0,
            True,
        ),
        (
            TerminalReason.SUBMITTED,
            _execution(AuditStatus.SYNTAX_ERROR, None),
            _execution(AuditStatus.PASSED),
            -0.2,
            True,
        ),
        (TerminalReason.LOOP, None, None, -0.4, True),
        (TerminalReason.MAX_ACTIONS, None, None, -0.4, True),
        (TerminalReason.UNSAFE_SQL, None, None, -1.0, True),
        (TerminalReason.INFRASTRUCTURE_ERROR, None, None, None, False),
        (
            TerminalReason.SUBMITTED,
            _execution(AuditStatus.PASSED),
            _execution(AuditStatus.TIMEOUT, None),
            None,
            False,
        ),
    ],
)
def test_reward_all_terminal_branches(
    terminal: TerminalReason,
    prediction: QueryExecution | None,
    gold: QueryExecution | None,
    reward: float | None,
    valid: bool,
) -> None:
    result = score_terminal(terminal, prediction=prediction, gold=gold)
    assert result.reward == reward
    assert result.valid_for_training is valid
