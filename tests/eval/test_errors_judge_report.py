from __future__ import annotations

import json

import pytest

from nl2sql_rl.eval.behavior import behavior_metrics
from nl2sql_rl.eval.errors import classify_error, classify_error_with_fallback
from nl2sql_rl.eval.judge import (
    BehaviorJudgeScore,
    build_blind_payload,
    validate_judge_score,
)
from nl2sql_rl.eval.report import render_report
from nl2sql_rl.models import (
    AgentAction,
    AuditStatus,
    EpisodeResult,
    EvaluationRecord,
    TaskView,
    TerminalReason,
    ToolObservation,
    TrajectoryEvent,
)


def _task() -> TaskView:
    return TaskView(
        task_id="judge_001",
        split="dev_final",
        db_id="fixture",
        question="问题",
        db_ref="fixture.sqlite",
    )


def _episode() -> EpisodeResult:
    action = AgentAction(action="submit_sql", arguments={"sql": "SELECT 1"})
    observation = ToolObservation(
        event_id="event_0",
        tool="submit_sql",
        ok=True,
        payload={"status": "passed"},
    )
    return EpisodeResult(
        episode_id="episode_0",
        task_id="judge_001",
        terminal_reason=TerminalReason.SUBMITTED,
        events=[
            TrajectoryEvent(
                event_id="event_0", step=0, action=action, observation=observation
            )
        ],
        submitted_sql="SELECT 1",
        reward=1.0,
        config_hash="fixture",
    )


@pytest.mark.parametrize(
    ("prediction", "gold", "expected"),
    [
        ("SELECT b FROM t", "SELECT a FROM t", "schema_linking"),
        ("SELECT a FROM t WHERE a = 1", "SELECT a FROM t WHERE a = 2", "filter_value"),
        ("SELECT COUNT(a) FROM t", "SELECT SUM(a) FROM t", "aggregation_grouping"),
        ("SELECT a FROM t ORDER BY a", "SELECT a FROM t", "order_limit"),
        (
            "SELECT a FROM t UNION SELECT a FROM u",
            "SELECT a FROM t INTERSECT SELECT a FROM u",
            "set_operation",
        ),
    ],
)
def test_ast_error_taxonomy(prediction: str, gold: str, expected: str) -> None:
    assert classify_error(prediction, gold, prediction_status=AuditStatus.PASSED) == expected


def test_llm_error_fallback_runs_only_for_other_semantic() -> None:
    class Fallback:
        calls = 0

        def classify(self, prediction_sql: str, gold_sql: str) -> str:
            del prediction_sql, gold_sql
            self.calls += 1
            return "join"

    fallback = Fallback()
    deterministic = classify_error_with_fallback(
        "SELECT a FROM t WHERE a = 1",
        "SELECT a FROM t WHERE a = 2",
        prediction_status=AuditStatus.PASSED,
        fallback=fallback,
    )
    ambiguous = classify_error_with_fallback(
        "SELECT a + 1 FROM t",
        "SELECT a + 2 FROM t",
        prediction_status=AuditStatus.PASSED,
        fallback=fallback,
    )
    assert deterministic == "filter_value"
    assert ambiguous == "join"
    assert fallback.calls == 1


def test_blind_judge_payload_excludes_correctness_and_validates_event_citations() -> None:
    episode = _episode()
    payload = build_blind_payload(_task(), episode)
    serialized = json.dumps(payload, ensure_ascii=False)
    assert "reward" not in serialized
    assert "gold" not in serialized.lower()
    assert "ex" not in payload
    valid = BehaviorJudgeScore(
        schema_grounding=3,
        tool_efficiency=4,
        error_recovery=3,
        submission_discipline=4,
        event_citations=["event_0"],
        rationale="引用了提交事件。",
    )
    validate_judge_score(valid, episode)
    invalid = valid.model_copy(update={"event_citations": ["missing"]})
    with pytest.raises(ValueError, match="不存在"):
        validate_judge_score(invalid, episode)


def test_behavior_metrics_and_report_are_deterministic() -> None:
    episode = _episode()
    metrics = behavior_metrics([episode])
    assert metrics["successful_submission_rate"] == 1.0
    assert metrics["average_steps"] == 1.0
    record = EvaluationRecord(
        task_id="judge_001",
        db_id="fixture",
        prediction_sql="SELECT 1",
        ex=1.0,
        soft_f1=1.0,
        prediction_status="passed",
        gold_status="passed",
    )
    report = render_report([record], [episode], official_count=1, project_git_sha="abc")
    assert "EX | 100.00%" in report
    assert "SQL 执行器是唯一正确性裁判" in report
    assert "项目 Git SHA：`abc`" in report
