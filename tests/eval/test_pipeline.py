from __future__ import annotations

import hashlib
from pathlib import Path

from nl2sql_rl.eval.pipeline import PredictionRecord, score_dataset, score_sql_pair
from nl2sql_rl.models import AuditStatus, HiddenAnswer, TaskView


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _task(db_name: str) -> TaskView:
    return TaskView(
        task_id="eval_001",
        split="dev_final",
        db_id="evaluation",
        question="有哪些类别？",
        db_ref=db_name,
    )


def _answer() -> HiddenAnswer:
    return HiddenAnswer(
        task_id="eval_001",
        gold_sql="SELECT category FROM items",
        audit_status=AuditStatus.PASSED,
    )


def test_score_sql_pair_uses_set_semantics_and_preserves_database(
    evaluation_db: Path,
) -> None:
    before = _sha(evaluation_db)
    record = score_sql_pair(
        _task(evaluation_db.name),
        _answer(),
        PredictionRecord(
            task_id="eval_001",
            prediction_sql="SELECT DISTINCT category FROM items ORDER BY category DESC",
        ),
        evaluation_db,
    )
    assert record.ex == 1.0
    assert record.soft_f1 is not None
    assert record.error_type is None
    assert _sha(evaluation_db) == before


def test_score_sql_pair_reports_execution_and_ast_errors(evaluation_db: Path) -> None:
    syntax = score_sql_pair(
        _task(evaluation_db.name),
        _answer(),
        PredictionRecord(task_id="eval_001", prediction_sql="SELECT FROM"),
        evaluation_db,
    )
    wrong_filter = score_sql_pair(
        _task(evaluation_db.name),
        _answer(),
        PredictionRecord(
            task_id="eval_001",
            prediction_sql="SELECT category FROM items WHERE category = 'a'",
        ),
        evaluation_db,
    )
    assert syntax.ex == 0.0 and syntax.error_type == "syntax"
    assert wrong_filter.ex == 0.0 and wrong_filter.error_type == "filter_value"


def test_dataset_summary_keeps_official_and_final_denominators(evaluation_db: Path) -> None:
    task = _task(evaluation_db.name)
    records, summary = score_dataset(
        [task],
        [_answer()],
        [PredictionRecord(task_id=task.task_id, prediction_sql="SELECT category FROM items")],
        evaluation_db.parent,
        official_count=500,
    )
    assert len(records) == 1
    assert summary["official_count"] == 500
    assert summary["final_n"] == 1
    assert summary["unverifiable_count"] == 499
