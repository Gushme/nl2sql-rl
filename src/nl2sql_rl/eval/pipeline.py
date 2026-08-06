"""统一执行预测与 Gold，并生成 Final-N 评测记录。"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from pydantic import Field

from nl2sql_rl.eval.errors import classify_error
from nl2sql_rl.eval.executor import execute_rows
from nl2sql_rl.eval.metrics import (
    exact_execution_match,
    official_soft_f1,
    rves_metadata,
    rves_score_from_ratios,
)
from nl2sql_rl.models import (
    AuditStatus,
    EvaluationRecord,
    HiddenAnswer,
    StrictRecord,
    TaskView,
)


class PredictionRecord(StrictRecord):
    task_id: str
    prediction_sql: str | None = None
    usage: dict[str, int] = Field(default_factory=dict)


def _rves_ratios(
    prediction_sql: str,
    gold_sql: str,
    db_path: Path,
    *,
    iterations: int,
    timeout_seconds: float,
) -> list[float]:
    ratios: list[float] = []
    for _ in range(iterations):
        prediction = execute_rows(
            prediction_sql, db_path, timeout_seconds=timeout_seconds
        )
        gold = execute_rows(gold_sql, db_path, timeout_seconds=timeout_seconds)
        if prediction.status is not AuditStatus.PASSED or gold.status is not AuditStatus.PASSED:
            return []
        if prediction.elapsed_ms <= 0:
            return []
        ratios.append(gold.elapsed_ms / prediction.elapsed_ms)
    return ratios


def score_sql_pair(
    task: TaskView,
    answer: HiddenAnswer,
    prediction: PredictionRecord,
    db_path: Path,
    *,
    timeout_seconds: float = 10.0,
    rves_iterations: int = 0,
) -> EvaluationRecord:
    if task.task_id != answer.task_id or task.task_id != prediction.task_id:
        raise ValueError("Task、Answer 与 Prediction 的 task_id 必须一致")
    if prediction.prediction_sql is None or not prediction.prediction_sql.strip():
        return EvaluationRecord(
            task_id=task.task_id,
            db_id=task.db_id,
            prediction_sql=prediction.prediction_sql,
            ex=0.0,
            soft_f1=0.0,
            r_ves=0.0 if rves_iterations else None,
            prediction_status="missing_prediction",
            gold_status=answer.audit_status.value,
            error_type="protocol",
        )
    predicted = execute_rows(
        prediction.prediction_sql,
        db_path,
        timeout_seconds=timeout_seconds,
    )
    gold = execute_rows(answer.gold_sql, db_path, timeout_seconds=timeout_seconds)
    if gold.status is not AuditStatus.PASSED or gold.result_too_large:
        return EvaluationRecord(
            task_id=task.task_id,
            db_id=task.db_id,
            prediction_sql=prediction.prediction_sql,
            prediction_status=predicted.status.value,
            gold_status=gold.status.value,
            error_type="infrastructure",
            infrastructure_status="invalid_gold",
            details={"gold_error": gold.error, "gold_result_too_large": gold.result_too_large},
        )
    if predicted.status is AuditStatus.PASSED and not predicted.result_too_large:
        ex = exact_execution_match(predicted.rows, gold.rows)
        soft_f1 = official_soft_f1(predicted.rows, gold.rows)
    else:
        ex = 0.0
        soft_f1 = 0.0
    r_ves: float | None = None
    ratios: list[float] = []
    if rves_iterations:
        if ex == 1.0:
            ratios = _rves_ratios(
                prediction.prediction_sql,
                answer.gold_sql,
                db_path,
                iterations=rves_iterations,
                timeout_seconds=timeout_seconds,
            )
            r_ves = rves_score_from_ratios(ratios)
        else:
            r_ves = 0.0
    error_type = (
        None
        if ex == 1.0
        else classify_error(
            prediction.prediction_sql,
            answer.gold_sql,
            prediction_status=predicted.status,
        )
    )
    infrastructure_status = (
        "prediction_result_too_large" if predicted.result_too_large else "ok"
    )
    return EvaluationRecord(
        task_id=task.task_id,
        db_id=task.db_id,
        prediction_sql=prediction.prediction_sql,
        ex=ex,
        soft_f1=soft_f1,
        r_ves=r_ves,
        prediction_status=predicted.status.value,
        gold_status=gold.status.value,
        error_type=error_type,
        infrastructure_status=infrastructure_status,
        details={
            "prediction_rows": len(predicted.rows),
            "gold_rows": len(gold.rows),
            "prediction_elapsed_ms": predicted.elapsed_ms,
            "gold_elapsed_ms": gold.elapsed_ms,
            "rves_ratios": ratios,
        },
    )


def summarize_records(
    records: list[EvaluationRecord], *, official_count: int = 500, rves_iterations: int = 0
) -> dict[str, Any]:
    valid = [record for record in records if record.ex is not None]

    def mean(field: str) -> float | None:
        values = [getattr(record, field) for record in valid]
        numeric = [float(value) for value in values if value is not None]
        return sum(numeric) / len(numeric) if numeric else None

    return {
        "schema_version": 1,
        "official_count": official_count,
        "final_n": len(valid),
        "unverifiable_count": official_count - len(valid),
        "metrics": {
            "ex": mean("ex"),
            "soft_f1": mean("soft_f1"),
            "r_ves": mean("r_ves"),
        },
        "error_counts": dict(
            sorted(Counter(record.error_type for record in valid if record.error_type).items())
        ),
        "infrastructure_counts": dict(
            sorted(Counter(record.infrastructure_status for record in records).items())
        ),
        "rves_environment": rves_metadata(rves_iterations).as_dict(),
    }


def score_dataset(
    tasks: list[TaskView],
    answers: list[HiddenAnswer],
    predictions: list[PredictionRecord],
    db_root: Path,
    *,
    timeout_seconds: float = 10.0,
    rves_iterations: int = 0,
    official_count: int = 500,
) -> tuple[list[EvaluationRecord], dict[str, Any]]:
    answer_by_id = {answer.task_id: answer for answer in answers}
    prediction_by_id = {prediction.task_id: prediction for prediction in predictions}
    records: list[EvaluationRecord] = []
    for task in tasks:
        answer = answer_by_id.get(task.task_id)
        if answer is None:
            raise ValueError(f"缺少 HiddenAnswer：{task.task_id}")
        prediction = prediction_by_id.get(
            task.task_id, PredictionRecord(task_id=task.task_id, prediction_sql=None)
        )
        records.append(
            score_sql_pair(
                task,
                answer,
                prediction,
                db_root / task.db_ref,
                timeout_seconds=timeout_seconds,
                rves_iterations=rves_iterations,
            )
        )
    return records, summarize_records(
        records, official_count=official_count, rves_iterations=rves_iterations
    )
