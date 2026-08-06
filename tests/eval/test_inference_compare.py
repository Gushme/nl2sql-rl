from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import pytest

from nl2sql_rl.config import RuntimeConfig
from nl2sql_rl.eval.inference import collect_inference_episodes
from nl2sql_rl.eval.report import render_comparison_report
from nl2sql_rl.models import (
    AgentAction,
    AuditStatus,
    EvaluationRecord,
    HiddenAnswer,
    TaskView,
)
from nl2sql_rl.teacher.client import LLMClientConfig, LLMCompletion


class FakeInferenceClient:
    real_api = False
    spent_usd = 0.0

    def __init__(self) -> None:
        self.config = LLMClientConfig(
            endpoint="https://mock.local/v1",
            model="fixture-model",
            real_api=False,
            retry_base_seconds=0,
        )
        self.calls = 0

    @property
    def config_hash(self) -> str:
        return self.config.fingerprint()

    async def complete_action(
        self, messages: list[dict[str, Any]], *, max_tokens: int | None = None
    ) -> LLMCompletion:
        assert max_tokens == 512
        self.calls += 1
        if messages[-1]["role"] == "tool":
            action = AgentAction(
                action="submit_sql",
                arguments={"sql": "SELECT COUNT(*) FROM items"},
            )
        else:
            action = AgentAction(action="list_tables", arguments={})
        return LLMCompletion(
            action=action,
            response_id=f"mock_{self.calls}",
            input_tokens=10,
            output_tokens=5,
            cost_usd=0.0,
        )


def _fixtures(root: Path) -> tuple[list[TaskView], list[HiddenAnswer]]:
    tasks: list[TaskView] = []
    answers: list[HiddenAnswer] = []
    for index in range(2):
        db_id = f"db_{index}"
        relative = Path(db_id) / f"{db_id}.sqlite"
        database = root / relative
        database.parent.mkdir(parents=True)
        with sqlite3.connect(database) as connection:
            connection.executescript(
                "CREATE TABLE items(id INTEGER); INSERT INTO items VALUES (1), (2);"
            )
        task = TaskView(
            task_id=f"task_{index}",
            split="dev",
            db_id=db_id,
            question="有多少条记录？",
            db_ref=str(relative),
        )
        tasks.append(task)
        answers.append(
            HiddenAnswer(
                task_id=task.task_id,
                gold_sql="SELECT COUNT(*) FROM items",
                audit_status=AuditStatus.PASSED,
            )
        )
    return tasks, answers


@pytest.mark.asyncio
async def test_inference_collects_every_task_and_resumes_idempotently(
    tmp_path: Path,
) -> None:
    tasks, answers = _fixtures(tmp_path)
    client = FakeInferenceClient()
    output = tmp_path / "inference.jsonl"
    first, summary = await collect_inference_episodes(
        tasks,
        answers,
        tmp_path,
        output,
        client,
        runtime=RuntimeConfig(),
        model_label="base",
        concurrency=2,
    )
    second, _ = await collect_inference_episodes(
        tasks,
        answers,
        tmp_path,
        output,
        client,
        runtime=RuntimeConfig(),
        model_label="base",
        concurrency=2,
    )
    assert len(first) == len(second) == 2
    assert client.calls == 4
    assert all(record.episode.reward == 1.0 for record in first)
    assert summary["submitted_count"] == 2
    assert len(summary["comparison_condition_hash"]) == 64


@pytest.mark.asyncio
async def test_inference_real_api_requires_confirmation(tmp_path: Path) -> None:
    tasks, answers = _fixtures(tmp_path)
    client = FakeInferenceClient()
    client.real_api = True
    with pytest.raises(PermissionError, match="显式"):
        await collect_inference_episodes(
            tasks,
            answers,
            tmp_path,
            tmp_path / "inference.jsonl",
            client,
            runtime=RuntimeConfig(),
            model_label="base",
        )


def _record(task_id: str, ex: float) -> EvaluationRecord:
    return EvaluationRecord(
        task_id=task_id,
        db_id="fixture",
        prediction_sql="SELECT 1",
        ex=ex,
        soft_f1=ex,
        prediction_status="passed",
        gold_status="passed",
    )


def test_comparison_requires_identical_tasks_and_inference_conditions() -> None:
    runs = {
        "base": [_record("task_1", 0.0), _record("task_2", 1.0)],
        "sft": [_record("task_1", 1.0), _record("task_2", 1.0)],
        "grpo": [_record("task_1", 1.0), _record("task_2", 1.0)],
    }
    hashes = {label: "a" * 64 for label in runs}
    report = render_comparison_report(runs, hashes, official_count=2, project_git_sha="sha")
    assert "| BASE | 2 | 50.00%" in report
    assert "+50.00 pp" in report

    mismatched = dict(hashes)
    mismatched["grpo"] = "b" * 64
    with pytest.raises(ValueError, match="推理条件"):
        render_comparison_report(runs, mismatched, official_count=2)
