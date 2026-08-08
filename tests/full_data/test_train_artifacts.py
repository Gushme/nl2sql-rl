import json
from collections import Counter
from pathlib import Path

import pytest

from nl2sql_rl.agent.fingerprint import harness_config_hash
from nl2sql_rl.config import load_project_config
from nl2sql_rl.io_utils import read_jsonl
from nl2sql_rl.models import HiddenAnswer, TaskView
from nl2sql_rl.teacher.sampling import (
    SAMPLING_VERSION,
    StratifiedScheduler,
    build_sampling_plan,
)


@pytest.mark.full_data
def test_full_train_audit_artifacts_are_complete_and_leak_free() -> None:
    manifest_root = Path("data/manifests")
    summary = json.loads(
        (manifest_root / "train_audit_summary.json").read_text(encoding="utf-8")
    )
    audit = read_jsonl(manifest_root / "train_audit.jsonl")
    tasks = read_jsonl(Path("outputs/data/train/tasks/rl_clean.jsonl"))
    answers = read_jsonl(Path("outputs/data/train/answers/rl_clean.jsonl"))

    assert summary["input_count"] == 9_428
    assert len(audit) == 9_428
    assert len({row["task_id"] for row in audit}) == 9_428
    assert sum(summary["status_counts"].values()) == 9_428
    assert Counter(row["status"] for row in audit) == Counter(summary["status_counts"])
    assert summary["database_hashes_unchanged"] is True
    assert len(tasks) == summary["rl_clean_count"]
    assert len(answers) == summary["rl_clean_count"]
    assert {row["task_id"] for row in tasks} == {row["task_id"] for row in answers}
    assert all("gold_sql" not in row and "reward" not in row for row in tasks)
    assert all(row["status"] == "passed" for row in audit if row["rl_clean"])


@pytest.mark.full_data
def test_teacher_pilot_prefix_preserves_split_and_complexity_ratios() -> None:
    tasks = [
        TaskView.model_validate(row)
        for row in read_jsonl(Path("outputs/data/splits/tasks/teacher_pool.jsonl"))
    ]
    answers = [
        HiddenAnswer.model_validate(row)
        for row in read_jsonl(Path("outputs/data/splits/answers/teacher_pool.jsonl"))
    ]
    plan = build_sampling_plan(
        tasks,
        answers,
        split_targets={"train": 900, "validation": 100},
        seed=42,
    )
    scheduler = StratifiedScheduler(plan, completed_ids=set(), accepted_ids=set())
    selected = scheduler.next_task_ids(100)
    task_by_id = {task.task_id: task for task in tasks}
    assert Counter(task_by_id[task_id].split for task_id in selected) == {
        "train": 90,
        "validation": 10,
    }
    assert Counter(plan.task_complexity[task_id].bucket.value for task_id in selected) == {
        "simple": 30,
        "moderate": 50,
        "challenging": 20,
    }
    train_targets = [
        target
        for (split, _), target in plan.database_targets.items()
        if split == "train"
    ]
    validation_targets = [
        target
        for (split, _), target in plan.database_targets.items()
        if split == "validation"
    ]
    assert len(train_targets) == 61 and min(train_targets) >= 3 and max(train_targets) <= 30
    assert len(validation_targets) == 8 and min(validation_targets) >= 8
    assert max(validation_targets) <= 17

    preflight = json.loads(
        Path("data/manifests/teacher_harness_preflight_summary.json").read_text(
            encoding="utf-8"
        )
    )
    runtime = load_project_config(Path("configs/project.yaml")).runtime
    assert preflight["model_api_called"] is False
    assert preflight["gold_sql_saved"] is False
    assert preflight["selected"] == preflight["passed"] == 100
    assert preflight["failed"] == 0
    assert preflight["database_hashes_unchanged"] is True
    assert preflight["sampling_version"] == SAMPLING_VERSION
    assert preflight["sampling_manifest_hash"] == plan.manifest_hash
    assert preflight["harness_config_hash"] == harness_config_hash(runtime)
