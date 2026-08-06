import json
from collections import Counter
from pathlib import Path

import pytest

from nl2sql_rl.io_utils import read_jsonl


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
