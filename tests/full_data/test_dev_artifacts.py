from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from nl2sql_rl.io_utils import read_jsonl, sha256_file


@pytest.mark.full_data
def test_full_dev_audit_split_and_leakage_artifacts_are_complete() -> None:
    manifest_root = Path("data/manifests")
    output_root = Path("outputs/data")
    dev_root = Path("dev500")
    inventory = json.loads((manifest_root / "dev_inventory.json").read_text(encoding="utf-8"))
    summary = json.loads(
        (manifest_root / "dev_audit_summary.json").read_text(encoding="utf-8")
    )
    split = json.loads((manifest_root / "split_manifest.json").read_text(encoding="utf-8"))
    leakage = json.loads((manifest_root / "leakage_report.json").read_text(encoding="utf-8"))
    audit = read_jsonl(manifest_root / "dev_audit.jsonl")
    dev_tasks = read_jsonl(output_root / "dev/tasks/final.jsonl")
    dev_answers = read_jsonl(output_root / "dev/answers/final.jsonl")
    train_tasks = read_jsonl(output_root / "splits/tasks/train.jsonl")
    validation_tasks = read_jsonl(output_root / "splits/tasks/validation.jsonl")
    train_answers = read_jsonl(output_root / "splits/answers/train.jsonl")
    validation_answers = read_jsonl(output_root / "splits/answers/validation.jsonl")

    assert inventory["row_count"] == 500
    assert inventory["database_count"] == 11
    assert inventory["database_source"] == "local"
    assert len(inventory["databases"]) == 11
    assert inventory["package_annotation_cross_check"]["metadata_mismatch_count"] == 0
    assert inventory["package_annotation_cross_check"]["sql_mismatch_count"] == 1
    assert inventory["gold_cross_check"]["mismatch_count"] == 4
    for metadata in inventory["databases"].values():
        database = dev_root / metadata["path"]
        source = dev_root / metadata["source_path"]
        assert database.is_file() and source.is_file()
        assert sha256_file(database) == metadata["sha256"]
        assert sha256_file(source) == metadata["sha256"]

    assert len(audit) == 500
    assert len({row["task_id"] for row in audit}) == 500
    assert sum(summary["status_counts"].values()) == 500
    assert Counter(row["status"] for row in audit) == Counter(summary["status_counts"])
    assert summary["status_counts"].get("missing_database", 0) == 0
    assert summary["status_counts"].get("infrastructure_error", 0) == 0
    assert summary["final_n"] + summary["unverifiable_count"] == 500
    assert summary["database_hashes_unchanged"] is True
    assert len(dev_tasks) == len(dev_answers) == summary["final_n"]
    assert {row["task_id"] for row in dev_tasks} == {
        row["task_id"] for row in dev_answers
    }
    assert all(
        key not in task
        for task in dev_tasks
        for key in ("gold_sql", "reward", "audit_status", "result_cache_ref")
    )

    assert len(train_tasks) == len(train_answers) == split["train_count"] == 7_667
    assert (
        len(validation_tasks)
        == len(validation_answers)
        == split["validation_count"]
        == 918
    )
    assert split["validation_count"] >= 100
    assert split["dev_final_count"] == summary["final_n"]
    assert set(split["validation_db_ids"]) == {
        "cs_semester",
        "simpson_episodes",
        "bike_share_1",
        "music_tracker",
        "airline",
        "authors",
        "donor",
        "address",
    }
    train_db_ids = {row["db_id"] for row in train_tasks}
    validation_db_ids = {row["db_id"] for row in validation_tasks}
    dev_db_ids = {row["db_id"] for row in dev_tasks}
    assert train_db_ids.isdisjoint(validation_db_ids)
    assert train_db_ids.isdisjoint(dev_db_ids)
    assert validation_db_ids.isdisjoint(dev_db_ids)

    assert leakage["hard_pass"] is True
    assert leakage["db_id_overlap"] == []
    assert leakage["database_sha256_overlap"] == []
    assert leakage["task_id_overlap"] == []
    assert leakage["exact_question_overlap"] == []
    assert leakage["near_question_pairs"] == []
