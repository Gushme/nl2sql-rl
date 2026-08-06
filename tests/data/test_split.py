from pathlib import Path

from nl2sql_rl.data.split import (
    build_splits_and_leakage_report,
    choose_validation_databases,
    normalize_question,
)
from nl2sql_rl.io_utils import read_jsonl, write_json, write_jsonl


def test_validation_split_is_deterministic_and_database_disjoint() -> None:
    counts = {f"db_{index}": 100 + index for index in range(12)}
    first = choose_validation_databases(counts, seed=42)
    second = choose_validation_databases(dict(reversed(list(counts.items()))), seed=42)
    assert first == second
    assert first
    assert set(first).issubset(counts)
    assert set(first) != set(counts)
    assert sum(counts[db_id] for db_id in first) >= 100


def test_question_normalization_removes_case_punctuation_and_spacing() -> None:
    assert normalize_question("  How MANY rows? ") == normalize_question("how many rows")


def test_split_writes_combined_teacher_pool_without_changing_split_labels(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "outputs"
    manifest_root = tmp_path / "manifests"
    tasks = [
        {
            "task_id": f"train_{index:03d}",
            "db_id": "train_a" if index < 100 else "train_b",
            "question": f"train question {index}",
            "split": "train_pool",
        }
        for index in range(200)
    ]
    answers = [{"task_id": task["task_id"], "gold_sql": "SELECT 1"} for task in tasks]
    dev_tasks = [
        {
            "task_id": "dev_1",
            "db_id": "dev_db",
            "question": "dev unique question",
            "split": "dev_final",
        }
    ]
    write_jsonl(output_root / "data/train/tasks/rl_clean.jsonl", tasks)
    write_jsonl(output_root / "data/train/answers/rl_clean.jsonl", answers)
    write_jsonl(output_root / "data/dev/tasks/final.jsonl", dev_tasks)
    write_json(
        manifest_root / "train_inventory.json",
        {
            "databases": {
                "train_a": {"sha256": "train-a"},
                "train_b": {"sha256": "train-b"},
            }
        },
    )
    write_json(
        manifest_root / "dev_inventory.json",
        {"databases": {"dev_db": {"sha256": "dev"}}},
    )

    manifest, leakage = build_splits_and_leakage_report(
        output_root, manifest_root, seed=42
    )
    pool = read_jsonl(output_root / "data/splits/tasks/teacher_pool.jsonl")
    answer_pool = read_jsonl(output_root / "data/splits/answers/teacher_pool.jsonl")
    assert leakage["hard_pass"] is True
    assert len(pool) == len(answer_pool) == 200
    assert {row["split"] for row in pool} == {"train", "validation"}
    assert manifest["validation_count"] >= 100
