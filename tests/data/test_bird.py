import json
from pathlib import Path

from nl2sql_rl.data.bird import load_train_examples


def test_load_train_examples_cross_checks_gold_file(tmp_path: Path) -> None:
    train_root = tmp_path / "train"
    train_root.mkdir()
    rows = [
        {"db_id": "db", "question": "q1", "evidence": "", "SQL": "SELECT 1"},
        {"db_id": "db", "question": "q2", "evidence": "", "SQL": "SELECT 2"},
    ]
    (train_root / "train.json").write_text(json.dumps(rows), encoding="utf-8")
    (train_root / "train_gold.sql").write_text(
        "SELECT 1\tdb\nSELECT wrong\tdb\n", encoding="utf-8"
    )

    examples = load_train_examples(train_root)

    assert examples[0].task_id == "bird_train_000000"
    assert examples[0].source_match
    assert not examples[1].source_match
    assert examples[1].gold_sql == "SELECT 2"
