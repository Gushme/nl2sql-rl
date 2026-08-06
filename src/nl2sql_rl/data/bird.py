"""BIRD train source loading and immutable inventory generation."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from pydantic import ConfigDict

from nl2sql_rl.io_utils import sha256_file
from nl2sql_rl.models import StrictRecord

EXPECTED_TRAIN_ROWS = 9_428
EXPECTED_TRAIN_DATABASES = 69


class BirdSourceExample(StrictRecord):
    model_config = ConfigDict(extra="forbid", frozen=True)

    source_index: int
    task_id: str
    db_id: str
    question: str
    evidence: str
    gold_sql: str
    gold_file_sql: str | None
    gold_file_db_id: str | None
    source_match: bool
    db_path: Path


def _parse_gold_line(line: str) -> tuple[str | None, str | None]:
    if "\t" not in line:
        return None, None
    sql, db_id = line.rsplit("\t", maxsplit=1)
    return sql, db_id.strip()


def load_train_examples(train_root: Path) -> list[BirdSourceExample]:
    source_path = train_root / "train.json"
    gold_path = train_root / "train_gold.sql"
    raw: Any = json.loads(source_path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"expected a JSON array: {source_path}")
    gold_lines = gold_path.read_text(encoding="utf-8").splitlines()
    if len(raw) != len(gold_lines):
        raise ValueError(f"source/gold length mismatch: {len(raw)} != {len(gold_lines)}")

    examples: list[BirdSourceExample] = []
    for index, (value, gold_line) in enumerate(zip(raw, gold_lines, strict=True)):
        if not isinstance(value, dict):
            raise ValueError(f"expected object at train.json index {index}")
        required = {"db_id", "question", "evidence", "SQL"}
        missing = required.difference(value)
        if missing:
            raise ValueError(f"missing {sorted(missing)} at train.json index {index}")
        db_id = str(value["db_id"])
        sql = str(value["SQL"])
        gold_sql, gold_db_id = _parse_gold_line(gold_line)
        examples.append(
            BirdSourceExample(
                source_index=index,
                task_id=f"bird_train_{index:06d}",
                db_id=db_id,
                question=str(value["question"]),
                evidence=str(value["evidence"]),
                gold_sql=sql,
                gold_file_sql=gold_sql,
                gold_file_db_id=gold_db_id,
                source_match=sql == gold_sql and db_id == gold_db_id,
                db_path=train_root / "train_databases" / db_id / f"{db_id}.sqlite",
            )
        )
    return examples


def build_train_inventory(train_root: Path) -> dict[str, Any]:
    examples = load_train_examples(train_root)
    db_counts = Counter(example.db_id for example in examples)
    database_root = train_root / "train_databases"
    database_paths = sorted(database_root.glob("*/*.sqlite"))
    databases: dict[str, dict[str, Any]] = {}
    for path in database_paths:
        db_id = path.stem
        databases[db_id] = {
            "path": str(path.relative_to(train_root)),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "task_count": db_counts.get(db_id, 0),
        }

    expected_db_ids = set(db_counts)
    actual_db_ids = set(databases)
    root_metadata = train_root / "train_tables.json"
    duplicate_metadata = database_root / "train_tables.json"
    report: dict[str, Any] = {
        "schema_version": 1,
        "dataset": "bird_train",
        "row_count": len(examples),
        "database_count": len(db_counts),
        "source_mismatch_count": sum(not example.source_match for example in examples),
        "missing_database_ids": sorted(expected_db_ids - actual_db_ids),
        "orphan_database_ids": sorted(actual_db_ids - expected_db_ids),
        "source_files": {
            name: {
                "bytes": (train_root / name).stat().st_size,
                "sha256": sha256_file(train_root / name),
            }
            for name in ("train.json", "train_gold.sql", "train_tables.json")
        },
        "metadata_duplicate": {
            "path": str(duplicate_metadata.relative_to(train_root)),
            "sha256": sha256_file(duplicate_metadata),
            "matches_root": sha256_file(root_metadata) == sha256_file(duplicate_metadata),
        },
        "databases": databases,
    }
    if report["row_count"] != EXPECTED_TRAIN_ROWS:
        raise ValueError(f"expected {EXPECTED_TRAIN_ROWS} train rows, got {report['row_count']}")
    if report["database_count"] != EXPECTED_TRAIN_DATABASES:
        raise ValueError(
            f"expected {EXPECTED_TRAIN_DATABASES} train db_ids, got {report['database_count']}"
        )
    if report["missing_database_ids"] or report["orphan_database_ids"]:
        raise ValueError("train database coverage mismatch")
    if not report["metadata_duplicate"]["matches_root"]:
        raise ValueError("duplicate train_tables.json does not match root metadata")
    return report
