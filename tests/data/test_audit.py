import hashlib
import sqlite3
from pathlib import Path

from nl2sql_rl.data.audit import _run_workers, _WorkerState
from nl2sql_rl.data.bird import BirdSourceExample
from nl2sql_rl.models import AuditStatus
from nl2sql_rl.sqlite_exec import canonical_result_digest, canonical_row, execute_read_only


def _database(path: Path) -> None:
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE items(id INTEGER PRIMARY KEY, name TEXT, payload BLOB, score REAL);
        INSERT INTO items(name, payload, score) VALUES
          ('alpha', X'00FF', 1.5),
          ('beta', NULL, 2.5),
          ('alpha', X'00FF', 1.5);
        """
    )
    connection.close()


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_read_only_execution_uses_set_semantics_and_preserves_database(tmp_path: Path) -> None:
    db = tmp_path / "fixture.sqlite"
    _database(db)
    before = _sha(db)
    outcome = execute_read_only(
        "SELECT name FROM items",
        db,
        timeout_seconds=1,
    )
    after = _sha(db)

    assert outcome.status is AuditStatus.PASSED
    assert outcome.row_count == 3
    assert outcome.unique_row_count == 2
    assert before == after
    expected = {canonical_row(("alpha",)), canonical_row(("beta",))}
    assert outcome.digest == canonical_result_digest(expected)


def test_read_only_execution_classifies_failures(tmp_path: Path) -> None:
    db = tmp_path / "fixture.sqlite"
    _database(db)
    cases = {
        "SELECT FROM": AuditStatus.SYNTAX_ERROR,
        "SELECT missing FROM items": AuditStatus.MISSING_TABLE_OR_COLUMN,
        "SELECT made_up_function(name) FROM items": AuditStatus.UNSUPPORTED_FUNCTION,
        "DELETE FROM items": AuditStatus.UNSAFE_SQL,
        "PRAGMA table_info(items)": AuditStatus.UNSAFE_SQL,
        "SELECT 1; SELECT 2": AuditStatus.UNSAFE_SQL,
    }
    for sql, expected in cases.items():
        result = execute_read_only(sql, db, timeout_seconds=1)
        assert result.status is expected, (sql, result)


def test_read_only_execution_enforces_timeout(tmp_path: Path) -> None:
    db = tmp_path / "fixture.sqlite"
    _database(db)
    result = execute_read_only(
        "WITH RECURSIVE cnt(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM cnt) "
        "SELECT max(x) FROM cnt",
        db,
        timeout_seconds=0.02,
    )
    assert result.status is AuditStatus.TIMEOUT


def test_read_only_execution_flags_empty_and_large_results(tmp_path: Path) -> None:
    db = tmp_path / "fixture.sqlite"
    _database(db)
    empty = execute_read_only(
        "SELECT name FROM items WHERE 0", db, timeout_seconds=1
    )
    large = execute_read_only(
        "SELECT name FROM items", db, timeout_seconds=1, max_rows=1
    )
    assert empty.status is AuditStatus.PASSED and empty.empty_result
    assert large.status is AuditStatus.PASSED and large.result_too_large


def test_database_worker_supervisor_drains_all_process_messages(tmp_path: Path) -> None:
    states = []
    for db_index in range(2):
        db_id = f"fixture_{db_index}"
        db = tmp_path / f"{db_id}.sqlite"
        _database(db)
        examples = [
            BirdSourceExample(
                source_index=db_index * 2 + query_index,
                task_id=f"task_{db_index}_{query_index}",
                db_id=db_id,
                question="question",
                evidence="",
                gold_sql="SELECT name FROM items",
                gold_file_sql="SELECT name FROM items",
                gold_file_db_id=db_id,
                source_match=True,
                db_path=db,
            )
            for query_index in range(2)
        ]
        states.append(
            _WorkerState(
                db_id=db_id,
                db_path=db,
                examples=examples,
                db_sha256=_sha(db),
            )
        )

    records = []
    _run_workers(
        states,
        timeout_seconds=1,
        max_workers=2,
        max_rows=100,
        max_result_bytes=10_000,
        on_record=records.append,
    )

    assert len(records) == 4
    assert {record.task_id for record in records} == {
        "task_0_0",
        "task_0_1",
        "task_1_0",
        "task_1_1",
    }
    assert all(record.status is AuditStatus.PASSED for record in records)
