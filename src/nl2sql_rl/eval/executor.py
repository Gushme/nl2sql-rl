"""为官方兼容评测保留原始 SQLite 行的只读执行器。"""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nl2sql_rl.models import AuditStatus
from nl2sql_rl.sqlite_exec import (
    UnsafeSQLError,
    canonical_row,
    classify_sqlite_error,
    readonly_authorizer,
    validate_read_only_sql,
)


@dataclass(frozen=True)
class RowExecution:
    status: AuditStatus
    rows: tuple[tuple[Any, ...], ...]
    columns: tuple[str, ...]
    elapsed_ms: float
    result_bytes: int
    result_too_large: bool
    error: str | None = None


def execute_rows(
    sql: str,
    db_path: Path,
    *,
    timeout_seconds: float,
    max_rows: int = 100_000,
    max_result_bytes: int = 64 * 1024 * 1024,
) -> RowExecution:
    started = time.perf_counter()
    deadline = time.monotonic() + timeout_seconds
    deadline_hit = False
    try:
        validate_read_only_sql(sql)
    except UnsafeSQLError as exc:
        return RowExecution(
            AuditStatus.UNSAFE_SQL, (), (), 0.0, 0, False, str(exc)[:500]
        )
    except ValueError as exc:
        return RowExecution(
            AuditStatus.SYNTAX_ERROR, (), (), 0.0, 0, False, str(exc)[:500]
        )
    if not db_path.is_file():
        return RowExecution(
            AuditStatus.INFRASTRUCTURE_ERROR,
            (),
            (),
            0.0,
            0,
            False,
            f"数据库不存在：{db_path}",
        )
    connection: sqlite3.Connection | None = None
    try:
        uri = f"{db_path.resolve().as_uri()}?mode=ro"
        connection = sqlite3.connect(uri, uri=True, timeout=1.0)
        connection.enable_load_extension(False)
        connection.execute("PRAGMA query_only = ON")
        connection.set_authorizer(readonly_authorizer)

        def progress() -> int:
            nonlocal deadline_hit
            if time.monotonic() >= deadline:
                deadline_hit = True
                return 1
            return 0

        connection.set_progress_handler(progress, 1_000)
        cursor = connection.execute(sql)
        columns = tuple(str(item[0]) for item in (cursor.description or ()))
        rows: list[tuple[Any, ...]] = []
        result_bytes = 0
        too_large = False
        while batch := cursor.fetchmany(1_000):
            for raw_row in batch:
                row = tuple(raw_row)
                rows.append(row)
                result_bytes += len(canonical_row(row).encode("utf-8"))
                if len(rows) > max_rows or result_bytes > max_result_bytes:
                    too_large = True
                    break
            if too_large:
                break
        return RowExecution(
            AuditStatus.PASSED,
            tuple(rows),
            columns,
            (time.perf_counter() - started) * 1_000,
            result_bytes,
            too_large,
        )
    except sqlite3.Error as exc:
        return RowExecution(
            classify_sqlite_error(str(exc), deadline_hit=deadline_hit),
            (),
            (),
            (time.perf_counter() - started) * 1_000,
            0,
            False,
            str(exc)[:500],
        )
    except Exception as exc:
        return RowExecution(
            AuditStatus.INFRASTRUCTURE_ERROR,
            (),
            (),
            (time.perf_counter() - started) * 1_000,
            0,
            False,
            f"{type(exc).__name__}: {exc}"[:500],
        )
    finally:
        if connection is not None:
            connection.close()
