"""Read-only SQLite execution primitives shared by cleaning, harness, and evaluation."""

from __future__ import annotations

import base64
import hashlib
import json
import math
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sqlglot import exp, parse
from sqlglot.errors import ParseError

from nl2sql_rl.models import AuditStatus

_DENIED_AUTHORIZER_ACTIONS = {
    getattr(sqlite3, name)
    for name in (
        "SQLITE_INSERT",
        "SQLITE_UPDATE",
        "SQLITE_DELETE",
        "SQLITE_CREATE_INDEX",
        "SQLITE_CREATE_TABLE",
        "SQLITE_CREATE_TEMP_INDEX",
        "SQLITE_CREATE_TEMP_TABLE",
        "SQLITE_CREATE_TEMP_TRIGGER",
        "SQLITE_CREATE_TEMP_VIEW",
        "SQLITE_CREATE_TRIGGER",
        "SQLITE_CREATE_VIEW",
        "SQLITE_DROP_INDEX",
        "SQLITE_DROP_TABLE",
        "SQLITE_DROP_TEMP_INDEX",
        "SQLITE_DROP_TEMP_TABLE",
        "SQLITE_DROP_TEMP_TRIGGER",
        "SQLITE_DROP_TEMP_VIEW",
        "SQLITE_DROP_TRIGGER",
        "SQLITE_DROP_VIEW",
        "SQLITE_ALTER_TABLE",
        "SQLITE_REINDEX",
        "SQLITE_ANALYZE",
        "SQLITE_PRAGMA",
        "SQLITE_ATTACH",
        "SQLITE_DETACH",
    )
    if hasattr(sqlite3, name)
}


class UnsafeSQLError(ValueError):
    """Raised before execution when SQL is not a single read-only query."""


@dataclass(frozen=True)
class QueryExecution:
    status: AuditStatus
    digest: str | None
    row_count: int
    unique_row_count: int
    result_bytes: int
    elapsed_ms: float
    empty_result: bool
    result_too_large: bool
    error: str | None = None


def validate_read_only_sql(sql: str) -> None:
    if not sql.strip():
        raise UnsafeSQLError("SQL is empty")
    try:
        statements = parse(sql, read="sqlite")
    except ParseError as exc:
        raise ValueError(str(exc)) from exc
    if len(statements) != 1:
        raise UnsafeSQLError("exactly one SQL statement is required")
    statement = statements[0]
    if statement is None or not isinstance(statement, exp.Query):
        statement_name = "unknown" if statement is None else statement.key
        raise UnsafeSQLError(f"only SELECT/CTE queries are allowed, got {statement_name}")
    forbidden_names = (
        "Insert",
        "Update",
        "Delete",
        "Create",
        "Drop",
        "Alter",
        "Attach",
        "Detach",
        "Pragma",
        "Command",
        "Transaction",
        "Merge",
        "Copy",
    )
    forbidden_types = tuple(
        node_type
        for name in forbidden_names
        if isinstance((node_type := getattr(exp, name, None)), type)
    )
    if forbidden_types and any(statement.find(node_type) for node_type in forbidden_types):
        raise UnsafeSQLError("query contains a forbidden operation")


def _typed_value(value: Any) -> list[Any]:
    if value is None:
        return ["null", None]
    if isinstance(value, bytes):
        return ["bytes", base64.b64encode(value).decode("ascii")]
    if isinstance(value, int):
        return ["int", str(value)]
    if isinstance(value, float):
        if math.isnan(value):
            encoded = "nan"
        elif math.isinf(value):
            encoded = "+inf" if value > 0 else "-inf"
        else:
            encoded = value.hex()
        return ["float", encoded]
    if isinstance(value, str):
        return ["str", value]
    return [type(value).__name__, repr(value)]


def canonical_row(row: tuple[Any, ...]) -> str:
    return json.dumps(
        [_typed_value(value) for value in row],
        ensure_ascii=False,
        separators=(",", ":"),
    )


def canonical_result_digest(rows: set[str]) -> str:
    digest = hashlib.sha256()
    for row in sorted(rows):
        encoded = row.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def classify_sqlite_error(message: str, *, deadline_hit: bool = False) -> AuditStatus:
    lowered = message.lower()
    if deadline_hit or "interrupted" in lowered:
        return AuditStatus.TIMEOUT
    if "no such function" in lowered:
        return AuditStatus.UNSUPPORTED_FUNCTION
    if "no such table" in lowered or "no such column" in lowered:
        return AuditStatus.MISSING_TABLE_OR_COLUMN
    if "syntax error" in lowered or "incomplete input" in lowered or lowered.startswith("near "):
        return AuditStatus.SYNTAX_ERROR
    if "not authorized" in lowered or "authorization denied" in lowered:
        return AuditStatus.UNSAFE_SQL
    return AuditStatus.SQLITE_ERROR


def _authorizer(
    action: int,
    argument_1: str | None,
    argument_2: str | None,
    _database: str | None,
    _trigger: str | None,
) -> int:
    if action in _DENIED_AUTHORIZER_ACTIONS:
        return sqlite3.SQLITE_DENY
    if action == sqlite3.SQLITE_FUNCTION:
        function_name = (argument_2 or argument_1 or "").lower()
        if function_name == "load_extension":
            return sqlite3.SQLITE_DENY
    return sqlite3.SQLITE_OK


def execute_read_only(
    sql: str,
    db_path: Path,
    *,
    timeout_seconds: float,
    max_rows: int = 100_000,
    max_result_bytes: int = 64 * 1024 * 1024,
) -> QueryExecution:
    started = time.monotonic()
    deadline = started + timeout_seconds
    deadline_hit = False
    try:
        validate_read_only_sql(sql)
    except UnsafeSQLError as exc:
        return QueryExecution(
            status=AuditStatus.UNSAFE_SQL,
            digest=None,
            row_count=0,
            unique_row_count=0,
            result_bytes=0,
            elapsed_ms=(time.monotonic() - started) * 1000,
            empty_result=False,
            result_too_large=False,
            error=str(exc)[:500],
        )
    except ValueError as exc:
        return QueryExecution(
            status=AuditStatus.SYNTAX_ERROR,
            digest=None,
            row_count=0,
            unique_row_count=0,
            result_bytes=0,
            elapsed_ms=(time.monotonic() - started) * 1000,
            empty_result=False,
            result_too_large=False,
            error=str(exc)[:500],
        )

    uri = f"{db_path.resolve().as_uri()}?mode=ro"
    connection: sqlite3.Connection | None = None
    try:
        connection = sqlite3.connect(uri, uri=True, timeout=1.0)
        connection.enable_load_extension(False)
        connection.execute("PRAGMA query_only = ON")
        connection.set_authorizer(_authorizer)

        def progress() -> int:
            nonlocal deadline_hit
            if time.monotonic() >= deadline:
                deadline_hit = True
                return 1
            return 0

        connection.set_progress_handler(progress, 1_000)
        cursor = connection.execute(sql)
        canonical_rows: set[str] = set()
        row_count = 0
        result_bytes = 0
        too_large = False
        while batch := cursor.fetchmany(1_000):
            for raw_row in batch:
                row = canonical_row(tuple(raw_row))
                row_count += 1
                result_bytes += len(row.encode("utf-8"))
                canonical_rows.add(row)
                if row_count > max_rows or result_bytes > max_result_bytes:
                    too_large = True
                    break
            if too_large:
                break
        elapsed = (time.monotonic() - started) * 1000
        return QueryExecution(
            status=AuditStatus.PASSED,
            digest=canonical_result_digest(canonical_rows),
            row_count=row_count,
            unique_row_count=len(canonical_rows),
            result_bytes=result_bytes,
            elapsed_ms=elapsed,
            empty_result=row_count == 0,
            result_too_large=too_large,
        )
    except sqlite3.Error as exc:
        status = classify_sqlite_error(str(exc), deadline_hit=deadline_hit)
        return QueryExecution(
            status=status,
            digest=None,
            row_count=0,
            unique_row_count=0,
            result_bytes=0,
            elapsed_ms=(time.monotonic() - started) * 1000,
            empty_result=False,
            result_too_large=False,
            error=str(exc)[:500],
        )
    except Exception as exc:  # defensive boundary around native SQLite calls
        return QueryExecution(
            status=AuditStatus.INFRASTRUCTURE_ERROR,
            digest=None,
            row_count=0,
            unique_row_count=0,
            result_bytes=0,
            elapsed_ms=(time.monotonic() - started) * 1000,
            empty_result=False,
            result_too_large=False,
            error=f"{type(exc).__name__}: {exc}"[:500],
        )
    finally:
        if connection is not None:
            connection.close()
