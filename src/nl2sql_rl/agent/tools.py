"""BIRD SQLite 环境的五个只读工具。"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from pathlib import Path
from typing import Any

from pydantic import Field, ValidationError

from nl2sql_rl.io_utils import stable_json
from nl2sql_rl.models import AgentAction, AuditStatus, StrictRecord, ToolObservation
from nl2sql_rl.sqlite_exec import QueryExecution, execute_read_only


class _NoArguments(StrictRecord):
    pass


class _DescribeArguments(StrictRecord):
    tables: list[str] = Field(min_length=1, max_length=20)


class _SearchArguments(StrictRecord):
    table: str = Field(min_length=1)
    column: str = Field(min_length=1)
    query: str
    limit: int = Field(default=20, ge=1, le=20)


class _SQLArguments(StrictRecord):
    sql: str = Field(min_length=1)


def _quote_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def _payload_bytes(payload: dict[str, Any]) -> int:
    return len(stable_json(payload).encode("utf-8"))


def _limit_payload(payload: dict[str, Any], max_bytes: int) -> tuple[dict[str, Any], bool]:
    if _payload_bytes(payload) <= max_bytes:
        return payload, False
    mutable = dict(payload)
    rows = mutable.get("rows")
    if isinstance(rows, list):
        while rows and _payload_bytes(mutable) > max_bytes:
            rows.pop()
        mutable["returned_rows"] = len(rows)
        if _payload_bytes(mutable) <= max_bytes:
            return mutable, True
    encoded = stable_json(payload).encode("utf-8")
    return {
        "message": "观察结果超过字节上限，已替换为摘要",
        "original_bytes": len(encoded),
        "sha256": hashlib.sha256(encoded).hexdigest(),
    }, True


class SQLiteToolbox:
    """把模型动作映射到受限的只读 SQLite 操作。"""

    def __init__(
        self,
        db_path: Path,
        *,
        exploration_timeout_seconds: float = 2.0,
        submission_timeout_seconds: float = 10.0,
        max_observation_bytes: int = 8_192,
    ) -> None:
        self.db_path = db_path
        self.exploration_timeout_seconds = exploration_timeout_seconds
        self.submission_timeout_seconds = submission_timeout_seconds
        self.max_observation_bytes = max_observation_bytes
        self.last_submission: QueryExecution | None = None

    def _connect_catalog(self) -> sqlite3.Connection:
        if not self.db_path.is_file():
            raise FileNotFoundError(f"数据库不存在：{self.db_path}")
        uri = f"{self.db_path.resolve().as_uri()}?mode=ro"
        connection = sqlite3.connect(uri, uri=True, timeout=1.0)
        connection.execute("PRAGMA query_only = ON")
        return connection

    def _table_names(self) -> list[str]:
        with self._connect_catalog() as connection:
            rows = connection.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type IN ('table', 'view') AND name NOT LIKE 'sqlite_%' ORDER BY name"
            ).fetchall()
        return [str(row[0]) for row in rows]

    def _schema_payload(self, arguments: _DescribeArguments) -> dict[str, Any]:
        known = set(self._table_names())
        missing = sorted(set(arguments.tables).difference(known))
        if missing:
            raise ValueError(f"未知表：{missing}")
        placeholders = ",".join("?" for _ in arguments.tables)
        sql = (
            "SELECT name, type, sql FROM sqlite_master "
            f"WHERE name IN ({placeholders}) ORDER BY name"
        )
        with self._connect_catalog() as connection:
            rows = connection.execute(sql, arguments.tables).fetchall()
        return {
            "schemas": [
                {"name": str(name), "type": str(kind), "ddl": str(ddl or "")}
                for name, kind, ddl in rows
            ]
        }

    def _search_sql(self, arguments: _SearchArguments) -> str:
        known = set(self._table_names())
        if arguments.table not in known:
            raise ValueError(f"未知表：{arguments.table}")
        with self._connect_catalog() as connection:
            columns = {
                str(row[1])
                for row in connection.execute(
                    f"PRAGMA table_info({_quote_identifier(arguments.table)})"
                ).fetchall()
            }
        if arguments.column not in columns:
            raise ValueError(f"未知列：{arguments.table}.{arguments.column}")
        table = _quote_identifier(arguments.table)
        column = _quote_identifier(arguments.column)
        query = arguments.query.replace("'", "''")
        return (
            f"SELECT DISTINCT {column} FROM {table} "
            f"WHERE instr(CAST({column} AS TEXT), '{query}') > 0 "
            f"LIMIT {arguments.limit}"
        )

    def _query_payload(self, execution: QueryExecution) -> dict[str, Any]:
        rows = [json.loads(row) for row in execution.preview_rows]
        return {
            "status": execution.status.value,
            "columns": list(execution.columns),
            "rows": rows,
            "returned_rows": len(rows),
            "scanned_rows": execution.row_count,
            "result_too_large": execution.result_too_large,
            "result_digest": execution.digest,
        }

    def call(self, action: AgentAction, *, event_id: str) -> ToolObservation:
        started = time.monotonic()
        payload: dict[str, Any] = {}
        status = AuditStatus.PASSED
        error: str | None = None
        try:
            if action.action == "list_tables":
                _NoArguments.model_validate(action.arguments)
                payload = {"tables": self._table_names()}
            elif action.action == "describe_schema":
                describe_args = _DescribeArguments.model_validate(action.arguments)
                payload = self._schema_payload(describe_args)
            elif action.action == "search_values":
                search_args = _SearchArguments.model_validate(action.arguments)
                execution = execute_read_only(
                    self._search_sql(search_args),
                    self.db_path,
                    timeout_seconds=self.exploration_timeout_seconds,
                    max_rows=20,
                    max_result_bytes=self.max_observation_bytes,
                    preview_limit=20,
                )
                status = execution.status
                error = execution.error
                payload = self._query_payload(execution)
            else:
                sql_args = _SQLArguments.model_validate(action.arguments)
                is_submit = action.action == "submit_sql"
                execution = execute_read_only(
                    sql_args.sql,
                    self.db_path,
                    timeout_seconds=(
                        self.submission_timeout_seconds
                        if is_submit
                        else self.exploration_timeout_seconds
                    ),
                    max_rows=100_000 if is_submit else 50,
                    max_result_bytes=64 * 1024 * 1024 if is_submit else self.max_observation_bytes,
                    preview_limit=0 if is_submit else 50,
                )
                if is_submit:
                    self.last_submission = execution
                status = execution.status
                error = execution.error
                payload = self._query_payload(execution)
        except (ValidationError, ValueError, FileNotFoundError, sqlite3.Error) as exc:
            status = AuditStatus.SQLITE_ERROR
            error = f"{type(exc).__name__}: {exc}"[:500]
        elapsed_ms = (time.monotonic() - started) * 1000
        if error is not None:
            payload["message"] = error
        payload, truncated = _limit_payload(payload, self.max_observation_bytes)
        return ToolObservation(
            event_id=event_id,
            tool=action.action,
            ok=status is AuditStatus.PASSED,
            payload=payload,
            error_code=None if status is AuditStatus.PASSED else status.value,
            elapsed_ms=elapsed_ms,
            truncated=truncated,
        )
