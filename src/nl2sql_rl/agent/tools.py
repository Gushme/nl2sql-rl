"""BIRD SQLite 环境的五个只读、结构化且可审计工具。"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from pathlib import Path
from typing import Any

from pydantic import Field, ValidationError

from nl2sql_rl.agent.sql_semantics import SQLSemanticError, normalize_sql, physical_tables
from nl2sql_rl.io_utils import stable_json
from nl2sql_rl.models import AgentAction, AuditStatus, StrictRecord, ToolObservation
from nl2sql_rl.sqlite_exec import (
    QueryExecution,
    UnsafeSQLError,
    execute_read_only,
    validate_read_only_sql,
)


class _NoArguments(StrictRecord):
    pass


class _DescribeArguments(StrictRecord):
    tables: list[str] = Field(min_length=1, max_length=5)


class _SearchArguments(StrictRecord):
    table: str = Field(min_length=1)
    column: str = Field(min_length=1)
    query: str
    limit: int = Field(default=20, ge=1, le=20)


class _SQLArguments(StrictRecord):
    sql: str = Field(min_length=1)


class SchemaPayloadTooLarge(ValueError):
    """单表结构化 schema 本身超过 observation 上限。"""


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
        rows = list(rows)
        mutable["rows"] = rows
        while rows and _payload_bytes(mutable) > max_bytes:
            rows.pop()
        mutable["returned_rows"] = len(rows)
        if _payload_bytes(mutable) <= max_bytes:
            return mutable, True
    tables = mutable.get("tables")
    if isinstance(tables, list):
        tables = list(tables)
        original_count = len(tables)
        mutable["tables"] = tables
        while tables and _payload_bytes(mutable) > max_bytes:
            tables.pop()
        mutable["omitted_table_count"] = original_count - len(tables)
        if _payload_bytes(mutable) <= max_bytes:
            return mutable, True
    encoded = stable_json(payload).encode("utf-8")
    return {
        "message": "观察结果超过字节上限，已替换为摘要",
        "original_bytes": len(encoded),
        "sha256": hashlib.sha256(encoded).hexdigest(),
    }, True


class SQLiteToolbox:
    """把模型动作映射到只读 SQLite，并强制提交前验证约束。"""

    def __init__(
        self,
        db_path: Path,
        *,
        exploration_timeout_seconds: float = 10.0,
        submission_timeout_seconds: float = 10.0,
        max_observation_bytes: int = 8_192,
    ) -> None:
        self.db_path = db_path
        self.exploration_timeout_seconds = exploration_timeout_seconds
        self.submission_timeout_seconds = submission_timeout_seconds
        self.max_observation_bytes = max_observation_bytes
        self.last_submission: QueryExecution | None = None
        self.last_successful_execution_sql: str | None = None
        self.described_tables: set[str] = set()

    def _connect_catalog(self) -> sqlite3.Connection:
        if not self.db_path.is_file():
            raise FileNotFoundError(f"数据库不存在：{self.db_path}")
        uri = f"{self.db_path.resolve().as_uri()}?mode=ro"
        connection = sqlite3.connect(uri, uri=True, timeout=1.0)
        connection.execute("PRAGMA query_only = ON")
        return connection

    def _table_catalog(self) -> dict[str, str]:
        with self._connect_catalog() as connection:
            rows = connection.execute(
                "SELECT name, type FROM sqlite_master "
                "WHERE type IN ('table', 'view') AND name NOT LIKE 'sqlite_%' ORDER BY name"
            ).fetchall()
        return {str(name): str(kind) for name, kind in rows}

    def _table_names(self) -> list[str]:
        return list(self._table_catalog())

    def _structured_schema(
        self,
        connection: sqlite3.Connection,
        table: str,
        object_type: str,
    ) -> dict[str, Any]:
        quoted = _quote_identifier(table)
        column_rows = connection.execute(f"PRAGMA table_info({quoted})").fetchall()
        foreign_key_rows = connection.execute(f"PRAGMA foreign_key_list({quoted})").fetchall()
        columns = [
            {
                "name": str(row[1]),
                "type": str(row[2] or ""),
                "not_null": bool(row[3]),
                "default": row[4],
                "primary_key_position": int(row[5]),
            }
            for row in column_rows
        ]
        primary_key = [
            str(row[1])
            for row in sorted(column_rows, key=lambda item: int(item[5]) or 10**9)
            if int(row[5]) > 0
        ]
        foreign_keys = [
            {
                "id": int(row[0]),
                "sequence": int(row[1]),
                "to_table": str(row[2]),
                "from_column": str(row[3]),
                "to_column": str(row[4]),
                "on_update": str(row[5]),
                "on_delete": str(row[6]),
            }
            for row in foreign_key_rows
        ]
        return {
            "name": table,
            "object_type": object_type,
            "columns": columns,
            "primary_key": primary_key,
            "foreign_keys": foreign_keys,
        }

    def _schema_payload(
        self, arguments: _DescribeArguments
    ) -> tuple[dict[str, Any], bool]:
        if len(arguments.tables) != len(set(arguments.tables)):
            raise ValueError("tables 不允许重复")
        catalog = self._table_catalog()
        missing = sorted(set(arguments.tables).difference(catalog))
        if missing:
            raise ValueError(f"未知表：{missing}")
        with self._connect_catalog() as connection:
            schemas = [
                self._structured_schema(connection, table, catalog[table])
                for table in arguments.tables
            ]
        kept: list[dict[str, Any]] = []
        for index, schema in enumerate(schemas):
            candidate = {
                "schemas": [*kept, schema],
                "returned_tables": len(kept) + 1,
                "omitted_tables": arguments.tables[index + 1 :],
            }
            if _payload_bytes(candidate) > self.max_observation_bytes:
                break
            kept.append(schema)
        if not kept:
            raise SchemaPayloadTooLarge(
                f"表 {arguments.tables[0]} 的结构化 schema 超过 observation 上限"
            )
        omitted = arguments.tables[len(kept) :]
        return {
            "schemas": kept,
            "returned_tables": len(kept),
            "omitted_tables": omitted,
        }, bool(omitted)

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

    def _submission_gate(self, sql: str) -> str | None:
        try:
            validate_read_only_sql(sql)
        except UnsafeSQLError:
            raise
        except ValueError as exc:
            raise SQLSemanticError(str(exc)) from exc
        normalized = normalize_sql(sql)
        if self.last_successful_execution_sql is None:
            return "submission_not_executed"
        if normalized != self.last_successful_execution_sql:
            return "submission_sql_mismatch"
        missing = sorted(physical_tables(sql).difference(self.described_tables))
        if missing:
            return "undescribed_table"
        return None

    def call(self, action: AgentAction, *, event_id: str) -> ToolObservation:
        started = time.monotonic()
        payload: dict[str, Any] = {}
        status = AuditStatus.PASSED
        error: str | None = None
        explicit_error_code: str | None = None
        truncated = False
        try:
            if action.action == "list_tables":
                _NoArguments.model_validate(action.arguments)
                payload = {"tables": self._table_names()}
            elif action.action == "describe_schema":
                describe_args = _DescribeArguments.model_validate(action.arguments)
                payload, truncated = self._schema_payload(describe_args)
                self.described_tables.update(
                    str(schema["name"]).casefold() for schema in payload["schemas"]
                )
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
                if is_submit:
                    explicit_error_code = self._submission_gate(sql_args.sql)
                    if explicit_error_code is not None:
                        error = {
                            "submission_not_executed": "尚无成功的 execute_sql",
                            "submission_sql_mismatch": "提交 SQL 与最后一次成功执行 SQL 不一致",
                            "undescribed_table": "提交 SQL 引用了尚未成功描述的物理表",
                        }[explicit_error_code]
                    else:
                        execution = execute_read_only(
                            sql_args.sql,
                            self.db_path,
                            timeout_seconds=self.submission_timeout_seconds,
                            max_rows=100_000,
                            max_result_bytes=64 * 1024 * 1024,
                            preview_limit=0,
                        )
                        self.last_submission = execution
                        status = execution.status
                        error = execution.error
                        payload = self._query_payload(execution)
                else:
                    execution = execute_read_only(
                        sql_args.sql,
                        self.db_path,
                        timeout_seconds=self.exploration_timeout_seconds,
                        max_rows=50,
                        max_result_bytes=self.max_observation_bytes,
                        preview_limit=50,
                    )
                    status = execution.status
                    error = execution.error
                    payload = self._query_payload(execution)
                    if status is AuditStatus.PASSED:
                        self.last_successful_execution_sql = normalize_sql(sql_args.sql)
        except ValidationError as exc:
            explicit_error_code = "invalid_arguments"
            error = f"{type(exc).__name__}: {exc}"[:500]
        except SchemaPayloadTooLarge as exc:
            explicit_error_code = "schema_too_large"
            error = str(exc)[:500]
        except UnsafeSQLError as exc:
            status = AuditStatus.UNSAFE_SQL
            error = str(exc)[:500]
        except SQLSemanticError as exc:
            status = AuditStatus.SYNTAX_ERROR
            error = str(exc)[:500]
        except ValueError as exc:
            explicit_error_code = "invalid_arguments"
            error = f"{type(exc).__name__}: {exc}"[:500]
        except FileNotFoundError as exc:
            status = AuditStatus.INFRASTRUCTURE_ERROR
            error = f"{type(exc).__name__}: {exc}"[:500]
        except sqlite3.Error as exc:
            status = AuditStatus.SQLITE_ERROR
            error = f"{type(exc).__name__}: {exc}"[:500]
        elapsed_ms = (time.monotonic() - started) * 1000
        if error is not None:
            payload["message"] = error
        if action.action != "describe_schema":
            payload, payload_truncated = _limit_payload(payload, self.max_observation_bytes)
            truncated = truncated or payload_truncated
        ok = status is AuditStatus.PASSED and explicit_error_code is None
        return ToolObservation(
            event_id=event_id,
            tool=action.action,
            ok=ok,
            payload=payload,
            error_code=(
                None
                if ok
                else explicit_error_code or status.value
            ),
            elapsed_ms=elapsed_ms,
            truncated=truncated,
        )
