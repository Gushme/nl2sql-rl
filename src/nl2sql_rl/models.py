"""数据、Agent、训练和评测共享的版本化记录。"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class StrictRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AuditStatus(StrEnum):
    PASSED = "passed"
    SOURCE_MISMATCH = "source_mismatch"
    MISSING_DATABASE = "missing_database"
    UNSAFE_SQL = "unsafe_sql"
    SYNTAX_ERROR = "syntax_error"
    SQLITE_ERROR = "sqlite_error"
    UNSUPPORTED_FUNCTION = "unsupported_function"
    MISSING_TABLE_OR_COLUMN = "missing_table_or_column"
    TIMEOUT = "timeout"
    NONDETERMINISTIC_RESULT = "nondeterministic_result"
    INFRASTRUCTURE_ERROR = "infrastructure_error"


class TerminalReason(StrEnum):
    SUBMITTED = "submitted"
    LOOP = "loop"
    MAX_ACTIONS = "max_actions"
    UNSAFE_SQL = "unsafe_sql"
    INFRASTRUCTURE_ERROR = "infrastructure_error"


class TaskView(StrictRecord):
    schema_version: int = 1
    task_id: str
    split: str
    db_id: str
    question: str
    evidence: str = ""
    db_ref: str

    def actor_payload(self) -> dict[str, Any]:
        """返回唯一允许进入 Actor 上下文的任务表示。"""
        return self.model_dump(mode="json")


class HiddenAnswer(StrictRecord):
    schema_version: int = 1
    task_id: str
    gold_sql: str
    audit_status: AuditStatus
    result_cache_ref: str | None = None


ToolName = Literal[
    "list_tables",
    "describe_schema",
    "search_values",
    "execute_sql",
    "submit_sql",
]
ObservationKind = ToolName | Literal["protocol"]


class AgentAction(StrictRecord):
    action: ToolName
    arguments: dict[str, Any] = Field(default_factory=dict)


class ToolObservation(StrictRecord):
    event_id: str
    tool: ObservationKind
    ok: bool
    payload: dict[str, Any] = Field(default_factory=dict)
    error_code: str | None = None
    elapsed_ms: float = Field(default=0.0, ge=0)
    truncated: bool = False


class TrajectoryEvent(StrictRecord):
    event_id: str
    step: int = Field(ge=0)
    action: AgentAction | None
    observation: ToolObservation


class EpisodeResult(StrictRecord):
    schema_version: int = 1
    episode_id: str
    task_id: str
    terminal_reason: TerminalReason
    events: list[TrajectoryEvent] = Field(default_factory=list)
    submitted_sql: str | None = None
    reward: float | None = None
    valid_for_training: bool = True
    usage: dict[str, int] = Field(default_factory=dict)
    database_sha256: str | None = None
    infrastructure_error_code: str | None = None
    infrastructure_error_detail: dict[str, str | int | bool] = Field(
        default_factory=dict
    )
    infrastructure_request_sent: bool | None = None
    config_hash: str


class EvaluationRecord(StrictRecord):
    schema_version: int = 1
    task_id: str
    db_id: str | None = None
    prediction_sql: str | None
    ex: float | None = None
    soft_f1: float | None = None
    r_ves: float | None = None
    prediction_status: str | None = None
    gold_status: str | None = None
    error_type: str | None = None
    infrastructure_status: str = "ok"
    details: dict[str, Any] = Field(default_factory=dict)
