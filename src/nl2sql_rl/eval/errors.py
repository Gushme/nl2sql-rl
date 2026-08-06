"""优先使用执行状态和 pred/gold AST 差异的确定性错误分类。"""

from __future__ import annotations

from typing import Protocol

from sqlglot import exp, parse_one
from sqlglot.errors import ParseError

from nl2sql_rl.models import AuditStatus, TerminalReason
from nl2sql_rl.sqlite_exec import UnsafeSQLError, validate_read_only_sql

SEMANTIC_FALLBACK_LABELS = {
    "schema_linking",
    "join",
    "filter_value",
    "aggregation_grouping",
    "order_limit",
    "set_operation",
    "other_semantic",
}


class SemanticErrorFallback(Protocol):
    """只在确定性规则返回 other_semantic 时使用的可选 LLM 接口。"""

    def classify(self, prediction_sql: str, gold_sql: str) -> str: ...


def _sql_set(node: exp.Expression, kind: type[exp.Expression]) -> set[str]:
    return {value.sql(dialect="sqlite", normalize=True) for value in node.find_all(kind)}


def classify_error(
    prediction_sql: str | None,
    gold_sql: str,
    *,
    prediction_status: AuditStatus | None = None,
    terminal_reason: TerminalReason | None = None,
) -> str:
    if terminal_reason is TerminalReason.UNSAFE_SQL or prediction_status is AuditStatus.UNSAFE_SQL:
        return "safety"
    if prediction_sql is None or not prediction_sql.strip():
        return "protocol"
    status_mapping = {
        AuditStatus.SYNTAX_ERROR: "syntax",
        AuditStatus.MISSING_TABLE_OR_COLUMN: "table_or_column",
        AuditStatus.UNSUPPORTED_FUNCTION: "function",
        AuditStatus.TIMEOUT: "timeout",
        AuditStatus.INFRASTRUCTURE_ERROR: "infrastructure",
    }
    if prediction_status in status_mapping:
        return status_mapping[prediction_status]
    try:
        validate_read_only_sql(prediction_sql)
    except UnsafeSQLError:
        return "safety"
    except ValueError:
        return "syntax"
    try:
        predicted = parse_one(prediction_sql, read="sqlite")
        gold = parse_one(gold_sql, read="sqlite")
    except ParseError:
        return "syntax"
    if _sql_set(predicted, exp.Table) != _sql_set(gold, exp.Table):
        return "schema_linking"
    if _sql_set(predicted, exp.Column) != _sql_set(gold, exp.Column):
        return "schema_linking"
    if _sql_set(predicted, exp.Join) != _sql_set(gold, exp.Join):
        return "join"
    if _sql_set(predicted, exp.Where) != _sql_set(gold, exp.Where):
        return "filter_value"
    aggregate_types: tuple[type[exp.Expression], ...] = (
        exp.AggFunc,
        exp.Group,
        exp.Having,
    )
    if any(_sql_set(predicted, kind) != _sql_set(gold, kind) for kind in aggregate_types):
        return "aggregation_grouping"
    ordering_types: tuple[type[exp.Expression], ...] = (exp.Order, exp.Limit)
    if any(_sql_set(predicted, kind) != _sql_set(gold, kind) for kind in ordering_types):
        return "order_limit"
    set_types = tuple(
        kind
        for name in ("Union", "Intersect", "Except")
        if isinstance((kind := getattr(exp, name, None)), type)
    )
    if any(_sql_set(predicted, kind) != _sql_set(gold, kind) for kind in set_types):
        return "set_operation"
    return "other_semantic"


def classify_error_with_fallback(
    prediction_sql: str | None,
    gold_sql: str,
    *,
    prediction_status: AuditStatus | None = None,
    terminal_reason: TerminalReason | None = None,
    fallback: SemanticErrorFallback | None = None,
) -> str:
    category = classify_error(
        prediction_sql,
        gold_sql,
        prediction_status=prediction_status,
        terminal_reason=terminal_reason,
    )
    if category != "other_semantic" or fallback is None or prediction_sql is None:
        return category
    proposed = fallback.classify(prediction_sql, gold_sql)
    return proposed if proposed in SEMANTIC_FALLBACK_LABELS else "other_semantic"
