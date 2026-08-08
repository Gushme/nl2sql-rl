"""提交一致性与 schema 覆盖共用的 SQL AST 语义。"""

from __future__ import annotations

from sqlglot import exp, parse_one
from sqlglot.errors import ParseError


class SQLSemanticError(ValueError):
    """SQL 无法规范化或提取物理表。"""


def normalize_sql(sql: str) -> str:
    """生成 SQLite 方言的稳定 AST 序列化，用于一致性检查。"""
    try:
        tree = parse_one(sql, read="sqlite")
    except ParseError as exc:
        raise SQLSemanticError(str(exc)) from exc
    if tree is None:
        raise SQLSemanticError("SQL AST 为空")
    return tree.sql(dialect="sqlite", pretty=False, normalize=True)


def physical_tables(sql: str) -> set[str]:
    """提取物理表名，并排除 CTE 名称与大小写差异。"""
    try:
        tree = parse_one(sql, read="sqlite")
    except ParseError as exc:
        raise SQLSemanticError(str(exc)) from exc
    if tree is None:
        raise SQLSemanticError("SQL AST 为空")
    cte_names = {
        cte.alias_or_name.casefold()
        for cte in tree.find_all(exp.CTE)
        if cte.alias_or_name
    }
    tables = {
        table.name.casefold()
        for table in tree.find_all(exp.Table)
        if table.name and table.name.casefold() not in cte_names
    }
    return tables
