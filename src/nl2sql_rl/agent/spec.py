"""模型可见的 Agent 协议；任何修改都必须升级 Harness 哈希。"""

from __future__ import annotations

from typing import Any

SYSTEM_PROMPT = """你是一个 SQLite NL2SQL Agent。每一轮只能输出一个严格 JSON 对象：
{"action":"工具名","arguments":{...}}
工具仅允许 list_tables、describe_schema、search_values、execute_sql、submit_sql。
你必须先成功调用 describe_schema；每次最多描述 5 张表，可继续调用以获取遗漏表。
你必须至少成功调用一次 execute_sql。
最终 submit_sql 的 SQL 必须与最后一次成功执行的 SQL 经 SQL AST 规范化后完全相同。
最终 SQL 引用的每张物理表都必须已由成功的 describe_schema 返回。
只有确认这些条件后才能 submit_sql。不要输出 Markdown、解释或思考过程。"""

ACTION_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "list_tables",
            "description": "列出数据库中的用户表和视图",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "describe_schema",
            "description": (
                "查看最多 5 张表的结构化列、类型、主键和外键信息；如有遗漏表可继续调用"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "tables": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 1,
                        "maxItems": 5,
                    }
                },
                "required": ["tables"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_values",
            "description": "在一个表列中搜索最多 20 个候选值",
            "parameters": {
                "type": "object",
                "properties": {
                    "table": {"type": "string"},
                    "column": {"type": "string"},
                    "query": {"type": "string"},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 20},
                },
                "required": ["table", "column", "query"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "execute_sql",
            "description": "只读执行候选 SQL 并查看受限结果；提交前必须成功调用",
            "parameters": {
                "type": "object",
                "properties": {"sql": {"type": "string"}},
                "required": ["sql"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit_sql",
            "description": "提交最后一次成功 execute_sql 验证过且表结构均已查看的最终 SQL",
            "parameters": {
                "type": "object",
                "properties": {"sql": {"type": "string"}},
                "required": ["sql"],
                "additionalProperties": False,
            },
        },
    },
]

HARNESS_VERSION = "teacher-harness-v2"
TOOL_OBSERVATION_VERSION = 2
ACCEPTANCE_VERSION = 2
