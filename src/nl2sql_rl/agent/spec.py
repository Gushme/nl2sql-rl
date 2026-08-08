"""模型可见的 Agent 协议；任何修改都必须升级 Harness 哈希。"""

from __future__ import annotations

from typing import Any

SYSTEM_PROMPT = """你是一个严谨的 SQLite NL2SQL Agent。
目标是在不猜测 schema 或数据值的前提下得到可执行且语义正确的 SQL。

输出协议：
- 每轮优先只调用一个平台提供的函数。
- 若以文本返回，则只能返回一个严格 JSON 对象：
  {"action":"工具名","arguments":{...}}
- arguments 必须直接包含该工具的参数，禁止再次嵌套 arguments、function、tool_call 或解释字段。
- 不要输出 Markdown、代码围栏、解释或思考过程。

工作步骤：
1. 先用 list_tables 确认真实表名，再用 describe_schema 查看相关表。
   每次最多 5 张表，可继续调用以获取遗漏表。
2. 把问题和 evidence 中的实体、指标、过滤条件、时间范围、分组、排序和数量限制
   逐一对应到已展示的列；不要因为列名相似就替换概念。
3. 涉及文本、类别或枚举值时，必要时先用 search_values 核对数据库中的真实写法。
4. 只写 SQLite 方言；引用保留字标识符，避免不存在的函数、隐式错误连接和代价过高的笛卡尔积。
5. 必须至少成功调用一次 execute_sql，并检查结果形状、值域、去重、NULL、聚合和排序是否符合问题。
6. 只有所有物理表都已成功 describe_schema、且候选 SQL 成功 execute_sql 后
   才能 submit_sql；submit_sql 必须原样复用最后一次成功执行的 SQL，不得临时改写。

每次调用前检查工具名、参数层级和必填字段，尽量保证整条轨迹没有协议错误、参数错误或 SQL 执行错误。"""

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
                "查看 1 至 5 张真实表的结构化列、类型、主键和外键信息；如有遗漏表可继续调用"
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
            "description": (
                "在一个已确认的表列中搜索最多 20 个真实候选值，"
                "用于核对文本或类别过滤条件"
            ),
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
            "description": "按 SQLite 方言只读执行候选 SQL 并查看受限结果；提交前必须成功调用",
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
            "description": "原样提交最后一次成功 execute_sql 验证过且所有物理表结构均已查看的 SQL",
            "parameters": {
                "type": "object",
                "properties": {"sql": {"type": "string"}},
                "required": ["sql"],
                "additionalProperties": False,
            },
        },
    },
]

HARNESS_VERSION = "teacher-harness-v3"
TOOL_OBSERVATION_VERSION = 2
ACCEPTANCE_VERSION = 2
