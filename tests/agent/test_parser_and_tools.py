from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

import pytest

from nl2sql_rl.agent.parser import ActionParseError, parse_action
from nl2sql_rl.agent.sql_semantics import normalize_sql, physical_tables
from nl2sql_rl.agent.tools import SQLiteToolbox
from nl2sql_rl.models import AgentAction


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_action_parser_accepts_only_strict_json_schema() -> None:
    action = parse_action('{"action":"list_tables","arguments":{}}')
    assert action.action == "list_tables"
    for invalid in (
        "```json\n{}\n```",
        "[]",
        '{"action":"shell","arguments":{}}',
        '{"action":"list_tables","arguments":{},"extra":1}',
    ):
        with pytest.raises(ActionParseError):
            parse_action(invalid)


def test_five_tools_are_read_only_and_return_bounded_observations(agent_db: Path) -> None:
    before = _sha(agent_db)
    toolbox = SQLiteToolbox(agent_db, max_observation_bytes=2_048)
    actions = [
        AgentAction(action="list_tables", arguments={}),
        AgentAction(action="describe_schema", arguments={"tables": ["employees"]}),
        AgentAction(
            action="search_values",
            arguments={"table": "employees", "column": "name", "query": "li", "limit": 20},
        ),
        AgentAction(action="execute_sql", arguments={"sql": "SELECT name FROM employees"}),
        AgentAction(action="submit_sql", arguments={"sql": "SELECT name FROM employees"}),
    ]
    observations = [
        toolbox.call(action, event_id=f"event_{index}")
        for index, action in enumerate(actions)
    ]
    assert all(observation.ok for observation in observations)
    assert observations[0].payload["tables"] == ["departments", "employees"]
    employee_schema = observations[1].payload["schemas"][0]
    assert employee_schema["name"] == "employees"
    assert {column["name"] for column in employee_schema["columns"]} >= {
        "id",
        "name",
    }
    assert observations[2].payload["returned_rows"] == 1
    assert observations[3].payload["returned_rows"] == 3
    assert toolbox.last_submission is not None
    assert all(
        len(json.dumps(observation.payload, ensure_ascii=False).encode()) <= 2_048
        for observation in observations
    )
    assert _sha(agent_db) == before


@pytest.mark.parametrize(
    "sql",
    [
        "DELETE FROM employees",
        "PRAGMA table_info(employees)",
        "ATTACH DATABASE '/tmp/attack.sqlite' AS attack",
        "SELECT 1; DROP TABLE employees",
        "CREATE TABLE stolen(value TEXT)",
    ],
)
def test_sql_guard_blocks_write_and_escape_attempts(agent_db: Path, sql: str) -> None:
    before = _sha(agent_db)
    observation = SQLiteToolbox(agent_db).call(
        AgentAction(action="execute_sql", arguments={"sql": sql}),
        event_id="attack",
    )
    assert not observation.ok
    assert observation.error_code == "unsafe_sql"
    assert _sha(agent_db) == before


def test_tool_validation_errors_are_observations_not_exceptions(agent_db: Path) -> None:
    observation = SQLiteToolbox(agent_db).call(
        AgentAction(
            action="search_values",
            arguments={"table": "employees", "column": "missing", "query": "x"},
        ),
        event_id="bad_column",
    )
    assert not observation.ok
    assert observation.error_code == "invalid_arguments"


def test_schema_pagination_keeps_structured_columns_and_reports_omissions(
    tmp_path: Path,
) -> None:
    database = tmp_path / "wide.sqlite"
    with sqlite3.connect(database) as connection:
        for index in range(6):
            connection.execute(
                f"CREATE TABLE table_{index}(id INTEGER PRIMARY KEY, name TEXT, value REAL)"
            )
    toolbox = SQLiteToolbox(database, max_observation_bytes=900)
    too_many = toolbox.call(
        AgentAction(
            action="describe_schema",
            arguments={"tables": [f"table_{index}" for index in range(6)]},
        ),
        event_id="too_many",
    )
    assert not too_many.ok
    assert too_many.error_code == "invalid_arguments"

    paged = toolbox.call(
        AgentAction(
            action="describe_schema",
            arguments={"tables": [f"table_{index}" for index in range(5)]},
        ),
        event_id="paged",
    )
    assert paged.ok
    assert paged.truncated
    assert paged.payload["omitted_tables"]
    assert all(schema["columns"] for schema in paged.payload["schemas"])
    assert "sha256" not in paged.payload
    assert len(json.dumps(paged.payload, ensure_ascii=False).encode()) <= 900


def test_submit_requires_last_execute_and_all_physical_tables_described(
    agent_db: Path,
) -> None:
    sql = (
        "SELECT e.name FROM employees e JOIN departments d "
        "ON e.department_id = d.id"
    )
    toolbox = SQLiteToolbox(agent_db)
    no_execute = toolbox.call(
        AgentAction(action="submit_sql", arguments={"sql": sql}),
        event_id="no_execute",
    )
    assert no_execute.error_code == "submission_not_executed"

    toolbox.call(
        AgentAction(action="describe_schema", arguments={"tables": ["employees"]}),
        event_id="describe_employees",
    )
    assert toolbox.call(
        AgentAction(action="execute_sql", arguments={"sql": sql}),
        event_id="execute",
    ).ok
    missing_schema = toolbox.call(
        AgentAction(action="submit_sql", arguments={"sql": sql}),
        event_id="missing_schema",
    )
    assert missing_schema.error_code == "undescribed_table"

    toolbox.call(
        AgentAction(action="describe_schema", arguments={"tables": ["departments"]}),
        event_id="describe_departments",
    )
    mismatch = toolbox.call(
        AgentAction(action="submit_sql", arguments={"sql": "SELECT name FROM employees"}),
        event_id="mismatch",
    )
    assert mismatch.error_code == "submission_sql_mismatch"
    equivalent = toolbox.call(
        AgentAction(action="submit_sql", arguments={"sql": sql + ";"}),
        event_id="equivalent",
    )
    assert equivalent.ok
    assert normalize_sql(sql) == normalize_sql(sql + ";")


def test_sql_table_extraction_ignores_cte_aliases() -> None:
    sql = (
        "WITH recent AS (SELECT id FROM employees) "
        "SELECT recent.id FROM recent JOIN departments d ON recent.id = d.id"
    )
    assert physical_tables(sql) == {"employees", "departments"}


def test_exploration_timeout_defaults_to_ten_seconds_and_can_interrupt(
    agent_db: Path,
) -> None:
    toolbox = SQLiteToolbox(agent_db)
    assert toolbox.exploration_timeout_seconds == 10.0
    impatient = SQLiteToolbox(agent_db, exploration_timeout_seconds=0.001)
    recursive = impatient.call(
        AgentAction(
            action="execute_sql",
            arguments={
                "sql": (
                    "WITH RECURSIVE counter(x) AS (SELECT 1 UNION ALL "
                    "SELECT x + 1 FROM counter) SELECT max(x) FROM counter"
                )
            },
        ),
        event_id="timeout",
    )
    assert recursive.error_code == "timeout"
