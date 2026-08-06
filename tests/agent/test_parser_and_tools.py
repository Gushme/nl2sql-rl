from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from nl2sql_rl.agent.parser import ActionParseError, parse_action
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
        AgentAction(action="submit_sql", arguments={"sql": "SELECT COUNT(*) FROM employees"}),
    ]
    observations = [
        toolbox.call(action, event_id=f"event_{index}")
        for index, action in enumerate(actions)
    ]
    assert all(observation.ok for observation in observations)
    assert observations[0].payload["tables"] == ["departments", "employees"]
    assert "CREATE TABLE employees" in observations[1].payload["schemas"][0]["ddl"]
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
    assert observation.error_code == "sqlite_error"
