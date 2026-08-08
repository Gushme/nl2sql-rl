from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from nl2sql_rl.agent.loop import ScriptedPolicy, run_episode
from nl2sql_rl.config import RuntimeConfig
from nl2sql_rl.models import (
    AgentAction,
    AuditStatus,
    EpisodeResult,
    HiddenAnswer,
    TaskView,
    TerminalReason,
)
from nl2sql_rl.training.grpo import episode_to_fake_rollout
from nl2sql_rl.training.grpo_reward import (
    InvalidRolloutError,
    adapt_episode_reward,
    masked_group_advantages,
)


class CharacterTokenizer:
    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        assert add_special_tokens is False
        return [ord(character) for character in text]


def _database(path: Path) -> None:
    with sqlite3.connect(path) as connection:
        connection.executescript(
            """
            CREATE TABLE departments(id INTEGER PRIMARY KEY, name TEXT);
            CREATE TABLE employees(
                id INTEGER PRIMARY KEY,
                name TEXT,
                department_id INTEGER REFERENCES departments(id),
                salary REAL
            );
            INSERT INTO departments VALUES (1, '研发'), (2, '销售');
            INSERT INTO employees VALUES
                (1, 'Alice', 1, 100.0),
                (2, 'Bob', 1, 120.0),
                (3, 'Carol', 2, 90.0);
            """
        )


def _task_answer() -> tuple[TaskView, HiddenAnswer]:
    task = TaskView(
        task_id="bird_fixture_1",
        split="train",
        db_id="agent",
        question="列出研发部门员工",
        db_ref="agent.sqlite",
    )
    answer = HiddenAnswer(
        task_id=task.task_id,
        gold_sql=(
            "SELECT e.name FROM employees e JOIN departments d "
            "ON e.department_id=d.id WHERE d.name='研发'"
        ),
        audit_status=AuditStatus.PASSED,
    )
    return task, answer


def test_fake_rollout_masks_actions_but_not_tool_observations(tmp_path: Path) -> None:
    database = tmp_path / "agent.sqlite"
    _database(database)
    task, answer = _task_answer()
    actions = [
        AgentAction(
            action="describe_schema",
            arguments={"tables": ["employees", "departments"]},
        ),
        AgentAction(
            action="execute_sql",
            arguments={
                "sql": (
                    "SELECT e.name FROM employees e JOIN departments d "
                    "ON e.department_id=d.id WHERE d.name='研发'"
                )
            },
        ),
        AgentAction(
            action="submit_sql",
            arguments={
                "sql": (
                    "SELECT e.name FROM employees e JOIN departments d "
                    "ON e.department_id=d.id WHERE d.name='研发'"
                )
            },
        ),
    ]
    episode = run_episode(
        task,
        answer,
        database,
        ScriptedPolicy(actions),
        runtime=RuntimeConfig(),
        config_hash="fixture",
        episode_id="fake-rollout",
    )
    output = episode_to_fake_rollout(task, episode, CharacterTokenizer())
    assert output.reward_score == 1.0
    assert len(output.response_ids) == len(output.response_mask)
    assert 1 in output.response_mask
    assert 0 in output.response_mask
    assert sum(output.response_mask) < len(output.response_mask)
    assert output.extra_fields["valid_for_advantage"] is True


def test_invalid_rollout_never_enters_advantage_calculation() -> None:
    invalid = EpisodeResult(
        episode_id="invalid",
        task_id="task",
        terminal_reason=TerminalReason.INFRASTRUCTURE_ERROR,
        reward=None,
        valid_for_training=False,
        config_hash="fixture",
    )
    with pytest.raises(InvalidRolloutError):
        adapt_episode_reward(invalid)

    advantages = masked_group_advantages([1.0, 0.0, None, -1.0], group_size=4)
    assert advantages[2] is None
    valid = [value for value in advantages if value is not None]
    assert sum(valid) == pytest.approx(0.0)


def test_reward_adapter_preserves_terminal_penalties() -> None:
    episode = EpisodeResult(
        episode_id="loop",
        task_id="task",
        terminal_reason=TerminalReason.LOOP,
        reward=-0.4,
        valid_for_training=True,
        config_hash="fixture",
    )
    assert adapt_episode_reward(episode).score == -0.4
