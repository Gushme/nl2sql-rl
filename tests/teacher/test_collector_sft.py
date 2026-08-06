from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import pytest

from nl2sql_rl.config import RuntimeConfig
from nl2sql_rl.io_utils import read_jsonl
from nl2sql_rl.models import AgentAction, AuditStatus, HiddenAnswer, TaskView
from nl2sql_rl.teacher.client import LLMCompletion
from nl2sql_rl.teacher.collector import (
    CollectorConfig,
    TeacherAttempt,
    collect_trajectories,
)
from nl2sql_rl.training.sft_data import build_sft_conversations, tokenize_action_only


class FakeTeacherClient:
    config_hash = "fake-client-config"
    real_api = False
    spent_usd = 0.0

    def __init__(self) -> None:
        self.calls = 0

    async def complete_action(
        self, messages: list[dict[str, Any]], *, max_tokens: int | None = None
    ) -> LLMCompletion:
        assert messages and max_tokens
        self.calls += 1
        return LLMCompletion(
            action=AgentAction(
                action="submit_sql", arguments={"sql": "SELECT COUNT(*) FROM items"}
            ),
            response_id=f"mock_{self.calls}",
            input_tokens=10,
            output_tokens=5,
            cost_usd=0.0,
        )


class CharacterTokenizer:
    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        assert not add_special_tokens
        return [ord(character) for character in text]


def _database(path: Path) -> None:
    connection = sqlite3.connect(path)
    connection.executescript(
        "CREATE TABLE items(id INTEGER); INSERT INTO items VALUES (1), (2);"
    )
    connection.close()


def _fixtures(db_name: str) -> tuple[list[TaskView], list[HiddenAnswer]]:
    tasks = [
        TaskView(
            task_id=f"task_{split}",
            split=split,
            db_id="fixture",
            question="有多少条记录？",
            db_ref=db_name,
        )
        for split in ("train", "validation")
    ]
    answers = [
        HiddenAnswer(
            task_id=task.task_id,
            gold_sql="SELECT COUNT(*) FROM items",
            audit_status=AuditStatus.PASSED,
        )
        for task in tasks
    ]
    return tasks, answers


@pytest.mark.asyncio
async def test_collector_resumes_without_duplicate_requests_and_builds_sft(
    tmp_path: Path,
) -> None:
    database = tmp_path / "fixture.sqlite"
    _database(database)
    tasks, answers = _fixtures(database.name)
    output = tmp_path / "attempts.jsonl"
    client = FakeTeacherClient()
    config = CollectorConfig(
        target_total=2,
        train_quota=1,
        validation_quota=1,
        max_attempts=2,
        concurrency=2,
    )
    first = await collect_trajectories(
        tasks,
        answers,
        tmp_path,
        output,
        client,
        runtime=RuntimeConfig(),
        config=config,
    )
    second = await collect_trajectories(
        tasks,
        answers,
        tmp_path,
        output,
        client,
        runtime=RuntimeConfig(),
        config=config,
    )
    assert first["complete"] is True and second["complete"] is True
    assert client.calls == 2
    assert len(read_jsonl(output)) == 2

    attempts = [TeacherAttempt.model_validate(row) for row in read_jsonl(output)]
    conversations = build_sft_conversations(tasks, attempts)
    assert {row.split for row in conversations} == {"train", "validation"}
    serialized = json.dumps(
        [row.model_dump(mode="json") for row in conversations], ensure_ascii=False
    )
    assert "gold_sql" not in serialized
    assert '"reward"' not in serialized

    conversation = conversations[0]
    tokenized = tokenize_action_only(
        conversation, CharacterTokenizer(), max_length=16_384
    )
    expected = [
        ord(character)
        for index in conversation.assistant_message_indexes
        for character in str(conversation.messages[index]["content"])
    ]
    assert [label for label in tokenized.labels if label != -100] == expected
    assert len(tokenized.input_ids) == len(tokenized.labels) == len(tokenized.attention_mask)
    rendered = "".join(chr(token_id) for token_id in tokenized.input_ids)
    assert "<|im_start|>user\n<tool_response>" in rendered
    assert "<|im_start|>tool" not in rendered


@pytest.mark.asyncio
async def test_real_api_requires_explicit_confirmation(tmp_path: Path) -> None:
    class RealClient(FakeTeacherClient):
        real_api = True

    database = tmp_path / "fixture.sqlite"
    _database(database)
    tasks, answers = _fixtures(database.name)
    with pytest.raises(PermissionError, match="显式"):
        await collect_trajectories(
            tasks,
            answers,
            tmp_path,
            tmp_path / "attempts.jsonl",
            RealClient(),
            runtime=RuntimeConfig(),
            config=CollectorConfig(
                target_total=2,
                train_quota=1,
                validation_quota=1,
                max_attempts=2,
            ),
        )
