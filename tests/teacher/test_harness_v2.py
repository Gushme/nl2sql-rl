from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import pytest

from nl2sql_rl.agent.fingerprint import harness_config_hash
from nl2sql_rl.config import RuntimeConfig
from nl2sql_rl.io_utils import read_jsonl, stable_json, write_jsonl
from nl2sql_rl.models import AgentAction, AuditStatus, HiddenAnswer, TaskView
from nl2sql_rl.teacher.campaign import prepare_campaign_state, register_campaign_attempt
from nl2sql_rl.teacher.client import LLMCompletion
from nl2sql_rl.teacher.collector import (
    CollectorConfig,
    TeacherAttempt,
    collect_trajectories,
)
from nl2sql_rl.teacher.diagnostics import diagnose_attempts
from nl2sql_rl.teacher.migration import migrate_accepted_attempts
from nl2sql_rl.teacher.preflight import preflight_scheduled_gold
from nl2sql_rl.teacher.sampling import ComplexityBucket, build_sampling_plan
from nl2sql_rl.training.sft_data import build_sft_conversations


class CharacterTokenizer:
    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        assert not add_special_tokens
        return [ord(character) for character in text]


class InflatingTokenizer:
    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        assert not add_special_tokens
        return [0] * (len(text) * 20)


class GoodClient:
    config_hash = "good-client"
    real_api = False
    spent_usd = 0.0

    def __init__(self) -> None:
        self.calls = 0

    async def complete_action(
        self, messages: list[dict[str, Any]], *, max_tokens: int | None = None
    ) -> LLMCompletion:
        assert max_tokens is None
        self.calls += 1
        action_count = sum(message.get("role") == "assistant" for message in messages)
        if action_count == 0:
            action = AgentAction(
                action="describe_schema", arguments={"tables": ["items"]}
            )
        elif action_count == 1:
            action = AgentAction(
                action="execute_sql", arguments={"sql": "SELECT COUNT(*) FROM items"}
            )
        else:
            action = AgentAction(
                action="submit_sql", arguments={"sql": "SELECT COUNT(*) FROM items"}
            )
        return LLMCompletion(
            action=action,
            response_id=f"good_{self.calls}",
            input_tokens=10,
            output_tokens=5,
            cost_usd=0.0,
        )


class InvalidProtocolClient(GoodClient):
    config_hash = "invalid-client"

    async def complete_action(
        self, messages: list[dict[str, Any]], *, max_tokens: int | None = None
    ) -> LLMCompletion:
        del messages, max_tokens
        self.calls += 1
        return LLMCompletion(
            action=None,
            action_text="{}",
            response_id=f"invalid_{self.calls}",
            input_tokens=10,
            output_tokens=2,
            cost_usd=0.0,
            normalization_error="Action schema 不合法",
        )


class CorrectingClient(GoodClient):
    config_hash = "correcting-client"

    async def complete_action(
        self, messages: list[dict[str, Any]], *, max_tokens: int | None = None
    ) -> LLMCompletion:
        assert max_tokens is None
        self.calls += 1
        action_count = sum(message.get("role") == "assistant" for message in messages)
        if action_count == 0:
            action = AgentAction(
                action="describe_schema", arguments={"tables": ["items"]}
            )
        elif action_count == 1:
            action = AgentAction(
                action="execute_sql", arguments={"sql": "SELECT missing FROM items"}
            )
        elif action_count == 2:
            action = AgentAction(
                action="execute_sql", arguments={"sql": "SELECT COUNT(*) FROM items"}
            )
        else:
            action = AgentAction(
                action="submit_sql", arguments={"sql": "SELECT COUNT(*) FROM items"}
            )
        return LLMCompletion(
            action=action,
            response_id=f"correcting_{self.calls}",
            input_tokens=10,
            output_tokens=5,
            cost_usd=0.0,
        )


def _fixtures(root: Path, count: int = 2) -> tuple[list[TaskView], list[HiddenAnswer]]:
    database = root / "items.sqlite"
    with sqlite3.connect(database) as connection:
        connection.executescript(
            "CREATE TABLE items(id INTEGER); INSERT INTO items VALUES (1), (2);"
        )
    tasks = [
        TaskView(
            task_id=f"task_{index:03d}",
            split="train" if count > 2 or index == 0 else "validation",
            db_id="items",
            question="有多少条记录？",
            db_ref=database.name,
        )
        for index in range(count)
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


def _config(total: int = 2) -> CollectorConfig:
    return CollectorConfig(
        target_total=total,
        train_quota=total if total > 2 else 1,
        validation_quota=0 if total > 2 else 1,
        max_attempts=total,
        run_attempt_limit=total,
        diagnostic_batch_size=20,
        concurrency=2,
        simple_ratio=1.0,
        moderate_ratio=0.0,
        challenging_ratio=0.0,
    )


def test_gold_harness_preflight_saves_no_hidden_sql(tmp_path: Path) -> None:
    tasks, answers = _fixtures(tmp_path)
    report = preflight_scheduled_gold(
        tasks,
        answers,
        tmp_path,
        runtime=RuntimeConfig(),
        limit=2,
        train_quota=1,
        validation_quota=1,
        complexity_weights={ComplexityBucket.SIMPLE: 1.0},
    )
    assert report["passed"] == 2
    assert report["failed"] == 0
    assert report["database_hashes_unchanged"] is True
    assert report["gold_sql_saved"] is False
    assert "SELECT COUNT" not in stable_json(report)


@pytest.mark.asyncio
async def test_protocol_threshold_pauses_before_twenty_attempts(tmp_path: Path) -> None:
    tasks, answers = _fixtures(tmp_path, count=20)
    client = InvalidProtocolClient()
    summary = await collect_trajectories(
        tasks,
        answers,
        tmp_path,
        tmp_path / "invalid.jsonl",
        client,
        runtime=RuntimeConfig(),
        config=_config(total=20),
        tokenizer=CharacterTokenizer(),
        diagnostics_dir=tmp_path / "diagnostics",
    )
    assert summary["paused"] is True
    assert "protocol_or_argument_error_rate_gt_5pct" in summary["pause_reasons"]
    assert summary["attempts"] == 2
    assert len(read_jsonl(tmp_path / "invalid.jsonl")) == 2


@pytest.mark.asyncio
async def test_context_limit_stops_before_teacher_request(tmp_path: Path) -> None:
    tasks, answers = _fixtures(tmp_path)
    client = GoodClient()
    summary = await collect_trajectories(
        tasks,
        answers,
        tmp_path,
        tmp_path / "context.jsonl",
        client,
        runtime=RuntimeConfig(max_context_tokens=1_024),
        config=_config(),
        tokenizer=InflatingTokenizer(),
    )
    attempts = [
        TeacherAttempt.model_validate(row)
        for row in read_jsonl(tmp_path / "context.jsonl")
    ]
    assert client.calls == 0
    assert summary["paused"] is True
    assert "context_overflow_rate_gt_1pct" in summary["pause_reasons"]
    assert all(row.episode.infrastructure_request_sent is False for row in attempts)


@pytest.mark.asyncio
async def test_correct_final_sql_is_rejected_after_any_tool_error(tmp_path: Path) -> None:
    tasks, answers = _fixtures(tmp_path)
    output = tmp_path / "corrected.jsonl"
    summary = await collect_trajectories(
        tasks,
        answers,
        tmp_path,
        output,
        CorrectingClient(),
        runtime=RuntimeConfig(),
        config=_config(),
        tokenizer=CharacterTokenizer(),
    )
    attempts = [TeacherAttempt.model_validate(row) for row in read_jsonl(output)]
    assert summary["accepted_total"] == 0
    assert all(row.episode.reward == 1.0 for row in attempts)
    assert all(row.rejection_reason == "tool_error:missing_table_or_column" for row in attempts)


@pytest.mark.asyncio
async def test_timeout_only_harness_upgrade_can_migrate_exact_replay(
    tmp_path: Path,
) -> None:
    tasks, answers = _fixtures(tmp_path)
    source = tmp_path / "v1.jsonl"
    first_client = GoodClient()
    first = await collect_trajectories(
        tasks,
        answers,
        tmp_path,
        source,
        first_client,
        runtime=RuntimeConfig(exploration_timeout_seconds=5.0),
        config=_config(),
        tokenizer=CharacterTokenizer(),
    )
    assert first["complete"] is True

    target = tmp_path / "v2.jsonl"
    second_client = GoodClient()
    second = await collect_trajectories(
        tasks,
        answers,
        tmp_path,
        target,
        second_client,
        runtime=RuntimeConfig(exploration_timeout_seconds=6.0),
        config=_config(),
        tokenizer=CharacterTokenizer(),
        migration_source=source,
    )
    assert second["complete"] is True
    assert second["migration"]["migrated"] == 2
    assert second_client.calls == 0
    migrated = [TeacherAttempt.model_validate(row) for row in read_jsonl(target)]
    assert all(row.migrated_from for row in migrated)
    assert len({row.harness_config_hash for row in migrated}) == 1


def test_visible_protocol_change_rejects_migration_and_sft_hash_mixing(
    tmp_path: Path,
) -> None:
    tasks, answers = _fixtures(tmp_path)
    source = tmp_path / "source.jsonl"
    client = GoodClient()
    import asyncio

    asyncio.run(
        collect_trajectories(
            tasks,
            answers,
            tmp_path,
            source,
            client,
            runtime=RuntimeConfig(),
            config=_config(),
            tokenizer=CharacterTokenizer(),
        )
    )
    rows = [TeacherAttempt.model_validate(row) for row in read_jsonl(source)]
    tampered = [
        row.model_copy(update={"visible_protocol_hash": "changed"}) for row in rows
    ]
    tampered_path = tmp_path / "tampered.jsonl"
    write_jsonl(
        tampered_path,
        [row.model_dump(mode="json") for row in tampered],
    )
    plan = build_sampling_plan(
        tasks,
        answers,
        split_targets={"train": 1, "validation": 1},
        complexity_weights=_config().complexity_weights(),
    )
    report = migrate_accepted_attempts(
        tampered_path,
        tmp_path / "rejected.jsonl",
        tasks=tasks,
        answers=answers,
        db_root=tmp_path,
        runtime=RuntimeConfig(),
        tokenizer=CharacterTokenizer(),
        plan=plan,
        new_config_hash="new-config",
        new_harness_config_hash=harness_config_hash(RuntimeConfig()),
    )
    assert report["migrated"] == 0
    assert report["rejected"] == {"visible_prompt_or_tools_changed": 2}

    mixed = [rows[0], rows[1].model_copy(update={"config_hash": "other"})]
    with pytest.raises(ValueError, match="禁止混合"):
        build_sft_conversations(tasks, mixed)

    duplicate = [rows[0], rows[0]]
    with pytest.raises(ValueError, match="重复 task_id"):
        build_sft_conversations(tasks, duplicate)

    invalid_episode = rows[0].episode.model_copy(update={"reward": 0.0})
    invalid_attempt = rows[0].model_copy(update={"episode": invalid_episode})
    with pytest.raises(ValueError, match="终局不合格"):
        build_sft_conversations(tasks, [invalid_attempt])

    final_event = rows[0].episode.events[-1].model_copy(
        update={
            "action": AgentAction(
                action="submit_sql", arguments={"sql": "SELECT id FROM items"}
            )
        }
    )
    violated_episode = rows[0].episode.model_copy(
        update={"events": [*rows[0].episode.events[:-1], final_event]}
    )
    violated_attempt = rows[0].model_copy(update={"episode": violated_episode})
    diagnostics = diagnose_attempts(
        [violated_attempt],
        tasks=tasks,
        answers=answers,
        db_root=tmp_path,
        runtime=RuntimeConfig(),
        replay_accepted=False,
    )
    assert diagnostics["accepted_invariant_violations"] == 1
    assert "accepted_trajectory_invariant_violation" in diagnostics["pause_reasons"]
    assert diagnostics["response_calls"] == 3
    assert diagnostics["text_json_calls"] == 3


def test_campaign_state_counts_attempts_once_and_freezes_limits(tmp_path: Path) -> None:
    tasks, answers = _fixtures(tmp_path)
    attempts_path = tmp_path / "attempts.jsonl"
    import asyncio

    asyncio.run(
        collect_trajectories(
            tasks,
            answers,
            tmp_path,
            attempts_path,
            GoodClient(),
            runtime=RuntimeConfig(),
            config=_config(),
            tokenizer=CharacterTokenizer(),
        )
    )
    state_path = tmp_path / "campaign.json"
    state = prepare_campaign_state(
        state_path,
        attempt_paths=[attempts_path],
        target_total=2,
        max_attempts=2,
        cost_limit_usd=20.0,
        token_limit=1_000,
        sampling_manifest_hash="sampling-v1",
        teacher_behavior_hash="teacher-v1",
        pricing_hash="pricing-v1",
    )
    assert state.attempts == 2
    assert state.sampling_manifest_hash == "sampling-v1"
    assert state.teacher_behavior_hash == "teacher-v1"
    assert state.teacher_behavior_hashes == ["teacher-v1"]
    assert state.pricing_hash == "pricing-v1"
    assert state.used_tokens == 90
    first_episode_id = TeacherAttempt.model_validate(
        read_jsonl(attempts_path)[0]
    ).episode.episode_id
    unchanged = register_campaign_attempt(
        state_path,
        state,
        episode_id=first_episode_id,
        spent_usd=state.spent_usd,
        used_tokens=state.used_tokens,
        harness_config_hash="harness",
    )
    assert unchanged.attempts == 2
    with pytest.raises(ValueError, match="费用上限"):
        prepare_campaign_state(
            state_path,
            attempt_paths=[attempts_path],
            target_total=2,
            max_attempts=2,
            cost_limit_usd=21.0,
            token_limit=1_000,
        )
    with pytest.raises(ValueError, match="采样清单"):
        prepare_campaign_state(
            state_path,
            attempt_paths=[attempts_path],
            target_total=2,
            max_attempts=2,
            cost_limit_usd=20.0,
            token_limit=1_000,
            sampling_manifest_hash="sampling-v2",
        )
    with pytest.raises(ValueError, match="Teacher 行为配置"):
        prepare_campaign_state(
            state_path,
            attempt_paths=[attempts_path],
            target_total=2,
            max_attempts=2,
            cost_limit_usd=20.0,
            token_limit=1_000,
            teacher_behavior_hash="teacher-v2",
        )
    with pytest.raises(ValueError, match="计费口径"):
        prepare_campaign_state(
            state_path,
            attempt_paths=[attempts_path],
            target_total=2,
            max_attempts=2,
            cost_limit_usd=20.0,
            token_limit=1_000,
            pricing_hash="pricing-v2",
        )
    with pytest.raises(ValueError, match="Token 上限"):
        prepare_campaign_state(
            state_path,
            attempt_paths=[attempts_path],
            target_total=2,
            max_attempts=2,
            cost_limit_usd=20.0,
            token_limit=999,
        )
    upgraded = prepare_campaign_state(
        state_path,
        attempt_paths=[attempts_path],
        target_total=2,
        max_attempts=2,
        cost_limit_usd=20.0,
        token_limit=1_000,
        teacher_behavior_hash="teacher-v2",
        allow_teacher_behavior_upgrade=True,
    )
    assert upgraded.teacher_behavior_hash == "teacher-v2"
    assert upgraded.teacher_behavior_hashes == ["teacher-v1", "teacher-v2"]
    assert upgraded.attempts == state.attempts
    assert upgraded.used_tokens == state.used_tokens


def test_campaign_state_supports_token_only_budget(tmp_path: Path) -> None:
    state_path = tmp_path / "token_campaign.json"
    state = prepare_campaign_state(
        state_path,
        attempt_paths=[],
        target_total=1_000,
        max_attempts=1_500,
        token_limit=1_000_000,
    )
    assert state.cost_limit_usd is None
    assert state.token_limit == 1_000_000
    assert state.used_tokens == 0
    with pytest.raises(ValueError, match="费用上限或 Token 上限"):
        prepare_campaign_state(
            tmp_path / "invalid_campaign.json",
            attempt_paths=[],
            target_total=1_000,
            max_attempts=1_500,
        )
