"""按分层配额采集 Teacher 轨迹，并提供可审计的断点续采。"""

from __future__ import annotations

import asyncio
import hashlib
import os
from collections import Counter
from concurrent.futures import Future
from pathlib import Path
from typing import Any, Protocol

from pydantic import Field

from nl2sql_rl.agent.loop import ModelResponse, run_episode
from nl2sql_rl.config import RuntimeConfig
from nl2sql_rl.io_utils import read_jsonl, stable_json
from nl2sql_rl.models import EpisodeResult, HiddenAnswer, StrictRecord, TaskView, TerminalReason
from nl2sql_rl.teacher.client import LLMCompletion
from nl2sql_rl.teacher.sampling import (
    ComplexityBucket,
    StratifiedScheduler,
    build_sampling_plan,
)


class CollectorConfig(StrictRecord):
    target_total: int = Field(default=1_000, ge=1)
    train_quota: int = Field(default=900, ge=0)
    validation_quota: int = Field(default=100, ge=0)
    max_attempts: int = Field(default=1_500, ge=1)
    run_attempt_limit: int = Field(default=100, ge=1)
    concurrency: int = Field(default=4, ge=1, le=32)
    seed: int = 42
    simple_ratio: float = Field(default=0.30, ge=0)
    moderate_ratio: float = Field(default=0.50, ge=0)
    challenging_ratio: float = Field(default=0.20, ge=0)
    confirm_real_api: bool = False

    def validate_quotas(self) -> None:
        if self.train_quota + self.validation_quota != self.target_total:
            raise ValueError("train_quota + validation_quota 必须等于 target_total")
        if self.max_attempts < self.target_total:
            raise ValueError("max_attempts 不能小于 target_total")
        if self.simple_ratio + self.moderate_ratio + self.challenging_ratio <= 0:
            raise ValueError("复杂度配比之和必须大于 0")

    def complexity_weights(self) -> dict[ComplexityBucket, float]:
        return {
            ComplexityBucket.SIMPLE: self.simple_ratio,
            ComplexityBucket.MODERATE: self.moderate_ratio,
            ComplexityBucket.CHALLENGING: self.challenging_ratio,
        }

    def behavior_payload(self) -> dict[str, Any]:
        """排除并发、单次运行上限和确认开关等运维字段。"""
        return {
            "target_total": self.target_total,
            "train_quota": self.train_quota,
            "validation_quota": self.validation_quota,
            "seed": self.seed,
            "complexity_weights": {
                bucket.value: weight
                for bucket, weight in self.complexity_weights().items()
            },
        }


class TeacherAttempt(StrictRecord):
    schema_version: int = 2
    task_id: str
    split: str
    db_id: str
    complexity: ComplexityBucket
    complexity_score: int = Field(ge=0)
    config_hash: str
    sampling_manifest_hash: str
    accepted: bool
    rejection_reason: str | None = None
    episode: EpisodeResult
    migrated_from: str | None = None


class TeacherActionClient(Protocol):
    @property
    def config_hash(self) -> str: ...

    @property
    def real_api(self) -> bool: ...

    @property
    def spent_usd(self) -> float: ...

    async def complete_action(
        self, messages: list[dict[str, Any]], *, max_tokens: int | None = None
    ) -> LLMCompletion: ...


class _ThreadBridgePolicy:
    def __init__(self, client: TeacherActionClient, loop: asyncio.AbstractEventLoop) -> None:
        self.client = client
        self.loop = loop

    def generate(self, messages: list[dict[str, Any]], *, max_tokens: int) -> ModelResponse:
        del max_tokens
        # Teacher 的上限包含隐藏思考；Action 的 512 token 上限由 Harness 单独校验。
        future: Future[LLMCompletion] = asyncio.run_coroutine_threadsafe(
            self.client.complete_action(messages), self.loop
        )
        completion = future.result()
        action_tokens = completion.action_tokens
        return ModelResponse(
            content=stable_json(completion.action.model_dump(mode="json")),
            usage={
                "input_tokens": completion.input_tokens,
                "output_tokens": action_tokens if action_tokens is not None else 0,
                "billed_output_tokens": completion.output_tokens,
                "reasoning_tokens": completion.reasoning_tokens or 0,
                "cached_input_tokens": completion.cached_input_tokens,
                "latency_ms": round(completion.latency_ms),
                "cost_micro_usd": round(completion.cost_usd * 1_000_000),
            },
        )


def _accepted(episode: EpisodeResult) -> tuple[bool, str | None]:
    if not episode.valid_for_training:
        return False, "invalid_infrastructure"
    if episode.terminal_reason is not TerminalReason.SUBMITTED:
        return False, episode.terminal_reason.value
    if episode.reward != 1.0:
        return False, "execution_not_correct"
    if any(event.action is None for event in episode.events):
        return False, "protocol_error"
    if any(event.observation.error_code == "unsafe_sql" for event in episode.events):
        return False, "unsafe_sql"
    return True, None


def _append_attempt(path: Path, attempt: TeacherAttempt) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(stable_json(attempt.model_dump(mode="json")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def historical_cost_usd(path: Path) -> float:
    """从已落盘记录恢复整个采集活动的累计费用。"""
    total_micro_usd = 0
    for row in read_jsonl(path):
        episode = row.get("episode")
        if not isinstance(episode, dict):
            continue
        usage = episode.get("usage")
        if isinstance(usage, dict):
            total_micro_usd += int(usage.get("cost_micro_usd", 0))
    return total_micro_usd / 1_000_000


def _client_behavior_hash(client: TeacherActionClient) -> str:
    value = getattr(client, "behavior_config_hash", None)
    return str(value) if value is not None else client.config_hash


def _collection_hash(
    client: TeacherActionClient,
    runtime: RuntimeConfig,
    config: CollectorConfig,
    sampling_manifest_hash: str,
) -> str:
    payload = {
        "schema_version": 2,
        "client_behavior": _client_behavior_hash(client),
        "collector": config.behavior_payload(),
        "runtime": runtime.model_dump(mode="json"),
        "sampling_manifest_hash": sampling_manifest_hash,
        "acceptance_version": 1,
    }
    return hashlib.sha256(stable_json(payload).encode("utf-8")).hexdigest()


async def collect_trajectories(
    tasks: list[TaskView],
    answers: list[HiddenAnswer],
    db_root: Path,
    output_path: Path,
    client: TeacherActionClient,
    *,
    runtime: RuntimeConfig,
    config: CollectorConfig,
) -> dict[str, Any]:
    config.validate_quotas()
    if client.real_api and not config.confirm_real_api:
        raise PermissionError("真实 Teacher API 必须显式设置 confirm_real_api=true")
    split_targets = {
        "train": config.train_quota,
        "validation": config.validation_quota,
    }
    plan = build_sampling_plan(
        tasks,
        answers,
        split_targets=split_targets,
        complexity_weights=config.complexity_weights(),
        seed=config.seed,
    )
    collection_hash = _collection_hash(client, runtime, config, plan.manifest_hash)
    answer_by_id = {answer.task_id: answer for answer in answers}
    task_by_id = {task.task_id: task for task in tasks}
    existing = [TeacherAttempt.model_validate(row) for row in read_jsonl(output_path)]
    mismatched = [row.task_id for row in existing if row.config_hash != collection_hash]
    if mismatched:
        raise ValueError("续采文件包含不同 config_hash，拒绝混合轨迹")
    completed_ids = {row.task_id for row in existing}
    accepted_ids = {row.task_id for row in existing if row.accepted}
    scheduler = StratifiedScheduler(
        plan,
        completed_ids=completed_ids,
        accepted_ids=accepted_ids,
    )
    attempts = len(existing)
    run_attempts = 0
    loop = asyncio.get_running_loop()

    async def collect_one(task: TaskView) -> TeacherAttempt:
        answer = answer_by_id[task.task_id]
        episode = await asyncio.to_thread(
            run_episode,
            task,
            answer,
            db_root / task.db_ref,
            _ThreadBridgePolicy(client, loop),
            runtime=runtime,
            config_hash=collection_hash,
        )
        accepted, reason = _accepted(episode)
        complexity = plan.task_complexity[task.task_id]
        return TeacherAttempt(
            task_id=task.task_id,
            split=task.split,
            db_id=task.db_id,
            complexity=complexity.bucket,
            complexity_score=complexity.score,
            config_hash=collection_hash,
            sampling_manifest_hash=plan.manifest_hash,
            accepted=accepted,
            rejection_reason=reason,
            episode=episode,
        )

    while (
        not scheduler.complete
        and attempts < config.max_attempts
        and run_attempts < config.run_attempt_limit
    ):
        batch_size = min(
            config.concurrency,
            config.max_attempts - attempts,
            config.run_attempt_limit - run_attempts,
        )
        task_ids = scheduler.next_task_ids(batch_size)
        if not task_ids:
            break
        results = await asyncio.gather(*(collect_one(task_by_id[task_id]) for task_id in task_ids))
        for attempt in results:
            _append_attempt(output_path, attempt)
            scheduler.register(attempt.task_id, accepted=attempt.accepted)
            completed_ids.add(attempt.task_id)
            attempts += 1
            run_attempts += 1

    all_attempts = existing + [
        TeacherAttempt.model_validate(row)
        for row in read_jsonl(output_path)[len(existing) :]
    ]
    accepted = [attempt for attempt in all_attempts if attempt.accepted]
    split_counts = Counter(attempt.split for attempt in accepted)
    complexity_counts = Counter(attempt.complexity.value for attempt in accepted)
    database_counts = Counter(f"{attempt.split}:{attempt.db_id}" for attempt in accepted)
    complete = scheduler.complete
    return {
        "schema_version": 2,
        "target_total": config.target_total,
        "accepted_total": len(accepted),
        "accepted_train": split_counts["train"],
        "accepted_validation": split_counts["validation"],
        "accepted_complexity": dict(sorted(complexity_counts.items())),
        "accepted_by_database": dict(sorted(database_counts.items())),
        "attempts": attempts,
        "attempts_this_run": run_attempts,
        "max_attempts": config.max_attempts,
        "run_attempt_limit": config.run_attempt_limit,
        "complete": complete,
        "paused_at_batch_boundary": not complete and run_attempts >= config.run_attempt_limit,
        "spent_usd": client.spent_usd,
        "config_hash": collection_hash,
        "sampling_manifest": plan.manifest(),
        "reallocations": scheduler.reallocations,
    }
