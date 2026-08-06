"""可断点续采、task 级幂等且按配额验收的 Teacher 轨迹采集器。"""

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


class CollectorConfig(StrictRecord):
    target_total: int = Field(default=1_000, ge=1)
    train_quota: int = Field(default=900, ge=0)
    validation_quota: int = Field(default=100, ge=0)
    max_attempts: int = Field(default=1_500, ge=1)
    concurrency: int = Field(default=4, ge=1, le=32)
    confirm_real_api: bool = False

    def validate_quotas(self) -> None:
        if self.train_quota + self.validation_quota != self.target_total:
            raise ValueError("train_quota + validation_quota 必须等于 target_total")


class TeacherAttempt(StrictRecord):
    schema_version: int = 1
    task_id: str
    split: str
    config_hash: str
    accepted: bool
    rejection_reason: str | None = None
    episode: EpisodeResult


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
        future: Future[LLMCompletion] = asyncio.run_coroutine_threadsafe(
            self.client.complete_action(messages, max_tokens=max_tokens), self.loop
        )
        completion = future.result()
        return ModelResponse(
            content=stable_json(completion.action.model_dump(mode="json")),
            usage={
                "input_tokens": completion.input_tokens,
                "output_tokens": completion.output_tokens,
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
    collection_hash = hashlib.sha256(
        stable_json(
            {
                "client": client.config_hash,
                "collector": config.model_dump(mode="json"),
                "runtime": runtime.model_dump(mode="json"),
            }
        ).encode("utf-8")
    ).hexdigest()
    answer_by_id = {answer.task_id: answer for answer in answers}
    existing = [TeacherAttempt.model_validate(row) for row in read_jsonl(output_path)]
    mismatched = [row.task_id for row in existing if row.config_hash != collection_hash]
    if mismatched:
        raise ValueError("续采文件包含不同 config_hash，拒绝混合轨迹")
    completed_ids = {row.task_id for row in existing}
    accepted_counts = Counter(row.split for row in existing if row.accepted)
    attempts = len(existing)
    task_by_split: dict[str, list[TaskView]] = {}
    for split in ("train", "validation"):
        unique = {
            task.task_id: task
            for task in tasks
            if task.split == split and task.task_id not in completed_ids
        }
        task_by_split[split] = list(unique.values())
    quotas = {"train": config.train_quota, "validation": config.validation_quota}
    loop = asyncio.get_running_loop()

    async def collect_one(task: TaskView) -> TeacherAttempt:
        answer = answer_by_id.get(task.task_id)
        if answer is None:
            raise ValueError(f"缺少 HiddenAnswer：{task.task_id}")
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
        return TeacherAttempt(
            task_id=task.task_id,
            split=task.split,
            config_hash=collection_hash,
            accepted=accepted,
            rejection_reason=reason,
            episode=episode,
        )

    for split in ("train", "validation"):
        candidates = task_by_split[split]
        cursor = 0
        while accepted_counts[split] < quotas[split] and attempts < config.max_attempts:
            remaining_quota = quotas[split] - accepted_counts[split]
            remaining_attempts = config.max_attempts - attempts
            batch_size = min(config.concurrency, remaining_quota, remaining_attempts)
            batch = candidates[cursor : cursor + batch_size]
            if not batch:
                break
            cursor += len(batch)
            results = await asyncio.gather(*(collect_one(task) for task in batch))
            for attempt in results:
                _append_attempt(output_path, attempt)
                completed_ids.add(attempt.task_id)
                attempts += 1
                if attempt.accepted:
                    accepted_counts[split] += 1
    complete = all(accepted_counts[split] == quotas[split] for split in quotas)
    return {
        "schema_version": 1,
        "target_total": config.target_total,
        "accepted_total": accepted_counts["train"] + accepted_counts["validation"],
        "accepted_train": accepted_counts["train"],
        "accepted_validation": accepted_counts["validation"],
        "attempts": attempts,
        "max_attempts": config.max_attempts,
        "complete": complete,
        "spent_usd": client.spent_usd,
        "config_hash": collection_hash,
    }
