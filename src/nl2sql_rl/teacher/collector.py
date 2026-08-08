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

from nl2sql_rl.agent.fingerprint import (
    harness_config_hash,
    visible_protocol_hash,
)
from nl2sql_rl.agent.loop import ModelResponse, build_actor_messages, run_episode
from nl2sql_rl.agent.sql_semantics import SQLSemanticError, normalize_sql, physical_tables
from nl2sql_rl.config import RuntimeConfig
from nl2sql_rl.io_utils import read_jsonl, sha256_file, stable_json, write_json
from nl2sql_rl.models import EpisodeResult, HiddenAnswer, StrictRecord, TaskView, TerminalReason
from nl2sql_rl.teacher.campaign import CampaignState, register_campaign_attempt
from nl2sql_rl.teacher.client import LLMCompletion
from nl2sql_rl.teacher.diagnostics import diagnose_attempts
from nl2sql_rl.teacher.sampling import (
    ComplexityBucket,
    StratifiedScheduler,
    build_sampling_plan,
)
from nl2sql_rl.training.sft_data import (
    TokenizerLike,
    chat_messages_token_count,
    sft_token_count,
)


class ContextWindowExceeded(RuntimeError):
    """下一轮请求在发送前已经超过固定的 Qwen SFT 上下文上限。"""

    error_code = "context_overflow"
    request_sent = False


class CollectorConfig(StrictRecord):
    target_total: int = Field(default=1_000, ge=1)
    train_quota: int = Field(default=900, ge=0)
    validation_quota: int = Field(default=100, ge=0)
    max_attempts: int = Field(default=1_500, ge=1)
    run_attempt_limit: int = Field(default=100, ge=1)
    diagnostic_batch_size: int = Field(default=100, ge=20)
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
    schema_version: int = 3
    task_id: str
    split: str
    db_id: str
    complexity: ComplexityBucket
    complexity_score: int = Field(ge=0)
    config_hash: str
    harness_config_hash: str
    visible_protocol_hash: str
    actor_message_hash: str
    sampling_manifest_hash: str
    accepted: bool
    rejection_reason: str | None = None
    episode: EpisodeResult
    sft_tokens: int | None = Field(default=None, ge=1)
    database_size_bytes: int = Field(ge=0)
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
    def __init__(
        self,
        client: TeacherActionClient,
        loop: asyncio.AbstractEventLoop,
        tokenizer: TokenizerLike | None,
        max_context_tokens: int,
    ) -> None:
        self.client = client
        self.loop = loop
        self.tokenizer = tokenizer
        self.max_context_tokens = max_context_tokens

    def generate(self, messages: list[dict[str, Any]], *, max_tokens: int) -> ModelResponse:
        del max_tokens
        if (
            self.tokenizer is not None
            and chat_messages_token_count(messages, self.tokenizer)
            > self.max_context_tokens
        ):
            raise ContextWindowExceeded("下一轮 Teacher 请求超过 16K SFT 上下文上限")
        # Teacher 的上限包含隐藏思考；Action 的 512 token 上限由 Harness 单独校验。
        future: Future[LLMCompletion] = asyncio.run_coroutine_threadsafe(
            self.client.complete_action(messages), self.loop
        )
        completion = future.result()
        content = (
            stable_json(completion.action.model_dump(mode="json"))
            if completion.action is not None
            else completion.action_text or ""
        )
        action_tokens = (
            len(self.tokenizer.encode(content, add_special_tokens=False))
            if self.tokenizer is not None
            else completion.action_tokens
        )
        return ModelResponse(
            content=content,
            usage={
                "input_tokens": completion.input_tokens,
                "output_tokens": action_tokens if action_tokens is not None else 0,
                "action_tokens": action_tokens if action_tokens is not None else 0,
                "billed_output_tokens": completion.output_tokens,
                "reasoning_tokens": completion.reasoning_tokens or 0,
                "reasoning_tokens_reported": int(
                    completion.reasoning_tokens is not None
                ),
                "reasoning_present": int(completion.reasoning_present),
                "cached_input_tokens": completion.cached_input_tokens,
                "latency_ms": round(completion.latency_ms),
                "context_tokens": completion.input_tokens + (action_tokens or 0),
                "normalization_failed": int(completion.normalization_error is not None),
                "native_tool_call": int(completion.response_format == "native_tool_call"),
                "text_json": int(completion.response_format == "text_json"),
                "response_count": 1,
                "cost_micro_usd": round(completion.cost_usd * 1_000_000),
            },
        )


def _accepted(
    task: TaskView,
    episode: EpisodeResult,
    *,
    tokenizer: TokenizerLike | None,
    runtime: RuntimeConfig,
) -> tuple[bool, str | None, int | None]:
    if not episode.valid_for_training:
        return False, episode.infrastructure_error_code or "invalid_infrastructure", None
    if episode.terminal_reason is not TerminalReason.SUBMITTED:
        return False, episode.terminal_reason.value, None
    if episode.reward != 1.0:
        return False, "execution_not_correct", None
    if any(event.action is None for event in episode.events):
        return False, "protocol_error", None
    first_error = next(
        (
            event.observation.error_code or "observation_not_ok"
            for event in episode.events
            if not event.observation.ok or event.observation.error_code is not None
        ),
        None,
    )
    if first_error is not None:
        return False, f"tool_error:{first_error}", None
    described_tables: set[str] = set()
    successful_execute_sql: list[str] = []
    for event in episode.events:
        assert event.action is not None
        if event.action.action == "describe_schema":
            schemas = event.observation.payload.get("schemas")
            if isinstance(schemas, list):
                described_tables.update(
                    str(schema.get("name", "")).casefold()
                    for schema in schemas
                    if isinstance(schema, dict) and schema.get("columns")
                )
        elif event.action.action == "execute_sql":
            sql = event.action.arguments.get("sql")
            if isinstance(sql, str):
                successful_execute_sql.append(sql)
    if not described_tables:
        return False, "missing_schema_description", None
    if not successful_execute_sql:
        return False, "missing_successful_execute", None
    if not episode.events or episode.events[-1].action is None:
        return False, "missing_submit", None
    final_action = episode.events[-1].action
    if final_action.action != "submit_sql" or episode.submitted_sql is None:
        return False, "missing_submit", None
    try:
        if normalize_sql(successful_execute_sql[-1]) != normalize_sql(episode.submitted_sql):
            return False, "submission_sql_mismatch", None
        missing_tables = physical_tables(episode.submitted_sql).difference(described_tables)
    except SQLSemanticError:
        return False, "submission_sql_invalid", None
    if missing_tables:
        return False, "undescribed_table", None
    if tokenizer is None:
        return True, None, None
    for event in episode.events:
        assert event.action is not None
        action_text = stable_json(event.action.model_dump(mode="json"))
        if len(tokenizer.encode(action_text, add_special_tokens=False)) > runtime.max_action_tokens:
            return False, "action_too_long", None
    try:
        token_count = sft_token_count(task, episode, tokenizer)
    except ValueError:
        return False, "invalid_sft_sequence", None
    if token_count > runtime.max_context_tokens:
        return False, "context_overflow", token_count
    return True, None, token_count


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


def _client_used_tokens(client: TeacherActionClient) -> int:
    """兼容不启用 Token 闸门的 mock 客户端。"""
    return int(getattr(client, "used_tokens", 0))


def _client_token_limit(client: TeacherActionClient) -> int | None:
    value = getattr(client, "token_limit", None)
    return int(value) if value is not None else None


def _collection_hash(
    client: TeacherActionClient,
    runtime: RuntimeConfig,
    config: CollectorConfig,
    sampling_manifest_hash: str,
) -> str:
    payload = {
        "schema_version": 3,
        "client_behavior": _client_behavior_hash(client),
        "collector": config.behavior_payload(),
        "harness_config_hash": harness_config_hash(runtime),
        "sampling_manifest_hash": sampling_manifest_hash,
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
    tokenizer: TokenizerLike | None = None,
    diagnostics_dir: Path | None = None,
    campaign_state_path: Path | None = None,
    campaign_state: CampaignState | None = None,
    migration_source: Path | None = None,
) -> dict[str, Any]:
    config.validate_quotas()
    if client.real_api and not config.confirm_real_api:
        raise PermissionError("真实 Teacher API 必须显式设置 confirm_real_api=true")
    if client.real_api and tokenizer is None:
        raise ValueError("真实 Teacher 采集必须提供固定 Qwen tokenizer")
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
    if (
        campaign_state is not None
        and campaign_state.sampling_manifest_hash not in {None, plan.manifest_hash}
    ):
        raise ValueError("Campaign 账本与当前采样清单哈希不一致")
    collection_hash = _collection_hash(client, runtime, config, plan.manifest_hash)
    current_harness_hash = harness_config_hash(runtime)
    current_visible_protocol_hash = visible_protocol_hash()
    migration_report: dict[str, object] | None = None
    if migration_source is not None:
        if tokenizer is None:
            raise ValueError("轨迹迁移必须提供固定 tokenizer")
        from nl2sql_rl.teacher.migration import migrate_accepted_attempts

        migration_report = migrate_accepted_attempts(
            migration_source,
            output_path,
            tasks=tasks,
            answers=answers,
            db_root=db_root,
            runtime=runtime,
            tokenizer=tokenizer,
            plan=plan,
            new_config_hash=collection_hash,
            new_harness_config_hash=current_harness_hash,
        )
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
    campaign_attempts = campaign_state.attempts if campaign_state is not None else len(existing)
    if campaign_attempts > config.max_attempts:
        raise ValueError("Campaign 已记录的尝试数超过 max_attempts")
    run_attempts = 0
    version_paid_attempts = [row for row in existing if row.migrated_from is None]
    remainder = len(version_paid_attempts) % config.diagnostic_batch_size
    current_diagnostic_batch = version_paid_attempts[-remainder:] if remainder else []
    pause_reasons: list[str] = []
    latest_diagnostics: dict[str, Any] | None = None
    loop = asyncio.get_running_loop()
    database_hashes: dict[str, str] = {}
    database_hash_lock = asyncio.Lock()

    async def database_hash(task: TaskView) -> str:
        async with database_hash_lock:
            if task.db_ref not in database_hashes:
                database_hashes[task.db_ref] = await asyncio.to_thread(
                    sha256_file, db_root / task.db_ref
                )
            return database_hashes[task.db_ref]

    async def collect_one(task: TaskView) -> TeacherAttempt:
        answer = answer_by_id[task.task_id]
        expected_database_sha = await database_hash(task)
        episode = await asyncio.to_thread(
            run_episode,
            task,
            answer,
            db_root / task.db_ref,
            _ThreadBridgePolicy(
                client,
                loop,
                tokenizer,
                runtime.max_context_tokens,
            ),
            runtime=runtime,
            config_hash=collection_hash,
            expected_database_sha256=expected_database_sha,
            verify_database_sha=False,
        )
        accepted, reason, token_count = _accepted(
            task,
            episode,
            tokenizer=tokenizer,
            runtime=runtime,
        )
        complexity = plan.task_complexity[task.task_id]
        return TeacherAttempt(
            task_id=task.task_id,
            split=task.split,
            db_id=task.db_id,
            complexity=complexity.bucket,
            complexity_score=complexity.score,
            config_hash=collection_hash,
            harness_config_hash=current_harness_hash,
            visible_protocol_hash=current_visible_protocol_hash,
            actor_message_hash=hashlib.sha256(
                stable_json(build_actor_messages(task)).encode("utf-8")
            ).hexdigest(),
            sampling_manifest_hash=plan.manifest_hash,
            accepted=accepted,
            rejection_reason=reason,
            episode=episode,
            sft_tokens=token_count,
            database_size_bytes=(db_root / task.db_ref).stat().st_size,
        )

    while (
        not scheduler.complete
        and campaign_attempts < config.max_attempts
        and run_attempts < config.run_attempt_limit
        and not pause_reasons
    ):
        batch_size = min(
            config.concurrency,
            config.max_attempts - campaign_attempts,
            config.run_attempt_limit - run_attempts,
            config.diagnostic_batch_size - len(current_diagnostic_batch),
        )
        task_ids = scheduler.next_task_ids(batch_size)
        if not task_ids:
            break
        results = await asyncio.gather(*(collect_one(task_by_id[task_id]) for task_id in task_ids))
        for attempt in results:
            budget_error = attempt.episode.infrastructure_error_code in {
                "cost_limit_exceeded",
                "token_limit_exceeded",
            }
            if budget_error and attempt.episode.infrastructure_request_sent is False:
                pause_reasons.append(
                    f"{attempt.episode.infrastructure_error_code}_before_request"
                )
                continue
            _append_attempt(output_path, attempt)
            scheduler.register(attempt.task_id, accepted=attempt.accepted)
            completed_ids.add(attempt.task_id)
            campaign_attempts += 1
            run_attempts += 1
            version_paid_attempts.append(attempt)
            current_diagnostic_batch.append(attempt)
            if campaign_state is not None and campaign_state_path is not None:
                campaign_state = register_campaign_attempt(
                    campaign_state_path,
                    campaign_state,
                    episode_id=attempt.episode.episode_id,
                    spent_usd=client.spent_usd,
                    used_tokens=_client_used_tokens(client),
                    harness_config_hash=current_harness_hash,
                )
        if not current_diagnostic_batch:
            continue
        batch_complete = len(current_diagnostic_batch) >= config.diagnostic_batch_size
        latest_diagnostics = diagnose_attempts(
            current_diagnostic_batch,
            tasks=tasks,
            answers=answers,
            db_root=db_root,
            runtime=runtime,
            replay_accepted=batch_complete,
            cost_limit_usd=(
                campaign_state.cost_limit_usd if campaign_state is not None else None
            ),
            spent_usd=client.spent_usd,
            token_limit=_client_token_limit(client),
            used_tokens=_client_used_tokens(client),
        )
        latest_diagnostics.update(
            {
                "config_hash": collection_hash,
                "harness_config_hash": current_harness_hash,
                "visible_protocol_hash": current_visible_protocol_hash,
                "sampling_manifest_hash": plan.manifest_hash,
                "teacher_client_config_hash": client.config_hash,
                "teacher_behavior_hash": _client_behavior_hash(client),
                "pricing_hash": getattr(client, "pricing_config_hash", None),
                "campaign_attempts": campaign_attempts,
                "version_attempts": len(version_paid_attempts),
                "spent_usd": client.spent_usd,
                "used_tokens": _client_used_tokens(client),
                "token_limit": _client_token_limit(client),
            }
        )
        pause_reasons.extend(str(value) for value in latest_diagnostics["pause_reasons"])
        if batch_complete:
            if len(version_paid_attempts) == config.diagnostic_batch_size:
                split_accepted = Counter(
                    row.split for row in current_diagnostic_batch if row.accepted
                )
                accepted_in_batch = sum(row.accepted for row in current_diagnostic_batch)
                if accepted_in_batch < 75:
                    pause_reasons.append("pilot_accepted_lt_75")
                if split_accepted["train"] < 68:
                    pause_reasons.append("pilot_train_accepted_lt_68")
                if split_accepted["validation"] < 7:
                    pause_reasons.append("pilot_validation_accepted_lt_7")
                acceptance_rate = max(
                    1e-12,
                    accepted_in_batch / len(current_diagnostic_batch),
                )
                projected_attempts = config.target_total / acceptance_rate
                projected_cost = (
                    client.spent_usd / max(1, campaign_attempts) * projected_attempts
                )
                projected_tokens = (
                    _client_used_tokens(client)
                    / max(1, campaign_attempts)
                    * projected_attempts
                )
                latest_diagnostics["pilot_projection"] = {
                    "attempts": projected_attempts,
                    "cost_usd": projected_cost,
                    "tokens": projected_tokens,
                }
                if projected_attempts > config.max_attempts:
                    pause_reasons.append("projected_attempts_gt_1500")
                if (
                    campaign_state is not None
                    and campaign_state.cost_limit_usd is not None
                    and projected_cost > campaign_state.cost_limit_usd
                ):
                    pause_reasons.append("projected_cost_gt_campaign_limit")
                if (
                    campaign_state is not None
                    and campaign_state.token_limit is not None
                    and projected_tokens > campaign_state.token_limit
                ):
                    pause_reasons.append("projected_tokens_gt_campaign_limit")
            latest_diagnostics["pause_reasons"] = sorted(set(pause_reasons))
            latest_diagnostics["pause"] = bool(pause_reasons)
            if diagnostics_dir is not None:
                batch_number = (
                    (len(version_paid_attempts) - 1) // config.diagnostic_batch_size + 1
                )
                write_json(
                    diagnostics_dir / f"batch_{batch_number:04d}.json",
                    latest_diagnostics,
                )
            current_diagnostic_batch = []

    if not scheduler.complete and campaign_attempts >= config.max_attempts:
        pause_reasons.append("campaign_attempt_limit_reached")
    all_attempts = existing + [
        TeacherAttempt.model_validate(row)
        for row in read_jsonl(output_path)[len(existing) :]
    ]
    accepted = [attempt for attempt in all_attempts if attempt.accepted]
    split_counts = Counter(attempt.split for attempt in accepted)
    complexity_counts = Counter(attempt.complexity.value for attempt in accepted)
    database_counts = Counter(f"{attempt.split}:{attempt.db_id}" for attempt in accepted)
    complete = scheduler.complete
    if current_diagnostic_batch and latest_diagnostics is not None and diagnostics_dir is not None:
        latest_diagnostics["pause_reasons"] = sorted(set(pause_reasons))
        latest_diagnostics["pause"] = bool(pause_reasons)
        write_json(
            diagnostics_dir / f"partial_after_{campaign_attempts:04d}.json",
            latest_diagnostics,
        )
    return {
        "schema_version": 3,
        "target_total": config.target_total,
        "accepted_total": len(accepted),
        "accepted_train": split_counts["train"],
        "accepted_validation": split_counts["validation"],
        "accepted_complexity": dict(sorted(complexity_counts.items())),
        "accepted_by_database": dict(sorted(database_counts.items())),
        "attempts": campaign_attempts,
        "version_attempts": len(version_paid_attempts),
        "attempts_this_run": run_attempts,
        "max_attempts": config.max_attempts,
        "run_attempt_limit": config.run_attempt_limit,
        "complete": complete,
        "paused": bool(pause_reasons),
        "pause_reasons": sorted(set(pause_reasons)),
        "paused_at_batch_boundary": (
            not complete
            and not pause_reasons
            and run_attempts >= config.run_attempt_limit
        ),
        "spent_usd": client.spent_usd,
        "used_tokens": _client_used_tokens(client),
        "token_limit": _client_token_limit(client),
        "config_hash": collection_hash,
        "harness_config_hash": current_harness_hash,
        "visible_protocol_hash": current_visible_protocol_hash,
        "teacher_client_config_hash": client.config_hash,
        "teacher_behavior_hash": _client_behavior_hash(client),
        "pricing_hash": getattr(client, "pricing_config_hash", None),
        "sampling_manifest": plan.manifest(),
        "reallocations": scheduler.reallocations,
        "migration": migration_report,
        "latest_diagnostics": latest_diagnostics,
    }
