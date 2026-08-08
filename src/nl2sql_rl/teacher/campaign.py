"""跨 Harness 版本保存 Teacher 活动的累计尝试、费用与 Token。"""

from __future__ import annotations

import hashlib
from pathlib import Path

from pydantic import Field

from nl2sql_rl.io_utils import read_jsonl, stable_json, write_json
from nl2sql_rl.models import StrictRecord


class CampaignState(StrictRecord):
    schema_version: int = 3
    campaign_id: str
    target_total: int = Field(ge=1)
    max_attempts: int = Field(ge=1)
    cost_limit_usd: float | None = Field(default=None, gt=0)
    token_limit: int | None = Field(default=None, ge=1)
    attempts: int = Field(default=0, ge=0)
    spent_micro_usd: int = Field(default=0, ge=0)
    used_tokens: int = Field(default=0, ge=0)
    sampling_manifest_hash: str | None = None
    teacher_behavior_hash: str | None = None
    teacher_behavior_hashes: list[str] = Field(default_factory=list)
    pricing_hash: str | None = None
    recorded_episode_ids: list[str] = Field(default_factory=list)
    harness_config_hashes: list[str] = Field(default_factory=list)

    @property
    def spent_usd(self) -> float:
        return self.spent_micro_usd / 1_000_000


def _new_campaign_id(target_total: int, max_attempts: int) -> str:
    payload = {"target_total": target_total, "max_attempts": max_attempts, "kind": "teacher"}
    return hashlib.sha256(stable_json(payload).encode("utf-8")).hexdigest()[:20]


def prepare_campaign_state(
    path: Path,
    *,
    attempt_paths: list[Path],
    target_total: int,
    max_attempts: int,
    cost_limit_usd: float | None = None,
    token_limit: int | None = None,
    sampling_manifest_hash: str | None = None,
    teacher_behavior_hash: str | None = None,
    pricing_hash: str | None = None,
    allow_teacher_behavior_upgrade: bool = False,
) -> CampaignState:
    """先恢复所有已知落盘请求，再创建 API 客户端预算账本。"""
    if cost_limit_usd is None and token_limit is None:
        raise ValueError("Teacher campaign 必须设置费用上限或 Token 上限")
    if path.is_file():
        import json

        state = CampaignState.model_validate(json.loads(path.read_text(encoding="utf-8")))
        if state.schema_version < 3:
            state = state.model_copy(update={"schema_version": 3})
        if state.target_total != target_total or state.max_attempts != max_attempts:
            raise ValueError("Teacher campaign 的目标或尝试上限不能在续采时改变")
        if (state.cost_limit_usd is None) != (cost_limit_usd is None) or (
            state.cost_limit_usd is not None
            and cost_limit_usd is not None
            and abs(state.cost_limit_usd - cost_limit_usd) > 1e-12
        ):
            raise ValueError("Teacher campaign 的累计费用上限不能在续采时改变")
        if state.token_limit != token_limit:
            raise ValueError("Teacher campaign 的累计 Token 上限不能在续采时改变")
        if (
            sampling_manifest_hash is not None
            and state.sampling_manifest_hash is not None
            and state.sampling_manifest_hash != sampling_manifest_hash
        ):
            raise ValueError("Teacher campaign 的采样清单不能在续采时改变")
        if state.sampling_manifest_hash is None and sampling_manifest_hash is not None:
            state = state.model_copy(
                update={"sampling_manifest_hash": sampling_manifest_hash}
            )
        current_behavior = state.teacher_behavior_hash
        behavior_history = set(state.teacher_behavior_hashes)
        if current_behavior is not None:
            behavior_history.add(current_behavior)
        if (
            teacher_behavior_hash is not None
            and current_behavior is not None
            and current_behavior != teacher_behavior_hash
        ):
            # 只有显式提供旧轨迹迁移来源时，CLI 才会打开该升级开关。
            if not allow_teacher_behavior_upgrade:
                raise ValueError("Teacher campaign 的 Teacher 行为配置不能在续采时改变")
            current_behavior = teacher_behavior_hash
        elif current_behavior is None and teacher_behavior_hash is not None:
            current_behavior = teacher_behavior_hash
        if teacher_behavior_hash is not None:
            behavior_history.add(teacher_behavior_hash)
        state = state.model_copy(
            update={
                "teacher_behavior_hash": current_behavior,
                "teacher_behavior_hashes": sorted(behavior_history),
            }
        )
        if (
            pricing_hash is not None
            and state.pricing_hash is not None
            and state.pricing_hash != pricing_hash
        ):
            raise ValueError("Teacher campaign 的计费口径不能在续采时改变")
        if state.pricing_hash is None and pricing_hash is not None:
            state = state.model_copy(update={"pricing_hash": pricing_hash})
    else:
        state = CampaignState(
            campaign_id=_new_campaign_id(target_total, max_attempts),
            target_total=target_total,
            max_attempts=max_attempts,
            cost_limit_usd=cost_limit_usd,
            token_limit=token_limit,
            sampling_manifest_hash=sampling_manifest_hash,
            teacher_behavior_hash=teacher_behavior_hash,
            teacher_behavior_hashes=(
                [teacher_behavior_hash] if teacher_behavior_hash is not None else []
            ),
            pricing_hash=pricing_hash,
        )
    recorded = set(state.recorded_episode_ids)
    attempts = state.attempts
    spent_micro_usd = state.spent_micro_usd
    used_tokens = state.used_tokens
    for attempt_path in attempt_paths:
        for row in read_jsonl(attempt_path):
            if row.get("migrated_from") is not None:
                continue
            episode = row.get("episode")
            if not isinstance(episode, dict):
                continue
            episode_id = str(episode.get("episode_id", ""))
            if not episode_id or episode_id in recorded:
                continue
            usage = episode.get("usage")
            cost = int(usage.get("cost_micro_usd", 0)) if isinstance(usage, dict) else 0
            input_tokens = int(usage.get("input_tokens", 0)) if isinstance(usage, dict) else 0
            if isinstance(usage, dict):
                billed_output_value = usage.get("billed_output_tokens")
                billed_output_tokens = int(
                    usage.get("output_tokens", 0)
                    if billed_output_value is None
                    else billed_output_value
                )
            else:
                billed_output_tokens = 0
            recorded.add(episode_id)
            attempts += 1
            spent_micro_usd += cost
            used_tokens += input_tokens + billed_output_tokens
    state = state.model_copy(
        update={
            "attempts": attempts,
            "spent_micro_usd": spent_micro_usd,
            "used_tokens": used_tokens,
            "recorded_episode_ids": sorted(recorded),
        }
    )
    if (
        state.cost_limit_usd is not None
        and state.spent_usd > state.cost_limit_usd + 1e-12
    ):
        raise ValueError("Teacher campaign 历史费用已超过累计上限")
    if state.token_limit is not None and state.used_tokens > state.token_limit:
        raise ValueError("Teacher campaign 历史 Token 已超过累计上限")
    write_json(path, state.model_dump(mode="json"))
    return state


def register_campaign_attempt(
    path: Path,
    state: CampaignState,
    *,
    episode_id: str,
    spent_usd: float,
    used_tokens: int,
    harness_config_hash: str,
) -> CampaignState:
    recorded = set(state.recorded_episode_ids)
    attempts = state.attempts
    if episode_id not in recorded:
        recorded.add(episode_id)
        attempts += 1
    harness_hashes = sorted({*state.harness_config_hashes, harness_config_hash})
    updated = state.model_copy(
        update={
            "attempts": attempts,
            "spent_micro_usd": max(
                state.spent_micro_usd,
                round(spent_usd * 1_000_000),
            ),
            "used_tokens": max(state.used_tokens, used_tokens),
            "recorded_episode_ids": sorted(recorded),
            "harness_config_hashes": harness_hashes,
        }
    )
    write_json(path, updated.model_dump(mode="json"))
    return updated
