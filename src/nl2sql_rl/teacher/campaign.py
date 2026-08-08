"""跨 Harness 版本保存 Teacher 活动的累计尝试与费用。"""

from __future__ import annotations

import hashlib
from pathlib import Path

from pydantic import Field

from nl2sql_rl.io_utils import read_jsonl, stable_json, write_json
from nl2sql_rl.models import StrictRecord


class CampaignState(StrictRecord):
    schema_version: int = 1
    campaign_id: str
    target_total: int = Field(ge=1)
    max_attempts: int = Field(ge=1)
    cost_limit_usd: float = Field(gt=0)
    attempts: int = Field(default=0, ge=0)
    spent_micro_usd: int = Field(default=0, ge=0)
    sampling_manifest_hash: str | None = None
    teacher_behavior_hash: str | None = None
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
    cost_limit_usd: float,
    sampling_manifest_hash: str | None = None,
    teacher_behavior_hash: str | None = None,
    pricing_hash: str | None = None,
) -> CampaignState:
    """先恢复所有已知落盘请求，再创建 API 客户端费用账本。"""
    if path.is_file():
        import json

        state = CampaignState.model_validate(json.loads(path.read_text(encoding="utf-8")))
        if state.target_total != target_total or state.max_attempts != max_attempts:
            raise ValueError("Teacher campaign 的目标或尝试上限不能在续采时改变")
        if abs(state.cost_limit_usd - cost_limit_usd) > 1e-12:
            raise ValueError("Teacher campaign 的累计费用上限不能在续采时改变")
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
        for field_name, provided, label in (
            ("teacher_behavior_hash", teacher_behavior_hash, "Teacher 行为配置"),
            ("pricing_hash", pricing_hash, "计费口径"),
        ):
            current = getattr(state, field_name)
            if provided is not None and current is not None and current != provided:
                raise ValueError(f"Teacher campaign 的{label}不能在续采时改变")
            if current is None and provided is not None:
                state = state.model_copy(update={field_name: provided})
    else:
        state = CampaignState(
            campaign_id=_new_campaign_id(target_total, max_attempts),
            target_total=target_total,
            max_attempts=max_attempts,
            cost_limit_usd=cost_limit_usd,
            sampling_manifest_hash=sampling_manifest_hash,
            teacher_behavior_hash=teacher_behavior_hash,
            pricing_hash=pricing_hash,
        )
    recorded = set(state.recorded_episode_ids)
    attempts = state.attempts
    spent_micro_usd = state.spent_micro_usd
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
            recorded.add(episode_id)
            attempts += 1
            spent_micro_usd += cost
    state = state.model_copy(
        update={
            "attempts": attempts,
            "spent_micro_usd": spent_micro_usd,
            "recorded_episode_ids": sorted(recorded),
        }
    )
    if state.spent_usd > state.cost_limit_usd + 1e-12:
        raise ValueError("Teacher campaign 历史费用已超过累计上限")
    write_json(path, state.model_dump(mode="json"))
    return state


def register_campaign_attempt(
    path: Path,
    state: CampaignState,
    *,
    episode_id: str,
    spent_usd: float,
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
            "recorded_episode_ids": sorted(recorded),
            "harness_config_hashes": harness_hashes,
        }
    )
    write_json(path, updated.model_dump(mode="json"))
    return updated
