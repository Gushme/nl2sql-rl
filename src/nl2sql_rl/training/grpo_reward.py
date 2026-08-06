"""veRL GRPO 使用的确定性 Reward 适配与 CPU 优势校验。"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from nl2sql_rl.models import EpisodeResult


class InvalidRolloutError(RuntimeError):
    """轨迹包含基础设施故障，不能进入 GRPO 优势计算。"""


@dataclass(frozen=True)
class GRPOReward:
    score: float
    valid_for_advantage: bool
    reason: str


def adapt_episode_reward(episode: EpisodeResult) -> GRPOReward:
    """把 harness 的终局 Reward 转为 veRL 可消费的标量。"""
    if not episode.valid_for_training or episode.reward is None:
        raise InvalidRolloutError(
            f"轨迹 {episode.episode_id} 无效，已拒绝进入优势计算"
        )
    if not math.isfinite(episode.reward):
        raise InvalidRolloutError(f"轨迹 {episode.episode_id} 的 Reward 不是有限数")
    return GRPOReward(
        score=float(episode.reward),
        valid_for_advantage=True,
        reason=episode.terminal_reason.value,
    )


def masked_group_advantages(
    rewards: Sequence[float | None], *, group_size: int
) -> list[float | None]:
    """仅用有效样本计算逐组标准化优势；无效位置保持 None。"""
    if group_size < 2:
        raise ValueError("group_size 必须至少为 2")
    if len(rewards) % group_size != 0:
        raise ValueError("Reward 数量必须能被 group_size 整除")
    advantages: list[float | None] = []
    for offset in range(0, len(rewards), group_size):
        group = list(rewards[offset : offset + group_size])
        valid = [float(value) for value in group if value is not None]
        if not valid:
            advantages.extend([None] * group_size)
            continue
        mean = sum(valid) / len(valid)
        variance = sum((value - mean) ** 2 for value in valid) / len(valid)
        standard_deviation = math.sqrt(variance)
        for value in group:
            if value is None:
                advantages.append(None)
            elif standard_deviation == 0:
                advantages.append(0.0)
            else:
                advantages.append((float(value) - mean) / standard_deviation)
    return advantages
