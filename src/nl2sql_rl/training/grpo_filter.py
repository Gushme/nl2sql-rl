"""GRPO 同奖组动态过滤的纯 CPU 逻辑与诊断记录。"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from pydantic import Field

from nl2sql_rl.io_utils import stable_json
from nl2sql_rl.models import StrictRecord


class DynamicGroupFilterConfig(StrictRecord):
    """固定的同奖组过滤配置。"""

    enabled: bool = True
    metric: Literal["seq_reward"] = "seq_reward"
    max_generation_batches_per_update: int = Field(default=10, ge=1, le=100)


@dataclass(frozen=True)
class RewardGroup:
    """同一个 Prompt 产生的一组完整 rollout。"""

    uid: str
    rewards: tuple[float, ...]
    indices: tuple[int, ...]

    @property
    def uniform(self) -> bool:
        return all(reward == self.rewards[0] for reward in self.rewards[1:])


@dataclass(frozen=True)
class GroupFilterDecision:
    """一个生成批次内的同奖组判定结果。"""

    groups: tuple[RewardGroup, ...]
    uniform_groups: tuple[RewardGroup, ...]
    diverse_groups: tuple[RewardGroup, ...]


@dataclass(frozen=True)
class BatchGroupSelection:
    """当前生成批次需要保留到有效更新缓冲区的轨迹。"""

    selected_indices: tuple[int, ...]
    selected_group_ids: tuple[str, ...]
    surplus_group_ids: tuple[str, ...]
    ready: bool


class DynamicGroupFilterExhausted(RuntimeError):
    """达到补采上限后仍无法组成一个有效更新。"""

    def __init__(self, summary: dict[str, object]) -> None:
        super().__init__("达到同奖组动态过滤补采上限，当前有效组不足")
        self.summary = summary


def analyze_reward_groups(
    uids: Sequence[object],
    rewards: Sequence[float],
    *,
    group_size: int,
) -> GroupFilterDecision:
    """按 UID 分组，并严格验证每组的 Reward 数量和有限性。"""
    if group_size < 2:
        raise ValueError("group_size 必须至少为 2")
    if len(uids) != len(rewards):
        raise ValueError("UID 与 Reward 数量不一致")
    if not uids:
        raise ValueError("生成批次不能为空")

    indices_by_uid: dict[str, list[int]] = {}
    rewards_by_uid: dict[str, list[float]] = {}
    for index, (raw_uid, raw_reward) in enumerate(zip(uids, rewards, strict=True)):
        if not isinstance(raw_uid, str) or not raw_uid:
            raise ValueError("每条 rollout 必须带非空字符串 UID")
        try:
            reward = float(raw_reward)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Prompt 组 {raw_uid} 出现无法解析的 Reward") from exc
        if not math.isfinite(reward):
            raise ValueError(f"Prompt 组 {raw_uid} 出现非有限 Reward")
        indices_by_uid.setdefault(raw_uid, []).append(index)
        rewards_by_uid.setdefault(raw_uid, []).append(reward)

    groups: list[RewardGroup] = []
    for uid, indices in indices_by_uid.items():
        if len(indices) != group_size:
            raise ValueError(
                f"Prompt 组 {uid} 应有 {group_size} 条 rollout，实际为 {len(indices)}"
            )
        groups.append(
            RewardGroup(
                uid=uid,
                rewards=tuple(rewards_by_uid[uid]),
                indices=tuple(indices),
            )
        )
    uniform_groups = tuple(group for group in groups if group.uniform)
    diverse_groups = tuple(group for group in groups if not group.uniform)
    return GroupFilterDecision(
        groups=tuple(groups),
        uniform_groups=uniform_groups,
        diverse_groups=diverse_groups,
    )


def _reward_bucket(reward: float) -> str:
    """使用稳定字符串记录同奖组的 Reward，不引入近似分桶。"""
    return format(reward, ".12g")


@dataclass
class DynamicGroupFilterState:
    """聚合多个生成批次，直到凑齐一次有效 GRPO 更新。"""

    target_groups: int
    group_size: int
    max_generation_batches: int
    generation_batches: int = 0
    generated_groups: int = 0
    generated_rollouts: int = 0
    diverse_groups_seen: int = 0
    buffered_diverse_groups: int = 0
    discarded_uniform_groups: int = 0
    discarded_surplus_diverse_groups: int = 0
    uniform_reward_counts: Counter[str] = field(default_factory=Counter)
    _seen_uids: set[str] = field(default_factory=set, repr=False)

    def __post_init__(self) -> None:
        if self.target_groups < 1:
            raise ValueError("target_groups 必须至少为 1")
        if self.group_size < 2:
            raise ValueError("group_size 必须至少为 2")
        if self.max_generation_batches < 1:
            raise ValueError("max_generation_batches 必须至少为 1")

    @property
    def ready(self) -> bool:
        return self.buffered_diverse_groups == self.target_groups

    def register_batch(self, decision: GroupFilterDecision) -> BatchGroupSelection:
        """登记一个生成批次，并返回需要拼入有效缓冲区的完整组。"""
        if self.ready:
            raise RuntimeError("当前有效更新已经凑齐，不能继续登记生成批次")
        if self.generation_batches >= self.max_generation_batches:
            raise DynamicGroupFilterExhausted(self.summary(status="max_batches_exceeded"))

        batch_uids = {group.uid for group in decision.groups}
        duplicate_uids = batch_uids.intersection(self._seen_uids)
        if duplicate_uids:
            raise ValueError(f"同一有效更新内 UID 重复：{sorted(duplicate_uids)}")
        self._seen_uids.update(batch_uids)
        self.generation_batches += 1
        self.generated_groups += len(decision.groups)
        self.generated_rollouts += sum(len(group.indices) for group in decision.groups)
        self.diverse_groups_seen += len(decision.diverse_groups)
        self.discarded_uniform_groups += len(decision.uniform_groups)
        for group in decision.uniform_groups:
            self.uniform_reward_counts[_reward_bucket(group.rewards[0])] += 1

        remaining = self.target_groups - self.buffered_diverse_groups
        selected_groups = decision.diverse_groups[:remaining]
        surplus_groups = decision.diverse_groups[remaining:]
        self.buffered_diverse_groups += len(selected_groups)
        self.discarded_surplus_diverse_groups += len(surplus_groups)

        selected_uids = {group.uid for group in selected_groups}
        selected_indices = tuple(
            sorted(
                index
                for group in decision.groups
                for index in group.indices
                if group.uid in selected_uids
            )
        )
        selection = BatchGroupSelection(
            selected_indices=selected_indices,
            selected_group_ids=tuple(group.uid for group in selected_groups),
            surplus_group_ids=tuple(group.uid for group in surplus_groups),
            ready=self.ready,
        )
        if not self.ready and self.generation_batches >= self.max_generation_batches:
            raise DynamicGroupFilterExhausted(self.summary(status="max_batches_exceeded"))
        return selection

    def summary(self, *, status: str) -> dict[str, object]:
        """生成不含任务文本、SQL 或 UID 的聚合诊断。"""
        retained_groups = self.target_groups if self.ready else 0
        retained_rollouts = retained_groups * self.group_size
        return {
            "schema_version": 1,
            "status": status,
            "generation_batches": self.generation_batches,
            "generated_groups": self.generated_groups,
            "generated_rollouts": self.generated_rollouts,
            "diverse_groups_seen": self.diverse_groups_seen,
            "buffered_diverse_groups": self.buffered_diverse_groups,
            "retained_groups": retained_groups,
            "retained_rollouts": retained_rollouts,
            "discarded_uniform_groups": self.discarded_uniform_groups,
            "discarded_surplus_diverse_groups": self.discarded_surplus_diverse_groups,
            "uniform_reward_counts": dict(sorted(self.uniform_reward_counts.items())),
            "effective_group_rate": (
                retained_groups / self.generated_groups if self.generated_groups else 0.0
            ),
            "extra_rollouts": max(0, self.generated_rollouts - retained_rollouts),
        }


def append_filter_metrics(path: Path, payload: dict[str, object]) -> None:
    """由训练 driver 追加一条 JSONL 诊断，不记录轨迹明细。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(stable_json(payload))
        handle.write("\n")
