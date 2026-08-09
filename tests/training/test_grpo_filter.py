from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from nl2sql_rl.training.grpo_filter import (
    DynamicGroupFilterExhausted,
    DynamicGroupFilterState,
    analyze_reward_groups,
    append_filter_metrics,
)
from nl2sql_rl.training.grpo_reward import masked_group_advantages


def _decision(groups: list[tuple[str, list[float]]]):
    uids: list[str] = []
    rewards: list[float] = []
    for uid, values in groups:
        uids.extend([uid] * len(values))
        rewards.extend(values)
    return analyze_reward_groups(uids, rewards, group_size=4)


@pytest.mark.parametrize("reward", [1.0, 0.0, -0.2, -0.4, -1.0])
def test_all_equal_terminal_rewards_drop_the_whole_group(reward: float) -> None:
    decision = _decision([("prompt", [reward] * 4)])
    assert [group.uid for group in decision.uniform_groups] == ["prompt"]
    assert decision.diverse_groups == ()


def test_mixed_negative_rewards_still_form_an_effective_group() -> None:
    decision = _decision([("prompt", [-0.2, -0.4, -0.2, -1.0])])
    assert decision.uniform_groups == ()
    assert [group.uid for group in decision.diverse_groups] == ["prompt"]


def test_interleaved_uids_keep_complete_group_and_original_row_order() -> None:
    decision = analyze_reward_groups(
        ["mixed", "uniform", "mixed", "uniform", "mixed", "uniform", "mixed", "uniform"],
        [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, -0.2, 0.0],
        group_size=4,
    )
    state = DynamicGroupFilterState(target_groups=1, group_size=4, max_generation_batches=1)
    selection = state.register_batch(decision)
    assert selection.ready is True
    assert selection.selected_group_ids == ("mixed",)
    assert selection.selected_indices == (0, 2, 4, 6)


@pytest.mark.parametrize(
    ("uids", "rewards", "message"),
    [
        (["a"] * 3, [1.0, 0.0, 1.0], "应有 4 条"),
        (["a"] * 4, [1.0, 0.0, math.nan, 1.0], "非有限"),
        (["a"] * 4, [1.0, 0.0, math.inf, 1.0], "非有限"),
        (["a"] * 4, [1.0, 0.0, None, 1.0], "无法解析"),
        (["a"] * 4, [1.0, 0.0, 1.0], "数量不一致"),
    ],
)
def test_invalid_reward_groups_fail_closed(
    uids: list[str], rewards: list[float], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        analyze_reward_groups(uids, rewards, group_size=4)


def test_filter_accumulates_across_batches_and_drops_surplus_group() -> None:
    state = DynamicGroupFilterState(target_groups=2, group_size=4, max_generation_batches=3)
    first = state.register_batch(
        _decision(
            [
                ("uniform", [1.0] * 4),
                ("first-diverse", [1.0, 0.0, 0.0, 0.0]),
            ]
        )
    )
    assert first.ready is False
    assert first.selected_group_ids == ("first-diverse",)

    second = state.register_batch(
        _decision(
            [
                ("second-diverse", [-0.2, -0.4, -0.2, -0.4]),
                ("surplus-diverse", [1.0, 0.0, 1.0, 0.0]),
            ]
        )
    )
    assert second.ready is True
    assert second.selected_group_ids == ("second-diverse",)
    assert second.surplus_group_ids == ("surplus-diverse",)
    summary = state.summary(status="updated")
    assert summary == {
        "schema_version": 1,
        "status": "updated",
        "generation_batches": 2,
        "generated_groups": 4,
        "generated_rollouts": 16,
        "diverse_groups_seen": 3,
        "buffered_diverse_groups": 2,
        "retained_groups": 2,
        "retained_rollouts": 8,
        "discarded_uniform_groups": 1,
        "discarded_surplus_diverse_groups": 1,
        "uniform_reward_counts": {"1": 1},
        "effective_group_rate": 0.5,
        "extra_rollouts": 8,
    }


def test_filter_cap_raises_without_turning_uniform_groups_into_an_update() -> None:
    state = DynamicGroupFilterState(target_groups=2, group_size=4, max_generation_batches=2)
    first = state.register_batch(_decision([("a", [0.0] * 4), ("b", [-0.4] * 4)]))
    assert first.ready is False
    with pytest.raises(DynamicGroupFilterExhausted) as raised:
        state.register_batch(_decision([("c", [1.0] * 4), ("d", [-1.0] * 4)]))
    summary = raised.value.summary
    assert summary["status"] == "max_batches_exceeded"
    assert summary["retained_groups"] == 0
    assert summary["buffered_diverse_groups"] == 0
    assert summary["generated_rollouts"] == 16


def test_duplicate_uid_across_generation_batches_is_rejected() -> None:
    state = DynamicGroupFilterState(target_groups=2, group_size=4, max_generation_batches=2)
    state.register_batch(_decision([("same", [1.0, 0.0, 1.0, 0.0])]))
    with pytest.raises(ValueError, match="UID 重复"):
        state.register_batch(_decision([("same", [-0.2, -0.4, -0.2, -0.4])]))


def test_filter_metrics_are_appended_as_jsonl(tmp_path: Path) -> None:
    output = tmp_path / "metrics/dynamic_filter_metrics.jsonl"
    append_filter_metrics(output, {"global_step": 1, "status": "updated"})
    append_filter_metrics(output, {"global_step": 2, "status": "updated"})
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert rows == [
        {"global_step": 1, "status": "updated"},
        {"global_step": 2, "status": "updated"},
    ]


def test_fake_optimizer_runs_only_after_two_complete_diverse_groups() -> None:
    state = DynamicGroupFilterState(target_groups=2, group_size=4, max_generation_batches=3)
    update_calls = 0
    global_step = 0
    checkpoint_steps: list[int] = []
    kept_rewards: list[float] = []

    batches = [
        [("uniform-a", [1.0] * 4), ("uniform-b", [-0.4] * 4)],
        [("diverse-a", [1.0, 0.0, 1.0, 0.0]), ("uniform-c", [0.0] * 4)],
        [("diverse-b", [-0.2, -0.4, -0.2, -0.4]), ("uniform-d", [-1.0] * 4)],
    ]
    for groups in batches:
        decision = _decision(groups)
        selection = state.register_batch(decision)
        selected = set(selection.selected_group_ids)
        for group in decision.groups:
            if group.uid in selected:
                kept_rewards.extend(group.rewards)
        if selection.ready:
            update_calls += 1
            global_step += 1
            checkpoint_steps.append(global_step)

    assert update_calls == global_step == 1
    assert checkpoint_steps == [1]
    assert len(kept_rewards) == 8
    advantages = masked_group_advantages(kept_rewards, group_size=4)
    assert all(value is not None for value in advantages)
    assert any(value != 0 for value in advantages)
