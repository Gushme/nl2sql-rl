from __future__ import annotations

import math
from typing import Any

import pytest

from nl2sql_rl.eval.metrics import (
    exact_execution_match,
    official_soft_f1,
    official_ves_reward,
    rves_score_from_ratios,
)


def _reference_ex(predicted: list[tuple[Any, ...]], gold: list[tuple[Any, ...]]) -> int:
    return int(set(predicted) == set(gold))


def _reference_soft_f1(
    predicted: list[tuple[Any, ...]], gold: list[tuple[Any, ...]]
) -> float:
    if not predicted and not gold:
        return 1.0
    predicted = list(dict.fromkeys(predicted))
    gold = list(dict.fromkeys(gold))
    matches: list[float] = []
    pred_only: list[float] = []
    gold_only: list[float] = []
    for index, gold_row in enumerate(gold):
        if index >= len(predicted):
            matches.append(0)
            gold_only.append(1)
            continue
        pred_row = predicted[index]
        width = len(gold_row)
        matches.append(sum(value in gold_row for value in pred_row) / width)
        pred_only.append(sum(value not in gold_row for value in pred_row) / width)
        gold_only.append(sum(value not in pred_row for value in gold_row) / width)
    for _ in range(len(predicted) - len(gold)):
        matches.append(0)
        pred_only.append(1)
        gold_only.append(0)
    true_positive = sum(matches)
    false_positive = sum(pred_only)
    false_negative = sum(gold_only)
    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive
        else 0
    )
    recall = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative
        else 0
    )
    return 2 * precision * recall / (precision + recall) if precision + recall else 0


@pytest.mark.parametrize(
    ("predicted", "gold"),
    [
        ([], []),
        ([(1,), (1,)], [(1,)]),
        ([(2,), (1,)], [(1,), (2,)]),
        ([(1, 2)], [(1, 3)]),
        ([(1,), (2,)], [(1,)]),
    ],
)
def test_ex_and_soft_f1_match_fixed_official_reference(
    predicted: list[tuple[Any, ...]], gold: list[tuple[Any, ...]]
) -> None:
    assert exact_execution_match(predicted, gold) == _reference_ex(predicted, gold)
    assert official_soft_f1(predicted, gold) == pytest.approx(
        _reference_soft_f1(predicted, gold)
    )


@pytest.mark.parametrize(
    ("ratio", "reward"),
    [(0, 0), (0.1, 0.25), (0.25, 0.5), (0.5, 0.75), (1.0, 1.0), (2.0, 1.25)],
)
def test_rves_reward_buckets_match_official_reference(ratio: float, reward: float) -> None:
    assert official_ves_reward(ratio) == reward
    if ratio > 0:
        assert rves_score_from_ratios([ratio]) == pytest.approx(math.sqrt(reward) * 100)
