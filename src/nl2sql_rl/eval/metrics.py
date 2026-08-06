"""与 BIRD Mini-Dev 固定 revision 对齐的 EX、Soft-F1 与 R-VES。"""

from __future__ import annotations

import math
import platform
import sqlite3
import statistics
from dataclasses import asdict, dataclass
from typing import Any

OFFICIAL_REPOSITORY_REVISION = "b3d4bcbbae9a96934ad812551eb400c7a3b23c12"
OFFICIAL_EVALUATOR_SHA256 = {
    "evaluation_ex.py": "da1bbcd4530be83692d7c650c814ea9704bb710d0c953eb75d02ccb38233cf89",
    "evaluation_f1.py": "ba2700ee09b0c34d7e203788544757e6e6d6c99c0425bb92b3daa3927eee29bf",
    "evaluation_ves.py": "c23d606c002fd9eb75c17ce3cbabdc26d0a791b8291615a75cbf5e9c98ae67df",
}


def exact_execution_match(
    predicted: list[tuple[Any, ...]] | tuple[tuple[Any, ...], ...],
    gold: list[tuple[Any, ...]] | tuple[tuple[Any, ...], ...],
) -> float:
    """官方 EX：忽略行顺序和重复行，直接比较 tuple 集合。"""
    return float(set(predicted) == set(gold))


def _official_row_match(
    predicted_row: tuple[Any, ...], gold_row: tuple[Any, ...]
) -> tuple[float, float, float]:
    total_columns = len(gold_row)
    if total_columns == 0:
        return 0.0, 0.0, 0.0
    matches = sum(value in gold_row for value in predicted_row)
    predicted_only = sum(value not in gold_row for value in predicted_row)
    gold_only = sum(value not in predicted_row for value in gold_row)
    return (
        matches / total_columns,
        predicted_only / total_columns,
        gold_only / total_columns,
    )


def official_soft_f1(
    predicted: list[tuple[Any, ...]] | tuple[tuple[Any, ...], ...],
    gold: list[tuple[Any, ...]] | tuple[tuple[Any, ...], ...],
) -> float:
    """逐行复现官方 Soft-F1 的去重、位置对齐和列内成员匹配语义。"""
    if not predicted and not gold:
        return 1.0
    predicted_rows = list(dict.fromkeys(predicted))
    gold_rows = list(dict.fromkeys(gold))
    matches: list[float] = []
    predicted_only: list[float] = []
    gold_only: list[float] = []
    for index, gold_row in enumerate(gold_rows):
        if index >= len(predicted_rows):
            matches.append(0.0)
            gold_only.append(1.0)
            continue
        match, pred_extra, gold_extra = _official_row_match(predicted_rows[index], gold_row)
        matches.append(match)
        predicted_only.append(pred_extra)
        gold_only.append(gold_extra)
    for _ in range(len(predicted_rows) - len(gold_rows)):
        matches.append(0.0)
        predicted_only.append(1.0)
        gold_only.append(0.0)
    true_positive = sum(matches)
    false_positive = sum(predicted_only)
    false_negative = sum(gold_only)
    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive > 0
        else 0.0
    )
    recall = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative > 0
        else 0.0
    )
    return (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0.0
    )


def clean_official_timings(values: list[float]) -> list[float]:
    if len(values) < 2:
        return values
    mean = statistics.fmean(values)
    std = statistics.pstdev(values)
    if std == 0:
        return values
    cleaned = [value for value in values if mean - 3 * std < value < mean + 3 * std]
    return cleaned or values


def official_ves_reward(time_ratio: float) -> float:
    if time_ratio <= 0:
        return 0.0
    if time_ratio >= 2:
        return 1.25
    if time_ratio >= 1:
        return 1.0
    if time_ratio >= 0.5:
        return 0.75
    if time_ratio >= 0.25:
        return 0.5
    return 0.25


def rves_score_from_ratios(ratios: list[float]) -> float:
    cleaned = clean_official_timings(ratios)
    if not cleaned:
        return 0.0
    reward = official_ves_reward(statistics.fmean(cleaned))
    return math.sqrt(reward) * 100


@dataclass(frozen=True)
class RVESMetadata:
    iterations: int
    platform: str
    machine: str
    processor: str
    python_version: str
    sqlite_version: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def rves_metadata(iterations: int) -> RVESMetadata:
    return RVESMetadata(
        iterations=iterations,
        platform=platform.platform(),
        machine=platform.machine(),
        processor=platform.processor(),
        python_version=platform.python_version(),
        sqlite_version=sqlite3.sqlite_version,
    )
