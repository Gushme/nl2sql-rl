"""Teacher 分层配额、确定性候选顺序与失败替补调度。"""

from __future__ import annotations

import hashlib
import math
from collections import Counter, deque
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from sqlglot import exp, parse_one
from sqlglot.errors import ParseError

from nl2sql_rl.io_utils import stable_json
from nl2sql_rl.models import HiddenAnswer, TaskView


class ComplexityBucket(StrEnum):
    SIMPLE = "simple"
    MODERATE = "moderate"
    CHALLENGING = "challenging"


DEFAULT_COMPLEXITY_WEIGHTS: dict[ComplexityBucket, float] = {
    ComplexityBucket.SIMPLE: 0.30,
    ComplexityBucket.MODERATE: 0.50,
    ComplexityBucket.CHALLENGING: 0.20,
}
SAMPLING_VERSION = "stratified-v3-data-bound"


class SamplingInfeasible(RuntimeError):
    """剩余候选无法满足固定 split/复杂度配额。"""


@dataclass(frozen=True)
class ComplexityResult:
    bucket: ComplexityBucket
    score: int


@dataclass(frozen=True)
class SamplingPlan:
    seed: int
    task_complexity: dict[str, ComplexityResult]
    cell_targets: dict[tuple[str, str, ComplexityBucket], int]
    database_targets: dict[tuple[str, str], int]
    complexity_targets: dict[tuple[str, ComplexityBucket], int]
    ordered_task_ids: dict[tuple[str, str, ComplexityBucket], tuple[str, ...]]
    manifest_hash: str

    def manifest(self) -> dict[str, Any]:
        candidate_counts = {
            f"{split}:{db_id}:{bucket.value}": len(task_ids)
            for (split, db_id, bucket), task_ids in sorted(
                self.ordered_task_ids.items(),
                key=lambda item: (item[0][0], item[0][1], item[0][2].value),
            )
            if task_ids
        }
        candidates_by_split: Counter[str] = Counter()
        candidates_by_complexity: Counter[str] = Counter()
        for (split, _, bucket), task_ids in self.ordered_task_ids.items():
            candidates_by_split[split] += len(task_ids)
            candidates_by_complexity[f"{split}:{bucket.value}"] += len(task_ids)
        return {
            "schema_version": 1,
            "sampling_version": SAMPLING_VERSION,
            "seed": self.seed,
            "candidate_total": sum(candidates_by_split.values()),
            "candidates_by_split": dict(sorted(candidates_by_split.items())),
            "candidates_by_complexity": dict(sorted(candidates_by_complexity.items())),
            "candidate_cells": candidate_counts,
            "database_targets": {
                f"{split}:{db_id}": target
                for (split, db_id), target in sorted(self.database_targets.items())
            },
            "complexity_targets": {
                f"{split}:{bucket.value}": target
                for (split, bucket), target in sorted(
                    self.complexity_targets.items(), key=lambda item: (item[0][0], item[0][1].value)
                )
            },
            "cell_targets": {
                f"{split}:{db_id}:{bucket.value}": target
                for (split, db_id, bucket), target in sorted(
                    self.cell_targets.items(),
                    key=lambda item: (item[0][0], item[0][1], item[0][2].value),
                )
                if target
            },
            "manifest_hash": self.manifest_hash,
        }


def classify_sql_complexity(sql: str) -> ComplexityResult:
    """按固定 AST 规则生成非官方、仅用于采样的复杂度代理。"""
    try:
        tree = parse_one(sql, read="sqlite")
    except ParseError as exc:
        raise ValueError(f"Gold SQL 无法用于复杂度分桶：{exc}") from exc
    if tree is None:
        raise ValueError("Gold SQL 无法用于复杂度分桶：空 AST")
    joins = sum(1 for _ in tree.find_all(exp.Join))
    nested_selects = max(0, sum(1 for _ in tree.find_all(exp.Select)) - 1)
    set_operations = sum(
        1
        for node_type in (exp.Union, exp.Intersect, exp.Except)
        for _ in tree.find_all(node_type)
    )
    score = (
        min(3, joins)
        + 2 * nested_selects
        + 2 * set_operations
        + int(tree.find(exp.Group) is not None)
        + int(tree.find(exp.Having) is not None)
        + int(any(True for _ in tree.find_all(exp.AggFunc)))
        + 2 * sum(1 for _ in tree.find_all(exp.Window))
        + int(tree.find(exp.With) is not None)
    )
    if score <= 1:
        bucket = ComplexityBucket.SIMPLE
    elif score <= 3:
        bucket = ComplexityBucket.MODERATE
    else:
        bucket = ComplexityBucket.CHALLENGING
    return ComplexityResult(bucket=bucket, score=score)


def _tie_hash(seed: int, value: str) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


def _largest_remainder(
    total: int,
    weights: dict[str, float],
    *,
    seed: int,
) -> dict[str, int]:
    if total < 0 or not weights or any(value < 0 for value in weights.values()):
        raise ValueError("最大余数分配参数不合法")
    weight_sum = sum(weights.values())
    if weight_sum <= 0:
        raise ValueError("最大余数分配的权重和必须大于 0")
    raw = {key: total * value / weight_sum for key, value in weights.items()}
    result = {key: math.floor(value) for key, value in raw.items()}
    remaining = total - sum(result.values())
    order = sorted(
        weights,
        key=lambda key: (-(raw[key] - result[key]), _tie_hash(seed, key)),
    )
    for key in order[:remaining]:
        result[key] += 1
    return result


def _bounded_database_quotas(
    total: int,
    database_counts: Counter[str],
    *,
    seed: int,
) -> dict[str, int]:
    """按 sqrt 候选量分配，并保证每库至少一条且不超过候选量。"""
    if total < len(database_counts):
        raise SamplingInfeasible(
            f"数据库覆盖至少需要 {len(database_counts)} 条，当前目标只有 {total} 条"
        )
    quotas = {db_id: 1 for db_id in database_counts}
    remaining = total - len(quotas)
    while remaining:
        eligible = {
            db_id: math.sqrt(count)
            for db_id, count in database_counts.items()
            if quotas[db_id] < count
        }
        if not eligible:
            raise SamplingInfeasible("数据库候选总量不足以满足采样目标")
        additions = _largest_remainder(remaining, eligible, seed=seed)
        progressed = 0
        for db_id, requested in additions.items():
            increment = min(requested, database_counts[db_id] - quotas[db_id])
            quotas[db_id] += increment
            progressed += increment
        if progressed == 0:
            # 最大余数法在极端小配额下仍应分配，但保留确定性兜底。
            db_id = min(eligible, key=lambda value: _tie_hash(seed, value))
            quotas[db_id] += 1
            progressed = 1
        remaining -= progressed
    return quotas


def _maximum_flow_allocations(
    capacities: dict[tuple[str, ComplexityBucket], int],
    database_targets: dict[str, int],
    complexity_targets: dict[ComplexityBucket, int],
) -> dict[tuple[str, ComplexityBucket], int]:
    """在数据库行配额和复杂度列配额之间求确定性整数流。"""
    databases = sorted(database_targets)
    complexities = list(ComplexityBucket)
    source = 0
    database_offset = 1
    complexity_offset = database_offset + len(databases)
    sink = complexity_offset + len(complexities)
    residual: list[dict[int, int]] = [{} for _ in range(sink + 1)]
    original: dict[tuple[int, int], int] = {}

    def add_edge(left: int, right: int, capacity: int) -> None:
        residual[left][right] = residual[left].get(right, 0) + capacity
        residual[right].setdefault(left, 0)
        original[left, right] = original.get((left, right), 0) + capacity

    for index, db_id in enumerate(databases):
        add_edge(source, database_offset + index, database_targets[db_id])
    for db_index, db_id in enumerate(databases):
        for complexity_index, bucket in enumerate(complexities):
            add_edge(
                database_offset + db_index,
                complexity_offset + complexity_index,
                capacities.get((db_id, bucket), 0),
            )
    for index, bucket in enumerate(complexities):
        add_edge(complexity_offset + index, sink, complexity_targets[bucket])

    flow = 0
    while True:
        parent: list[int | None] = [None] * (sink + 1)
        parent[source] = -1
        queue: deque[int] = deque([source])
        while queue and parent[sink] is None:
            node = queue.popleft()
            for neighbor in sorted(residual[node]):
                if residual[node][neighbor] > 0 and parent[neighbor] is None:
                    parent[neighbor] = node
                    queue.append(neighbor)
        if parent[sink] is None:
            break
        increment = 10**9
        node = sink
        while node != source:
            previous = parent[node]
            assert previous is not None
            increment = min(increment, residual[previous][node])
            node = previous
        node = sink
        while node != source:
            previous = parent[node]
            assert previous is not None
            residual[previous][node] -= increment
            residual[node][previous] += increment
            node = previous
        flow += increment

    required = sum(database_targets.values())
    if flow != required:
        raise SamplingInfeasible(f"交叉配额不可行：最大流 {flow} != {required}")
    allocations: dict[tuple[str, ComplexityBucket], int] = {}
    for db_index, db_id in enumerate(databases):
        left = database_offset + db_index
        for complexity_index, bucket in enumerate(complexities):
            right = complexity_offset + complexity_index
            capacity = original.get((left, right), 0)
            allocations[db_id, bucket] = capacity - residual[left].get(right, 0)
    return allocations


def build_sampling_plan(
    tasks: list[TaskView],
    answers: list[HiddenAnswer],
    *,
    split_targets: dict[str, int],
    complexity_weights: dict[ComplexityBucket, float] | None = None,
    seed: int = 42,
) -> SamplingPlan:
    answer_by_id = {answer.task_id: answer for answer in answers}
    task_ids = [task.task_id for task in tasks]
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("Teacher 候选包含重复 task_id")
    if set(task_ids) != set(answer_by_id):
        raise ValueError("Teacher TaskView 与 HiddenAnswer 的 task_id 不完全一致")
    selected_weights = complexity_weights or DEFAULT_COMPLEXITY_WEIGHTS
    normalized_weights = {
        bucket: float(selected_weights.get(bucket, 0.0)) for bucket in ComplexityBucket
    }
    task_complexity = {
        task.task_id: classify_sql_complexity(answer_by_id[task.task_id].gold_sql)
        for task in tasks
    }
    ordered_task_ids: dict[tuple[str, str, ComplexityBucket], tuple[str, ...]] = {}
    cell_targets: dict[tuple[str, str, ComplexityBucket], int] = {}
    database_targets: dict[tuple[str, str], int] = {}
    complexity_targets: dict[tuple[str, ComplexityBucket], int] = {}

    for split, target in split_targets.items():
        split_tasks = [task for task in tasks if task.split == split]
        if target < 0 or target > len(split_tasks):
            raise SamplingInfeasible(
                f"split={split} 的目标 {target} 超过候选数量 {len(split_tasks)}"
            )
        database_counts: Counter[str] = Counter(task.db_id for task in split_tasks)
        db_quotas = _bounded_database_quotas(target, database_counts, seed=seed)
        complexity_quotas_raw = _largest_remainder(
            target,
            {bucket.value: weight for bucket, weight in normalized_weights.items()},
            seed=seed,
        )
        complexity_quotas = {
            bucket: complexity_quotas_raw[bucket.value] for bucket in ComplexityBucket
        }
        capacities: Counter[tuple[str, ComplexityBucket]] = Counter(
            (task.db_id, task_complexity[task.task_id].bucket) for task in split_tasks
        )
        allocations = _maximum_flow_allocations(capacities, db_quotas, complexity_quotas)
        for db_id, quota in db_quotas.items():
            database_targets[split, db_id] = quota
        for bucket, quota in complexity_quotas.items():
            complexity_targets[split, bucket] = quota
        for db_id in sorted(database_counts):
            for bucket in ComplexityBucket:
                cell = (split, db_id, bucket)
                cell_targets[cell] = allocations[db_id, bucket]
                candidate_ids = [
                    task.task_id
                    for task in split_tasks
                    if task.db_id == db_id and task_complexity[task.task_id].bucket is bucket
                ]
                ordered_task_ids[cell] = tuple(
                    sorted(candidate_ids, key=lambda task_id: _tie_hash(seed, task_id))
                )

    # 把 Actor 输入、Gold 内容和分桶成员关系都绑定到清单哈希，防止数据静默漂移。
    candidate_fingerprints = []
    task_by_id = {task.task_id: task for task in tasks}
    for task_id in sorted(task_ids):
        task = task_by_id[task_id]
        answer = answer_by_id[task_id]
        candidate_fingerprints.append(
            {
                "task": task.actor_payload(),
                "gold_sha256": hashlib.sha256(answer.gold_sql.encode("utf-8")).hexdigest(),
                "audit_status": answer.audit_status.value,
                "complexity": task_complexity[task_id].bucket.value,
                "complexity_score": task_complexity[task_id].score,
            }
        )
    manifest_payload = {
        "sampling_version": SAMPLING_VERSION,
        "seed": seed,
        "candidate_fingerprints": candidate_fingerprints,
        "cell_targets": {
            f"{split}:{db_id}:{bucket.value}": target
            for (split, db_id, bucket), target in sorted(
                cell_targets.items(), key=lambda item: (item[0][0], item[0][1], item[0][2].value)
            )
        },
        "ordered_task_ids": {
            f"{split}:{db_id}:{bucket.value}": list(ordered_ids)
            for (split, db_id, bucket), ordered_ids in sorted(
                ordered_task_ids.items(),
                key=lambda item: (item[0][0], item[0][1], item[0][2].value),
            )
        },
    }
    manifest_hash = hashlib.sha256(stable_json(manifest_payload).encode()).hexdigest()
    return SamplingPlan(
        seed=seed,
        task_complexity=task_complexity,
        cell_targets=cell_targets,
        database_targets=database_targets,
        complexity_targets=complexity_targets,
        ordered_task_ids=ordered_task_ids,
        manifest_hash=manifest_hash,
    )


class StratifiedScheduler:
    """按目标完成率确定性轮转，并在桶耗尽时做受限同复杂度重分配。"""

    def __init__(
        self,
        plan: SamplingPlan,
        *,
        completed_ids: set[str],
        accepted_ids: set[str],
    ) -> None:
        self.plan = plan
        self.targets = dict(plan.cell_targets)
        self.queues = {
            cell: deque(task_id for task_id in task_ids if task_id not in completed_ids)
            for cell, task_ids in plan.ordered_task_ids.items()
        }
        self.task_to_cell = {
            task_id: cell
            for cell, task_ids in plan.ordered_task_ids.items()
            for task_id in task_ids
        }
        self.accepted: Counter[tuple[str, str, ComplexityBucket]] = Counter()
        self.attempted: Counter[tuple[str, str, ComplexityBucket]] = Counter()
        for cell, task_ids in plan.ordered_task_ids.items():
            task_set = set(task_ids)
            self.accepted[cell] = len(task_set.intersection(accepted_ids))
            self.attempted[cell] = len(task_set.intersection(completed_ids))
        self.attempted_by_split: Counter[str] = Counter()
        self.attempted_by_complexity: Counter[tuple[str, ComplexityBucket]] = Counter()
        for (split, _, bucket), count in self.attempted.items():
            self.attempted_by_split[split] += count
            self.attempted_by_complexity[split, bucket] += count
        self.reallocations: list[dict[str, Any]] = []

    @property
    def complete(self) -> bool:
        return all(self.accepted[cell] >= target for cell, target in self.targets.items())

    def register(self, task_id: str, *, accepted: bool) -> None:
        cell = self._cell_for_task(task_id)
        if accepted:
            self.accepted[cell] += 1

    def next_task_ids(self, limit: int) -> list[str]:
        selected: list[str] = []
        pending: Counter[tuple[str, str, ComplexityBucket]] = Counter()
        while len(selected) < limit and not self.complete:
            available = [
                cell
                for cell, target in self.targets.items()
                if self.accepted[cell] + pending[cell] < target and self.queues[cell]
            ]
            if not available:
                if selected:
                    break
                if not self._reallocate_exhausted():
                    deficits = {
                        f"{split}:{db_id}:{bucket.value}": self.targets[cell]
                        - self.accepted[cell]
                        for cell in self.targets
                        if (self.targets[cell] - self.accepted[cell]) > 0
                        for split, db_id, bucket in (cell,)
                    }
                    raise SamplingInfeasible(f"剩余分层候选不足：{deficits}")
                continue
            cell = min(
                available,
                key=lambda item: (
                    (
                        self.attempted_by_split[item[0]]
                        + 1
                    )
                    / sum(
                        target
                        for (split, _), target in self.plan.database_targets.items()
                        if split == item[0]
                    ),
                    (
                        self.attempted_by_complexity[item[0], item[2]]
                        + 1
                    )
                    / max(1, self.plan.complexity_targets[item[0], item[2]]),
                    (self.attempted[item] + 1) / max(1, self.targets[item]),
                    (self.accepted[item] + pending[item]) / max(1, self.targets[item]),
                    _tie_hash(self.plan.seed, f"{item[0]}:{item[1]}:{item[2].value}"),
                ),
            )
            task_id = self.queues[cell].popleft()
            self.attempted[cell] += 1
            pending[cell] += 1
            self.attempted_by_split[cell[0]] += 1
            self.attempted_by_complexity[cell[0], cell[2]] += 1
            selected.append(task_id)
        return selected

    def _cell_for_task(self, task_id: str) -> tuple[str, str, ComplexityBucket]:
        try:
            return self.task_to_cell[task_id]
        except KeyError as exc:
            raise KeyError(f"采样计划不存在 task_id：{task_id}") from exc

    def _database_target_now(self, split: str, db_id: str) -> int:
        return sum(
            target
            for (cell_split, cell_db, _), target in self.targets.items()
            if cell_split == split and cell_db == db_id
        )

    def _reallocate_exhausted(self) -> bool:
        exhausted = sorted(
            (
                cell
                for cell, target in self.targets.items()
                if self.accepted[cell] < target and not self.queues[cell]
            ),
            key=lambda cell: (cell[0], cell[1], cell[2].value),
        )
        for source in exhausted:
            split, source_db, bucket = source
            original_db_target = self.plan.database_targets[split, source_db]
            tolerance = max(1, math.ceil(original_db_target * 0.20))
            lower = max(1, original_db_target - tolerance)
            if self._database_target_now(split, source_db) <= lower:
                continue
            destinations = []
            for destination, queue in self.queues.items():
                destination_split, destination_db, destination_bucket = destination
                if (
                    destination_split != split
                    or destination_bucket is not bucket
                    or not queue
                    or destination_db == source_db
                ):
                    continue
                destination_original = self.plan.database_targets[split, destination_db]
                destination_tolerance = max(1, math.ceil(destination_original * 0.20))
                upper = destination_original + destination_tolerance
                if self._database_target_now(split, destination_db) < upper:
                    destinations.append(destination)
            if not destinations:
                continue
            destination = min(
                destinations,
                key=lambda cell: (
                    self._database_target_now(split, cell[1])
                    / self.plan.database_targets[split, cell[1]],
                    _tie_hash(self.plan.seed, f"{cell[0]}:{cell[1]}:{cell[2].value}"),
                ),
            )
            self.targets[source] -= 1
            self.targets[destination] += 1
            self.reallocations.append(
                {
                    "split": split,
                    "complexity": bucket.value,
                    "from_db_id": source_db,
                    "to_db_id": destination[1],
                }
            )
            return True
        return False
