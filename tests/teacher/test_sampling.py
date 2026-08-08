from __future__ import annotations

from collections import Counter

from nl2sql_rl.models import AuditStatus, HiddenAnswer, TaskView
from nl2sql_rl.teacher.sampling import (
    ComplexityBucket,
    StratifiedScheduler,
    build_sampling_plan,
    classify_sql_complexity,
)


def test_complexity_proxy_uses_sql_ast_features() -> None:
    assert classify_sql_complexity("SELECT name FROM items").bucket is ComplexityBucket.SIMPLE
    assert (
        classify_sql_complexity(
            "SELECT COUNT(*) FROM items i JOIN groups g ON i.group_id = g.id"
        ).bucket
        is ComplexityBucket.MODERATE
    )
    challenging = classify_sql_complexity(
        "WITH ranked AS (SELECT id, ROW_NUMBER() OVER (ORDER BY id) AS rn FROM items) "
        "SELECT id FROM ranked UNION SELECT id FROM archived"
    )
    assert challenging.bucket is ComplexityBucket.CHALLENGING
    assert challenging.score >= 4


def _plan_fixtures() -> tuple[list[TaskView], list[HiddenAnswer]]:
    sql_by_bucket = {
        ComplexityBucket.SIMPLE: "SELECT name FROM items",
        ComplexityBucket.MODERATE: (
            "SELECT COUNT(*) FROM items i JOIN groups g ON i.group_id = g.id"
        ),
        ComplexityBucket.CHALLENGING: (
            "WITH ranked AS (SELECT id, ROW_NUMBER() OVER (ORDER BY id) AS rn FROM items) "
            "SELECT id FROM ranked UNION SELECT id FROM archived"
        ),
    }
    layout = [
        ComplexityBucket.SIMPLE,
        ComplexityBucket.SIMPLE,
        ComplexityBucket.MODERATE,
        ComplexityBucket.MODERATE,
        ComplexityBucket.CHALLENGING,
        ComplexityBucket.SIMPLE,
        ComplexityBucket.MODERATE,
        ComplexityBucket.MODERATE,
        ComplexityBucket.MODERATE,
        ComplexityBucket.CHALLENGING,
    ]
    tasks: list[TaskView] = []
    answers: list[HiddenAnswer] = []
    for split in ("train", "validation"):
        for index, bucket in enumerate(layout):
            db_id = f"{split}_db_{index // 5}"
            task_id = f"{split}_{index:02d}"
            tasks.append(
                TaskView(
                    task_id=task_id,
                    split=split,
                    db_id=db_id,
                    question=f"问题 {task_id}",
                    db_ref=f"{db_id}/{db_id}.sqlite",
                )
            )
            answers.append(
                HiddenAnswer(
                    task_id=task_id,
                    gold_sql=sql_by_bucket[bucket],
                    audit_status=AuditStatus.PASSED,
                )
            )
    return tasks, answers


def test_sampling_plan_satisfies_cross_quotas_and_is_deterministic() -> None:
    tasks, answers = _plan_fixtures()
    first = build_sampling_plan(
        tasks,
        answers,
        split_targets={"train": 10, "validation": 10},
        seed=42,
    )
    second = build_sampling_plan(
        list(reversed(tasks)),
        list(reversed(answers)),
        split_targets={"train": 10, "validation": 10},
        seed=42,
    )
    assert first.manifest_hash == second.manifest_hash
    for split in ("train", "validation"):
        assert sum(
            target
            for (target_split, _), target in first.database_targets.items()
            if target_split == split
        ) == 10
        assert {
            bucket: first.complexity_targets[split, bucket]
            for bucket in ComplexityBucket
        } == {
            ComplexityBucket.SIMPLE: 3,
            ComplexityBucket.MODERATE: 5,
            ComplexityBucket.CHALLENGING: 2,
        }
        assert all(
            target >= 1
            for (target_split, _), target in first.database_targets.items()
            if target_split == split
        )

    scheduler = StratifiedScheduler(first, completed_ids=set(), accepted_ids=set())
    selected: list[str] = []
    while not scheduler.complete:
        batch = scheduler.next_task_ids(4)
        selected.extend(batch)
        for task_id in batch:
            scheduler.register(task_id, accepted=True)
    counts = Counter(first.task_complexity[task_id].bucket for task_id in selected)
    assert len(selected) == 20
    assert counts == {
        ComplexityBucket.SIMPLE: 6,
        ComplexityBucket.MODERATE: 10,
        ComplexityBucket.CHALLENGING: 4,
    }


def test_sampling_manifest_binds_actor_and_hidden_answer_content() -> None:
    tasks, answers = _plan_fixtures()
    baseline = build_sampling_plan(
        tasks,
        answers,
        split_targets={"train": 10, "validation": 10},
        seed=42,
    )
    changed_tasks = list(tasks)
    changed_tasks[0] = changed_tasks[0].model_copy(update={"question": "已修改的问题"})
    changed_actor = build_sampling_plan(
        changed_tasks,
        answers,
        split_targets={"train": 10, "validation": 10},
        seed=42,
    )
    changed_answers = list(answers)
    changed_answers[0] = changed_answers[0].model_copy(
        update={"gold_sql": "SELECT name FROM items ORDER BY name"}
    )
    changed_gold = build_sampling_plan(
        tasks,
        changed_answers,
        split_targets={"train": 10, "validation": 10},
        seed=42,
    )
    assert changed_actor.manifest_hash != baseline.manifest_hash
    assert changed_gold.manifest_hash != baseline.manifest_hash
