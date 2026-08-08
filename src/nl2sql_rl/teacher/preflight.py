"""在发送 API 请求前，用隐藏 Gold 验证计划任务与 Harness 约束兼容。"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from nl2sql_rl.agent.fingerprint import harness_config_hash
from nl2sql_rl.agent.sql_semantics import SQLSemanticError, physical_tables
from nl2sql_rl.agent.tools import SQLiteToolbox
from nl2sql_rl.config import RuntimeConfig
from nl2sql_rl.io_utils import sha256_file
from nl2sql_rl.models import AgentAction, HiddenAnswer, TaskView
from nl2sql_rl.teacher.sampling import (
    ComplexityBucket,
    StratifiedScheduler,
    build_sampling_plan,
)


def preflight_scheduled_gold(
    tasks: list[TaskView],
    answers: list[HiddenAnswer],
    db_root: Path,
    *,
    runtime: RuntimeConfig,
    limit: int = 100,
    seed: int = 42,
    train_quota: int = 900,
    validation_quota: int = 100,
    complexity_weights: dict[ComplexityBucket, float] | None = None,
) -> dict[str, Any]:
    """按真实前缀顺序走 describe→execute→submit，不把 Gold 写入报告。"""
    plan = build_sampling_plan(
        tasks,
        answers,
        split_targets={"train": train_quota, "validation": validation_quota},
        complexity_weights=complexity_weights,
        seed=seed,
    )
    scheduler = StratifiedScheduler(plan, completed_ids=set(), accepted_ids=set())
    selected_ids = scheduler.next_task_ids(limit)
    task_by_id = {task.task_id: task for task in tasks}
    answer_by_id = {answer.task_id: answer for answer in answers}
    database_hashes_before: dict[str, str] = {}
    records: list[dict[str, Any]] = []

    for task_id in selected_ids:
        task = task_by_id[task_id]
        answer = answer_by_id[task_id]
        database = db_root / task.db_ref
        if task.db_ref not in database_hashes_before:
            database_hashes_before[task.db_ref] = sha256_file(database)
        toolbox = SQLiteToolbox(
            database,
            exploration_timeout_seconds=runtime.exploration_timeout_seconds,
            submission_timeout_seconds=runtime.gold_timeout_seconds,
            max_observation_bytes=runtime.max_observation_bytes,
        )
        error_code: str | None = None
        schema_calls = 0
        schema_truncations = 0
        try:
            required = physical_tables(answer.gold_sql)
        except SQLSemanticError:
            required = set()
            error_code = "gold_ast_error"
        table_list = toolbox.call(
            AgentAction(action="list_tables", arguments={}),
            event_id=f"preflight:{task_id}:list",
        )
        catalog = {
            str(name).casefold(): str(name)
            for name in table_list.payload.get("tables", [])
        }
        missing = sorted(required.difference(catalog))
        if not required:
            error_code = error_code or "gold_has_no_physical_table"
        elif missing:
            error_code = "gold_table_missing"
        else:
            pending = [catalog[name] for name in sorted(required)]
            while pending and error_code is None:
                requested = pending[:5]
                del pending[:5]
                observation = toolbox.call(
                    AgentAction(
                        action="describe_schema",
                        arguments={"tables": requested},
                    ),
                    event_id=f"preflight:{task_id}:schema:{schema_calls}",
                )
                schema_calls += 1
                schema_truncations += int(observation.truncated)
                if not observation.ok:
                    error_code = observation.error_code or "schema_error"
                    break
                omitted = observation.payload.get("omitted_tables")
                if isinstance(omitted, list):
                    pending = [str(name) for name in omitted] + pending
        execute_elapsed_ms: float | None = None
        if error_code is None:
            execution = toolbox.call(
                AgentAction(action="execute_sql", arguments={"sql": answer.gold_sql}),
                event_id=f"preflight:{task_id}:execute",
            )
            execute_elapsed_ms = execution.elapsed_ms
            if not execution.ok:
                error_code = execution.error_code or "execute_error"
        if error_code is None:
            submission = toolbox.call(
                AgentAction(action="submit_sql", arguments={"sql": answer.gold_sql}),
                event_id=f"preflight:{task_id}:submit",
            )
            if not submission.ok:
                error_code = submission.error_code or "submit_error"
        records.append(
            {
                "task_id": task.task_id,
                "split": task.split,
                "db_id": task.db_id,
                "complexity": plan.task_complexity[task.task_id].bucket.value,
                "ok": error_code is None,
                "error_code": error_code,
                "schema_calls": schema_calls,
                "schema_truncations": schema_truncations,
                "execute_elapsed_ms": execute_elapsed_ms,
            }
        )

    database_hashes_after = {
        db_ref: sha256_file(db_root / db_ref) for db_ref in database_hashes_before
    }
    errors = Counter(
        str(record["error_code"]) for record in records if record["error_code"] is not None
    )
    return {
        "schema_version": 1,
        "harness_config_hash": harness_config_hash(runtime),
        "sampling_manifest_hash": plan.manifest_hash,
        "selected": len(records),
        "passed": sum(bool(record["ok"]) for record in records),
        "failed": sum(not bool(record["ok"]) for record in records),
        "error_counts": dict(sorted(errors.items())),
        "selected_by_split": dict(
            sorted(Counter(str(record["split"]) for record in records).items())
        ),
        "selected_by_complexity": dict(
            sorted(Counter(str(record["complexity"]) for record in records).items())
        ),
        "schema_truncations": sum(int(record["schema_truncations"]) for record in records),
        "database_hashes_unchanged": database_hashes_before == database_hashes_after,
        "records": records,
        "gold_sql_saved": False,
    }
