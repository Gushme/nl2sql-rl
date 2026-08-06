"""Resumable, process-isolated Gold SQL audit pipeline."""

from __future__ import annotations

import hashlib
import multiprocessing as mp
import os
import queue
import time
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import Field

from nl2sql_rl.data.bird import BirdSourceExample, build_train_inventory, load_train_examples
from nl2sql_rl.io_utils import read_jsonl, sha256_file, stable_json, write_json, write_jsonl
from nl2sql_rl.models import AuditStatus, HiddenAnswer, StrictRecord, TaskView
from nl2sql_rl.sqlite_exec import execute_read_only

AUDITOR_VERSION = "1"
DEFAULT_MAX_ROWS = 100_000
DEFAULT_MAX_RESULT_BYTES = 64 * 1024 * 1024


class GoldAuditRecord(StrictRecord):
    schema_version: int = 1
    auditor_version: str = AUDITOR_VERSION
    task_id: str
    source_index: int
    db_id: str
    status: AuditStatus
    executable: bool
    rl_clean: bool
    empty_result: bool = False
    result_too_large: bool = False
    deterministic: bool | None = None
    row_count: int | None = None
    unique_row_count: int | None = None
    result_bytes: int | None = None
    result_digest: str | None = None
    elapsed_ms: list[float] = Field(default_factory=list)
    error: str | None = None
    sql_sha256: str
    db_sha256: str | None
    fingerprint: str


@dataclass
class _WorkerState:
    db_id: str
    db_path: Path
    examples: list[BirdSourceExample]
    db_sha256: str
    next_index: int = 0
    process: Any = None
    generation: int = 0
    process_started: float | None = None
    current_task_id: str | None = None
    current_started: float | None = None


def audit_fingerprint(
    example: BirdSourceExample,
    db_sha256: str | None,
    timeout_seconds: float,
    max_rows: int,
    max_result_bytes: int,
) -> str:
    payload = {
        "auditor_version": AUDITOR_VERSION,
        "task_id": example.task_id,
        "sql": example.gold_sql,
        "source_match": example.source_match,
        "db_sha256": db_sha256,
        "timeout_seconds": timeout_seconds,
        "max_rows": max_rows,
        "max_result_bytes": max_result_bytes,
    }
    return hashlib.sha256(stable_json(payload).encode("utf-8")).hexdigest()


def _base_record(
    example: BirdSourceExample,
    db_sha256: str | None,
    timeout_seconds: float,
    max_rows: int,
    max_result_bytes: int,
    *,
    status: AuditStatus,
    error: str | None,
) -> GoldAuditRecord:
    return GoldAuditRecord(
        task_id=example.task_id,
        source_index=example.source_index,
        db_id=example.db_id,
        status=status,
        executable=False,
        rl_clean=False,
        error=error,
        sql_sha256=hashlib.sha256(example.gold_sql.encode("utf-8")).hexdigest(),
        db_sha256=db_sha256,
        fingerprint=audit_fingerprint(
            example, db_sha256, timeout_seconds, max_rows, max_result_bytes
        ),
    )


def _audit_example(
    example: BirdSourceExample,
    db_sha256: str,
    timeout_seconds: float,
    max_rows: int,
    max_result_bytes: int,
) -> GoldAuditRecord:
    first = execute_read_only(
        example.gold_sql,
        example.db_path,
        timeout_seconds=timeout_seconds,
        max_rows=max_rows,
        max_result_bytes=max_result_bytes,
    )
    if first.status is not AuditStatus.PASSED:
        record = _base_record(
            example,
            db_sha256,
            timeout_seconds,
            max_rows,
            max_result_bytes,
            status=first.status,
            error=first.error,
        )
        return record.model_copy(update={"elapsed_ms": [first.elapsed_ms]})

    if first.result_too_large:
        return GoldAuditRecord(
            task_id=example.task_id,
            source_index=example.source_index,
            db_id=example.db_id,
            status=AuditStatus.PASSED,
            executable=True,
            rl_clean=False,
            empty_result=first.empty_result,
            result_too_large=True,
            deterministic=None,
            row_count=first.row_count,
            unique_row_count=first.unique_row_count,
            result_bytes=first.result_bytes,
            result_digest=first.digest,
            elapsed_ms=[first.elapsed_ms],
            sql_sha256=hashlib.sha256(example.gold_sql.encode("utf-8")).hexdigest(),
            db_sha256=db_sha256,
            fingerprint=audit_fingerprint(
                example, db_sha256, timeout_seconds, max_rows, max_result_bytes
            ),
        )

    second = execute_read_only(
        example.gold_sql,
        example.db_path,
        timeout_seconds=timeout_seconds,
        max_rows=max_rows,
        max_result_bytes=max_result_bytes,
    )
    if second.status is not AuditStatus.PASSED:
        record = _base_record(
            example,
            db_sha256,
            timeout_seconds,
            max_rows,
            max_result_bytes,
            status=second.status,
            error=f"second execution failed: {second.error}",
        )
        return record.model_copy(update={"elapsed_ms": [first.elapsed_ms, second.elapsed_ms]})
    deterministic = first.digest == second.digest
    status = AuditStatus.PASSED if deterministic else AuditStatus.NONDETERMINISTIC_RESULT
    return GoldAuditRecord(
        task_id=example.task_id,
        source_index=example.source_index,
        db_id=example.db_id,
        status=status,
        executable=deterministic,
        rl_clean=deterministic and not first.empty_result,
        empty_result=first.empty_result,
        result_too_large=False,
        deterministic=deterministic,
        row_count=first.row_count,
        unique_row_count=first.unique_row_count,
        result_bytes=first.result_bytes,
        result_digest=first.digest if deterministic else None,
        elapsed_ms=[first.elapsed_ms, second.elapsed_ms],
        error=None if deterministic else "canonical result digest changed between executions",
        sql_sha256=hashlib.sha256(example.gold_sql.encode("utf-8")).hexdigest(),
        db_sha256=db_sha256,
        fingerprint=audit_fingerprint(
            example, db_sha256, timeout_seconds, max_rows, max_result_bytes
        ),
    )


def _database_worker(
    db_id: str,
    db_sha256: str,
    examples: list[BirdSourceExample],
    timeout_seconds: float,
    max_rows: int,
    max_result_bytes: int,
    generation: int,
    result_queue: Any,
) -> None:
    for example in examples:
        result_queue.put(
            {
                "kind": "start",
                "db_id": db_id,
                "generation": generation,
                "task_id": example.task_id,
            }
        )
        record = _audit_example(
            example, db_sha256, timeout_seconds, max_rows, max_result_bytes
        )
        result_queue.put(
            {
                "kind": "result",
                "db_id": db_id,
                "generation": generation,
                "record": record.model_dump(mode="json"),
            }
        )
    result_queue.put({"kind": "done", "db_id": db_id, "generation": generation})


def _timeout_record(
    state: _WorkerState,
    timeout_seconds: float,
    max_rows: int,
    max_result_bytes: int,
    reason: str,
) -> GoldAuditRecord:
    example = state.examples[state.next_index]
    return _base_record(
        example,
        state.db_sha256,
        timeout_seconds,
        max_rows,
        max_result_bytes,
        status=AuditStatus.TIMEOUT,
        error=reason,
    ).model_copy(update={"elapsed_ms": [timeout_seconds * 1000]})


def _run_workers(
    states: list[_WorkerState],
    *,
    timeout_seconds: float,
    max_workers: int,
    max_rows: int,
    max_result_bytes: int,
    on_record: Any,
) -> None:
    context = mp.get_context("spawn")
    result_queue = context.Queue()
    waiting = deque(states)
    active: dict[str, _WorkerState] = {}
    hard_timeout = timeout_seconds * 2 + 5.0

    def launch(state: _WorkerState) -> None:
        remaining = state.examples[state.next_index :]
        state.generation += 1
        process = context.Process(
            target=_database_worker,
            args=(
                state.db_id,
                state.db_sha256,
                remaining,
                timeout_seconds,
                max_rows,
                max_result_bytes,
                state.generation,
                result_queue,
            ),
            name=f"audit-{state.db_id}",
        )
        process.start()
        state.process = process
        state.process_started = time.monotonic()
        state.current_task_id = None
        state.current_started = None
        active[state.db_id] = state

    def close_process(state: _WorkerState, *, terminate: bool = False) -> None:
        if state.process is None:
            return
        if terminate and state.process.is_alive():
            state.process.terminate()
        state.process.join(timeout=2)
        if state.process.is_alive():
            state.process.kill()
            state.process.join(timeout=2)
        state.process.close()
        state.process = None
        state.process_started = None

    try:
        while waiting or active:
            while waiting and len(active) < max_workers:
                launch(waiting.popleft())
            try:
                message = result_queue.get(timeout=0.1)
            except queue.Empty:
                message = None
            if message is not None:
                kind = message["kind"]
                state = active.get(message["db_id"])
                if state is None or message["generation"] != state.generation:
                    # A terminated/restarted process may have already flushed a stale
                    # queue message. Its current task has been recorded by the watchdog.
                    continue
                if kind == "start":
                    state.current_task_id = message["task_id"]
                    state.current_started = time.monotonic()
                elif kind == "result":
                    record = GoldAuditRecord.model_validate(message["record"])
                    on_record(record)
                    state.next_index += 1
                    state.current_task_id = None
                    state.current_started = None
                elif kind == "done":
                    close_process(state)
                    active.pop(state.db_id)

            now = time.monotonic()
            for db_id, state in list(active.items()):
                process = state.process
                if state.current_started is not None and now - state.current_started > hard_timeout:
                    close_process(state, terminate=True)
                    record = _timeout_record(
                        state,
                        timeout_seconds,
                        max_rows,
                        max_result_bytes,
                        "parent watchdog terminated unresponsive worker",
                    )
                    on_record(record)
                    state.next_index += 1
                    active.pop(db_id)
                    if state.next_index < len(state.examples):
                        waiting.appendleft(state)
                elif (
                    process is not None
                    and not process.is_alive()
                    and state.process_started is not None
                    and now - state.process_started > 1.0
                ):
                    close_process(state)
                    active.pop(db_id)
                    if state.next_index < len(state.examples):
                        example = state.examples[state.next_index]
                        record = _base_record(
                            example,
                            state.db_sha256,
                            timeout_seconds,
                            max_rows,
                            max_result_bytes,
                            status=AuditStatus.INFRASTRUCTURE_ERROR,
                            error="database audit worker exited unexpectedly",
                        )
                        on_record(record)
                        state.next_index += 1
                    if state.next_index < len(state.examples):
                        waiting.appendleft(state)
    finally:
        for state in active.values():
            close_process(state, terminate=True)
        result_queue.close()
        result_queue.join_thread()


def _append_partial(handle: Any, record: GoldAuditRecord) -> None:
    handle.write(stable_json(record.model_dump(mode="json")) + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def run_train_audit(
    train_root: Path,
    output_root: Path,
    manifest_root: Path,
    *,
    timeout_seconds: float = 10.0,
    max_workers: int = 4,
    max_rows: int = DEFAULT_MAX_ROWS,
    max_result_bytes: int = DEFAULT_MAX_RESULT_BYTES,
    limit: int | None = None,
    resume: bool = True,
) -> dict[str, Any]:
    inventory = build_train_inventory(train_root)
    write_json(manifest_root / "train_inventory.json", inventory)
    examples = load_train_examples(train_root)
    if limit is not None:
        examples = examples[:limit]

    db_hashes = {
        db_id: str(value["sha256"])
        for db_id, value in inventory["databases"].items()
    }
    local_root = output_root / "data" / "train"
    local_root.mkdir(parents=True, exist_ok=True)
    partial_path = local_root / "audit.partial.jsonl"
    previous: dict[str, GoldAuditRecord] = {}
    if resume:
        for raw in read_jsonl(partial_path):
            record = GoldAuditRecord.model_validate(raw)
            previous[record.task_id] = record

    records: dict[str, GoldAuditRecord] = {}
    pending_by_db: dict[str, list[BirdSourceExample]] = {}
    append_mode = "a" if resume else "w"
    with partial_path.open(append_mode, encoding="utf-8") as partial:
        for example in examples:
            db_sha = db_hashes.get(example.db_id)
            fingerprint = audit_fingerprint(
                example, db_sha, timeout_seconds, max_rows, max_result_bytes
            )
            cached = previous.get(example.task_id)
            if cached is not None and cached.fingerprint == fingerprint:
                records[example.task_id] = cached
                continue
            if not example.source_match:
                record = _base_record(
                    example,
                    db_sha,
                    timeout_seconds,
                    max_rows,
                    max_result_bytes,
                    status=AuditStatus.SOURCE_MISMATCH,
                    error="train.json SQL/db_id differs from train_gold.sql",
                )
                records[example.task_id] = record
                _append_partial(partial, record)
            elif db_sha is None or not example.db_path.is_file():
                record = _base_record(
                    example,
                    db_sha,
                    timeout_seconds,
                    max_rows,
                    max_result_bytes,
                    status=AuditStatus.MISSING_DATABASE,
                    error="SQLite database file is missing",
                )
                records[example.task_id] = record
                _append_partial(partial, record)
            else:
                pending_by_db.setdefault(example.db_id, []).append(example)

        states = [
            _WorkerState(
                db_id=db_id,
                db_path=items[0].db_path,
                examples=items,
                db_sha256=db_hashes[db_id],
            )
            for db_id, items in sorted(pending_by_db.items())
        ]

        def accept(record: GoldAuditRecord) -> None:
            records[record.task_id] = record
            _append_partial(partial, record)

        _run_workers(
            states,
            timeout_seconds=timeout_seconds,
            max_workers=max_workers,
            max_rows=max_rows,
            max_result_bytes=max_result_bytes,
            on_record=accept,
        )

    expected_ids = {example.task_id for example in examples}
    if set(records) != expected_ids:
        missing = sorted(expected_ids - set(records))[:10]
        raise RuntimeError(f"audit did not produce exactly one record per task; missing={missing}")

    sorted_records = sorted(records.values(), key=lambda record: record.source_index)
    audit_rows = [record.model_dump(mode="json") for record in sorted_records]
    write_jsonl(local_root / "audit.jsonl", audit_rows)
    write_jsonl(manifest_root / "train_audit.jsonl", audit_rows)

    by_task = {example.task_id: example for example in examples}
    executable_records = [record for record in sorted_records if record.executable]
    rl_records = [record for record in sorted_records if record.rl_clean]
    for name, selected in (("executable", executable_records), ("rl_clean", rl_records)):
        tasks = []
        answers = []
        for record in selected:
            example = by_task[record.task_id]
            task = TaskView(
                task_id=example.task_id,
                split="train_pool",
                db_id=example.db_id,
                question=example.question,
                evidence=example.evidence,
                db_ref=str(example.db_path.relative_to(train_root)),
            )
            answer = HiddenAnswer(
                task_id=example.task_id,
                gold_sql=example.gold_sql,
                audit_status=record.status,
            )
            tasks.append(task.model_dump(mode="json"))
            answers.append(answer.model_dump(mode="json"))
        write_jsonl(local_root / "tasks" / f"{name}.jsonl", tasks)
        write_jsonl(local_root / "answers" / f"{name}.jsonl", answers)

    after_hashes = {
        db_id: sha256_file(train_root / str(value["path"]))
        for db_id, value in inventory["databases"].items()
    }
    unchanged = all(after_hashes[db_id] == db_hashes[db_id] for db_id in db_hashes)
    if not unchanged:
        raise RuntimeError("one or more BIRD train databases changed during read-only audit")

    status_counts = Counter(record.status.value for record in sorted_records)
    summary: dict[str, Any] = {
        "schema_version": 1,
        "dataset": "bird_train",
        "input_count": len(examples),
        "status_counts": dict(sorted(status_counts.items())),
        "executable_count": len(executable_records),
        "rl_clean_count": len(rl_records),
        "empty_result_count": sum(record.empty_result for record in sorted_records),
        "result_too_large_count": sum(record.result_too_large for record in sorted_records),
        "database_hashes_unchanged": unchanged,
        "timeout_seconds": timeout_seconds,
        "max_workers": max_workers,
        "max_rows": max_rows,
        "max_result_bytes": max_result_bytes,
        "auditor_version": AUDITOR_VERSION,
        "source_inventory_sha256": sha256_file(manifest_root / "train_inventory.json"),
    }
    write_json(local_root / "summary.json", summary)
    write_json(manifest_root / "train_audit_summary.json", summary)
    return summary
