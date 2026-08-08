"""按冻结批次生成 Harness 质量诊断与自动暂停判定。"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Sequence
from pathlib import Path
from statistics import mean
from typing import TYPE_CHECKING, Any

from nl2sql_rl.agent.loop import build_actor_messages
from nl2sql_rl.agent.parser import normalized_action
from nl2sql_rl.agent.replay import replay_episode
from nl2sql_rl.agent.sql_semantics import SQLSemanticError, normalize_sql, physical_tables
from nl2sql_rl.config import RuntimeConfig
from nl2sql_rl.io_utils import sha256_file, stable_json
from nl2sql_rl.models import HiddenAnswer, TaskView, TerminalReason

if TYPE_CHECKING:
    from nl2sql_rl.teacher.collector import TeacherAttempt
    from nl2sql_rl.training.sft_data import TokenizerLike


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * percentile)))
    return float(ordered[index])


def _database_size_bucket(size_bytes: int) -> str:
    if size_bytes < 10 * 1024 * 1024:
        return "small_lt_10_mib"
    if size_bytes < 100 * 1024 * 1024:
        return "medium_lt_100_mib"
    return "large_ge_100_mib"


def _usage_summary(values: list[float]) -> dict[str, float]:
    return {
        "mean": mean(values) if values else 0.0,
        "p50": _percentile(values, 0.50),
        "p95": _percentile(values, 0.95),
    }


def _group_metrics(
    attempts: Sequence[TeacherAttempt],
    key: str,
    actor_context_tokens: dict[str, int],
) -> dict[str, Any]:
    groups: dict[str, list[TeacherAttempt]] = defaultdict(list)
    for attempt in attempts:
        if key == "database":
            value = f"{attempt.split}:{attempt.db_id}"
        elif key == "complexity":
            value = f"{attempt.split}:{attempt.complexity.value}"
        elif key == "sampling_cell":
            value = f"{attempt.split}:{attempt.db_id}:{attempt.complexity.value}"
        else:
            value = _database_size_bucket(attempt.database_size_bytes)
        groups[value].append(attempt)
    report: dict[str, Any] = {}
    for value, rows in sorted(groups.items()):
        execute_calls = [
            event
            for attempt in rows
            for event in attempt.episode.events
            if event.action is not None and event.action.action == "execute_sql"
        ]
        timeouts = sum(
            event.observation.error_code == "timeout" for event in execute_calls
        )
        schema_calls = [
            event
            for attempt in rows
            for event in attempt.episode.events
            if event.action is not None and event.action.action == "describe_schema"
        ]
        protocol_attempts = sum(
            any(event.action is None for event in attempt.episode.events)
            for attempt in rows
        )
        invalid_argument_attempts = sum(
            any(
                event.observation.error_code == "invalid_arguments"
                for event in attempt.episode.events
            )
            for attempt in rows
        )
        tool_error_attempts = sum(
            any(
                not event.observation.ok or event.observation.error_code is not None
                for event in attempt.episode.events
            )
            for attempt in rows
        )
        loop_or_limit = sum(
            attempt.episode.terminal_reason
            in {TerminalReason.LOOP, TerminalReason.MAX_ACTIONS}
            for attempt in rows
        )
        unverified_submissions = sum(
            event.observation.error_code
            in {"submission_not_executed", "submission_sql_mismatch", "undescribed_table"}
            for attempt in rows
            for event in attempt.episode.events
        )
        accepted = sum(attempt.accepted for attempt in rows)
        report[value] = {
            "attempts": len(rows),
            "accepted": accepted,
            "acceptance_rate": accepted / len(rows),
            "rejection_reasons": dict(
                sorted(
                    Counter(
                        attempt.rejection_reason or "accepted"
                        for attempt in rows
                        if not attempt.accepted
                    ).items()
                )
            ),
            "execute_calls": len(execute_calls),
            "execute_timeouts": timeouts,
            "execute_timeout_rate": timeouts / len(execute_calls) if execute_calls else 0.0,
            "protocol_attempt_rate": protocol_attempts / len(rows),
            "invalid_argument_attempt_rate": invalid_argument_attempts / len(rows),
            "tool_error_attempt_rate": tool_error_attempts / len(rows),
            "schema_calls": len(schema_calls),
            "schema_truncation_rate": (
                sum(event.observation.truncated for event in schema_calls)
                / len(schema_calls)
                if schema_calls
                else 0.0
            ),
            "unverified_submissions": unverified_submissions,
            "loop_or_max_actions_rate": loop_or_limit / len(rows),
            "action_count": _usage_summary(
                [float(len(attempt.episode.events)) for attempt in rows]
            ),
            "context_tokens": _usage_summary(
                [
                    float(actor_context_tokens.get(attempt.episode.episode_id, 0))
                    for attempt in rows
                ]
            ),
            "latency_ms": _usage_summary(
                [float(attempt.episode.usage.get("latency_ms", 0)) for attempt in rows]
            ),
            "cost_micro_usd": _usage_summary(
                [float(attempt.episode.usage.get("cost_micro_usd", 0)) for attempt in rows]
            ),
        }
    return report


def _max_actor_context_tokens(
    task: TaskView,
    attempt: TeacherAttempt,
    tokenizer: TokenizerLike,
) -> int:
    """重建每次请求前的 Qwen ChatML，返回单轮最大上下文而非多轮总和。"""
    from nl2sql_rl.training.sft_data import chat_messages_token_count

    messages = build_actor_messages(task)
    counts: list[int] = []
    for event in attempt.episode.events:
        counts.append(chat_messages_token_count(messages, tokenizer))
        if event.action is not None:
            messages.append(
                {
                    "role": "assistant",
                    "content": stable_json(event.action.model_dump(mode="json")),
                }
            )
        messages.append(
            {
                "role": "tool",
                "event_id": event.event_id,
                "name": event.observation.tool,
                "content": stable_json(event.observation.model_dump(mode="json")),
            }
        )
    if (
        attempt.episode.terminal_reason is TerminalReason.INFRASTRUCTURE_ERROR
        and attempt.episode.infrastructure_request_sent is True
    ):
        counts.append(chat_messages_token_count(messages, tokenizer))
    return max(counts, default=0)


def _accepted_violation(attempt: TeacherAttempt) -> bool:
    if not attempt.accepted:
        return False
    events = attempt.episode.events
    if not events or any(
        event.action is None
        or not event.observation.ok
        or event.observation.error_code is not None
        for event in events
    ):
        return True
    episode = attempt.episode
    if (
        episode.terminal_reason is not TerminalReason.SUBMITTED
        or not episode.valid_for_training
        or episode.reward != 1.0
        or episode.submitted_sql is None
    ):
        return True
    described_tables: set[str] = set()
    executed_sql: list[str] = []
    for event in events:
        assert event.action is not None
        if event.action.action == "describe_schema":
            schemas = event.observation.payload.get("schemas")
            if isinstance(schemas, list):
                described_tables.update(
                    str(schema.get("name", "")).casefold()
                    for schema in schemas
                    if isinstance(schema, dict) and schema.get("columns")
                )
        elif event.action.action == "execute_sql":
            sql = event.action.arguments.get("sql")
            if isinstance(sql, str):
                executed_sql.append(sql)
    final_action = events[-1].action
    if (
        not described_tables
        or not executed_sql
        or final_action is None
        or final_action.action != "submit_sql"
    ):
        return True
    final_sql = final_action.arguments.get("sql")
    if not isinstance(final_sql, str):
        return True
    try:
        return (
            normalize_sql(executed_sql[-1]) != normalize_sql(final_sql)
            or normalize_sql(final_sql) != normalize_sql(episode.submitted_sql)
            or bool(physical_tables(final_sql).difference(described_tables))
        )
    except SQLSemanticError:
        return True


def diagnose_attempts(
    attempts: Sequence[TeacherAttempt],
    *,
    tasks: Sequence[TaskView],
    answers: Sequence[HiddenAnswer],
    db_root: Path,
    runtime: RuntimeConfig,
    replay_accepted: bool,
    cost_limit_usd: float | None = None,
    spent_usd: float | None = None,
    token_limit: int | None = None,
    used_tokens: int | None = None,
    tokenizer: TokenizerLike | None = None,
) -> dict[str, Any]:
    """统计一个批次；只有 accepted 轨迹参与确定性 replay 门禁。"""
    total = len(attempts)
    accepted_count = sum(attempt.accepted for attempt in attempts)
    rejection_reasons = Counter(
        attempt.rejection_reason or "unknown"
        for attempt in attempts
        if not attempt.accepted
    )
    action_counts = [len(attempt.episode.events) for attempt in attempts]
    tool_distribution: Counter[str] = Counter()
    protocol_attempts = 0
    invalid_argument_attempts = 0
    protocol_or_argument_attempts = 0
    schema_calls = 0
    schema_no_columns = 0
    schema_truncated = 0
    schema_omitted = 0
    schema_sizes: list[float] = []
    execute_calls = 0
    execute_timeouts = 0
    execute_sqlite_errors = 0
    unverified_submissions = 0
    submission_gate_errors: Counter[str] = Counter()
    repeated_tool_calls = 0
    ineffective_searches = 0
    attempts_with_tool_error = 0
    corrected_final_results = 0
    normalization_failures = 0
    multiple_native_tool_call_responses = 0
    invalid_text_action_responses = 0
    native_tool_calls = 0
    text_json_calls = 0
    response_calls = 0
    finish_reasons: Counter[str] = Counter()
    missing_reasoning_breakdown = 0
    observations_followed_by_action = 0
    observations_followed_by_distinct_action = 0
    usage_values: dict[str, list[float]] = defaultdict(list)
    infrastructure_codes: Counter[str] = Counter()
    infrastructure_details: Counter[str] = Counter()
    task_by_id = {task.task_id: task for task in tasks}
    actor_context_tokens: dict[str, int] = {}

    for attempt in attempts:
        episode = attempt.episode
        actor_context_tokens[episode.episode_id] = (
            _max_actor_context_tokens(task_by_id[attempt.task_id], attempt, tokenizer)
            if tokenizer is not None
            else int(episode.usage.get("context_tokens", 0))
        )
        normalization_failures += episode.usage.get("normalization_failed", 0)
        native_tool_calls += episode.usage.get("native_tool_call", 0)
        text_json_calls += episode.usage.get("text_json", 0)
        response_calls += episode.usage.get("response_count", 0)
        known_finish_reasons = 0
        for name, usage_key in (
            ("stop", "finish_reason_stop"),
            ("tool_calls", "finish_reason_tool_calls"),
            ("length", "finish_reason_length"),
            ("other", "finish_reason_other"),
        ):
            count = episode.usage.get(usage_key, 0)
            finish_reasons[name] += count
            known_finish_reasons += count
        finish_reasons["missing_or_legacy"] += max(
            0,
            episode.usage.get("response_count", 0) - known_finish_reasons,
        )
        if (
            episode.usage.get("reasoning_present", 0) > 0
            and episode.usage.get("reasoning_tokens_reported", 0) == 0
        ):
            missing_reasoning_breakdown += 1
        for name in (
            "input_tokens",
            "billed_output_tokens",
            "reasoning_tokens",
            "reasoning_tokens_reported",
            "reasoning_present",
            "action_tokens",
            "latency_ms",
            "cost_micro_usd",
        ):
            usage_values[name].append(float(episode.usage.get(name, 0)))
        usage_values["context_tokens"].append(
            float(actor_context_tokens[episode.episode_id])
        )
        if episode.infrastructure_error_code:
            infrastructure_codes[episode.infrastructure_error_code] += 1
        if episode.infrastructure_error_detail:
            infrastructure_details[stable_json(episode.infrastructure_error_detail)] += 1
        protocol_attempt = False
        invalid_argument_attempt = False
        tool_error_attempt = False
        previous_action: str | None = None
        for event in episode.events:
            if event.action is None:
                protocol_attempt = True
                message = event.observation.payload.get("message")
                if isinstance(message, str) and "native_tool_call_count" in message:
                    multiple_native_tool_call_responses += 1
                if isinstance(message, str) and message.startswith(
                    "Action 不是合法 JSON"
                ):
                    invalid_text_action_responses += 1
            else:
                tool_distribution[event.action.action] += 1
                current_action = normalized_action(event.action)
                if current_action == previous_action:
                    repeated_tool_calls += 1
                previous_action = current_action
            error_code = event.observation.error_code
            if error_code is not None:
                tool_error_attempt = True
            if error_code == "invalid_arguments":
                invalid_argument_attempt = True
            if error_code in {
                "submission_not_executed",
                "submission_sql_mismatch",
                "undescribed_table",
            }:
                unverified_submissions += 1
                submission_gate_errors[error_code] += 1
            if event.action is not None and event.action.action == "describe_schema":
                schema_calls += 1
                schema_sizes.append(
                    float(len(stable_json(event.observation.payload).encode("utf-8")))
                )
                schemas = event.observation.payload.get("schemas")
                has_columns = isinstance(schemas, list) and any(
                    isinstance(schema, dict) and bool(schema.get("columns"))
                    for schema in schemas
                )
                if (
                    event.observation.ok
                    or event.observation.error_code == "schema_too_large"
                ) and not has_columns:
                    schema_no_columns += 1
                schema_truncated += int(event.observation.truncated)
                schema_omitted += int(bool(event.observation.payload.get("omitted_tables")))
            if event.action is not None and event.action.action == "execute_sql":
                execute_calls += 1
                execute_timeouts += int(error_code == "timeout")
                execute_sqlite_errors += int(
                    error_code
                    in {
                        "sqlite_error",
                        "syntax_error",
                        "missing_table_or_column",
                        "unsupported_function",
                    }
                )
            if event.action is not None and event.action.action == "search_values":
                ineffective_searches += int(
                    not event.observation.ok
                    or event.observation.payload.get("returned_rows") == 0
                )
        protocol_attempts += int(protocol_attempt)
        invalid_argument_attempts += int(invalid_argument_attempt)
        protocol_or_argument_attempts += int(
            protocol_attempt or invalid_argument_attempt
        )
        attempts_with_tool_error += int(tool_error_attempt)
        corrected_final_results += int(tool_error_attempt and episode.reward == 1.0)
        for current, following in zip(episode.events, episode.events[1:], strict=False):
            if not current.observation.ok or following.action is None:
                continue
            observations_followed_by_action += 1
            if current.action is None or normalized_action(current.action) != normalized_action(
                following.action
            ):
                observations_followed_by_distinct_action += 1

    answer_by_id = {answer.task_id: answer for answer in answers}
    replay_checked = 0
    replay_mismatches = 0
    database_sha_mismatches = 0
    current_sha: dict[str, str] = {}
    if replay_accepted:
        for attempt in attempts:
            if not attempt.accepted:
                continue
            task = task_by_id[attempt.task_id]
            answer = answer_by_id[attempt.task_id]
            path = db_root / task.db_ref
            if task.db_ref not in current_sha:
                current_sha[task.db_ref] = sha256_file(path)
            database_sha_mismatches += int(
                attempt.episode.database_sha256 != current_sha[task.db_ref]
            )
            comparison = replay_episode(
                attempt.episode,
                task,
                answer,
                path,
                runtime=runtime,
                current_database_sha256=current_sha[task.db_ref],
            )
            replay_checked += 1
            replay_mismatches += int(not comparison.exact_event_replay)
        database_sha_after = {
            db_ref: sha256_file(db_root / db_ref) for db_ref in current_sha
        }
        database_sha_mismatches += sum(
            database_sha_after[db_ref] != digest
            for db_ref, digest in current_sha.items()
        )

    by_database = _group_metrics(attempts, "database", actor_context_tokens)
    by_complexity = _group_metrics(attempts, "complexity", actor_context_tokens)
    by_sampling_cell = _group_metrics(attempts, "sampling_cell", actor_context_tokens)
    by_database_size = _group_metrics(attempts, "database_size", actor_context_tokens)
    pause_reasons: list[str] = []
    if total >= 20 and not any(attempt.accepted for attempt in attempts[:20]):
        pause_reasons.append("first_20_without_accepted")
    # 比例阈值至少观察 20 条，避免 1/1 或 1/4 这类小样本误报。
    rate_sample_ready = total >= 20
    if rate_sample_ready and protocol_or_argument_attempts / total > 0.05:
        pause_reasons.append("protocol_or_argument_error_rate_gt_5pct")
    if schema_no_columns:
        pause_reasons.append("schema_without_columns")
    if rate_sample_ready and schema_calls and schema_truncated / schema_calls > 0.05:
        pause_reasons.append("schema_truncation_rate_gt_5pct")
    if rate_sample_ready and execute_calls and execute_timeouts / execute_calls > 0.02:
        pause_reasons.append("execute_timeout_rate_gt_2pct")
    if rate_sample_ready and any(
        group["execute_timeout_rate"] > 0.05
        for group in by_sampling_cell.values()
    ):
        pause_reasons.append("stratum_execute_timeout_rate_gt_5pct")
    context_overflows = sum(
        attempt.rejection_reason == "context_overflow"
        or attempt.episode.infrastructure_error_code == "context_overflow"
        for attempt in attempts
    )
    # 上下文在请求前即可确定，首例溢出就应暂停以避免无意义继续调度。
    if total and context_overflows / total > 0.01:
        pause_reasons.append("context_overflow_rate_gt_1pct")
    loop_or_limit = sum(
        attempt.episode.terminal_reason
        in {TerminalReason.LOOP, TerminalReason.MAX_ACTIONS}
        for attempt in attempts
    )
    if rate_sample_ready and loop_or_limit / total > 0.05:
        pause_reasons.append("loop_or_max_actions_rate_gt_5pct")
    if replay_mismatches or database_sha_mismatches:
        pause_reasons.append("replay_or_database_sha_mismatch")
    accepted_violations = sum(_accepted_violation(attempt) for attempt in attempts)
    if accepted_violations:
        pause_reasons.append("accepted_trajectory_invariant_violation")
    immediate_fatal_codes = {
        "authentication_error",
        "api_empty_response",
        "api_invalid_response",
        "cost_limit_exceeded",
        "token_limit_exceeded",
    }
    systemic_candidate_codes = {
        "model_or_request_error",
        "api_client_error",
        "api_transport_error",
        "api_retry_exhausted",
    }
    systemic_api_errors = sum(
        infrastructure_codes[code] for code in systemic_candidate_codes
    )
    repeated_systemic_code = any(
        infrastructure_codes[code] >= 2 for code in systemic_candidate_codes
    )
    if immediate_fatal_codes.intersection(infrastructure_codes) or (
        repeated_systemic_code
        or (rate_sample_ready and systemic_api_errors / total > 0.05)
    ):
        pause_reasons.append("systemic_api_or_budget_error")
    if missing_reasoning_breakdown:
        pause_reasons.append("reasoning_token_breakdown_missing")
    if (
        cost_limit_usd is not None
        and spent_usd is not None
        and spent_usd > cost_limit_usd + 1e-12
    ):
        pause_reasons.append("campaign_cost_limit_exceeded")
    if (
        token_limit is not None
        and used_tokens is not None
        and used_tokens > token_limit
    ):
        pause_reasons.append("campaign_token_limit_exceeded")

    return {
        "schema_version": 1,
        "attempts": total,
        "accepted": accepted_count,
        "acceptance_rate": accepted_count / total if total else 0.0,
        "rejection_reasons": dict(sorted(rejection_reasons.items())),
        "protocol_attempts": protocol_attempts,
        "invalid_argument_attempts": invalid_argument_attempts,
        "protocol_or_argument_attempts": protocol_or_argument_attempts,
        "normalization_failures": normalization_failures,
        "multiple_native_tool_call_responses": multiple_native_tool_call_responses,
        "invalid_text_action_responses": invalid_text_action_responses,
        "rate_threshold_sample_ready": rate_sample_ready,
        "native_tool_calls": native_tool_calls,
        "text_json_calls": text_json_calls,
        "response_calls": response_calls,
        "finish_reasons": dict(sorted(finish_reasons.items())),
        "normalization_failure_rate": (
            normalization_failures / response_calls if response_calls else 0.0
        ),
        "native_tool_call_rate": (
            native_tool_calls / response_calls if response_calls else 0.0
        ),
        "missing_reasoning_breakdown": missing_reasoning_breakdown,
        "schema": {
            "calls": schema_calls,
            "without_columns": schema_no_columns,
            "truncated": schema_truncated,
            "with_omitted_tables": schema_omitted,
            "bytes": _usage_summary(schema_sizes),
        },
        "execute_sql": {
            "calls": execute_calls,
            "timeouts": execute_timeouts,
            "sqlite_errors": execute_sqlite_errors,
        },
        "submissions": {
            "unverified": unverified_submissions,
            "gate_errors": dict(sorted(submission_gate_errors.items())),
        },
        "actions": {
            "count": _usage_summary([float(value) for value in action_counts]),
            "tool_distribution": dict(sorted(tool_distribution.items())),
            "repeated_calls": repeated_tool_calls,
            "ineffective_searches": ineffective_searches,
            "attempts_with_tool_error": attempts_with_tool_error,
            "corrected_final_results": corrected_final_results,
            "correction_success_rate": (
                corrected_final_results / attempts_with_tool_error
                if attempts_with_tool_error
                else 0.0
            ),
            "loop_or_max_actions": loop_or_limit,
            "observations_followed_by_action": observations_followed_by_action,
            "observations_followed_by_distinct_action": (
                observations_followed_by_distinct_action
            ),
            "observation_utilization_rate": (
                observations_followed_by_distinct_action
                / observations_followed_by_action
                if observations_followed_by_action
                else 0.0
            ),
        },
        "usage": {name: _usage_summary(values) for name, values in usage_values.items()},
        "context_token_semantics": (
            "max_qwen_chatml_before_request"
            if tokenizer is not None
            else "episode_usage_fallback"
        ),
        "infrastructure_codes": dict(sorted(infrastructure_codes.items())),
        "infrastructure_details": dict(sorted(infrastructure_details.items())),
        "systemic_api_error_attempts": systemic_api_errors,
        "replay": {
            "checked": replay_checked,
            "mismatches": replay_mismatches,
            "database_sha_mismatches": database_sha_mismatches,
            "database_sha256": dict(sorted(current_sha.items())),
        },
        "accepted_invariant_violations": accepted_violations,
        "budget": {
            "cost_limit_usd": cost_limit_usd,
            "spent_usd": spent_usd,
            "token_limit": token_limit,
            "used_tokens": used_tokens,
        },
        "by_database": by_database,
        "by_complexity": by_complexity,
        "by_sampling_cell": by_sampling_cell,
        "by_database_size": by_database_size,
        "pause": bool(pause_reasons),
        "pause_reasons": sorted(set(pause_reasons)),
    }
