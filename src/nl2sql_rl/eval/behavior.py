"""无需 LLM 的轨迹行为指标。"""

from __future__ import annotations

from collections import Counter
from typing import Any

from nl2sql_rl.models import EpisodeResult, TerminalReason


def behavior_metrics(episodes: list[EpisodeResult]) -> dict[str, Any]:
    count = len(episodes)
    tool_counts: Counter[str] = Counter()
    total_events = 0
    sql_events = 0
    sql_errors = 0
    episodes_with_error = 0
    corrected = 0
    token_totals: Counter[str] = Counter()
    elapsed_ms = 0.0
    for episode in episodes:
        total_events += len(episode.events)
        had_error = False
        later_success = False
        for event in episode.events:
            tool_counts[event.observation.tool] += 1
            elapsed_ms += event.observation.elapsed_ms
            if event.action is not None and event.action.action in ("execute_sql", "submit_sql"):
                sql_events += 1
                if not event.observation.ok:
                    sql_errors += 1
            if not event.observation.ok:
                had_error = True
            elif had_error:
                later_success = True
        if had_error:
            episodes_with_error += 1
        if had_error and later_success and episode.terminal_reason is TerminalReason.SUBMITTED:
            corrected += 1
        token_totals.update(episode.usage)
    denominator = count or 1
    return {
        "episode_count": count,
        "successful_submission_rate": sum(
            episode.terminal_reason is TerminalReason.SUBMITTED
            and bool(episode.events)
            and episode.events[-1].observation.ok
            for episode in episodes
        )
        / denominator,
        "average_steps": total_events / denominator,
        "tool_call_distribution": dict(sorted(tool_counts.items())),
        "execution_error_rate": sql_errors / (sql_events or 1),
        "loop_rate": sum(
            episode.terminal_reason is TerminalReason.LOOP for episode in episodes
        )
        / denominator,
        "correction_success_rate": corrected / (episodes_with_error or 1),
        "token_totals": dict(sorted(token_totals.items())),
        "average_tool_latency_ms": elapsed_ms / (total_events or 1),
    }
