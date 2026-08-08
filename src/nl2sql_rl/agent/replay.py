"""不调用模型、忽略时延后逐事件比较的确定性轨迹回放。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nl2sql_rl.agent.loop import ScriptedPolicy, run_episode
from nl2sql_rl.config import RuntimeConfig
from nl2sql_rl.models import EpisodeResult, HiddenAnswer, TaskView, TrajectoryEvent


def _event_semantics(event: TrajectoryEvent) -> dict[str, Any]:
    return {
        "step": event.step,
        "action": (
            event.action.model_dump(mode="json") if event.action is not None else None
        ),
        "observation": {
            "tool": event.observation.tool,
            "ok": event.observation.ok,
            "payload": event.observation.payload,
            "error_code": event.observation.error_code,
            "truncated": event.observation.truncated,
        },
    }


@dataclass(frozen=True)
class ReplayComparison:
    reproduced: EpisodeResult
    terminal_matches: bool
    submitted_sql_matches: bool
    reward_matches: bool
    event_count_matches: bool
    event_mismatches: tuple[int, ...]
    database_sha_matches: bool

    @property
    def events_match(self) -> bool:
        return self.event_count_matches and not self.event_mismatches

    @property
    def exact_terminal_outcome(self) -> bool:
        return self.terminal_matches and self.submitted_sql_matches and self.reward_matches

    @property
    def exact_event_replay(self) -> bool:
        return (
            self.exact_terminal_outcome
            and self.events_match
            and self.database_sha_matches
        )


def replay_episode(
    original: EpisodeResult,
    task: TaskView,
    answer: HiddenAnswer,
    db_path: Path,
    *,
    runtime: RuntimeConfig,
    config_hash: str | None = None,
    current_database_sha256: str | None = None,
) -> ReplayComparison:
    actions = [event.action for event in original.events if event.action is not None]
    reproduced = run_episode(
        task,
        answer,
        db_path,
        ScriptedPolicy(actions),
        runtime=runtime,
        config_hash=config_hash or original.config_hash,
        episode_id=f"replay_{original.episode_id}",
        expected_database_sha256=current_database_sha256,
        verify_database_sha=current_database_sha256 is None,
    )
    shared_count = min(len(original.events), len(reproduced.events))
    mismatches = tuple(
        index
        for index in range(shared_count)
        if _event_semantics(original.events[index])
        != _event_semantics(reproduced.events[index])
    )
    return ReplayComparison(
        reproduced=reproduced,
        terminal_matches=reproduced.terminal_reason is original.terminal_reason,
        submitted_sql_matches=reproduced.submitted_sql == original.submitted_sql,
        reward_matches=reproduced.reward == original.reward,
        event_count_matches=len(reproduced.events) == len(original.events),
        event_mismatches=mismatches,
        database_sha_matches=reproduced.database_sha256 == original.database_sha256,
    )
