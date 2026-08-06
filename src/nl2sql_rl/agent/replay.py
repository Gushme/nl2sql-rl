"""不调用模型的确定性轨迹回放。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from nl2sql_rl.agent.loop import ScriptedPolicy, run_episode
from nl2sql_rl.config import RuntimeConfig
from nl2sql_rl.models import EpisodeResult, HiddenAnswer, TaskView


@dataclass(frozen=True)
class ReplayComparison:
    reproduced: EpisodeResult
    terminal_matches: bool
    submitted_sql_matches: bool
    reward_matches: bool

    @property
    def exact_terminal_outcome(self) -> bool:
        return self.terminal_matches and self.submitted_sql_matches and self.reward_matches


def replay_episode(
    original: EpisodeResult,
    task: TaskView,
    answer: HiddenAnswer,
    db_path: Path,
    *,
    runtime: RuntimeConfig,
) -> ReplayComparison:
    actions = [event.action for event in original.events if event.action is not None]
    reproduced = run_episode(
        task,
        answer,
        db_path,
        ScriptedPolicy(actions),
        runtime=runtime,
        config_hash=original.config_hash,
        episode_id=f"replay_{original.episode_id}",
    )
    return ReplayComparison(
        reproduced=reproduced,
        terminal_matches=reproduced.terminal_reason is original.terminal_reason,
        submitted_sql_matches=reproduced.submitted_sql == original.submitted_sql,
        reward_matches=reproduced.reward == original.reward,
    )
