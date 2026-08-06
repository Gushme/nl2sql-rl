"""只依赖 SQL 执行结果的确定性终局 Reward。"""

from __future__ import annotations

from dataclasses import dataclass

from nl2sql_rl.models import AuditStatus, TerminalReason
from nl2sql_rl.sqlite_exec import QueryExecution


@dataclass(frozen=True)
class RewardDecision:
    reward: float | None
    valid_for_training: bool
    reason: str


def score_terminal(
    terminal_reason: TerminalReason,
    *,
    prediction: QueryExecution | None = None,
    gold: QueryExecution | None = None,
) -> RewardDecision:
    if terminal_reason is TerminalReason.INFRASTRUCTURE_ERROR:
        return RewardDecision(None, False, "infrastructure_error")
    if terminal_reason is TerminalReason.UNSAFE_SQL:
        return RewardDecision(-1.0, True, "unsafe_sql")
    if terminal_reason in (TerminalReason.LOOP, TerminalReason.MAX_ACTIONS):
        return RewardDecision(-0.4, True, terminal_reason.value)
    if gold is None or gold.status is not AuditStatus.PASSED or gold.digest is None:
        return RewardDecision(None, False, "invalid_gold")
    if prediction is None:
        return RewardDecision(-0.2, True, "submission_execution_error")
    if prediction.status is AuditStatus.INFRASTRUCTURE_ERROR:
        return RewardDecision(None, False, "prediction_infrastructure_error")
    if prediction.status is AuditStatus.UNSAFE_SQL:
        return RewardDecision(-1.0, True, "unsafe_sql")
    if prediction.status is not AuditStatus.PASSED or prediction.digest is None:
        return RewardDecision(-0.2, True, "submission_execution_error")
    if prediction.digest == gold.digest:
        return RewardDecision(1.0, True, "execution_match")
    return RewardDecision(0.0, True, "execution_mismatch")
