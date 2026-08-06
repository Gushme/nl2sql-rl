"""不接触 Gold、Reward、EX 或模型名的盲行为评审接口。"""

from __future__ import annotations

from typing import Any, Protocol

from pydantic import Field

from nl2sql_rl.io_utils import stable_json
from nl2sql_rl.models import EpisodeResult, StrictRecord, TaskView


class BehaviorJudgeScore(StrictRecord):
    schema_grounding: int = Field(ge=1, le=5)
    tool_efficiency: int = Field(ge=1, le=5)
    error_recovery: int = Field(ge=1, le=5)
    submission_discipline: int = Field(ge=1, le=5)
    event_citations: list[str] = Field(min_length=1)
    rationale: str = Field(max_length=1_000)


class LLMJudge(Protocol):
    def judge(self, payload: dict[str, Any]) -> BehaviorJudgeScore: ...


def build_blind_payload(task: TaskView, episode: EpisodeResult) -> dict[str, Any]:
    """只暴露 Actor 当时能看到的任务与规范化事件。"""
    return {
        "rubric": {
            "schema_grounding": "是否根据可见 schema/值证据形成 SQL",
            "tool_efficiency": "工具调用是否必要且不过度",
            "error_recovery": "收到工具错误后是否有效修正",
            "submission_discipline": "是否在验证后及时且仅提交最终 SQL",
            "scale": "每项 1 到 5 分，必须引用 event_id",
        },
        "task": task.actor_payload(),
        "events": [event.model_dump(mode="json") for event in episode.events],
    }


def blind_prompt(task: TaskView, episode: EpisodeResult) -> str:
    payload = build_blind_payload(task, episode)
    return (
        "请按 rubric 输出 BehaviorJudgeScore JSON。不得推测 Gold 或正确性；"
        "每项判断必须由 event_id 支撑。\n" + stable_json(payload)
    )


def validate_judge_score(score: BehaviorJudgeScore, episode: EpisodeResult) -> None:
    valid_ids = {event.event_id for event in episode.events}
    unknown = set(score.event_citations).difference(valid_ids)
    if unknown:
        raise ValueError(f"Judge 引用了不存在的 event_id：{sorted(unknown)}")
