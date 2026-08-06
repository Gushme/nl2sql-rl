"""从合格 Teacher 轨迹构建无 Gold 泄漏的 Action-only SFT 数据。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from pydantic import Field

from nl2sql_rl.agent.loop import build_actor_messages
from nl2sql_rl.io_utils import stable_json
from nl2sql_rl.models import StrictRecord, TaskView
from nl2sql_rl.teacher.collector import TeacherAttempt


class SFTConversation(StrictRecord):
    schema_version: int = 1
    task_id: str
    split: str
    messages: list[dict[str, Any]]
    assistant_message_indexes: list[int] = Field(min_length=1)
    source_config_hash: str


class TokenizerLike(Protocol):
    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]: ...


@dataclass(frozen=True)
class TokenizedSFTExample:
    input_ids: list[int]
    labels: list[int]
    attention_mask: list[int]


def build_sft_conversations(
    tasks: list[TaskView], attempts: list[TeacherAttempt]
) -> list[SFTConversation]:
    task_by_id = {task.task_id: task for task in tasks}
    conversations: list[SFTConversation] = []
    for attempt in attempts:
        if not attempt.accepted:
            continue
        task = task_by_id.get(attempt.task_id)
        if task is None:
            raise ValueError(f"缺少 TaskView：{attempt.task_id}")
        messages = build_actor_messages(task)
        assistant_indexes: list[int] = []
        for event in attempt.episode.events:
            if event.action is None:
                raise ValueError(f"合格轨迹包含空 Action：{attempt.task_id}")
            assistant_indexes.append(len(messages))
            messages.append(
                {
                    "role": "assistant",
                    "content": stable_json(event.action.model_dump(mode="json")),
                }
            )
            messages.append(
                {
                    "role": "tool",
                    "name": event.observation.tool,
                    "event_id": event.event_id,
                    "content": stable_json(event.observation.model_dump(mode="json")),
                }
            )
        serialized = stable_json(messages)
        for forbidden in ("gold_sql", '"reward"', "correctness"):
            if forbidden in serialized:
                raise ValueError(f"SFT Actor 数据出现隐藏字段：{forbidden}")
        conversations.append(
            SFTConversation(
                task_id=attempt.task_id,
                split=attempt.split,
                messages=messages,
                assistant_message_indexes=assistant_indexes,
                source_config_hash=attempt.config_hash,
            )
        )
    return conversations


def tokenize_action_only(
    conversation: SFTConversation,
    tokenizer: TokenizerLike,
    *,
    max_length: int = 16_384,
) -> TokenizedSFTExample:
    """按 Qwen ChatML 分段编码，只让 assistant Action 内容产生 label。"""
    input_ids: list[int] = []
    labels: list[int] = []
    assistant_indexes = set(conversation.assistant_message_indexes)
    actual_assistant_indexes = {
        index
        for index, message in enumerate(conversation.messages)
        if message.get("role") == "assistant"
    }
    if assistant_indexes != actual_assistant_indexes:
        raise ValueError("assistant_message_indexes 与消息角色不一致")
    for index, message in enumerate(conversation.messages):
        role = str(message["role"])
        content = str(message["content"])
        if role == "tool":
            role = "user"
            content = f"<tool_response>\n{content}\n</tool_response>"
        prefix = tokenizer.encode(f"<|im_start|>{role}\n", add_special_tokens=False)
        content_ids = tokenizer.encode(content, add_special_tokens=False)
        suffix = tokenizer.encode("<|im_end|>\n", add_special_tokens=False)
        input_ids.extend(prefix)
        labels.extend([-100] * len(prefix))
        input_ids.extend(content_ids)
        labels.extend(content_ids if index in assistant_indexes else [-100] * len(content_ids))
        input_ids.extend(suffix)
        labels.extend([-100] * len(suffix))
    if len(input_ids) > max_length:
        raise ValueError(f"SFT 样本超过 max_length：{len(input_ids)} > {max_length}")
    if not any(label != -100 for label in labels):
        raise ValueError("SFT 样本没有任何 assistant Action label")
    return TokenizedSFTExample(
        input_ids=input_ids,
        labels=labels,
        attention_mask=[1] * len(input_ids),
    )
