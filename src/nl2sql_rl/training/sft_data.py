"""从合格 Teacher 轨迹构建无 Gold 泄漏的 Action-only SFT 数据。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from pydantic import Field

from nl2sql_rl.agent.loop import build_actor_messages
from nl2sql_rl.agent.sql_semantics import SQLSemanticError, normalize_sql, physical_tables
from nl2sql_rl.io_utils import stable_json
from nl2sql_rl.models import EpisodeResult, StrictRecord, TaskView, TerminalReason

if TYPE_CHECKING:
    from nl2sql_rl.teacher.collector import TeacherAttempt


class SFTConversation(StrictRecord):
    schema_version: int = 1
    task_id: str
    split: str
    messages: list[dict[str, Any]]
    assistant_message_indexes: list[int] = Field(min_length=1)
    source_config_hash: str
    source_harness_config_hash: str


class TokenizerLike(Protocol):
    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]: ...


@dataclass(frozen=True)
class TokenizedSFTExample:
    input_ids: list[int]
    labels: list[int]
    attention_mask: list[int]


class JSONTokenizer:
    """把 tokenizer.json 包装成项目统一的最小编码接口。"""

    def __init__(self, path: Path) -> None:
        from tokenizers import Tokenizer  # type: ignore[import-untyped]

        self._tokenizer = Tokenizer.from_file(str(path))

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        encoded = self._tokenizer.encode(
            text,
            add_special_tokens=add_special_tokens,
        )
        return [int(token_id) for token_id in encoded.ids]


def _chat_message_token_ids(
    message: dict[str, Any], tokenizer: TokenizerLike
) -> tuple[list[int], list[int], list[int]]:
    role = str(message["role"])
    content = str(message["content"])
    if role == "tool":
        role = "user"
        content = f"<tool_response>\n{content}\n</tool_response>"
    return (
        tokenizer.encode(f"<|im_start|>{role}\n", add_special_tokens=False),
        tokenizer.encode(content, add_special_tokens=False),
        tokenizer.encode("<|im_end|>\n", add_special_tokens=False),
    )


def chat_messages_token_count(
    messages: list[dict[str, Any]], tokenizer: TokenizerLike
) -> int:
    """按与 SFT 完全相同的 ChatML 序列计算模型可见上下文长度。"""
    return sum(
        len(prefix) + len(content) + len(suffix)
        for prefix, content, suffix in (
            _chat_message_token_ids(message, tokenizer) for message in messages
        )
    )


def build_episode_sft_messages(
    task: TaskView, episode: EpisodeResult
) -> tuple[list[dict[str, Any]], list[int]]:
    """去掉终局 observation，只保留成功轨迹中的 Assistant Action。"""
    if episode.task_id != task.task_id:
        raise ValueError("SFT TaskView 与 Episode task_id 不一致")
    if (
        episode.terminal_reason is not TerminalReason.SUBMITTED
        or not episode.valid_for_training
        or episode.reward != 1.0
        or episode.submitted_sql is None
    ):
        raise ValueError(f"SFT 轨迹终局不合格：{task.task_id}")
    messages = build_actor_messages(task)
    assistant_indexes: list[int] = []
    described_tables: set[str] = set()
    executed_sql: list[str] = []
    if not episode.events:
        raise ValueError(f"合格轨迹没有 Action：{task.task_id}")
    for index, event in enumerate(episode.events):
        if event.action is None:
            raise ValueError(f"合格轨迹包含空 Action：{task.task_id}")
        if not event.observation.ok or event.observation.error_code is not None:
            raise ValueError(f"合格轨迹包含错误 observation：{task.task_id}")
        assistant_indexes.append(len(messages))
        messages.append(
            {
                "role": "assistant",
                "content": stable_json(event.action.model_dump(mode="json")),
            }
        )
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
        is_final_submit = index == len(episode.events) - 1 and event.action.action == "submit_sql"
        if is_final_submit:
            continue
        messages.append(
            {
                "role": "tool",
                "name": event.observation.tool,
                "event_id": event.event_id,
                "content": stable_json(event.observation.model_dump(mode="json")),
            }
        )
    if messages[-1].get("role") != "assistant":
        raise ValueError(f"SFT 序列必须以 submit_sql Assistant Action 结束：{task.task_id}")
    final_action = episode.events[-1].action
    if final_action is None or final_action.action != "submit_sql":
        raise ValueError(f"SFT 序列最后一个 Action 不是 submit_sql：{task.task_id}")
    final_sql = final_action.arguments.get("sql")
    if not described_tables or not executed_sql or not isinstance(final_sql, str):
        raise ValueError(f"SFT 轨迹缺少 schema、执行或提交证据：{task.task_id}")
    try:
        if (
            normalize_sql(executed_sql[-1]) != normalize_sql(final_sql)
            or normalize_sql(final_sql) != normalize_sql(episode.submitted_sql)
        ):
            raise ValueError(f"SFT execute/submit SQL 不一致：{task.task_id}")
        missing_tables = physical_tables(final_sql).difference(described_tables)
    except SQLSemanticError as exc:
        raise ValueError(f"SFT 最终 SQL 无法解析：{task.task_id}") from exc
    if missing_tables:
        raise ValueError(f"SFT 最终 SQL 包含未描述表：{task.task_id}")
    return messages, assistant_indexes


def build_sft_conversations(
    tasks: list[TaskView], attempts: list[TeacherAttempt]
) -> list[SFTConversation]:
    task_by_id = {task.task_id: task for task in tasks}
    accepted_hashes = {
        (attempt.config_hash, attempt.harness_config_hash)
        for attempt in attempts
        if attempt.accepted
    }
    if len(accepted_hashes) > 1:
        raise ValueError("SFT 数据禁止混合多个采集配置或 Harness 哈希")
    accepted_ids = [attempt.task_id for attempt in attempts if attempt.accepted]
    if len(accepted_ids) != len(set(accepted_ids)):
        raise ValueError("SFT 合格轨迹包含重复 task_id")
    conversations: list[SFTConversation] = []
    for attempt in attempts:
        if not attempt.accepted:
            continue
        task = task_by_id.get(attempt.task_id)
        if task is None:
            raise ValueError(f"缺少 TaskView：{attempt.task_id}")
        if attempt.split != task.split or attempt.db_id != task.db_id:
            raise ValueError(f"SFT 轨迹与 TaskView 的 split/db_id 不一致：{attempt.task_id}")
        if attempt.episode.config_hash != attempt.config_hash:
            raise ValueError(f"SFT 轨迹 Episode 配置哈希不一致：{attempt.task_id}")
        messages, assistant_indexes = build_episode_sft_messages(task, attempt.episode)
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
                source_harness_config_hash=attempt.harness_config_hash,
            )
        )
    return conversations


def sft_token_count(
    task: TaskView,
    episode: EpisodeResult,
    tokenizer: TokenizerLike,
) -> int:
    messages, assistant_indexes = build_episode_sft_messages(task, episode)
    conversation = SFTConversation(
        task_id=task.task_id,
        split=task.split,
        messages=messages,
        assistant_message_indexes=assistant_indexes,
        source_config_hash=episode.config_hash,
        source_harness_config_hash="length-check",
    )
    return len(tokenize_action_only(conversation, tokenizer, max_length=10**9).input_ids)


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
        prefix, content_ids, suffix = _chat_message_token_ids(message, tokenizer)
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
