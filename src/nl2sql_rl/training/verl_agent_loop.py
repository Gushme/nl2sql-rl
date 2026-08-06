"""veRL v0.8.0 自定义多轮 NL2SQL AgentLoop。"""

from __future__ import annotations

import asyncio
import hashlib
import importlib
import os
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

from nl2sql_rl.agent.parser import ActionParseError, normalized_action, parse_action
from nl2sql_rl.agent.reward import score_terminal
from nl2sql_rl.agent.tools import SQLiteToolbox
from nl2sql_rl.config import RuntimeConfig
from nl2sql_rl.io_utils import stable_json
from nl2sql_rl.models import (
    AgentAction,
    AuditStatus,
    EpisodeResult,
    HiddenAnswer,
    TaskView,
    TerminalReason,
    ToolObservation,
    TrajectoryEvent,
)
from nl2sql_rl.sqlite_exec import execute_read_only
from nl2sql_rl.training.grpo_reward import adapt_episode_reward


class _MissingVerlAgentLoopBase:
    """CPU 环境占位类，实例化时给出明确依赖错误。"""

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("NL2SQLAgentLoop 只能在固定 veRL v0.8.0 容器中实例化")


_VerlAgentLoopBase: Any = _MissingVerlAgentLoopBase
try:
    _verl_agent_module = importlib.import_module("verl.experimental.agent_loop.agent_loop")
    _VerlAgentLoopBase = _verl_agent_module.AgentLoopBase
except ModuleNotFoundError:
    pass


def _protocol_observation(event_id: str, message: str) -> ToolObservation:
    return ToolObservation(
        event_id=event_id,
        tool="protocol",
        ok=False,
        payload={"message": message[:500]},
        error_code="invalid_action",
    )


def _hidden_keys(value: Any) -> set[str]:
    keys: set[str] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            keys.add(str(key))
            keys.update(_hidden_keys(child))
    elif isinstance(value, list):
        for child in value:
            keys.update(_hidden_keys(child))
    return keys


class NL2SQLAgentLoop(_VerlAgentLoopBase):  # type: ignore[misc]
    """以 token 为边界串接模型 Action 和只读 SQLite observation。"""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.runtime = RuntimeConfig()
        self.response_length = int(self.rollout_config.response_length)

    def _encode_fragment(
        self,
        observation: ToolObservation,
        *,
        generated_ids: list[int],
        continue_generation: bool,
    ) -> list[int]:
        eos = getattr(self.tokenizer, "eos_token_id", None)
        eos_values = {int(eos)} if isinstance(eos, int) else set(eos or [])
        needs_assistant_end = not generated_ids or generated_ids[-1] not in eos_values
        prefix = "<|im_end|>\n" if needs_assistant_end else "\n"
        fragment = (
            prefix
            + "<|im_start|>user\n<tool_response>\n"
            + stable_json(observation.model_dump(mode="json"))
            + "\n</tool_response><|im_end|>\n"
        )
        if continue_generation:
            fragment += "<|im_start|>assistant\n"
        encoded = self.tokenizer.encode(fragment, add_special_tokens=False)
        return [int(token_id) for token_id in encoded]

    @staticmethod
    def _append_segment(
        response_ids: list[int],
        response_mask: list[int],
        segment: list[int],
        *,
        mask_value: int,
        limit: int,
    ) -> bool:
        remaining = max(0, limit - len(response_ids))
        accepted = segment[:remaining]
        response_ids.extend(accepted)
        response_mask.extend([mask_value] * len(accepted))
        return len(accepted) == len(segment)

    async def run(self, sampling_params: dict[str, Any], **kwargs: Any) -> Any:
        raw_prompt = list(kwargs["raw_prompt"])
        leaked = _hidden_keys(raw_prompt).intersection(
            {"gold_sql", "reward", "result_cache_ref", "audit_status"}
        )
        if leaked:
            raise ValueError(f"Actor raw_prompt 出现隐藏字段：{sorted(leaked)}")
        extra_info = kwargs.get("extra_info")
        if not isinstance(extra_info, dict):
            raise ValueError("veRL 样本缺少 extra_info")
        task = TaskView.model_validate(extra_info.get("task"))
        answer = HiddenAnswer.model_validate(extra_info.get("hidden_answer"))
        if task.task_id != answer.task_id:
            raise ValueError("veRL TaskView 与 HiddenAnswer 的 task_id 不一致")
        db_path = Path(str(extra_info.get("db_path", "")))
        if not db_path.is_file():
            raise FileNotFoundError(f"veRL 任务数据库不存在：{db_path}")

        prompt_ids = [int(token_id) for token_id in await self.apply_chat_template(raw_prompt)]
        response_ids: list[int] = []
        response_mask: list[int] = []
        events: list[TrajectoryEvent] = []
        toolbox = SQLiteToolbox(
            db_path,
            exploration_timeout_seconds=self.runtime.exploration_timeout_seconds,
            submission_timeout_seconds=self.runtime.gold_timeout_seconds,
            max_observation_bytes=self.runtime.max_observation_bytes,
        )
        request_id = uuid4().hex
        terminal_reason = TerminalReason.MAX_ACTIONS
        submitted_sql: str | None = None
        repeated_key: str | None = None
        repeated_count = 0
        generate_seconds = 0.0
        tool_seconds = 0.0
        score_seconds = 0.0
        assistant_turns = 0
        num_preempted = 0

        for step in range(self.runtime.max_episode_actions):
            remaining = self.response_length - len(response_ids)
            if remaining <= 0:
                break
            current_prompt_ids = prompt_ids + response_ids
            turn_sampling = dict(sampling_params)
            turn_sampling["max_tokens"] = min(self.runtime.max_action_tokens, remaining)
            started = time.monotonic()
            output = await self.server_manager.generate(
                request_id=request_id,
                prompt_ids=current_prompt_ids,
                sampling_params=turn_sampling,
            )
            generate_seconds += time.monotonic() - started
            assistant_turns += 1
            num_preempted += int(output.num_preempted or 0)
            generated_ids = [int(token_id) for token_id in output.token_ids]
            if not self._append_segment(
                response_ids,
                response_mask,
                generated_ids,
                mask_value=1,
                limit=self.response_length,
            ):
                terminal_reason = TerminalReason.MAX_ACTIONS
                break

            event_id = f"{request_id}:{step:02d}"
            content = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            action: AgentAction | None = None
            try:
                action = parse_action(content)
            except ActionParseError as exc:
                observation = _protocol_observation(event_id, str(exc))
                key = "invalid:" + hashlib.sha256(content.encode()).hexdigest()
                repeated_count = repeated_count + 1 if key == repeated_key else 1
                repeated_key = key
                terminal = repeated_count >= 3
                if terminal:
                    terminal_reason = TerminalReason.LOOP
            else:
                key = normalized_action(action)
                repeated_count = repeated_count + 1 if key == repeated_key else 1
                repeated_key = key
                if repeated_count >= 3:
                    observation = ToolObservation(
                        event_id=event_id,
                        tool=action.action,
                        ok=False,
                        payload={"message": "连续三次相同 Action，已终止循环"},
                        error_code="loop_detected",
                    )
                    terminal = True
                    terminal_reason = TerminalReason.LOOP
                else:
                    started = time.monotonic()
                    observation = await asyncio.to_thread(
                        toolbox.call, action, event_id=event_id
                    )
                    tool_seconds += time.monotonic() - started
                    terminal = False
                    if observation.error_code == AuditStatus.UNSAFE_SQL.value:
                        terminal = True
                        terminal_reason = TerminalReason.UNSAFE_SQL
                    elif action.action == "submit_sql":
                        sql_value = action.arguments.get("sql")
                        submitted_sql = sql_value if isinstance(sql_value, str) else None
                        terminal = True
                        terminal_reason = TerminalReason.SUBMITTED

            event = TrajectoryEvent(
                event_id=event_id,
                step=step,
                action=action,
                observation=observation,
            )
            events.append(event)
            observation_ids = self._encode_fragment(
                observation,
                generated_ids=generated_ids,
                continue_generation=not terminal,
            )
            complete = self._append_segment(
                response_ids,
                response_mask,
                observation_ids,
                mask_value=0,
                limit=self.response_length,
            )
            if terminal or not complete:
                if not complete and not terminal:
                    terminal_reason = TerminalReason.MAX_ACTIONS
                break

        started = time.monotonic()
        if terminal_reason is TerminalReason.SUBMITTED:
            gold_execution = await asyncio.to_thread(
                execute_read_only,
                answer.gold_sql,
                db_path,
                timeout_seconds=self.runtime.gold_timeout_seconds,
            )
            decision = score_terminal(
                terminal_reason,
                prediction=toolbox.last_submission,
                gold=gold_execution,
            )
        else:
            decision = score_terminal(terminal_reason)
        episode = EpisodeResult(
            episode_id=request_id,
            task_id=task.task_id,
            terminal_reason=terminal_reason,
            events=events,
            submitted_sql=submitted_sql,
            reward=decision.reward,
            valid_for_training=decision.valid_for_training,
            config_hash=os.environ.get("NL2SQL_GRPO_RUN_HASH", "verl-v0.8.0"),
        )
        reward = adapt_episode_reward(episode)
        score_seconds += time.monotonic() - started

        agent_module = importlib.import_module("verl.experimental.agent_loop.agent_loop")
        metrics = agent_module.AgentLoopMetrics(
            generate_sequences=generate_seconds,
            tool_calls=tool_seconds,
            compute_score=score_seconds,
            num_preempted=num_preempted,
        )
        return agent_module.AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            response_mask=response_mask,
            response_logprobs=None,
            reward_score=reward.score,
            num_turns=len(raw_prompt) + assistant_turns + len(events),
            metrics=metrics,
            extra_fields={
                "task_id": task.task_id,
                "terminal_reason": terminal_reason.value,
                "valid_for_advantage": True,
                "event_count": len(events),
            },
        )
