"""用统一 OpenAI-compatible backend 采集 Base/SFT/GRPO 评测轨迹。"""

from __future__ import annotations

import asyncio
import hashlib
import os
from collections.abc import Sequence
from concurrent.futures import Future
from pathlib import Path
from typing import Any, Protocol

from nl2sql_rl.agent.loop import SYSTEM_PROMPT, ModelResponse, run_episode
from nl2sql_rl.config import RuntimeConfig
from nl2sql_rl.io_utils import read_jsonl, stable_json
from nl2sql_rl.models import EpisodeResult, HiddenAnswer, StrictRecord, TaskView, TerminalReason
from nl2sql_rl.teacher.client import ACTION_TOOLS, LLMCompletion


class EvaluationActionClient(Protocol):
    @property
    def config_hash(self) -> str: ...

    @property
    def real_api(self) -> bool: ...

    @property
    def spent_usd(self) -> float: ...

    @property
    def config(self) -> Any: ...

    async def complete_action(
        self, messages: list[dict[str, Any]], *, max_tokens: int | None = None
    ) -> LLMCompletion: ...


class InferenceRecord(StrictRecord):
    schema_version: int = 1
    task_id: str
    model_label: str
    config_hash: str
    condition_hash: str
    episode: EpisodeResult


class _AsyncClientPolicy:
    def __init__(self, client: EvaluationActionClient, loop: asyncio.AbstractEventLoop) -> None:
        self.client = client
        self.loop = loop

    def generate(self, messages: list[dict[str, Any]], *, max_tokens: int) -> ModelResponse:
        future: Future[LLMCompletion] = asyncio.run_coroutine_threadsafe(
            self.client.complete_action(messages, max_tokens=max_tokens), self.loop
        )
        completion = future.result()
        return ModelResponse(
            content=stable_json(completion.action.model_dump(mode="json")),
            usage={
                "input_tokens": completion.input_tokens,
                "output_tokens": completion.output_tokens,
                "cost_micro_usd": round(completion.cost_usd * 1_000_000),
            },
        )


def inference_condition_hash(
    client: EvaluationActionClient, runtime: RuntimeConfig
) -> str:
    """排除模型名，仅哈希会影响三模型公平比较的推理条件。"""
    client_config = client.config.model_dump(mode="json")
    condition = {
        "endpoint": client_config.get("endpoint"),
        "temperature": client_config.get("temperature"),
        "seed": client_config.get("seed"),
        "max_completion_tokens": client_config.get("max_completion_tokens"),
        "runtime": runtime.model_dump(mode="json"),
        "system_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest(),
        "tools_sha256": hashlib.sha256(stable_json(ACTION_TOOLS).encode("utf-8")).hexdigest(),
    }
    return hashlib.sha256(stable_json(condition).encode("utf-8")).hexdigest()


def _run_hash(
    client: EvaluationActionClient,
    runtime: RuntimeConfig,
    *,
    model_label: str,
    condition_hash: str,
) -> str:
    payload = {
        "client": client.config_hash,
        "runtime": runtime.model_dump(mode="json"),
        "model_label": model_label,
        "condition_hash": condition_hash,
    }
    return hashlib.sha256(stable_json(payload).encode("utf-8")).hexdigest()


def _append_record(path: Path, record: InferenceRecord) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(stable_json(record.model_dump(mode="json")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


async def collect_inference_episodes(
    tasks: Sequence[TaskView],
    answers: Sequence[HiddenAnswer],
    db_root: Path,
    output_path: Path,
    client: EvaluationActionClient,
    *,
    runtime: RuntimeConfig,
    model_label: str,
    concurrency: int = 4,
    confirm_real_api: bool = False,
) -> tuple[list[InferenceRecord], dict[str, Any]]:
    """采集所有任务，无论是否答对都恰好保留一条可恢复记录。"""
    if not model_label.strip():
        raise ValueError("model_label 不能为空")
    if concurrency < 1 or concurrency > 32:
        raise ValueError("concurrency 必须位于 1..32")
    if client.real_api and not confirm_real_api:
        raise PermissionError("真实评测 API 必须显式设置 confirm_real_api=true")
    task_by_id = {task.task_id: task for task in tasks}
    answer_by_id = {answer.task_id: answer for answer in answers}
    if len(task_by_id) != len(tasks) or set(task_by_id) != set(answer_by_id):
        raise ValueError("评测 tasks/answers 的 task_id 不唯一或不一致")

    condition_hash = inference_condition_hash(client, runtime)
    run_hash = _run_hash(
        client,
        runtime,
        model_label=model_label,
        condition_hash=condition_hash,
    )
    existing = [InferenceRecord.model_validate(row) for row in read_jsonl(output_path)]
    if any(
        row.config_hash != run_hash or row.model_label != model_label for row in existing
    ):
        raise ValueError("续跑文件包含不同模型或推理配置")
    by_id = {row.task_id: row for row in existing}
    if len(by_id) != len(existing):
        raise ValueError("续跑文件包含重复 task_id")
    loop = asyncio.get_running_loop()

    async def run_one(task: TaskView) -> InferenceRecord:
        answer = answer_by_id[task.task_id]
        episode = await asyncio.to_thread(
            run_episode,
            task,
            answer,
            db_root / task.db_ref,
            _AsyncClientPolicy(client, loop),
            runtime=runtime,
            config_hash=run_hash,
            episode_id=f"{model_label}_{task.task_id}",
        )
        return InferenceRecord(
            task_id=task.task_id,
            model_label=model_label,
            config_hash=run_hash,
            condition_hash=condition_hash,
            episode=episode,
        )

    pending = [task for task in tasks if task.task_id not in by_id]
    for offset in range(0, len(pending), concurrency):
        batch = pending[offset : offset + concurrency]
        results = await asyncio.gather(*(run_one(task) for task in batch))
        for record in results:
            _append_record(output_path, record)
            by_id[record.task_id] = record

    ordered = [by_id[task.task_id] for task in tasks]
    summary = {
        "schema_version": 1,
        "model_label": model_label,
        "model": str(client.config.model),
        "config_hash": run_hash,
        "comparison_condition_hash": condition_hash,
        "seed": int(client.config.seed),
        "task_count": len(tasks),
        "completed_count": len(ordered),
        "submitted_count": sum(
            row.episode.terminal_reason is TerminalReason.SUBMITTED for row in ordered
        ),
        "valid_count": sum(row.episode.valid_for_training for row in ordered),
        "spent_usd": client.spent_usd,
        "output": str(output_path),
    }
    return ordered, summary
