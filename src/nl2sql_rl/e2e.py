"""不依赖 GPU、外部 API 或 BIRD 原始数据的端到端交付演练。"""

from __future__ import annotations

import asyncio
import hashlib
import platform
import sqlite3
import subprocess
import sys
from pathlib import Path
from typing import Any

from nl2sql_rl.agent.loop import ScriptedPolicy, run_episode
from nl2sql_rl.agent.replay import replay_episode
from nl2sql_rl.config import RuntimeConfig
from nl2sql_rl.eval.inference import InferenceRecord, collect_inference_episodes
from nl2sql_rl.eval.pipeline import PredictionRecord, score_dataset
from nl2sql_rl.eval.report import render_report, write_report
from nl2sql_rl.io_utils import read_jsonl, sha256_file, stable_json, write_json, write_jsonl
from nl2sql_rl.models import AgentAction, AuditStatus, HiddenAnswer, TaskView
from nl2sql_rl.sqlite_exec import execute_read_only
from nl2sql_rl.teacher.client import LLMClientConfig, LLMCompletion
from nl2sql_rl.teacher.collector import CollectorConfig, TeacherAttempt, collect_trajectories
from nl2sql_rl.training.grpo import (
    GRPOConfig,
    episode_to_fake_rollout,
    preflight_grpo,
    prepare_grpo_datasets,
)
from nl2sql_rl.training.sft_data import build_sft_conversations, tokenize_action_only


class _CharacterTokenizer:
    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        if add_special_tokens:
            raise ValueError("CPU fixture tokenizer 不接受特殊 token 自动注入")
        return [ord(character) for character in text]


class _MockActionClient:
    real_api = False
    spent_usd = 0.0

    def __init__(self) -> None:
        self.config = LLMClientConfig(
            endpoint="https://cpu-fixture.invalid/v1",
            model="cpu-fixture-policy",
            concurrency=2,
            retry_base_seconds=0,
            real_api=False,
        )
        self.calls = 0

    @property
    def config_hash(self) -> str:
        return self.config.fingerprint()

    async def complete_action(
        self, messages: list[dict[str, Any]], *, max_tokens: int | None = None
    ) -> LLMCompletion:
        del max_tokens
        self.calls += 1
        action_count = sum(message.get("role") == "assistant" for message in messages)
        if action_count == 0:
            action = AgentAction(
                action="describe_schema", arguments={"tables": ["items"]}
            )
        elif action_count == 1:
            action = AgentAction(
                action="execute_sql",
                arguments={"sql": "SELECT COUNT(*) FROM items"},
            )
        else:
            action = (
            AgentAction(
                action="submit_sql",
                arguments={"sql": "SELECT COUNT(*) FROM items"},
            )
            )
        return LLMCompletion(
            action=action,
            response_id=f"cpu_fixture_{self.calls}",
            input_tokens=10,
            output_tokens=5,
            cost_usd=0.0,
        )


def _create_database(path: Path, row_count: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE items(id INTEGER PRIMARY KEY, name TEXT)")
        connection.executemany(
            "INSERT INTO items(id, name) VALUES (?, ?)",
            [(index, f"item_{index}") for index in range(1, row_count + 1)],
        )


def _fixture_data(root: Path) -> tuple[list[TaskView], list[HiddenAnswer]]:
    tasks: list[TaskView] = []
    answers: list[HiddenAnswer] = []
    for index, split in enumerate(("train", "validation"), start=1):
        db_id = f"cpu_{split}_db"
        db_ref = f"databases/{db_id}/{db_id}.sqlite"
        _create_database(root / db_ref, row_count=index + 1)
        task = TaskView(
            task_id=f"cpu_{split}_task",
            split=split,
            db_id=db_id,
            question="items 表有多少行？",
            evidence="行数使用 COUNT(*) 计算。",
            db_ref=db_ref,
        )
        tasks.append(task)
        answers.append(
            HiddenAnswer(
                task_id=task.task_id,
                gold_sql="SELECT COUNT(*) FROM items",
                audit_status=AuditStatus.PASSED,
            )
        )
    return tasks, answers


def _create_fixture_handoff(path: Path, source_run_hash: str) -> None:
    path.mkdir(parents=True)
    (path / "config.json").write_text('{"model_type":"qwen2"}', encoding="utf-8")
    (path / "model.safetensors").write_bytes(b"cpu-fixture-not-a-real-checkpoint")
    files = {
        name: {
            "bytes": (path / name).stat().st_size,
            "sha256": sha256_file(path / name),
        }
        for name in ("config.json", "model.safetensors")
    }
    handoff: dict[str, Any] = {
        "schema_version": 1,
        "kind": "sft_merged_checkpoint",
        "base_model": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
        "model_revision": "2e1fd397ee46e1388853d2af2c993145b0f1098a",
        "source_run_hash": source_run_hash,
        "files": files,
        "fixture_only": True,
    }
    handoff["handoff_hash"] = hashlib.sha256(
        stable_json(handoff).encode("utf-8")
    ).hexdigest()
    write_json(path / "handoff_manifest.json", handoff)


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


async def _run_async_components(
    tasks: list[TaskView],
    answers: list[HiddenAnswer],
    root: Path,
    runtime: RuntimeConfig,
) -> tuple[list[InferenceRecord], dict[str, Any], list[TeacherAttempt], dict[str, Any]]:
    inference_client = _MockActionClient()
    inference, inference_summary = await collect_inference_episodes(
        tasks,
        answers,
        root,
        root / "inference/records.jsonl",
        inference_client,
        runtime=runtime,
        model_label="cpu_fixture",
        concurrency=2,
    )
    teacher_client = _MockActionClient()
    teacher_summary = await collect_trajectories(
        tasks,
        answers,
        root,
        root / "teacher/attempts.jsonl",
        teacher_client,
        runtime=runtime,
        config=CollectorConfig(
            target_total=2,
            train_quota=1,
            validation_quota=1,
            max_attempts=2,
            run_attempt_limit=2,
            concurrency=2,
            simple_ratio=1.0,
            moderate_ratio=0.0,
            challenging_ratio=0.0,
        ),
        tokenizer=_CharacterTokenizer(),
    )
    attempts = [
        TeacherAttempt.model_validate(row)
        for row in read_jsonl(root / "teacher/attempts.jsonl")
    ]
    return inference, inference_summary, attempts, teacher_summary


def run_cpu_e2e(output_root: Path, *, agent_loop_config: Path) -> dict[str, Any]:
    """运行完整 CPU fixture，并把所有中间产物写入一个全新目录。"""
    if output_root.exists():
        raise FileExistsError(f"CPU E2E 输出目录已存在，请换一个新目录：{output_root}")
    output_root.mkdir(parents=True)
    runtime = RuntimeConfig()
    tasks, answers = _fixture_data(output_root)
    write_jsonl(
        output_root / "splits/tasks/train.jsonl",
        [tasks[0].model_dump(mode="json")],
    )
    write_jsonl(
        output_root / "splits/tasks/validation.jsonl",
        [tasks[1].model_dump(mode="json")],
    )
    write_jsonl(
        output_root / "splits/answers/train.jsonl",
        [answers[0].model_dump(mode="json")],
    )
    write_jsonl(
        output_root / "splits/answers/validation.jsonl",
        [answers[1].model_dump(mode="json")],
    )

    database_hashes_before = {
        task.task_id: sha256_file(output_root / task.db_ref) for task in tasks
    }
    gold_runs = [
        (
            execute_read_only(
                answer.gold_sql,
                output_root / task.db_ref,
                timeout_seconds=runtime.gold_timeout_seconds,
            ),
            execute_read_only(
                answer.gold_sql,
                output_root / task.db_ref,
                timeout_seconds=runtime.gold_timeout_seconds,
            ),
        )
        for task, answer in zip(tasks, answers, strict=True)
    ]
    gold_deterministic = all(
        first.status is AuditStatus.PASSED and first.digest == second.digest
        for first, second in gold_runs
    )

    correction_actions = [
        AgentAction(action="describe_schema", arguments={"tables": ["items"]}),
        AgentAction(
            action="execute_sql",
            arguments={"sql": "SELECT missing_column FROM items"},
        ),
        AgentAction(
            action="execute_sql",
            arguments={"sql": "SELECT COUNT(*) FROM items"},
        ),
        AgentAction(
            action="submit_sql",
            arguments={"sql": "SELECT COUNT(*) FROM items"},
        ),
    ]
    correction_episode = run_episode(
        tasks[0],
        answers[0],
        output_root / tasks[0].db_ref,
        ScriptedPolicy(correction_actions),
        runtime=runtime,
        config_hash="cpu-e2e",
        episode_id="cpu_correction_episode",
    )
    replay = replay_episode(
        correction_episode,
        tasks[0],
        answers[0],
        output_root / tasks[0].db_ref,
        runtime=runtime,
    )
    write_json(
        output_root / "harness/correction_episode.json",
        correction_episode.model_dump(mode="json"),
    )

    inference, inference_summary, attempts, teacher_summary = asyncio.run(
        _run_async_components(tasks, answers, output_root, runtime)
    )
    predictions = [
        PredictionRecord(
            task_id=record.task_id,
            prediction_sql=record.episode.submitted_sql,
            usage=record.episode.usage,
        )
        for record in inference
    ]
    evaluation_records, evaluation_summary = score_dataset(
        tasks,
        answers,
        predictions,
        output_root,
        official_count=2,
    )
    write_jsonl(
        output_root / "evaluation/records.jsonl",
        [record.model_dump(mode="json") for record in evaluation_records],
    )
    write_report(
        output_root / "evaluation/report.md",
        "# CPU fixture 报告（不代表 BIRD 模型指标）\n\n"
        + render_report(
            evaluation_records,
            [record.episode for record in inference],
            official_count=2,
            project_git_sha=_git_sha(),
            inference_manifest=inference_summary,
        ),
    )

    conversations = build_sft_conversations(tasks, attempts)
    tokenizer = _CharacterTokenizer()
    tokenized = [tokenize_action_only(row, tokenizer) for row in conversations]
    for split in ("train", "validation"):
        write_jsonl(
            output_root / f"sft/{split}.jsonl",
            [
                row.model_dump(mode="json")
                for row in conversations
                if row.split == split
            ],
        )
    source_run_hash = hashlib.sha256(
        stable_json([row.model_dump(mode="json") for row in conversations]).encode("utf-8")
    ).hexdigest()
    checkpoint = output_root / "checkpoints/sft-merged-fixture"
    _create_fixture_handoff(checkpoint, source_run_hash)
    grpo_config = GRPOConfig(
        checkpoint_dir=checkpoint,
        train_tasks=output_root / "splits/tasks/train.jsonl",
        train_answers=output_root / "splits/answers/train.jsonl",
        validation_tasks=output_root / "splits/tasks/validation.jsonl",
        validation_answers=output_root / "splits/answers/validation.jsonl",
        database_root=output_root,
        train_dataset=output_root / "grpo/train.jsonl",
        validation_dataset=output_root / "grpo/validation.jsonl",
        output_dir=output_root / "checkpoints/grpo-fixture",
        agent_loop_config=agent_loop_config.resolve(),
    )
    grpo_data = prepare_grpo_datasets(grpo_config)
    grpo_preflight = preflight_grpo(grpo_config, dry_run=True)
    fake_rollout = episode_to_fake_rollout(tasks[0], correction_episode, tokenizer)

    database_hashes_after = {
        task.task_id: sha256_file(output_root / task.db_ref) for task in tasks
    }
    report = {
        "schema_version": 1,
        "fixture_only": True,
        "project_git_sha": _git_sha(),
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "sqlite": sqlite3.sqlite_version,
        },
        "data": {
            "tasks": len(tasks),
            "databases": len(database_hashes_before),
            "gold_deterministic": gold_deterministic,
            "database_hashes_unchanged": database_hashes_before == database_hashes_after,
        },
        "harness": {
            "reward": correction_episode.reward,
            "steps": len(correction_episode.events),
            "first_error": next(
                (
                    event.observation.error_code
                    for event in correction_episode.events
                    if event.observation.error_code is not None
                ),
                None,
            ),
            "replay_exact": replay.exact_event_replay,
        },
        "teacher": teacher_summary,
        "sft": {
            "conversations": len(conversations),
            "supervised_action_tokens": sum(
                sum(label != -100 for label in row.labels) for row in tokenized
            ),
            "gpu_training_executed": False,
        },
        "inference": inference_summary,
        "evaluation": evaluation_summary,
        "grpo": {
            "run_hash": grpo_preflight["run_hash"],
            "nominal_rollouts": grpo_preflight["nominal_rollouts"],
            "effective_optimizer_steps": grpo_preflight["effective_optimizer_steps"],
            "retained_rollouts": grpo_preflight["retained_rollouts"],
            "generated_rollouts_min": grpo_preflight["generated_rollouts_min"],
            "generated_rollouts_max": grpo_preflight["generated_rollouts_max"],
            "dynamic_group_filter": grpo_preflight["dynamic_group_filter"],
            "dataset_train_count": grpo_data["train"]["count"],
            "assistant_mask_tokens": sum(fake_rollout.response_mask),
            "tool_mask_tokens": len(fake_rollout.response_mask)
            - sum(fake_rollout.response_mask),
            "gpu_training_executed": False,
        },
        "external_teacher_api_called": False,
        "model_inference_executed": False,
    }
    write_json(output_root / "report.json", report)
    return report
