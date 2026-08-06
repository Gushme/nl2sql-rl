from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

import pytest
from pydantic import ValidationError

from nl2sql_rl.io_utils import sha256_file, stable_json, write_json, write_jsonl
from nl2sql_rl.models import AuditStatus, HiddenAnswer, TaskView
from nl2sql_rl.training.grpo import (
    VERL_COMMIT,
    GRPOConfig,
    build_verl_command,
    preflight_grpo,
    prepare_grpo_datasets,
    validate_sft_handoff,
)


def _database(path: Path, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE numbers(value INTEGER)")
        connection.execute("INSERT INTO numbers VALUES (?)", (value,))


def _handoff(path: Path) -> None:
    path.mkdir(parents=True)
    (path / "config.json").write_text('{"model_type":"qwen2"}', encoding="utf-8")
    (path / "model.safetensors").write_bytes(b"tiny-random-weight-fixture")
    files = {
        filename: {
            "bytes": (path / filename).stat().st_size,
            "sha256": sha256_file(path / filename),
        }
        for filename in ("config.json", "model.safetensors")
    }
    manifest: dict[str, object] = {
        "schema_version": 1,
        "kind": "sft_merged_checkpoint",
        "base_model": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
        "model_revision": "2e1fd397ee46e1388853d2af2c993145b0f1098a",
        "source_run_hash": "a" * 64,
        "files": files,
    }
    manifest["handoff_hash"] = hashlib.sha256(
        stable_json(manifest).encode("utf-8")
    ).hexdigest()
    write_json(path / "handoff_manifest.json", manifest)


def _task(task_id: str, split: str, db_id: str) -> TaskView:
    return TaskView(
        task_id=task_id,
        split=split,
        db_id=db_id,
        question="返回唯一数字",
        evidence="",
        db_ref=f"train_databases/{db_id}/{db_id}.sqlite",
    )


def _config(tmp_path: Path) -> GRPOConfig:
    database_root = tmp_path / "train"
    train_task = _task("train_1", "train", "train_db")
    validation_task = _task("validation_1", "validation", "validation_db")
    for task, value in ((train_task, 1), (validation_task, 2)):
        _database(database_root / task.db_ref, value)

    train_tasks = tmp_path / "splits/tasks/train.jsonl"
    train_answers = tmp_path / "splits/answers/train.jsonl"
    validation_tasks = tmp_path / "splits/tasks/validation.jsonl"
    validation_answers = tmp_path / "splits/answers/validation.jsonl"
    write_jsonl(train_tasks, [train_task.model_dump(mode="json")])
    write_jsonl(validation_tasks, [validation_task.model_dump(mode="json")])
    write_jsonl(
        train_answers,
        [
            HiddenAnswer(
                task_id=train_task.task_id,
                gold_sql="SELECT value FROM numbers",
                audit_status=AuditStatus.PASSED,
            ).model_dump(mode="json")
        ],
    )
    write_jsonl(
        validation_answers,
        [
            HiddenAnswer(
                task_id=validation_task.task_id,
                gold_sql="SELECT value FROM numbers",
                audit_status=AuditStatus.PASSED,
            ).model_dump(mode="json")
        ],
    )
    checkpoint = tmp_path / "sft-merged"
    _handoff(checkpoint)
    agent_config = tmp_path / "verl_agent_loop.yaml"
    agent_config.write_text(
        "- name: nl2sql_agent_loop\n"
        "  _target_: nl2sql_rl.training.verl_agent_loop.NL2SQLAgentLoop\n",
        encoding="utf-8",
    )
    return GRPOConfig(
        checkpoint_dir=checkpoint,
        train_tasks=train_tasks,
        train_answers=train_answers,
        validation_tasks=validation_tasks,
        validation_answers=validation_answers,
        database_root=database_root,
        train_dataset=tmp_path / "grpo/train.jsonl",
        validation_dataset=tmp_path / "grpo/validation.jsonl",
        output_dir=tmp_path / "grpo-checkpoints",
        agent_loop_config=agent_config,
    )


def test_grpo_preflight_requires_verified_sft_handoff_and_fixed_settings(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    first = preflight_grpo(config, dry_run=True)
    second = preflight_grpo(config, dry_run=True)
    assert first["run_hash"] == second["run_hash"]
    assert first["nominal_rollouts"] == 800
    assert first["checkpoint_steps"] == [25, 50, 75, 100]
    assert first["data"]["db_id_overlap"] == []
    assert first["prepared_datasets"]["train"]["matches_source"] is False
    assert validate_sft_handoff(config.checkpoint_dir)["source_run_hash"] == "a" * 64
    invalid_config = config.model_dump()
    invalid_config["group_size"] = 8
    with pytest.raises(ValidationError, match="group_size"):
        GRPOConfig.model_validate(invalid_config)


def test_handoff_rejects_base_directory_and_modified_weight(tmp_path: Path) -> None:
    base_only = tmp_path / "base"
    base_only.mkdir()
    (base_only / "config.json").write_text("{}", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="SFT 合并 checkpoint"):
        validate_sft_handoff(base_only)

    merged = tmp_path / "merged"
    _handoff(merged)
    (merged / "model.safetensors").write_bytes(b"modified")
    with pytest.raises(ValueError, match=r"文件大小|文件哈希"):
        validate_sft_handoff(merged)


def test_prepare_grpo_jsonl_separates_actor_prompt_from_hidden_answer(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    report = prepare_grpo_datasets(config)
    row = json.loads(config.train_dataset.read_text(encoding="utf-8"))
    prompt = stable_json(row["prompt"])
    assert "gold_sql" not in prompt
    assert "SELECT value FROM numbers" not in prompt
    assert row["agent_name"] == "nl2sql_agent_loop"
    assert row["extra_info"]["hidden_answer"]["gold_sql"] == "SELECT value FROM numbers"
    assert report["train"]["dataset_sha256"] == sha256_file(config.train_dataset)
    assert preflight_grpo(config, dry_run=True)["prepared_datasets"]["train"][
        "matches_source"
    ]


def test_verl_command_contains_multiturn_grpo_and_checkpoint_handoff(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    command = build_verl_command(config)
    serialized = "\n".join(command)
    assert command[:3] == ["python", "-m", "verl.trainer.main_ppo"]
    assert "algorithm.adv_estimator=grpo" in command
    assert "actor_rollout_ref.rollout.mode=async" in command
    assert "actor_rollout_ref.rollout.n=4" in command
    assert "actor_rollout_ref.actor.use_kl_loss=False" in command
    assert "algorithm.use_kl_in_reward=False" in command
    assert "trainer.total_training_steps=100" in command
    assert "trainer.save_freq=25" in command
    assert "actor_rollout_ref.model.lora_rank=16" in command
    assert str(config.checkpoint_dir) in serialized
    assert VERL_COMMIT not in serialized
