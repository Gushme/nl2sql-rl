"""Qwen2.5-Coder-1.5B 的 veRL v0.8.0 Agentic GRPO 配置与入口。"""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import os
import platform
import subprocess
from pathlib import Path
from typing import Any, Protocol

import yaml
from pydantic import Field, model_validator

from nl2sql_rl.agent.loop import build_actor_messages
from nl2sql_rl.io_utils import read_jsonl, sha256_file, stable_json, write_json, write_jsonl
from nl2sql_rl.models import AuditStatus, EpisodeResult, HiddenAnswer, StrictRecord, TaskView
from nl2sql_rl.training.grpo_filter import DynamicGroupFilterConfig
from nl2sql_rl.training.grpo_reward import adapt_episode_reward
from nl2sql_rl.training.model_assets import BASE_MODEL, MODEL_REVISION

VERL_VERSION = "0.8.0"
VERL_COMMIT = "7aed6b230776f963fa09509c10d9c3a767d1102c"
VERL_IMAGE = (
    "verlai/verl:vllm020.dev1@"
    "sha256:7b89de392419de3c532b96ea29bf7ac93a083ab965dbb4ec4252adf436d950c3"
)
AGENT_LOOP_NAME = "nl2sql_agent_loop"
VERL_DYNAMIC_FILTER_PATCH_ID = "nl2sql-dynamic-group-filter-v1"
VERL_RAY_TRAINER_SOURCE_SHA256 = (
    "de58d295cf86656a28196b0718168d4a11666f3e30957b7e166914496c2a6d66"
)
VERL_PATCHED_RAY_TRAINER_SHA256 = (
    "1de682b934bc07f36a6ef2813a0f0add66544f866cc4e7f0b0d4a07dc1f751cc"
)


class TokenizerLike(Protocol):
    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]: ...


class GRPOConfig(StrictRecord):
    schema_version: int = 1
    base_model: str = BASE_MODEL
    model_revision: str = MODEL_REVISION
    verl_version: str = VERL_VERSION
    verl_commit: str = VERL_COMMIT
    container_image: str = VERL_IMAGE
    checkpoint_dir: Path
    train_tasks: Path
    train_answers: Path
    validation_tasks: Path
    validation_answers: Path
    database_root: Path
    train_dataset: Path
    validation_dataset: Path
    output_dir: Path
    agent_loop_config: Path
    seed: int = 42
    total_steps: int = 100
    save_steps: int = 25
    prompt_batch_size: int = 2
    group_size: int = 4
    temperature: float = 0.7
    top_p: float = 0.9
    clip_ratio: float = 0.2
    kl_enabled: bool = False
    learning_rate: float = 1e-6
    lora_r: int = 16
    lora_alpha: int = 32
    max_actions: int = 10
    max_action_tokens: int = 512
    max_prompt_length: int = 4_096
    max_response_length: int = 12_288
    n_gpus_per_node: int = 1
    agent_workers: int = Field(default=4, ge=1, le=32)
    gpu_memory_utilization: float = Field(default=0.6, gt=0, lt=1)
    dynamic_group_filter: DynamicGroupFilterConfig = Field(
        default_factory=DynamicGroupFilterConfig
    )

    @model_validator(mode="after")
    def validate_frozen_settings(self) -> GRPOConfig:
        expected: dict[str, object] = {
            "base_model": BASE_MODEL,
            "model_revision": MODEL_REVISION,
            "verl_version": VERL_VERSION,
            "verl_commit": VERL_COMMIT,
            "container_image": VERL_IMAGE,
            "seed": 42,
            "total_steps": 100,
            "save_steps": 25,
            "prompt_batch_size": 2,
            "group_size": 4,
            "temperature": 0.7,
            "top_p": 0.9,
            "clip_ratio": 0.2,
            "kl_enabled": False,
            "learning_rate": 1e-6,
            "lora_r": 16,
            "lora_alpha": 32,
            "max_actions": 10,
            "max_action_tokens": 512,
            "n_gpus_per_node": 1,
            "dynamic_group_filter": DynamicGroupFilterConfig(),
        }
        for name, value in expected.items():
            if getattr(self, name) != value:
                raise ValueError(f"GRPO 固定参数 {name} 必须是 {value}")
        if self.max_prompt_length + self.max_response_length != 16_384:
            raise ValueError("GRPO prompt 与 response 长度之和必须为 16384")
        if self.total_steps % self.save_steps != 0:
            raise ValueError("save_steps 必须整除 total_steps")
        return self

    @property
    def nominal_rollouts(self) -> int:
        """兼容旧报告字段，表示进入优化器的有效轨迹数。"""
        return self.total_steps * self.prompt_batch_size * self.group_size

    @property
    def retained_rollouts(self) -> int:
        return self.nominal_rollouts

    @property
    def minimum_generated_rollouts(self) -> int:
        return self.retained_rollouts

    @property
    def maximum_generated_rollouts(self) -> int:
        return (
            self.retained_rollouts
            * self.dynamic_group_filter.max_generation_batches_per_update
        )

    @property
    def maximum_candidate_prompts(self) -> int:
        return (
            self.total_steps
            * self.prompt_batch_size
            * self.dynamic_group_filter.max_generation_batches_per_update
        )

    def resolved(self, config_dir: Path) -> GRPOConfig:
        path_names = (
            "checkpoint_dir",
            "train_tasks",
            "train_answers",
            "validation_tasks",
            "validation_answers",
            "database_root",
            "train_dataset",
            "validation_dataset",
            "output_dir",
            "agent_loop_config",
        )
        updates = {
            name: (
                value
                if (value := getattr(self, name)).is_absolute()
                else (config_dir / value).resolve()
            )
            for name in path_names
        }
        return self.model_copy(update=updates)


class FakeRolloutOutput(StrictRecord):
    prompt_ids: list[int]
    response_ids: list[int]
    response_mask: list[int]
    reward_score: float
    num_turns: int
    extra_fields: dict[str, Any] = Field(default_factory=dict)


def load_grpo_config(path: Path) -> GRPOConfig:
    raw: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("GRPO 配置必须是 YAML mapping")
    return GRPOConfig.model_validate(raw).resolved(path.parent)


def _project_root() -> Path:
    """定位包含 pyproject.toml 的源码仓库，用于读取受控补丁。"""
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    raise FileNotFoundError("无法定位 nl2sql-rl 项目根目录")


def verl_patch_identity() -> dict[str, str]:
    """返回会进入 run hash 的 veRL 补丁身份。"""
    patch_path = _project_root() / "patches/verl-v0.8.0-dynamic-group-filter.patch"
    if not patch_path.is_file():
        raise FileNotFoundError(f"veRL 动态过滤补丁不存在：{patch_path}")
    return {
        "patch_id": VERL_DYNAMIC_FILTER_PATCH_ID,
        "patch_path": str(patch_path),
        "patch_sha256": sha256_file(patch_path),
        "upstream_ray_trainer_sha256": VERL_RAY_TRAINER_SOURCE_SHA256,
        "patched_ray_trainer_sha256": VERL_PATCHED_RAY_TRAINER_SHA256,
    }


def _handoff_hash(payload: dict[str, Any]) -> str:
    unsigned = dict(payload)
    unsigned.pop("handoff_hash", None)
    return hashlib.sha256(stable_json(unsigned).encode("utf-8")).hexdigest()


def validate_sft_handoff(checkpoint_dir: Path) -> dict[str, Any]:
    manifest_path = checkpoint_dir / "handoff_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            "GRPO 只接受带 handoff_manifest.json 的 SFT 合并 checkpoint"
        )
    raw: Any = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("handoff_manifest.json 必须是 JSON 对象")
    if raw.get("kind") != "sft_merged_checkpoint":
        raise ValueError("GRPO checkpoint 不是 SFT 合并产物")
    if raw.get("base_model") != BASE_MODEL or raw.get("model_revision") != MODEL_REVISION:
        raise ValueError("SFT checkpoint 的 Base 模型或 revision 不匹配")
    source_run_hash = raw.get("source_run_hash")
    if not isinstance(source_run_hash, str) or len(source_run_hash) != 64:
        raise ValueError("SFT checkpoint 缺少合法 source_run_hash")
    claimed_hash = raw.get("handoff_hash")
    if not isinstance(claimed_hash, str) or claimed_hash != _handoff_hash(raw):
        raise ValueError("SFT checkpoint handoff_hash 不匹配")
    files = raw.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("SFT checkpoint manifest 缺少文件哈希")
    has_config = False
    has_weights = False
    for relative, metadata in files.items():
        if not isinstance(relative, str) or not isinstance(metadata, dict):
            raise ValueError("SFT checkpoint 文件清单格式错误")
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError("SFT checkpoint 文件路径越界")
        path = checkpoint_dir / relative_path
        if not path.is_file():
            raise FileNotFoundError(f"SFT checkpoint 缺少文件：{relative}")
        if metadata.get("bytes") != path.stat().st_size:
            raise ValueError(f"SFT checkpoint 文件大小不匹配：{relative}")
        if metadata.get("sha256") != sha256_file(path):
            raise ValueError(f"SFT checkpoint 文件哈希不匹配：{relative}")
        has_config = has_config or relative == "config.json"
        has_weights = has_weights or path.suffix in {".safetensors", ".bin"}
    if not has_config or not has_weights:
        raise ValueError("SFT checkpoint 必须同时包含 config.json 和模型权重")
    return raw


def _resolved_database_path(database_root: Path, task: TaskView) -> Path:
    root = database_root.resolve()
    path = (root / task.db_ref).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"db_ref 越过 database_root：{task.task_id}") from exc
    if not path.is_file():
        raise FileNotFoundError(f"任务数据库不存在：{path}")
    return path


def _jsonl_rows_hash(rows: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(stable_json(row).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _inspect_split(
    tasks_path: Path,
    answers_path: Path,
    database_root: Path,
    *,
    expected_split: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not tasks_path.is_file() or not answers_path.is_file():
        raise FileNotFoundError(f"GRPO {expected_split} tasks/answers 尚未生成")
    tasks = [TaskView.model_validate(row) for row in read_jsonl(tasks_path)]
    answers = [HiddenAnswer.model_validate(row) for row in read_jsonl(answers_path)]
    if not tasks or len(tasks) != len(answers):
        raise ValueError(f"GRPO {expected_split} 数据为空或 tasks/answers 数量不等")
    task_by_id = {task.task_id: task for task in tasks}
    answer_by_id = {answer.task_id: answer for answer in answers}
    if len(task_by_id) != len(tasks) or len(answer_by_id) != len(answers):
        raise ValueError(f"GRPO {expected_split} task_id 不唯一")
    if set(task_by_id) != set(answer_by_id):
        raise ValueError(f"GRPO {expected_split} tasks/answers task_id 不一致")

    rows: list[dict[str, Any]] = []
    for index, task in enumerate(tasks):
        answer = answer_by_id[task.task_id]
        if task.split != expected_split:
            raise ValueError(f"任务 {task.task_id} 的 split 不是 {expected_split}")
        if answer.audit_status is not AuditStatus.PASSED:
            raise ValueError(f"任务 {task.task_id} 的 Gold 未通过审计")
        database = _resolved_database_path(database_root, task)
        prompt = build_actor_messages(task)
        serialized_prompt = stable_json(prompt)
        if any(
            f'"{key}"' in serialized_prompt
            for key in ("gold_sql", "reward", "result_cache_ref", "audit_status")
        ):
            raise ValueError(f"任务 {task.task_id} 的 Actor prompt 出现隐藏字段")
        rows.append(
            {
                "data_source": "bird_sqlite_nl2sql",
                "agent_name": AGENT_LOOP_NAME,
                "prompt": prompt,
                "ability": "nl2sql",
                "reward_model": {"style": "agent_loop", "ground_truth": "hidden"},
                "extra_info": {
                    "index": index,
                    "task": task.model_dump(mode="json"),
                    "hidden_answer": answer.model_dump(mode="json"),
                    "db_path": str(database),
                },
            }
        )
    report = {
        "count": len(rows),
        "tasks_sha256": sha256_file(tasks_path),
        "answers_sha256": sha256_file(answers_path),
        "task_ids": sorted(task_by_id),
        "db_ids": sorted({task.db_id for task in tasks}),
        "expected_dataset_sha256": _jsonl_rows_hash(rows),
    }
    return rows, report


def inspect_grpo_data(config: GRPOConfig) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    train_rows, train_report = _inspect_split(
        config.train_tasks,
        config.train_answers,
        config.database_root,
        expected_split="train",
    )
    validation_rows, validation_report = _inspect_split(
        config.validation_tasks,
        config.validation_answers,
        config.database_root,
        expected_split="validation",
    )
    overlap = set(train_report["task_ids"]).intersection(validation_report["task_ids"])
    if overlap:
        raise ValueError(f"GRPO train/validation task_id 交叉：{len(overlap)}")
    database_overlap = set(train_report["db_ids"]).intersection(validation_report["db_ids"])
    if database_overlap:
        raise ValueError(f"GRPO train/validation db_id 交叉：{len(database_overlap)}")
    return (
        {"train": train_rows, "validation": validation_rows},
        {
            "schema_version": 1,
            "train": train_report,
            "validation": validation_report,
            "task_id_overlap": [],
            "db_id_overlap": [],
        },
    )


def prepare_grpo_datasets(config: GRPOConfig) -> dict[str, Any]:
    rows_by_split, report = inspect_grpo_data(config)
    write_jsonl(config.train_dataset, rows_by_split["train"])
    write_jsonl(config.validation_dataset, rows_by_split["validation"])
    report["train"]["dataset"] = str(config.train_dataset)
    report["train"]["dataset_sha256"] = sha256_file(config.train_dataset)
    report["validation"]["dataset"] = str(config.validation_dataset)
    report["validation"]["dataset_sha256"] = sha256_file(config.validation_dataset)
    return report


def _validate_agent_loop_config(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"veRL AgentLoop 配置不存在：{path}")
    raw: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("veRL AgentLoop 配置必须是 YAML list")
    expected_target = "nl2sql_rl.training.verl_agent_loop.NL2SQLAgentLoop"
    matching = [row for row in raw if isinstance(row, dict) and row.get("name") == AGENT_LOOP_NAME]
    if len(matching) != 1 or matching[0].get("_target_") != expected_target:
        raise ValueError("veRL AgentLoop 名称或 _target_ 与固定实现不一致")


def _gpu_runtime_report(config: GRPOConfig) -> dict[str, Any]:
    if platform.system() != "Linux" or platform.machine() not in {"x86_64", "AMD64"}:
        raise RuntimeError("正式 GRPO 只允许在 Linux x86_64 GPU 容器中启动")
    verl = importlib.import_module("verl")
    torch = importlib.import_module("torch")
    if getattr(verl, "__version__", None) != config.verl_version:
        raise RuntimeError(f"veRL 版本必须是 {config.verl_version}")
    if os.environ.get("NL2SQL_VERL_COMMIT") != config.verl_commit:
        raise RuntimeError("NL2SQL_VERL_COMMIT 未设置或与固定 commit 不一致")
    expected_digest = config.container_image.rsplit("@", maxsplit=1)[-1]
    if os.environ.get("NL2SQL_VERL_IMAGE_DIGEST") != expected_digest:
        raise RuntimeError("当前容器镜像 digest 未通过固定值校验")
    patch_identity = verl_patch_identity()
    if os.environ.get("NL2SQL_VERL_PATCH_SHA256") != patch_identity["patch_sha256"]:
        raise RuntimeError("NL2SQL_VERL_PATCH_SHA256 未设置或与受控补丁不一致")
    trainer_file = Path(str(getattr(verl, "__file__", ""))).resolve().parent / (
        "trainer/ppo/ray_trainer.py"
    )
    if not trainer_file.is_file():
        raise RuntimeError("无法定位已安装的 veRL ray_trainer.py")
    installed_trainer_sha256 = sha256_file(trainer_file)
    if installed_trainer_sha256 != VERL_PATCHED_RAY_TRAINER_SHA256:
        raise RuntimeError("已安装的 veRL trainer 未通过动态过滤补丁哈希校验")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("正式 GRPO 需要支持 BF16 的 NVIDIA CUDA GPU")
    capability = tuple(int(value) for value in torch.cuda.get_device_capability(0))
    if capability < (8, 0):
        raise RuntimeError("正式 GRPO 需要 NVIDIA Ampere 或更新架构")
    return {
        "platform": platform.platform(),
        "verl_version": verl.__version__,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "compute_capability": list(capability),
        "dynamic_filter_patch": patch_identity,
        "installed_ray_trainer": str(trainer_file),
        "installed_ray_trainer_sha256": installed_trainer_sha256,
    }


def grpo_run_hash(
    config: GRPOConfig, data_report: dict[str, Any], handoff: dict[str, Any]
) -> str:
    payload = {
        "config": config.model_dump(mode="json"),
        "data": data_report,
        "handoff_hash": handoff["handoff_hash"],
        "verl_patch": verl_patch_identity(),
    }
    return hashlib.sha256(stable_json(payload).encode("utf-8")).hexdigest()


def preflight_grpo(config: GRPOConfig, *, dry_run: bool) -> dict[str, Any]:
    _, data_report = inspect_grpo_data(config)
    _validate_agent_loop_config(config.agent_loop_config)
    handoff = validate_sft_handoff(config.checkpoint_dir)
    patch_identity = verl_patch_identity()
    dependencies = {
        name: importlib.util.find_spec(name) is not None
        for name in ("torch", "verl", "vllm", "ray")
    }
    runtime = None if dry_run else _gpu_runtime_report(config)
    datasets: dict[str, dict[str, Any]] = {}
    for split, path in (
        ("train", config.train_dataset),
        ("validation", config.validation_dataset),
    ):
        actual_hash = sha256_file(path) if path.is_file() else None
        expected_hash = data_report[split]["expected_dataset_sha256"]
        datasets[split] = {
            "path": str(path),
            "exists": path.is_file(),
            "sha256": actual_hash,
            "matches_source": actual_hash == expected_hash,
        }
    train_prompt_count = int(data_report["train"]["count"])
    candidate_pool_sufficient = train_prompt_count >= config.maximum_candidate_prompts
    if not dry_run and not candidate_pool_sufficient:
        raise ValueError(
            "GRPO 训练池不足以覆盖动态过滤最坏情况："
            f"{train_prompt_count} < {config.maximum_candidate_prompts}"
        )
    return {
        "schema_version": 1,
        "dry_run": dry_run,
        "run_hash": grpo_run_hash(config, data_report, handoff),
        "sft_handoff_hash": handoff["handoff_hash"],
        "nominal_rollouts": config.nominal_rollouts,
        "effective_optimizer_steps": config.total_steps,
        "retained_rollouts": config.retained_rollouts,
        "generated_rollouts_min": config.minimum_generated_rollouts,
        "generated_rollouts_max": config.maximum_generated_rollouts,
        "maximum_candidate_prompts": config.maximum_candidate_prompts,
        "candidate_pool_sufficient": candidate_pool_sufficient,
        "dynamic_group_filter": config.dynamic_group_filter.model_dump(mode="json"),
        "verl_patch": patch_identity,
        "checkpoint_steps": list(
            range(config.save_steps, config.total_steps + 1, config.save_steps)
        ),
        "dependencies": dependencies,
        "runtime": runtime,
        "data": data_report,
        "prepared_datasets": datasets,
    }


def episode_to_fake_rollout(
    task: TaskView,
    episode: EpisodeResult,
    tokenizer: TokenizerLike,
) -> FakeRolloutOutput:
    """把 CPU harness episode 编码成与 AgentLoopOutput 同构的假 rollout。"""
    if task.task_id != episode.task_id:
        raise ValueError("TaskView 与 EpisodeResult 的 task_id 不一致")
    prompt_ids = tokenizer.encode(
        stable_json(build_actor_messages(task)), add_special_tokens=False
    )
    response_ids: list[int] = []
    response_mask: list[int] = []
    for event in episode.events:
        if event.action is not None:
            action_ids = tokenizer.encode(
                stable_json(event.action.model_dump(mode="json")),
                add_special_tokens=False,
            )
            response_ids.extend(action_ids)
            response_mask.extend([1] * len(action_ids))
        observation_ids = tokenizer.encode(
            stable_json(event.observation.model_dump(mode="json")),
            add_special_tokens=False,
        )
        response_ids.extend(observation_ids)
        response_mask.extend([0] * len(observation_ids))
    reward = adapt_episode_reward(episode)
    return FakeRolloutOutput(
        prompt_ids=prompt_ids,
        response_ids=response_ids,
        response_mask=response_mask,
        reward_score=reward.score,
        num_turns=len(episode.events) * 2 + 2,
        extra_fields={
            "task_id": task.task_id,
            "terminal_reason": episode.terminal_reason.value,
            "valid_for_advantage": True,
        },
    )


def build_verl_command(config: GRPOConfig) -> list[str]:
    target_modules = "['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj']"
    total_context = config.max_prompt_length + config.max_response_length
    overrides = [
        "algorithm.adv_estimator=grpo",
        "algorithm.use_kl_in_reward=False",
        "critic.enable=False",
        "reward.reward_model.enable=False",
        f"data.train_files={config.train_dataset}",
        f"data.val_files={config.validation_dataset}",
        f"data.train_batch_size={config.prompt_batch_size}",
        f"data.gen_batch_size={config.prompt_batch_size}",
        f"data.val_batch_size={config.prompt_batch_size}",
        f"data.max_prompt_length={config.max_prompt_length}",
        f"data.max_response_length={config.max_response_length}",
        "data.return_raw_chat=True",
        "data.filter_overlong_prompts=True",
        "data.truncation=error",
        f"data.seed={config.seed}",
        f"actor_rollout_ref.model.path={config.checkpoint_dir}",
        f"actor_rollout_ref.model.lora_rank={config.lora_r}",
        f"actor_rollout_ref.model.lora_alpha={config.lora_alpha}",
        f"actor_rollout_ref.model.target_modules={target_modules}",
        "actor_rollout_ref.model.trust_remote_code=False",
        "actor_rollout_ref.model.use_remove_padding=True",
        "actor_rollout_ref.model.enable_gradient_checkpointing=True",
        f"actor_rollout_ref.actor.optim.lr={config.learning_rate}",
        f"actor_rollout_ref.actor.ppo_mini_batch_size={config.prompt_batch_size}",
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1",
        f"actor_rollout_ref.actor.ppo_max_token_len_per_gpu={total_context}",
        f"actor_rollout_ref.actor.clip_ratio={config.clip_ratio}",
        f"actor_rollout_ref.actor.clip_ratio_low={config.clip_ratio}",
        f"actor_rollout_ref.actor.clip_ratio_high={config.clip_ratio}",
        "actor_rollout_ref.actor.use_kl_loss=False",
        "actor_rollout_ref.rollout.name=vllm",
        "actor_rollout_ref.rollout.mode=async",
        f"actor_rollout_ref.rollout.n={config.group_size}",
        f"actor_rollout_ref.rollout.temperature={config.temperature}",
        f"actor_rollout_ref.rollout.top_p={config.top_p}",
        "actor_rollout_ref.rollout.top_k=-1",
        "actor_rollout_ref.rollout.tensor_model_parallel_size=1",
        f"actor_rollout_ref.rollout.gpu_memory_utilization={config.gpu_memory_utilization}",
        "actor_rollout_ref.rollout.load_format=safetensors",
        "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1",
        f"actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu={total_context}",
        "actor_rollout_ref.rollout.multi_turn.enable=True",
        f"actor_rollout_ref.rollout.multi_turn.max_assistant_turns={config.max_actions}",
        f"actor_rollout_ref.rollout.agent.num_workers={config.agent_workers}",
        f"actor_rollout_ref.rollout.agent.default_agent_loop={AGENT_LOOP_NAME}",
        f"actor_rollout_ref.rollout.agent.agent_loop_config_path={config.agent_loop_config}",
        "+algorithm.filter_groups.enable=True",
        f"+algorithm.filter_groups.metric={config.dynamic_group_filter.metric}",
        "+algorithm.filter_groups.max_num_gen_batches="
        f"{config.dynamic_group_filter.max_generation_batches_per_update}",
        "trainer.logger=['console']",
        "trainer.project_name=bird_nl2sql",
        "trainer.experiment_name=qwen2_5_coder_1_5b_agentic_grpo",
        "trainer.nnodes=1",
        f"trainer.n_gpus_per_node={config.n_gpus_per_node}",
        f"trainer.total_training_steps={config.total_steps}",
        "trainer.total_epochs=1",
        f"trainer.save_freq={config.save_steps}",
        "trainer.test_freq=-1",
        "trainer.val_before_train=False",
        "trainer.resume_mode=auto",
        f"trainer.default_local_dir={config.output_dir}",
    ]
    return ["python", "-m", "verl.trainer.main_ppo", *overrides]


def _run_manifest_payload(config: GRPOConfig, preflight: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "kind": "grpo_run",
        "run_hash": preflight["run_hash"],
        "sft_handoff_hash": preflight["sft_handoff_hash"],
        "verl_commit": config.verl_commit,
        "container_image": config.container_image,
        "verl_patch": preflight["verl_patch"],
        "dynamic_group_filter": preflight["dynamic_group_filter"],
        "effective_optimizer_steps": preflight["effective_optimizer_steps"],
        "retained_rollouts": preflight["retained_rollouts"],
        "generated_rollouts_min": preflight["generated_rollouts_min"],
        "generated_rollouts_max": preflight["generated_rollouts_max"],
    }


def validate_or_write_run_manifest(
    config: GRPOConfig, preflight: dict[str, Any]
) -> Path:
    """阻止自动恢复到配置或补丁身份不同的旧 GRPO checkpoint。"""
    manifest_path = config.output_dir / "run_manifest.json"
    expected = _run_manifest_payload(config, preflight)
    if manifest_path.is_file():
        actual: Any = json.loads(manifest_path.read_text(encoding="utf-8"))
        if actual != expected:
            raise ValueError("现有 GRPO run_manifest 与本次配置不一致，请使用新的输出目录")
        return manifest_path
    if config.output_dir.exists() and any(config.output_dir.iterdir()):
        raise ValueError("GRPO 输出目录非空但缺少 run_manifest，拒绝自动恢复")
    write_json(manifest_path, expected)
    return manifest_path


def run_grpo(config: GRPOConfig) -> int:
    preflight = preflight_grpo(config, dry_run=False)
    if not all(
        row["matches_source"] for row in preflight["prepared_datasets"].values()
    ):
        raise FileNotFoundError("veRL JSONL 数据缺失或已过期，请先使用 --prepare 重新生成")
    validate_or_write_run_manifest(config, preflight)
    environment = dict(os.environ)
    environment["NL2SQL_GRPO_RUN_HASH"] = str(preflight["run_hash"])
    environment["NL2SQL_GRPO_FILTER_METRICS"] = str(
        config.output_dir / "dynamic_filter_metrics.jsonl"
    )
    environment["NL2SQL_VERL_PATCH_SHA256"] = str(
        preflight["verl_patch"]["patch_sha256"]
    )
    completed = subprocess.run(
        build_verl_command(config), check=False, env=environment
    )
    if completed.returncode != 0:
        raise RuntimeError(f"veRL GRPO 退出码为 {completed.returncode}")
    return completed.returncode
