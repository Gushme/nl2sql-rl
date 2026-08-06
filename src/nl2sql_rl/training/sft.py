"""Qwen2.5-Coder-1.5B 的 Action-only LoRA SFT 入口。"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any

import yaml
from pydantic import Field, model_validator

from nl2sql_rl.io_utils import read_jsonl, sha256_file, stable_json, write_json
from nl2sql_rl.models import StrictRecord
from nl2sql_rl.training.model_assets import BASE_MODEL, MODEL_REVISION
from nl2sql_rl.training.sft_data import SFTConversation, tokenize_action_only


class SFTConfig(StrictRecord):
    schema_version: int = 1
    base_model: str = BASE_MODEL
    model_revision: str = MODEL_REVISION
    train_data: Path
    validation_data: Path
    output_dir: Path
    seed: int = 42
    max_length: int = 16_384
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list[str] = Field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )
    learning_rate: float = 1e-4
    epochs: float = 3.0
    per_device_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    bf16: bool = True
    save_steps: int = 100
    logging_steps: int = 5

    @model_validator(mode="after")
    def validate_frozen_settings(self) -> SFTConfig:
        expected = {
            "base_model": BASE_MODEL,
            "model_revision": MODEL_REVISION,
            "lora_r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            "learning_rate": 1e-4,
            "epochs": 3.0,
            "per_device_batch_size": 1,
            "gradient_accumulation_steps": 8,
            "max_length": 16_384,
            "seed": 42,
            "bf16": True,
        }
        for name, value in expected.items():
            if getattr(self, name) != value:
                raise ValueError(f"SFT 固定参数 {name} 必须是 {value}")
        return self

    def resolved(self, config_dir: Path) -> SFTConfig:
        def resolve(path: Path) -> Path:
            return path if path.is_absolute() else (config_dir / path).resolve()

        return self.model_copy(
            update={
                "train_data": resolve(self.train_data),
                "validation_data": resolve(self.validation_data),
                "output_dir": resolve(self.output_dir),
            }
        )


def load_sft_config(path: Path) -> SFTConfig:
    raw: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("SFT 配置必须是 YAML mapping")
    return SFTConfig.model_validate(raw).resolved(path.parent)


def _load_conversations(path: Path) -> list[SFTConversation]:
    return [SFTConversation.model_validate(row) for row in read_jsonl(path)]


def inspect_sft_data(config: SFTConfig) -> dict[str, Any]:
    reports: dict[str, dict[str, Any]] = {}
    all_ids: dict[str, set[str]] = {}
    for split, path in (
        ("train", config.train_data),
        ("validation", config.validation_data),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"SFT {split} 数据不存在：{path}")
        rows = _load_conversations(path)
        if not rows:
            raise ValueError(f"SFT {split} 数据为空")
        task_ids = [row.task_id for row in rows]
        if len(task_ids) != len(set(task_ids)):
            raise ValueError(f"SFT {split} task_id 不唯一")
        if any(row.split != split for row in rows):
            raise ValueError(f"SFT {split} 文件包含错误 split")
        serialized = stable_json([row.model_dump(mode="json") for row in rows])
        if "gold_sql" in serialized or '"reward"' in serialized:
            raise ValueError(f"SFT {split} 数据出现隐藏字段")
        reports[split] = {
            "path": str(path),
            "count": len(rows),
            "assistant_actions": sum(len(row.assistant_message_indexes) for row in rows),
            "sha256": sha256_file(path),
        }
        all_ids[split] = set(task_ids)
    overlap = all_ids["train"].intersection(all_ids["validation"])
    if overlap:
        raise ValueError(f"SFT train/validation task_id 交叉：{len(overlap)}")
    return {"schema_version": 1, "splits": reports, "task_id_overlap": []}


def sft_run_hash(config: SFTConfig, data_report: dict[str, Any]) -> str:
    payload = {
        "config": config.model_dump(mode="json"),
        "data": data_report,
    }
    return hashlib.sha256(stable_json(payload).encode("utf-8")).hexdigest()


def preflight_sft(
    config: SFTConfig,
    *,
    dry_run: bool,
    resume_from_checkpoint: Path | None = None,
) -> dict[str, Any]:
    data_report = inspect_sft_data(config)
    if resume_from_checkpoint is not None and not (
        resume_from_checkpoint / "trainer_state.json"
    ).is_file():
        raise FileNotFoundError("恢复目录缺少 trainer_state.json")
    dependencies = {
        name: importlib.util.find_spec(name) is not None
        for name in ("torch", "transformers", "peft", "accelerate")
    }
    if not dry_run:
        missing = [name for name, present in dependencies.items() if not present]
        if missing:
            raise RuntimeError(f"缺少 SFT 可选依赖：{missing}，请运行 uv sync --extra sft")
        torch = importlib.import_module("torch")

        if not torch.cuda.is_available():
            raise RuntimeError("正式 SFT 只允许在 CUDA GPU 环境启动")
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("当前 GPU 不支持 BF16")
    return {
        "schema_version": 1,
        "dry_run": dry_run,
        "base_model": config.base_model,
        "model_revision": config.model_revision,
        "run_hash": sft_run_hash(config, data_report),
        "effective_batch_size": (
            config.per_device_batch_size * config.gradient_accumulation_steps
        ),
        "dependencies": dependencies,
        "data": data_report,
        "resume_from_checkpoint": (
            str(resume_from_checkpoint) if resume_from_checkpoint is not None else None
        ),
    }


def inject_lora(model: Any, config: SFTConfig) -> Any:
    peft = importlib.import_module("peft")

    lora = peft.LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.target_modules,
        bias="none",
        task_type=peft.TaskType.CAUSAL_LM,
    )
    return peft.get_peft_model(model, lora)


def _collator(tokenizer: Any) -> Any:
    torch = importlib.import_module("torch")

    pad_id = int(tokenizer.pad_token_id)

    def collate(features: list[dict[str, list[int]]]) -> dict[str, Any]:
        width = max(len(feature["input_ids"]) for feature in features)

        def pad(values: list[int], fill: int) -> list[int]:
            return values + [fill] * (width - len(values))

        return {
            "input_ids": torch.tensor(
                [pad(feature["input_ids"], pad_id) for feature in features], dtype=torch.long
            ),
            "labels": torch.tensor(
                [pad(feature["labels"], -100) for feature in features], dtype=torch.long
            ),
            "attention_mask": torch.tensor(
                [pad(feature["attention_mask"], 0) for feature in features],
                dtype=torch.long,
            ),
        }

    return collate


def run_sft_training(
    config: SFTConfig, *, resume_from_checkpoint: Path | None = None
) -> dict[str, Any]:
    preflight = preflight_sft(
        config, dry_run=False, resume_from_checkpoint=resume_from_checkpoint
    )
    torch = importlib.import_module("torch")
    transformers = importlib.import_module("transformers")

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        config.base_model, revision=config.model_revision, trust_remote_code=False
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = transformers.AutoModelForCausalLM.from_pretrained(
        config.base_model,
        revision=config.model_revision,
        torch_dtype=torch.bfloat16,
        trust_remote_code=False,
    )
    model = inject_lora(model, config)

    def tokenize(path: Path) -> list[dict[str, list[int]]]:
        dataset = []
        for row in _load_conversations(path):
            item = tokenize_action_only(row, tokenizer, max_length=config.max_length)
            dataset.append(
                {
                    "input_ids": item.input_ids,
                    "labels": item.labels,
                    "attention_mask": item.attention_mask,
                }
            )
        return dataset

    arguments = transformers.TrainingArguments(
        output_dir=str(config.output_dir),
        seed=config.seed,
        data_seed=config.seed,
        per_device_train_batch_size=config.per_device_batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        num_train_epochs=config.epochs,
        bf16=config.bf16,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        eval_strategy="steps",
        eval_steps=config.save_steps,
        save_total_limit=3,
        remove_unused_columns=False,
        report_to=[],
    )
    trainer = transformers.Trainer(
        model=model,
        args=arguments,
        train_dataset=tokenize(config.train_data),
        eval_dataset=tokenize(config.validation_data),
        data_collator=_collator(tokenizer),
    )
    trainer.train(
        resume_from_checkpoint=(
            str(resume_from_checkpoint) if resume_from_checkpoint is not None else None
        )
    )
    adapter_dir = config.output_dir / "final_adapter"
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(adapter_dir)
    manifest = {
        "schema_version": 1,
        "kind": "sft_lora_adapter",
        "base_model": config.base_model,
        "model_revision": config.model_revision,
        "run_hash": preflight["run_hash"],
        "adapter_dir": str(adapter_dir),
    }
    write_json(adapter_dir / "adapter_manifest.json", manifest)
    return manifest


def _tree_hashes(root: Path) -> dict[str, dict[str, Any]]:
    return {
        str(path.relative_to(root)): {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def export_merged_checkpoint(
    config: SFTConfig,
    adapter_dir: Path,
    output_dir: Path,
    *,
    dry_run: bool,
) -> dict[str, Any]:
    adapter_manifest_path = adapter_dir / "adapter_manifest.json"
    adapter_config_path = adapter_dir / "adapter_config.json"
    if not adapter_config_path.is_file():
        raise FileNotFoundError("LoRA adapter 缺少 adapter_config.json")
    if not adapter_manifest_path.is_file():
        raise FileNotFoundError("LoRA adapter 缺少 adapter_manifest.json")
    adapter_manifest: Any = json.loads(adapter_manifest_path.read_text(encoding="utf-8"))
    if not isinstance(adapter_manifest, dict) or adapter_manifest.get("run_hash") is None:
        raise ValueError("adapter_manifest.json 不合法")
    if dry_run:
        return {
            "schema_version": 1,
            "dry_run": True,
            "kind": "sft_merged_checkpoint",
            "base_model": config.base_model,
            "model_revision": config.model_revision,
            "source_run_hash": adapter_manifest["run_hash"],
            "adapter_dir": str(adapter_dir),
            "output_dir": str(output_dir),
        }
    torch = importlib.import_module("torch")
    peft = importlib.import_module("peft")
    transformers = importlib.import_module("transformers")

    if not torch.cuda.is_available():
        raise RuntimeError("合并 1.5B checkpoint 只允许在 CUDA 环境执行")
    base = transformers.AutoModelForCausalLM.from_pretrained(
        config.base_model,
        revision=config.model_revision,
        torch_dtype=torch.bfloat16,
        trust_remote_code=False,
    )
    merged = peft.PeftModel.from_pretrained(base, adapter_dir).merge_and_unload()
    output_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(output_dir, safe_serialization=True)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        config.base_model, revision=config.model_revision, trust_remote_code=False
    )
    tokenizer.save_pretrained(output_dir)
    files = _tree_hashes(output_dir)
    handoff = {
        "schema_version": 1,
        "kind": "sft_merged_checkpoint",
        "base_model": config.base_model,
        "model_revision": config.model_revision,
        "source_run_hash": adapter_manifest["run_hash"],
        "files": files,
    }
    handoff_hash = hashlib.sha256(stable_json(handoff).encode("utf-8")).hexdigest()
    handoff["handoff_hash"] = handoff_hash
    write_json(output_dir / "handoff_manifest.json", handoff)
    return handoff
