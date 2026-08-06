from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
from pydantic import ValidationError

from nl2sql_rl.io_utils import write_jsonl
from nl2sql_rl.training.model_assets import METADATA_FILES, download_model_metadata
from nl2sql_rl.training.sft import (
    SFTConfig,
    export_merged_checkpoint,
    inspect_sft_data,
    preflight_sft,
)
from nl2sql_rl.training.sft_data import SFTConversation


def _conversation(task_id: str, split: str) -> SFTConversation:
    return SFTConversation(
        task_id=task_id,
        split=split,
        messages=[
            {"role": "system", "content": "系统"},
            {"role": "user", "content": "问题"},
            {
                "role": "assistant",
                "content": '{"action":"submit_sql","arguments":{"sql":"SELECT 1"}}',
            },
        ],
        assistant_message_indexes=[2],
        source_config_hash="teacher",
    )


def _config(tmp_path: Path) -> SFTConfig:
    train = tmp_path / "train.jsonl"
    validation = tmp_path / "validation.jsonl"
    write_jsonl(train, [_conversation("train_1", "train").model_dump(mode="json")])
    write_jsonl(
        validation,
        [_conversation("validation_1", "validation").model_dump(mode="json")],
    )
    return SFTConfig(
        train_data=train,
        validation_data=validation,
        output_dir=tmp_path / "checkpoint",
    )


def test_sft_preflight_validates_data_hashes_and_frozen_hyperparameters(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    data = inspect_sft_data(config)
    first = preflight_sft(config, dry_run=True)
    second = preflight_sft(config, dry_run=True)
    assert data["splits"]["train"]["count"] == 1
    assert first["effective_batch_size"] == 8
    assert first["run_hash"] == second["run_hash"]
    with pytest.raises(ValidationError, match="lora_r"):
        SFTConfig(
            train_data=config.train_data,
            validation_data=config.validation_data,
            output_dir=config.output_dir,
            lora_r=8,
        )


def test_sft_resume_and_export_require_checkpoint_manifests(tmp_path: Path) -> None:
    config = _config(tmp_path)
    resume = tmp_path / "resume"
    resume.mkdir()
    with pytest.raises(FileNotFoundError, match="trainer_state"):
        preflight_sft(config, dry_run=True, resume_from_checkpoint=resume)

    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    (adapter / "adapter_manifest.json").write_text(
        json.dumps({"run_hash": "sft-run"}), encoding="utf-8"
    )
    report = export_merged_checkpoint(
        config, adapter, tmp_path / "merged", dry_run=True
    )
    assert report["kind"] == "sft_merged_checkpoint"
    assert report["source_run_hash"] == "sft-run"


def test_metadata_downloader_excludes_weights_and_hashes_every_file(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        filename = request.url.path.rsplit("/", maxsplit=1)[-1]
        if filename == "config.json":
            content = b'{"model_type":"qwen2","architectures":["Qwen2ForCausalLM"]}'
        elif filename == "tokenizer_config.json":
            content = b'{"chat_template":"template"}'
        else:
            content = b"fixture"
        return httpx.Response(200, request=request, content=content)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        report = download_model_metadata(tmp_path / "model", client=client)
    assert report["weights_downloaded"] is False
    assert set(report["files"]) == set(METADATA_FILES)
    assert all(value["sha256"] for value in report["files"].values())
