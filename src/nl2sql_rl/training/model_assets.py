"""只下载并校验 Qwen tokenizer/config，显式排除模型权重。"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import httpx

from nl2sql_rl.io_utils import sha256_file

BASE_MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
MODEL_REVISION = "2e1fd397ee46e1388853d2af2c993145b0f1098a"
METADATA_FILES = (
    "config.json",
    "generation_config.json",
    "merges.txt",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
)


def download_model_metadata(
    destination: Path,
    *,
    force: bool = False,
    client: httpx.Client | None = None,
) -> dict[str, Any]:
    owns_client = client is None
    http = client or httpx.Client(follow_redirects=True, timeout=180.0)
    try:
        files: dict[str, dict[str, Any]] = {}
        for filename in METADATA_FILES:
            target = destination / filename
            if force or not target.is_file():
                target.parent.mkdir(parents=True, exist_ok=True)
                partial = target.with_suffix(f"{target.suffix}.part")
                url = f"https://huggingface.co/{BASE_MODEL}/resolve/{MODEL_REVISION}/{filename}"
                with http.stream("GET", url) as response:
                    response.raise_for_status()
                    with partial.open("wb") as handle:
                        for chunk in response.iter_bytes(chunk_size=1024 * 1024):
                            handle.write(chunk)
                os.replace(partial, target)
            files[filename] = {
                "bytes": target.stat().st_size,
                "sha256": sha256_file(target),
            }
        config: Any = json.loads((destination / "config.json").read_text(encoding="utf-8"))
        tokenizer_config: Any = json.loads(
            (destination / "tokenizer_config.json").read_text(encoding="utf-8")
        )
        if not isinstance(config, dict) or config.get("model_type") != "qwen2":
            raise ValueError("下载的 config.json 不是 Qwen2 模型配置")
        if not isinstance(tokenizer_config, dict) or "chat_template" not in tokenizer_config:
            raise ValueError("tokenizer_config.json 缺少 chat_template")
        unexpected_weights = sorted(
            path.name
            for path in destination.iterdir()
            if path.suffix in (".safetensors", ".bin", ".pt", ".pth")
        )
        if unexpected_weights:
            raise RuntimeError(f"metadata 目录意外出现权重：{unexpected_weights}")
        return {
            "schema_version": 1,
            "model": BASE_MODEL,
            "revision": MODEL_REVISION,
            "source": f"https://huggingface.co/{BASE_MODEL}/tree/{MODEL_REVISION}",
            "weights_downloaded": False,
            "model_type": config["model_type"],
            "architectures": config.get("architectures", []),
            "files": files,
        }
    finally:
        if owns_client:
            http.close()
