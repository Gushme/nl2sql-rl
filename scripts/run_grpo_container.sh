#!/usr/bin/env bash
set -euo pipefail

# 该脚本只负责固定 GPU 运行环境；训练参数仍由 configs/grpo.yaml 冻结。
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VERL_COMMIT="7aed6b230776f963fa09509c10d9c3a767d1102c"
VERL_DIGEST="sha256:7b89de392419de3c532b96ea29bf7ac93a083ab965dbb4ec4252adf436d950c3"
VERL_IMAGE="verlai/verl:vllm020.dev1@${VERL_DIGEST}"

if [[ ! -f "${PROJECT_ROOT}/checkpoints/sft-merged/handoff_manifest.json" ]]; then
  echo "缺少 SFT 合并 checkpoint：checkpoints/sft-merged/handoff_manifest.json" >&2
  exit 2
fi

if [[ ! -f "${PROJECT_ROOT}/outputs/data/splits/tasks/train.jsonl" ]]; then
  echo "缺少训练切分；请先完成 Dev500 审计并运行 data split" >&2
  exit 2
fi

# 首次运行需要联网拉取固定镜像、veRL release commit 和模型相关依赖。
docker run --rm --gpus all --ipc=host --shm-size=16g \
  -e "HF_TOKEN=${HF_TOKEN:-}" \
  -e "NL2SQL_VERL_COMMIT=${VERL_COMMIT}" \
  -e "NL2SQL_VERL_IMAGE_DIGEST=${VERL_DIGEST}" \
  -v "${PROJECT_ROOT}:/workspace/nl2sql-rl" \
  -v nl2sql-hf-cache:/root/.cache/huggingface \
  -w /workspace/nl2sql-rl \
  "${VERL_IMAGE}" \
  bash -lc '
    set -euo pipefail
    python -m pip install --no-deps --force-reinstall -r requirements/verl-v0.8.0.txt
    python -m pip install -e .
    nl2sql-rl train grpo --grpo-config configs/grpo.yaml --prepare --run
  '
