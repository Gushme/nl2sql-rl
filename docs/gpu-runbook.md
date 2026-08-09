# GPU 训练运行手册

目标环境为 Linux x86_64、单张 NVIDIA Ampere 或更新 GPU，并要求 BF16。CPU 开发固定使用 uv 管理的 Python 3.11；veRL 运行在独立、按 digest 固定的官方 GPU 容器中。

## 1. 前置产物

开始训练前必须具备：

- `outputs/sft/train.jsonl` 与 `outputs/sft/validation.jsonl`；
- 固定 Qwen revision 可从 Hugging Face 访问；
- 足够的磁盘空间保存 Base、adapter、合并 checkpoint 和 GRPO checkpoint；
- `HF_TOKEN`（模型访问需要时）。

先在 GPU 主机进行只读检查：

```bash
uv sync --frozen --dev --extra sft
uv run nl2sql-rl train sft --dry-run
```

## 2. SFT LoRA

正式运行会加载 Qwen2.5-Coder-1.5B-Instruct 权重，配置固定为 LoRA r=16、alpha=32、dropout=0.05、BF16、batch=1、gradient accumulation=8、LR=1e-4、3 epochs、16K：

```bash
uv run nl2sql-rl train sft --run
```

中断后从包含 `trainer_state.json` 的目录恢复：

```bash
uv run nl2sql-rl train sft \
  --resume-from-checkpoint checkpoints/sft/checkpoint-STEP \
  --run
```

合并 LoRA，并生成带完整文件 SHA256 的强制 handoff：

```bash
uv run nl2sql-rl train export \
  --adapter checkpoints/sft/final_adapter \
  --output checkpoints/sft-merged \
  --run
```

GRPO 若看不到或无法验证 `checkpoints/sft-merged/handoff_manifest.json` 会直接拒绝启动，因此不会静默从 Base 开始训练。

## 3. GRPO

固定运行环境：

- veRL tag `v0.8.0`，commit `7aed6b230776f963fa09509c10d9c3a767d1102c`；
- 镜像 `verlai/verl:vllm020.dev1@sha256:7b89de392419de3c532b96ea29bf7ac93a083ab965dbb4ec4252adf436d950c3`；
- async 自定义 `AgentLoop`，Action token mask=1、工具 observation mask=0；
- 100 次有效更新、prompt batch 2、group size 4，即固定 800 条轨迹进入优化器；
- 每个 Prompt 组 Reward 全相同时整组剔除，每次更新最多补采 10 个生成批次，因此实际生成量为 800–8,000 条；
- temperature 0.7、top-p 0.9、clip 0.2、KL off、LoRA r=16/alpha=32、LR=1e-6；
- 第 25、50、75、100 步保存 checkpoint。

主机先验证 Docker 与 GPU：

```bash
docker version
nvidia-smi
```

再启动固定容器：

```bash
HF_TOKEN=... bash scripts/run_grpo_container.sh
```

脚本会检出固定 veRL commit，依次校验原始 `ray_trainer.py`、动态过滤补丁和补丁后文件的 SHA256，再安装受控源码并执行正式 preflight。以下任何条件都会在训练前失败：SFT handoff 哈希变化、训练数据过期、veRL/补丁/镜像身份不符、训练池不足 2,000 个最坏情况候选 Prompt、无 CUDA/BF16、GPU 架构早于 Ampere。

动态过滤发生在终局 Reward 产生之后、old log-prob 和 Actor 更新之前。全 `+1`、全 `0` 或全负奖励等零方差组都会被剔除；包含任意两个不同 Reward 的组保留。补采期间 Actor 权重不变，凑齐 2 个有效组后才休眠 rollout replicas 并完成一次更新。达到 10 批仍不足时写入失败诊断并终止，不会回填同奖组。

## 4. Checkpoint 与恢复

GRPO checkpoint 写入 `checkpoints/grpo/`，veRL 使用 `trainer.resume_mode=auto`。保留以下复现材料：

- `outputs/grpo/preflight.json`；
- `checkpoints/grpo/run_manifest.json`；
- `checkpoints/grpo/dynamic_filter_metrics.jsonl`；
- `checkpoints/sft-merged/handoff_manifest.json`；
- `configs/grpo.yaml`；
- 项目 Git SHA、镜像 digest、veRL 上游/补丁/补丁后 SHA；
- 训练日志和每 25 步 checkpoint。

`run_manifest.json` 与本次 run hash 不一致时会拒绝自动恢复，必须改用新的输出目录。不要把模型、轨迹、日志或 checkpoint 提交到 Git。
