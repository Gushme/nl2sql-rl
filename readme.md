# BIRD Agentic NL2SQL Post-Training

一个以 BIRD SQLite 为数据基础、以
`Qwen/Qwen2.5-Coder-1.5B-Instruct` 为基座的多步 Agentic NL2SQL 后训练项目。

项目按“数据审计 → 只读 SQL Harness → Teacher 轨迹 → Action-only SFT → 在线 GRPO
→ 统一执行评测”组织。当前开发阶段只运行 CPU 数据处理与测试；真实 Teacher API、模型推理、
SFT 和 GRPO GPU 训练需要后续显式启动。

## 安全与数据边界

- `train/`、`dev500/`、`outputs/`、模型权重、轨迹和 checkpoint 不进入 Git。
- Actor 只读取 `TaskView`；Gold SQL 使用独立 `HiddenAnswer` 保存。
- SQLite 只读执行、执行结果等价是唯一正确性裁判，LLM Judge 不参与 Reward 或 EX。
- 训练和内部验证按 `db_id` 隔离，最终评测固定使用 BIRD Mini-Dev SQLite 500。

## 本地开发

```bash
UV_CACHE_DIR=/tmp/nl2sql-rl-uv-cache uv sync --dev
UV_CACHE_DIR=/tmp/nl2sql-rl-uv-cache uv run nl2sql-rl show-config
UV_CACHE_DIR=/tmp/nl2sql-rl-uv-cache uv run pytest -m "not full_data and not network and not gpu"
```

要求 Python 3.11。完整数据、Harness、训练和评测命令会随对应阶段补充。

## Train Gold SQL 审计

现有 BIRD train 保持原样，`train.json` 是 SQL 主来源，`train_gold.sql` 用于逐行交叉校验：

```bash
UV_CACHE_DIR=/tmp/nl2sql-rl-uv-cache uv run nl2sql-rl data inventory
UV_CACHE_DIR=/tmp/nl2sql-rl-uv-cache uv run nl2sql-rl data clean-train --resume
UV_CACHE_DIR=/tmp/nl2sql-rl-uv-cache uv run pytest -m full_data
```

当前固定来源包含 9,428 条任务和 69 个 SQLite 数据库。10 秒只读双执行审计得到：

- 8,905 条可执行且结果稳定；
- 350 条缺表或缺列，173 条超时；
- 可执行项中 271 条为空结果、49 条结果超过限制；
- 最终 8,585 条进入 `train_rl_clean`。

完整 Actor task 和隐藏 answer 位于本地 `outputs/data/train/`，Git 只保存不含问题与 Gold
SQL 的审计清单。审计前后 69 个数据库的 SHA256 全部一致。
