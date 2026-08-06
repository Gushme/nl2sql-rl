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
