# BIRD × Qwen2.5-Coder-1.5B Agentic NL2SQL 后训练

这是一个完整但默认安全停在 CPU 侧的 NL2SQL 后训练项目：以 BIRD SQLite 为数据，以 [Qwen2.5-Coder-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct) 为基座，先采集可执行的多轮 Teacher 轨迹并做 Action-only SFT，再用 veRL v0.8.0 进行带 SQLite 环境反馈的多步 GRPO。

SQL 执行器是 EX 和 Reward 的唯一正确性裁判。LLM Judge 只评价 Agent 行为，不能修改正确性指标。

## 当前交付状态

代码已覆盖数据构建、只读 SQLite harness、Teacher 客户端、SFT、GRPO、统一模型 rollout、确定性评测、错误分类、盲评输入和三模型对比报告。

本机已经实际完成 Train 全量 Gold 审计：

- 输入 9,428 条、69 个数据库；
- 8,905 条可执行且结果确定；
- 350 条缺表或缺列，173 条超时；
- 271 条空结果，49 条结果超过限制；
- `train_rl_clean` 为 8,585 条；
- 审计前后数据库 SHA256 全部一致。

本机也已完成 Dev500 全量审计：固定 annotation 包含 500 条、11 个数据库，498 条通过双执行审计，2 条超时，因此实际评测口径为 `Official-500 / Final-N=498`。Train/Validation/Dev 的 DB ID、数据库 SHA、task ID、标准化问题文本及近重复检查均无交叉。Base/SFT/GRPO 模型指标尚未运行。详见 [CPU 交付状态](docs/cpu-handoff.md)。

Teacher 采集前的 Harness v2 离线 Gold 预检也已完成：按真实采样顺序取前 100 条，Train/Validation 为 90/10，简单/中等/挑战为 30/50/20，100 条全部通过 `describe_schema → execute_sql → submit_sql`，数据库 SHA256 全部不变。该结果不包含模型调用，也不代表 Teacher 合格率。

真实 Teacher Action 兼容性探针已完成：模型以文本 JSON 返回合法 `list_tables` Action，thinking 与 token 明细正常，共使用 783 Token，未保存思考内容。该结果只证明 API 与 Action 协议兼容，不代表 NL2SQL 轨迹质量；完整摘要见 `data/manifests/teacher_api_probe_summary.json`。

首个真实 Pilot 小批次在 4 条任务后按协议错误阈值自动暂停：1 条合格，23 次模型响应中有 2 次在同一轮返回两个原生 tool call；schema、SQL 执行、超时与数据库基础设施均正常。该问题通过显式禁用并行 tool call 修正，模型可见 Prompt、工具与 observation 未改变；旧版唯一合格轨迹会在新行为配置下确定性重放后迁移。脱敏证据见 `data/manifests/teacher_pilot_transport_v1_summary.json`。

## 1. Fresh clone

CPU 开发环境固定使用 Python 3.11 和 uv。本项目额外兼容 Python 3.12，仅用于固定的 veRL GPU 容器；本机 Python 3.13 不作为运行环境。

```bash
git clone https://github.com/Gushme/nl2sql-rl.git
cd nl2sql-rl
uv python install 3.11
uv sync --frozen --dev
uv run nl2sql-rl version
uv run pytest -m "not full_data and not network and not gpu"
```

完整 CPU fixture 演练不会下载数据、调用外部 API、加载模型或使用 GPU：

```bash
uv run nl2sql-rl cpu-e2e --output outputs/cpu_e2e/run
```

报告写入 `outputs/cpu_e2e/run/report.json`。该结果只证明管线连通，不代表 BIRD 模型指标。

## Makefile 快速入口

根目录 Makefile 是现有 CLI 的薄编排层，兼容 macOS 自带的 GNU Make 3.81。默认只显示帮助，不存在会自动调用 API 或启动 GPU 训练的 `make all`：

```bash
make
make setup
make check
make test-full
make data-all
make cpu-e2e E2E_OUTPUT=outputs/cpu_e2e/manual-run
make pipeline-cpu
```

`data-all` 和 `pipeline-cpu` 要求本地已经存在 `train/` 与 `dev500/raw/minidev`。路径可通过 `DEV_PACKAGE_ROOT`、`E2E_OUTPUT`、`PROJECT_CONFIG` 等 Make 变量覆盖。

真实 Teacher API 和 GPU 操作必须显式设置 `CONFIRM=1`；Teacher 还要求累计 Token 上限或完整美元费用闸门。API key 只从环境变量读取且不会出现在命令行：

```bash
read -s TEACHER_API_KEY
export TEACHER_API_KEY
CONFIRM=1 \
TEACHER_ENDPOINT=https://provider.example/v1 \
TEACHER_MODEL=deepseek-v4-flash-0731 \
TEACHER_TOKEN_LIMIT=1000000 \
make teacher-collect

CONFIRM=1 make sft-run
CONFIRM=1 make sft-export
CONFIRM=1 make grpo-run
```

评测目标固定使用本地 OpenAI-compatible backend；真实外部评测 API 仍应直接使用 CLI 的 `--real-api --confirm-real-api`：

```bash
INFERENCE_API_KEY=local \
INFERENCE_ENDPOINT=http://127.0.0.1:8000/v1 \
BASE_MODEL_NAME=qwen-base \
make eval-base
```

完整参数和底层命令仍以 `uv run nl2sql-rl --help` 及下文 CLI 示例为准。

## 2. 数据准备与 Gold 清洗

原始目录永远保持只读并被 Git 忽略：

```text
train/
  train.json
  train_gold.sql
  train_tables.json
  train_databases/<db_id>/<db_id>.sqlite
```

先审计已有 Train：

```bash
uv run nl2sql-rl data inventory
uv run nl2sql-rl data clean-train --resume
uv run pytest -m full_data
```

随后准备固定版本的 [BIRD Mini-Dev SQLite 500](https://github.com/bird-bench/mini_dev)，执行相同的只读双执行清洗，并按 `db_id` 切分。若完整包已解压到 `dev500/raw/minidev`，使用完全离线的本地导入：

```bash
uv run nl2sql-rl data download-dev \
  --database-source local \
  --local-package-root dev500/raw/minidev
uv run nl2sql-rl data clean-dev --resume
uv run nl2sql-rl data split
```

若本地尚无完整包，可从固定镜像下载：

```bash
uv run nl2sql-rl data download-dev --database-source mirror
uv run nl2sql-rl data clean-dev --resume
uv run nl2sql-rl data split
```

镜像不可用时可改用 `--database-source drive`。本地导入不会发起网络请求；下载与审计支持断点。固定 HF annotation 是唯一 Gold 来源，包内 annotation/Gold 差异只写入 provenance，不会替换 Gold。禁止把 `dev500/` 或 `outputs/` 提交到 Git。

清洗产物严格分层：

- `TaskView` 位于 `tasks/`，只含 Actor 可见字段；
- `HiddenAnswer` 位于 `answers/`，只供 Reward/Evaluator；
- Train/Validation 按数据库隔离；
- `teacher_pool.jsonl` 只是两个内部 split 的拼接，不改变 split 标签；
- Dev 空结果只要可执行且确定就保留，模型指标只在实际 `Final-N` 上计算。

## 3. 只读 Agent harness

模型每轮只能输出一个严格 JSON Action：

```json
{"action":"execute_sql","arguments":{"sql":"SELECT COUNT(*) FROM items"}}
```

允许的五个工具为 `list_tables`、`describe_schema`、`search_values`、`execute_sql` 和 `submit_sql`。SQL 同时经过 sqlglot guard、SQLite authorizer、`query_only=ON` 和只读 URI；DDL/DML、PRAGMA、ATTACH、扩展加载、多语句和写入型 CTE 均被拒绝。

Teacher Harness v2 额外强制以下接受条件：

- `describe_schema` 每次最多 5 张表，返回结构化的列、类型、主键和外键；8 KiB 超限时按表分页并返回 `omitted_tables`，不会退化为只有哈希的摘要；
- 每条轨迹必须至少成功描述一次 schema、至少成功执行一次 SQL；探索和最终提交均使用 10 秒上限；
- `submit_sql` 必须与最后一次成功 `execute_sql` 经 sqlglot 规范化后相同，并且最终 SQL 的所有物理表都已成功描述；
- 任意参数、工具、协议或执行错误都会使轨迹退出主 SFT 候选，即使模型随后纠正成功；
- OpenAI-compatible 请求固定设置 `parallel_tool_calls=false`，确保原生 Function Calling 也遵守每轮一个 Action；
- replay 会逐事件比较 Action 与 observation 的 `ok/error_code/payload/truncated`，仅忽略耗时，并核对最终 SQL、Reward 和数据库 SHA。

可用预生成 Action 做确定性 CPU 运行与回放：

```bash
uv run nl2sql-rl agent run \
  --task-path task.json \
  --answer-path answer.json \
  --database path/to/database.sqlite \
  --actions actions.jsonl

uv run nl2sql-rl agent replay \
  --episode-path outputs/episodes/episode.json \
  --task-path task.json \
  --answer-path answer.json \
  --database path/to/database.sqlite
```

## 4. Teacher 轨迹与 SFT 数据

真实 Teacher 默认目标是 1,000 条最终 EX 正确且协议安全的轨迹：900 train、100 validation，复杂度固定为简单 30%、中等 50%、挑战 20%。采样器先按数据库候选量的平方根分配配额，再用 Gold SQL AST 做交叉分层；桶内顺序由 seed 42 和 task ID 的 SHA256 固定。整个活动最多 1,500 次尝试，每次运行默认只推进 100 次，便于在批次边界检查 Harness。

先离线生成配额清单，不会调用 API：

```bash
make teacher-plan
make teacher-harness-preflight
```

当前计划池为 8,585 条，实际配额可行性已经验证：Train 61 个数据库、Validation 8 个数据库均有目标，计划前 100 条严格保持 90/10 和 30/50/20。离线 Gold 预检将完整逐任务记录写到忽略目录 `outputs/teacher/harness_preflight.json`，终端只显示无 Gold 的摘要。

真实请求必须同时提供 API key、显式确认，以及累计 Token 上限或完整美元费用闸门。Token 模式直接累计接口返回的 `prompt_tokens + completion_tokens`，并在并发请求前保守预留；因此不需要虚构未知的美元单价。[阿里云 DeepSeek API 文档](https://help.aliyun.com/en/model-studio/deepseek-api)说明思考 token 按输出 token 计费。若另需美元费用报告，输入与输出单价及费用上限必须一起提供，且活动期间禁止变更口径。`reasoning_content` 只在响应解析期间读取，绝不保存或回传。下面的 endpoint 必须通过参数或 Make 变量注入，不写入仓库：

```bash
read -s TEACHER_API_KEY
export TEACHER_API_KEY

# 先发起恰好一次受保护的 Action 兼容性探针。
CONFIRM=1 \
TEACHER_ENDPOINT=https://provider.example/v1 \
TEACHER_MODEL=deepseek-v4-flash-0731 \
TEACHER_TOKEN_LIMIT=1000000 \
make teacher-probe

uv run nl2sql-rl teacher collect \
  --tasks outputs/data/splits/tasks/teacher_pool.jsonl \
  --answers outputs/data/splits/answers/teacher_pool.jsonl \
  --db-root train \
  --endpoint https://provider.example/v1 \
  --model deepseek-v4-flash-0731 \
  --token-limit 1000000 \
  --run-attempt-limit 100 \
  --enable-thinking \
  --reasoning-effort high \
  --confirm-real-api
```

采集器按 task 和 Harness 版本幂等、断点续采，处理 429、5xx 和超时。活动账本 `campaign_state.json` 跨运行、跨 Harness 版本累计探针与正式请求，1,500 次尝试、Token 上限和可选费用上限都不会因升级或续采重置。每 100 次尝试冻结一个诊断批次；协议/参数错误、schema 信息丢失、超时、上下文溢出、循环、replay 不一致、数据库 SHA 改变、预算耗尽或系统性 API 异常达到门槛时会自动停止后续请求。

如果模型可见 Prompt、工具 schema、observation、超时、SQL Guard 或接受标准变化，必须使用新的 Harness 哈希和输出文件；Teacher 请求行为变化也必须生成新的采集配置哈希和输出文件。旧轨迹只有在初始 Actor 消息、可见协议、逐事件 observation、最终 SQL、Reward 和数据库 SHA 全部一致时才能迁移；否则任务回到原分层队列重新采集。活动账本保留所有 Teacher 行为哈希，并累计旧版本的尝试与 Token。最终 SFT 只接受一个采集配置哈希。

把 1,000 条合格轨迹转换为 SFT 对话：

```bash
uv run nl2sql-rl teacher build-sft \
  --tasks outputs/data/splits/tasks/teacher_pool.jsonl \
  --attempts outputs/teacher/attempts.jsonl
```

只有 assistant Action 内容产生 label；system、user 和工具 observation 都被 mask 为 `-100`。终局 observation、思考内容、Gold、Reward、EX 和拒绝轨迹不会进入 SFT 数据。

## 5. SFT 与 checkpoint handoff

只下载 tokenizer/config、明确不下载 1.5B 权重：

```bash
uv run nl2sql-rl train fetch-model-metadata
```

GPU 主机执行：

```bash
uv sync --frozen --dev --extra sft
uv run nl2sql-rl train sft --dry-run
uv run nl2sql-rl train sft --run

uv run nl2sql-rl train export \
  --adapter checkpoints/sft/final_adapter \
  --output checkpoints/sft-merged \
  --run
```

SFT 固定 LoRA r=16、alpha=32、dropout=0.05、BF16、batch 1、gradient accumulation 8、LR 1e-4、3 epochs、16K、seed 42。导出会生成 `handoff_manifest.json`，GRPO 会重算其中每个文件和整体清单的 SHA256；缺失或被修改时直接失败。

## 6. Base / SFT / GRPO 统一评测

依次把三个模型放到同一个 OpenAI-compatible inference backend，保持 endpoint、prompt、harness、temperature 和 token 上限一致。下面以本地 endpoint 为例；本地服务也需要设置一个占位 key：

```bash
export INFERENCE_API_KEY=local
```

Base：

```bash
uv run nl2sql-rl eval rollout \
  --tasks outputs/data/dev/tasks/final.jsonl \
  --answers outputs/data/dev/answers/final.jsonl \
  --db-root dev500 \
  --endpoint http://127.0.0.1:8000/v1 \
  --model qwen-base \
  --model-label base \
  --cost-limit-usd 0.01 \
  --local-api \
  --output outputs/evaluation/base/inference.jsonl \
  --predictions-output outputs/evaluation/base/predictions.jsonl \
  --episodes-output outputs/evaluation/base/episodes.jsonl \
  --summary-output outputs/evaluation/base/inference_summary.json

uv run nl2sql-rl eval score \
  --tasks outputs/data/dev/tasks/final.jsonl \
  --answers outputs/data/dev/answers/final.jsonl \
  --predictions outputs/evaluation/base/predictions.jsonl \
  --db-root dev500 \
  --output outputs/evaluation/base/records.jsonl \
  --summary-output outputs/evaluation/base/score_summary.json
```

SFT 与 GRPO 使用同样命令，只替换服务中的 checkpoint、`--model`、`--model-label` 和输出目录：

```bash
# SFT
uv run nl2sql-rl eval rollout --tasks outputs/data/dev/tasks/final.jsonl --answers outputs/data/dev/answers/final.jsonl --db-root dev500 --endpoint http://127.0.0.1:8000/v1 --model qwen-sft --model-label sft --cost-limit-usd 0.01 --local-api --output outputs/evaluation/sft/inference.jsonl --predictions-output outputs/evaluation/sft/predictions.jsonl --episodes-output outputs/evaluation/sft/episodes.jsonl --summary-output outputs/evaluation/sft/inference_summary.json
uv run nl2sql-rl eval score --tasks outputs/data/dev/tasks/final.jsonl --answers outputs/data/dev/answers/final.jsonl --predictions outputs/evaluation/sft/predictions.jsonl --db-root dev500 --output outputs/evaluation/sft/records.jsonl --summary-output outputs/evaluation/sft/score_summary.json

# GRPO
uv run nl2sql-rl eval rollout --tasks outputs/data/dev/tasks/final.jsonl --answers outputs/data/dev/answers/final.jsonl --db-root dev500 --endpoint http://127.0.0.1:8000/v1 --model qwen-grpo --model-label grpo --cost-limit-usd 0.01 --local-api --output outputs/evaluation/grpo/inference.jsonl --predictions-output outputs/evaluation/grpo/predictions.jsonl --episodes-output outputs/evaluation/grpo/episodes.jsonl --summary-output outputs/evaluation/grpo/inference_summary.json
uv run nl2sql-rl eval score --tasks outputs/data/dev/tasks/final.jsonl --answers outputs/data/dev/answers/final.jsonl --predictions outputs/evaluation/grpo/predictions.jsonl --db-root dev500 --output outputs/evaluation/grpo/records.jsonl --summary-output outputs/evaluation/grpo/score_summary.json
```

生成单模型报告、盲评输入和三模型比较：

```bash
uv run nl2sql-rl eval report --records outputs/evaluation/base/records.jsonl --episodes outputs/evaluation/base/episodes.jsonl --inference-summary outputs/evaluation/base/inference_summary.json --output outputs/evaluation/base/report.md
uv run nl2sql-rl eval judge --tasks outputs/data/dev/tasks/final.jsonl --episodes outputs/evaluation/base/episodes.jsonl --output outputs/evaluation/base/blind_judge_inputs.jsonl

uv run nl2sql-rl eval compare \
  --base-records outputs/evaluation/base/records.jsonl \
  --sft-records outputs/evaluation/sft/records.jsonl \
  --grpo-records outputs/evaluation/grpo/records.jsonl \
  --base-inference outputs/evaluation/base/inference_summary.json \
  --sft-inference outputs/evaluation/sft/inference_summary.json \
  --grpo-inference outputs/evaluation/grpo/inference_summary.json
```

`eval compare` 会硬检查三组 task ID 与推理条件哈希完全一致，否则拒绝生成报告。EX 使用 [BIRD 官方 set 结果语义](https://github.com/bird-bench/mini_dev/blob/main/evaluation/evaluation_ex.py)；Soft-F1 与 R-VES 是辅助指标，R-VES 不进入 Reward。

## 7. Agentic GRPO

GRPO 必须读取 SFT 合并 checkpoint。CPU 侧 dry-run：

```bash
uv run nl2sql-rl train grpo --prepare --dry-run
```

正式训练固定 veRL [v0.8.0](https://github.com/volcengine/verl/tree/v0.8.0) 和镜像 digest；运行：

```bash
HF_TOKEN=... bash scripts/run_grpo_container.sh
```

配置为 100 steps、prompt batch 2、group size 4、名义 800 条 rollout、每 25 步保存；temperature 0.7、top-p 0.9、clip 0.2、KL off、LoRA r=16/alpha=32、LR 1e-6。每个 rollout 最多 10 个 Action，工具错误作为下一轮 observation，因此这是多步 Agentic RL，而非单轮 SQL RLVR。完整说明见 [GPU 运行手册](docs/gpu-runbook.md)。

## 8. 测试与复现边界

```bash
uv run ruff check .
uv run mypy
uv run pytest -m "not full_data and not network and not gpu"
uv run pytest -m full_data
uv build
```

CI 无 BIRD、无 GPU、无 API key 也能通过 fixture 测试。完整数据、模型、轨迹、checkpoint、日志、`.env` 和任何 API key 都不会进入 Git；Git 只保存代码、配置、fixture、来源 revision/SHA、task status 审计与汇总指标。
