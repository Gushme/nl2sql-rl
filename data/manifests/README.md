# Reproducibility manifests

该目录只保存可公开、可审计的小型派生清单，不保存 BIRD 问题、Gold SQL、数据库或结果表。

- `train_inventory.json`：原始文件与 69 个 SQLite 数据库的大小、SHA256 和任务计数。
- `train_audit.jsonl`：每个 task 的状态、SQL/数据库哈希、运行耗时和结果摘要。
- `train_audit_summary.json`：可执行、空结果、超时和 RL-clean 数量汇总。
- `dev_inventory.json`：固定 HF annotation、本地完整包 provenance，以及 11 个 Dev SQLite 数据库的大小、SHA256 和任务数。
- `dev_audit.jsonl`：500 条 Dev Gold 的双执行状态、耗时和结果摘要，不包含 Gold SQL 原文。
- `dev_audit_summary.json`：`Official-500 / Final-N=498`、2 条超时及数据库哈希不变证明。
- `split_manifest.json`：seed 42 固定的 Train 7,667、Validation 918 和 Dev Final-N 498 task/db 清单。
- `leakage_report.json`：DB ID、数据库 SHA、task ID、标准化问题与近重复检查结果；本次全部为空。
- `qwen_tokenizer_config.json`：固定模型 revision 的 tokenizer/config 文件哈希，明确无权重。
- `verl_v0_8_0.json`：veRL release commit、AgentLoop 源码位置和 GPU 镜像 digest。
- `teacher_harness_preflight_summary.json`：按真实分层顺序对前 100 条隐藏 Gold 做 Harness v2 离线预检的汇总。
- `teacher_api_probe_summary.json`：真实 Teacher Action 兼容性探针的无密钥、无思考内容摘要与累计 Token 账本快照。
- `teacher_pilot_transport_v1_summary.json`：首批真实 Pilot 在 4 条任务后自动暂停的脱敏诊断，记录并行 tool call 问题和 Token 用量。

完整的 Actor task、隐藏 answer、partial checkpoint 和运行缓存位于被 Git 忽略的
`outputs/data/`。Train 清单可由 `nl2sql-rl data clean-train --resume` 重建；Dev 清单可由
`nl2sql-rl data download-dev --database-source local`、`data clean-dev --resume` 和
`data split` 重建。仓库仍不包含 Base/SFT/GRPO 模型评测指标。
