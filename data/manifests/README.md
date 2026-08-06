# Reproducibility manifests

该目录只保存可公开、可审计的小型派生清单，不保存 BIRD 问题、Gold SQL、数据库或结果表。

- `train_inventory.json`：原始文件与 69 个 SQLite 数据库的大小、SHA256 和任务计数。
- `train_audit.jsonl`：每个 task 的状态、SQL/数据库哈希、运行耗时和结果摘要。
- `train_audit_summary.json`：可执行、空结果、超时和 RL-clean 数量汇总。
- `qwen_tokenizer_config.json`：固定模型 revision 的 tokenizer/config 文件哈希，明确无权重。
- `verl_v0_8_0.json`：veRL release commit、AgentLoop 源码位置和 GPU 镜像 digest。

完整的 Actor task、隐藏 answer、partial checkpoint 和运行缓存位于被 Git 忽略的
`outputs/data/`。清单可由 `nl2sql-rl data clean-train --resume` 完整重建。

Dev500 数据库下载与全量审计目前明确延后，因此仓库中没有 `dev_inventory.json`、
`dev_audit_summary.json`、`split_manifest.json` 或模型评测指标；这些文件只能由实际数据运行生成。
