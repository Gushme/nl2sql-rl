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
- `verl_v0_8_0.json`：veRL release commit、AgentLoop/trainer 来源、GPU 镜像 digest，以及动态同奖组过滤补丁前后 SHA256。
- `teacher_harness_preflight_summary.json`：按真实分层顺序对前 100 条隐藏 Gold 做当前 Harness v3 离线预检的汇总。
- `teacher_harness_v2_preflight_summary.json`：升级 Prompt 前 Harness v2 的 100 条历史预检证据。
- `teacher_api_probe_summary.json`：真实 Teacher Action 兼容性探针的无密钥、无思考内容摘要与累计 Token 账本快照。
- `teacher_pilot_transport_v1_summary.json`：首批真实 Pilot 在 4 条任务后自动暂停的脱敏诊断，记录并行 tool call 问题和 Token 用量。
- `teacher_pilot_transport_v2_canary_summary.json`：禁用并行 tool call 后的真实 canary 脱敏诊断，记录单条样本误触发比例阈值及累计 Token 用量。
- `teacher_pilot_transport_v2_partial_summary.json`：Transport v2 追加 8 条后的脱敏诊断，记录合格率、Harness 指标、单次 4xx 暂停和累计 Token 用量。
- `teacher_pilot_transport_v2_window20_summary.json`：Transport v2 满 20 条诊断窗口，记录 35% 合格率、暂停原因、Gold 对照诊断和确定性 replay。
- `teacher_pilot_transport_v3_window20_summary.json`：Harness v3 满 20 条诊断窗口，记录 45% 合格率、v2/v3 同题对照、Wilson 可行性上界、确定性 replay 和累计 Token 用量。

完整的 Actor task、隐藏 answer、partial checkpoint 和运行缓存位于被 Git 忽略的
`outputs/data/`。Train 清单可由 `nl2sql-rl data clean-train --resume` 重建；Dev 清单可由
`nl2sql-rl data download-dev --database-source local`、`data clean-dev --resume` 和
`data split` 重建。仓库仍不包含 Base/SFT/GRPO 模型评测指标。
