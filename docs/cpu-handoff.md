# CPU 交付状态

更新日期：2026-08-08。

## 已实际完成

- BIRD Train 原始输入：9,428 条、69 个 SQLite 数据库。
- Gold SQL 以 10 秒上限、最多 4 个数据库 worker、每条独立连接双执行完成全量审计。
- 审计结果：8,905 条可执行且确定，350 条缺表或缺列，173 条超时。
- 可执行结果中 271 条为空，49 条超过 100,000 行或 64 MiB 限制；最终 `train_rl_clean` 为 8,585 条。
- 审计前后 69 个数据库 SHA256 全部一致。
- BIRD Mini-Dev SQLite 500 已从本地完整包离线导入，固定 HF annotation 的 SHA256 为 `88ceb0710163cae46a256ecea8f0a8c98286599530b60587fda5c3cfe57d45d2`。
- Dev Gold 全量双执行结果为 498 条通过、2 条超时，无空结果或结果超限；数据库 SHA256 前后全部一致，实际口径为 `Official-500 / Final-N=498`。
- seed 42 的内部切分为 Train 7,667、Validation 918；Train/Validation/Dev 的 DB ID、数据库 SHA、task ID、标准化问题和近重复检查均无交叉。
- 完整包 annotation 与固定 HF annotation 有 1 条 SQL 差异，包内 Gold 文件有 4 条差异；两者只记录 provenance，固定 HF annotation 始终是唯一 Gold 来源。
- Qwen2.5-Coder-1.5B-Instruct 固定 revision 的 tokenizer/config 已下载并校验；未下载权重。
- CPU fixture 端到端演练已通过，覆盖只读双执行、错误后纠正、mock Teacher、统一评测 rollout、Action-only SFT、EX、SFT handoff 和 GRPO action mask。
- Teacher 分层采样计划已在 8,585 条候选上生成：Train/Validation 目标为 900/100，简单/中等/挑战目标为 300/500/200，并覆盖 Train 61 个、Validation 8 个数据库。
- Teacher Harness v2 已用计划前缀 100 条隐藏 Gold 做离线预检：90 条 Train、10 条 Validation，复杂度为 30/50/20，100 条全部通过，数据库 SHA256 全部不变；1 次 schema 分页仍保留结构化列信息。
- 实测 5 秒探索上限会让预检中的一条 Gold 偶发超时，因此在任何付费请求前把 `execute_sql` 探索上限调整为 10 秒，并写入新的 Harness 配置哈希 `b1a4977fc6ca9758e37e25770d9aa0a97b3920a425472ab13befec2da5359ba4`。
- 真实 Teacher Action 兼容性探针已完成：鉴权、thinking、reasoning token 明细和严格 JSON Action 均正常；模型以文本 JSON 返回合法 `list_tables` Action，共使用 783 Token，未保存思考内容。原先“必须原生 tool call”的探针判据与项目同时支持两种 Action 传输格式的协议不一致，已修正；模型可见 Harness 未改变。

权威机器可读结果见 `data/manifests/` 下的 Train/Dev inventory、审计 summary、
`split_manifest.json`、`leakage_report.json`、Teacher 预检/探针摘要和模型依赖清单。

## Dev 数据恢复命令

原始数据和 `outputs/` 不进入 Git。已有相同本地包时，可用以下命令重建全部 Dev 派生产物：

```bash
uv run nl2sql-rl data download-dev \
  --database-source local \
  --local-package-root dev500/raw/minidev
uv run nl2sql-rl data clean-dev --resume
uv run nl2sql-rl data split
```

## 本轮未执行

- 100 次真实 Teacher Pilot 与 1,000 条轨迹采集；
- Base、SFT、GRPO 模型推理；
- GPU SFT、checkpoint 合并和 GPU GRPO；
- LLM 行为 Judge。

这些入口均已实现并由 mock、tiny 随机模型或 dry-run 覆盖，必须由用户在具备 API/GPU 和费用授权后显式启动。
仓库没有生成或声称任何 Base/SFT/GRPO 的真实 Mini-Dev 指标。

真实 Teacher 密钥只通过无回显环境变量注入，仓库配置只保留占位 endpoint。若已有明确 Token 额度，可用 `--token-limit` 直接建立累计硬闸门；若需要美元费用报告，则必须从 Model Studio 控制台读取输入、输出单价并同时设置费用上限。探针、正式采集、累计 Token、可选费用、尝试次数、采样清单、Teacher 行为配置、计费口径、Harness 版本、诊断暂停和轨迹迁移共用同一个本地活动账本。
