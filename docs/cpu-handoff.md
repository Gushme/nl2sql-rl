# CPU 交付状态

更新日期：2026-08-06。

## 已实际完成

- BIRD Train 原始输入：9,428 条、69 个 SQLite 数据库。
- Gold SQL 以 10 秒上限、最多 4 个数据库 worker、每条独立连接双执行完成全量审计。
- 审计结果：8,905 条可执行且确定，350 条缺表或缺列，173 条超时。
- 可执行结果中 271 条为空，49 条超过 100,000 行或 64 MiB 限制；最终 `train_rl_clean` 为 8,585 条。
- 审计前后 69 个数据库 SHA256 全部一致。
- Qwen2.5-Coder-1.5B-Instruct 固定 revision 的 tokenizer/config 已下载并校验；未下载权重。
- CPU fixture 端到端演练已通过，覆盖只读双执行、错误后纠正、mock Teacher、统一评测 rollout、Action-only SFT、EX、SFT handoff 和 GRPO action mask。

权威机器可读结果见 `data/manifests/train_audit_summary.json`、
`data/manifests/train_inventory.json` 和 `data/manifests/qwen_tokenizer_config.json`。

## 明确延后

BIRD Mini-Dev SQLite 500 的 annotation 已固定为 500 条、11 个数据库，但完整数据库包在本机下载速度过低。根据项目约定，本轮未等待下载，因此以下内容没有伪造结果：

- Dev500 数据库完整性与 SHA 清单；
- Dev Gold SQL 全量执行清洗；
- `Official-500` / `Final-N` 实际计数；
- Train/Validation/Dev 泄漏审计；
- Base/SFT/GRPO 模型指标。

恢复时从以下命令继续，下载器支持固定 Hugging Face mirror 或官方 Drive 包：

```bash
uv run nl2sql-rl data download-dev --database-source mirror
uv run nl2sql-rl data clean-dev --resume
uv run nl2sql-rl data split
```

## 本轮未执行

- 真实外部 Teacher API 请求与 1,000 条轨迹采集；
- Base、SFT、GRPO 模型推理；
- GPU SFT、checkpoint 合并和 GPU GRPO；
- LLM 行为 Judge。

这些入口均已实现并由 mock、tiny 随机模型或 dry-run 覆盖，必须由用户在具备 API/GPU 和费用授权后显式启动。
