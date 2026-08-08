SHELL := /bin/bash
.DEFAULT_GOAL := help
.NOTPARALLEL:
.DELETE_ON_ERROR:

UV ?= uv
CLI = $(UV) run nl2sql-rl
SFT_CLI = $(UV) run --extra sft nl2sql-rl
PYTHON_VERSION ?= 3.11
PROJECT_CONFIG ?= configs/project.yaml
SFT_CONFIG ?= configs/sft.yaml
GRPO_CONFIG ?= configs/grpo.yaml

DEV_PACKAGE_ROOT ?= dev500/raw/minidev
DEV_TASKS ?= outputs/data/dev/tasks/final.jsonl
DEV_ANSWERS ?= outputs/data/dev/answers/final.jsonl
DEV_DB_ROOT ?= dev500
TEACHER_TASKS ?= outputs/data/splits/tasks/teacher_pool.jsonl
TEACHER_ANSWERS ?= outputs/data/splits/answers/teacher_pool.jsonl
TEACHER_DB_ROOT ?= train
TEACHER_ATTEMPTS ?= outputs/teacher/attempts.jsonl
TEACHER_SUMMARY ?= outputs/teacher/summary.json
TEACHER_PLAN ?= outputs/teacher/sampling_plan.json
TEACHER_PROBE_OUTPUT ?= outputs/teacher/probe.json
TEACHER_PREFLIGHT_REPORT ?= outputs/teacher/harness_preflight.json
TEACHER_DIAGNOSTICS_DIR ?= outputs/teacher/diagnostics
TEACHER_CAMPAIGN_STATE ?= outputs/teacher/campaign_state.json
TEACHER_MIGRATE_FROM ?=
E2E_OUTPUT ?= outputs/cpu_e2e/make-$(shell date +%Y%m%d-%H%M%S)

CONFIRM ?= 0
TEACHER_ENDPOINT ?=
TEACHER_MODEL ?= deepseek-v4-flash-0731
TEACHER_COST_LIMIT_USD ?=
TEACHER_TOKEN_LIMIT ?=
TEACHER_INPUT_PRICE_PER_MILLION ?=
TEACHER_OUTPUT_PRICE_PER_MILLION ?=
TEACHER_MAX_REQUEST_COST_USD ?= 0.01
TEACHER_TARGET_TOTAL ?= 1000
TEACHER_TRAIN_QUOTA ?= 900
TEACHER_VALIDATION_QUOTA ?= 100
TEACHER_MAX_ATTEMPTS ?= 1500
TEACHER_RUN_ATTEMPT_LIMIT ?= 100
TEACHER_CONCURRENCY ?= 4

SFT_ADAPTER ?= checkpoints/sft/final_adapter
SFT_MERGED_CHECKPOINT ?= checkpoints/sft-merged

INFERENCE_ENDPOINT ?= http://127.0.0.1:8000/v1
INFERENCE_COST_LIMIT_USD ?= 0.01
INFERENCE_CONCURRENCY ?= 4
BASE_MODEL_NAME ?= qwen-base
SFT_MODEL_NAME ?= qwen-sft
GRPO_MODEL_NAME ?= qwen-grpo
EVALUATION_ROOT ?= outputs/evaluation
EVAL_COMPARISON_OUTPUT ?= $(EVALUATION_ROOT)/comparison.md
EVAL_LABEL ?=
EVAL_MODEL ?=
EVAL_ROOT = $(EVALUATION_ROOT)/$(EVAL_LABEL)

.PHONY: help setup lint typecheck test test-full build check
.PHONY: data-train data-dev-local data-split data-all cpu-e2e pipeline-cpu
.PHONY: teacher-plan teacher-harness-preflight teacher-probe teacher-collect teacher-build-sft
.PHONY: sft-preflight sft-run sft-export
.PHONY: grpo-preflight grpo-run eval-base eval-sft eval-grpo eval-compare
.PHONY: guard-confirm guard-teacher guard-eval _eval-model

# 默认目标只展示可用命令，不执行数据处理、API 请求或训练。
help:
	@printf '%s\n' \
		'BIRD Agentic NL2SQL Make targets' \
		'' \
		'环境与验证：' \
		'  setup             安装 Python 3.11 和冻结的开发依赖' \
		'  check             依次运行 lint、类型检查、CPU 测试和构建' \
		'  test-full         运行本地 Train/Dev 全量数据验收' \
		'' \
		'数据与 CPU：' \
		'  data-train        审计 Train Gold（支持断点续跑）' \
		'  data-dev-local    从 DEV_PACKAGE_ROOT 离线导入并审计 Dev500' \
		'  data-split        生成 Train/Validation/Dev 切分与泄漏报告' \
		'  data-all          依次运行全部数据阶段和 full-data 测试' \
		'  cpu-e2e           运行无 API/GPU 的 fixture 端到端演练' \
		'  pipeline-cpu      串联 check、data-all 和 cpu-e2e' \
		'' \
		'显式外部操作：' \
		'  teacher-plan      离线生成分层采样配额，不调用 API' \
		'  teacher-harness-preflight 用前 100 个 Gold 离线验证 Harness' \
		'  teacher-probe     单次 Function Calling 探针；要求 CONFIRM=1' \
		'  teacher-collect   真实 Teacher API；要求 CONFIRM=1 和预算上限' \
		'  teacher-build-sft 构建 Action-only SFT 数据' \
		'  sft-preflight     SFT 配置与数据预检' \
		'  sft-run           GPU SFT；要求 CONFIRM=1' \
		'  sft-export        合并 SFT checkpoint；要求 CONFIRM=1' \
		'  grpo-preflight    GRPO 数据与 handoff 预检' \
		'  grpo-run          容器化 GPU GRPO；要求 CONFIRM=1' \
		'' \
		'本地模型评测：' \
		'  eval-base         评测 BASE_MODEL_NAME' \
		'  eval-sft          评测 SFT_MODEL_NAME' \
		'  eval-grpo         评测 GRPO_MODEL_NAME' \
		'  eval-compare      生成三模型公平比较报告'

setup:
	$(UV) python install $(PYTHON_VERSION)
	$(UV) sync --frozen --dev

lint:
	$(UV) run ruff check .

typecheck:
	$(UV) run mypy

test:
	$(UV) run pytest -m "not full_data and not network and not gpu"

test-full:
	$(UV) run pytest -m full_data

build:
	$(UV) build

check: lint typecheck test build

data-train:
	$(CLI) data inventory --config "$(PROJECT_CONFIG)"
	$(CLI) data clean-train --config "$(PROJECT_CONFIG)" --resume

data-dev-local:
	$(CLI) data download-dev --config "$(PROJECT_CONFIG)" --database-source local --local-package-root "$(DEV_PACKAGE_ROOT)"
	$(CLI) data clean-dev --config "$(PROJECT_CONFIG)" --resume

data-split:
	$(CLI) data split --config "$(PROJECT_CONFIG)"

data-all: data-train data-dev-local data-split test-full

cpu-e2e:
	$(CLI) cpu-e2e --output "$(E2E_OUTPUT)"

pipeline-cpu: check data-all cpu-e2e

guard-confirm:
	@if [[ "$(CONFIRM)" != "1" ]]; then echo "拒绝执行：该目标要求显式设置 CONFIRM=1"; exit 2; fi

guard-teacher: guard-confirm
	@if [[ -z "$$TEACHER_API_KEY" ]]; then echo "拒绝执行：环境变量 TEACHER_API_KEY 未设置"; exit 2; fi
	@if [[ -z "$(TEACHER_ENDPOINT)" ]]; then echo "拒绝执行：TEACHER_ENDPOINT 未设置"; exit 2; fi
	@if [[ -z "$(TEACHER_MODEL)" ]]; then echo "拒绝执行：TEACHER_MODEL 未设置"; exit 2; fi
	@if [[ -z "$(TEACHER_TOKEN_LIMIT)" && -z "$(TEACHER_COST_LIMIT_USD)" ]]; then echo "拒绝执行：必须显式设置 TEACHER_TOKEN_LIMIT 或完整美元费用闸门"; exit 2; fi
	@if [[ -n "$(TEACHER_COST_LIMIT_USD)" || -n "$(TEACHER_INPUT_PRICE_PER_MILLION)" || -n "$(TEACHER_OUTPUT_PRICE_PER_MILLION)" ]]; then \
		if [[ -z "$(TEACHER_COST_LIMIT_USD)" ]]; then echo "拒绝执行：美元计费模式缺少 TEACHER_COST_LIMIT_USD"; exit 2; fi; \
		if [[ -z "$(TEACHER_INPUT_PRICE_PER_MILLION)" ]]; then echo "拒绝执行：美元计费模式缺少 TEACHER_INPUT_PRICE_PER_MILLION"; exit 2; fi; \
		if [[ -z "$(TEACHER_OUTPUT_PRICE_PER_MILLION)" ]]; then echo "拒绝执行：美元计费模式缺少 TEACHER_OUTPUT_PRICE_PER_MILLION"; exit 2; fi; \
	fi

teacher-plan:
	$(CLI) teacher plan \
		--tasks "$(TEACHER_TASKS)" \
		--answers "$(TEACHER_ANSWERS)" \
		--output "$(TEACHER_PLAN)" \
		--target-total "$(TEACHER_TARGET_TOTAL)" \
		--train-quota "$(TEACHER_TRAIN_QUOTA)" \
		--validation-quota "$(TEACHER_VALIDATION_QUOTA)"

teacher-harness-preflight:
	$(CLI) teacher harness-preflight \
		--tasks "$(TEACHER_TASKS)" \
		--answers "$(TEACHER_ANSWERS)" \
		--db-root "$(TEACHER_DB_ROOT)" \
		--output "$(TEACHER_PREFLIGHT_REPORT)" \
		--limit 100 \
		--config "$(PROJECT_CONFIG)"

teacher-probe: guard-teacher
	$(CLI) teacher probe \
		--endpoint "$(TEACHER_ENDPOINT)" \
		--model "$(TEACHER_MODEL)" $(if $(TEACHER_COST_LIMIT_USD),--cost-limit-usd "$(TEACHER_COST_LIMIT_USD)") $(if $(TEACHER_TOKEN_LIMIT),--token-limit "$(TEACHER_TOKEN_LIMIT)") \
		--output "$(TEACHER_PROBE_OUTPUT)" \
		--campaign-state-output "$(TEACHER_CAMPAIGN_STATE)" $(if $(TEACHER_INPUT_PRICE_PER_MILLION),--input-price-per-million "$(TEACHER_INPUT_PRICE_PER_MILLION)") $(if $(TEACHER_OUTPUT_PRICE_PER_MILLION),--output-price-per-million "$(TEACHER_OUTPUT_PRICE_PER_MILLION)") \
		--max-request-cost-usd "$(TEACHER_MAX_REQUEST_COST_USD)" \
		--target-total "$(TEACHER_TARGET_TOTAL)" \
		--max-attempts "$(TEACHER_MAX_ATTEMPTS)" \
		--confirm-real-api

teacher-collect: guard-teacher
	$(CLI) teacher collect \
		--tasks "$(TEACHER_TASKS)" \
		--answers "$(TEACHER_ANSWERS)" \
		--db-root "$(TEACHER_DB_ROOT)" \
		--endpoint "$(TEACHER_ENDPOINT)" \
		--model "$(TEACHER_MODEL)" $(if $(TEACHER_COST_LIMIT_USD),--cost-limit-usd "$(TEACHER_COST_LIMIT_USD)") $(if $(TEACHER_TOKEN_LIMIT),--token-limit "$(TEACHER_TOKEN_LIMIT)") \
		--output "$(TEACHER_ATTEMPTS)" \
		--summary-output "$(TEACHER_SUMMARY)" \
		--diagnostics-dir "$(TEACHER_DIAGNOSTICS_DIR)" \
		--campaign-state-output "$(TEACHER_CAMPAIGN_STATE)" $(if $(TEACHER_MIGRATE_FROM),--migrate-from "$(TEACHER_MIGRATE_FROM)") $(if $(TEACHER_INPUT_PRICE_PER_MILLION),--input-price-per-million "$(TEACHER_INPUT_PRICE_PER_MILLION)") $(if $(TEACHER_OUTPUT_PRICE_PER_MILLION),--output-price-per-million "$(TEACHER_OUTPUT_PRICE_PER_MILLION)") \
		--max-request-cost-usd "$(TEACHER_MAX_REQUEST_COST_USD)" \
		--target-total "$(TEACHER_TARGET_TOTAL)" \
		--train-quota "$(TEACHER_TRAIN_QUOTA)" \
		--validation-quota "$(TEACHER_VALIDATION_QUOTA)" \
		--max-attempts "$(TEACHER_MAX_ATTEMPTS)" \
		--run-attempt-limit "$(TEACHER_RUN_ATTEMPT_LIMIT)" \
		--concurrency "$(TEACHER_CONCURRENCY)" \
		--confirm-real-api

teacher-build-sft:
	$(CLI) teacher build-sft --tasks "$(TEACHER_TASKS)" --attempts "$(TEACHER_ATTEMPTS)"

sft-preflight:
	$(SFT_CLI) train sft --sft-config "$(SFT_CONFIG)" --dry-run

sft-run: guard-confirm
	$(SFT_CLI) train sft --sft-config "$(SFT_CONFIG)" --run

sft-export: guard-confirm
	$(SFT_CLI) train export \
		--sft-config "$(SFT_CONFIG)" \
		--adapter "$(SFT_ADAPTER)" \
		--output "$(SFT_MERGED_CHECKPOINT)" \
		--run

grpo-preflight:
	$(CLI) train grpo --grpo-config "$(GRPO_CONFIG)" --prepare --dry-run

grpo-run: guard-confirm
	bash scripts/run_grpo_container.sh

guard-eval:
	@if [[ -z "$$INFERENCE_API_KEY" ]]; then echo "拒绝执行：环境变量 INFERENCE_API_KEY 未设置"; exit 2; fi
	@if [[ -z "$(INFERENCE_ENDPOINT)" ]]; then echo "拒绝执行：INFERENCE_ENDPOINT 未设置"; exit 2; fi
	@if [[ -z "$(EVAL_LABEL)" ]]; then echo "拒绝执行：内部 EVAL_LABEL 未设置"; exit 2; fi
	@if [[ -z "$(EVAL_MODEL)" ]]; then echo "拒绝执行：评测模型名未设置"; exit 2; fi

_eval-model: guard-eval
	$(CLI) eval rollout \
		--tasks "$(DEV_TASKS)" \
		--answers "$(DEV_ANSWERS)" \
		--db-root "$(DEV_DB_ROOT)" \
		--endpoint "$(INFERENCE_ENDPOINT)" \
		--model "$(EVAL_MODEL)" \
		--model-label "$(EVAL_LABEL)" \
		--cost-limit-usd "$(INFERENCE_COST_LIMIT_USD)" \
		--concurrency "$(INFERENCE_CONCURRENCY)" \
		--local-api \
		--output "$(EVAL_ROOT)/inference.jsonl" \
		--predictions-output "$(EVAL_ROOT)/predictions.jsonl" \
		--episodes-output "$(EVAL_ROOT)/episodes.jsonl" \
		--summary-output "$(EVAL_ROOT)/inference_summary.json"
	$(CLI) eval score \
		--tasks "$(DEV_TASKS)" \
		--answers "$(DEV_ANSWERS)" \
		--predictions "$(EVAL_ROOT)/predictions.jsonl" \
		--db-root "$(DEV_DB_ROOT)" \
		--output "$(EVAL_ROOT)/records.jsonl" \
		--summary-output "$(EVAL_ROOT)/score_summary.json"
	$(CLI) eval report \
		--records "$(EVAL_ROOT)/records.jsonl" \
		--episodes "$(EVAL_ROOT)/episodes.jsonl" \
		--inference-summary "$(EVAL_ROOT)/inference_summary.json" \
		--output "$(EVAL_ROOT)/report.md"

eval-base:
	@$(MAKE) --no-print-directory _eval-model EVAL_LABEL=base EVAL_MODEL="$(BASE_MODEL_NAME)"

eval-sft:
	@$(MAKE) --no-print-directory _eval-model EVAL_LABEL=sft EVAL_MODEL="$(SFT_MODEL_NAME)"

eval-grpo:
	@$(MAKE) --no-print-directory _eval-model EVAL_LABEL=grpo EVAL_MODEL="$(GRPO_MODEL_NAME)"

eval-compare:
	$(CLI) eval compare \
		--base-records "$(EVALUATION_ROOT)/base/records.jsonl" \
		--sft-records "$(EVALUATION_ROOT)/sft/records.jsonl" \
		--grpo-records "$(EVALUATION_ROOT)/grpo/records.jsonl" \
		--base-inference "$(EVALUATION_ROOT)/base/inference_summary.json" \
		--sft-inference "$(EVALUATION_ROOT)/sft/inference_summary.json" \
		--grpo-inference "$(EVALUATION_ROOT)/grpo/inference_summary.json" \
		--output "$(EVAL_COMPARISON_OUTPUT)"
