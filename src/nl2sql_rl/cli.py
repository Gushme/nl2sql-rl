"""项目统一命令行入口。"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import subprocess
from collections import Counter
from pathlib import Path
from typing import Annotated, Any, Literal, cast

import typer

from nl2sql_rl import __version__
from nl2sql_rl.agent.fingerprint import harness_config_hash
from nl2sql_rl.agent.loop import ScriptedPolicy, run_episode
from nl2sql_rl.agent.replay import replay_episode
from nl2sql_rl.config import load_project_config
from nl2sql_rl.data.audit import run_train_audit
from nl2sql_rl.data.bird import build_train_inventory
from nl2sql_rl.data.dev import download_dev500, run_dev_audit
from nl2sql_rl.data.split import build_splits_and_leakage_report
from nl2sql_rl.e2e import run_cpu_e2e
from nl2sql_rl.eval.inference import InferenceRecord, collect_inference_episodes
from nl2sql_rl.eval.judge import blind_prompt, build_blind_payload
from nl2sql_rl.eval.pipeline import PredictionRecord, score_dataset
from nl2sql_rl.eval.report import render_comparison_report, render_report, write_report
from nl2sql_rl.io_utils import read_jsonl, sha256_file, stable_json, write_json, write_jsonl
from nl2sql_rl.models import (
    AgentAction,
    EpisodeResult,
    EvaluationRecord,
    HiddenAnswer,
    TaskView,
)
from nl2sql_rl.teacher.campaign import prepare_campaign_state, register_campaign_attempt
from nl2sql_rl.teacher.client import LLMClient, LLMClientConfig
from nl2sql_rl.teacher.collector import (
    CollectorConfig,
    TeacherAttempt,
    collect_trajectories,
)
from nl2sql_rl.teacher.preflight import preflight_scheduled_gold
from nl2sql_rl.teacher.probe import run_function_call_probe
from nl2sql_rl.teacher.sampling import build_sampling_plan
from nl2sql_rl.training.grpo import (
    build_verl_command,
    load_grpo_config,
    preflight_grpo,
    prepare_grpo_datasets,
    run_grpo,
)
from nl2sql_rl.training.model_assets import download_model_metadata
from nl2sql_rl.training.sft import (
    export_merged_checkpoint,
    load_sft_config,
    preflight_sft,
    run_sft_training,
)
from nl2sql_rl.training.sft_data import JSONTokenizer, build_sft_conversations

app = typer.Typer(help="BIRD SQLite Agentic NL2SQL post-training toolkit.")
data_app = typer.Typer(help="Build and audit BIRD data.")
agent_app = typer.Typer(help="Run and replay multi-turn SQL agents.")
teacher_app = typer.Typer(help="Collect audited teacher trajectories.")
train_app = typer.Typer(help="Validate and launch SFT/GRPO jobs.")
eval_app = typer.Typer(help="Score predictions and analyze trajectories.")

app.add_typer(data_app, name="data")
app.add_typer(agent_app, name="agent")
app.add_typer(teacher_app, name="teacher")
app.add_typer(train_app, name="train")
app.add_typer(eval_app, name="eval")


def _validate_teacher_budget(
    *,
    input_price_per_million: float,
    output_price_per_million: float,
    max_context_tokens: int,
    max_completion_tokens: int,
    max_request_cost_usd: float,
    cost_limit_usd: float | None,
    token_limit: int | None,
) -> int:
    """校验真实调用预算，并返回单次请求的保守 Token 上界。"""
    has_input_price = input_price_per_million > 0
    has_output_price = output_price_per_million > 0
    if has_input_price != has_output_price:
        raise typer.BadParameter("输入与输出 Token 单价必须同时设置或同时省略")
    has_usd_pricing = has_input_price and has_output_price
    if not has_usd_pricing and token_limit is None:
        raise typer.BadParameter("真实 Teacher 调用必须设置 Token 上限或完整费用闸门")
    # 工具 JSON schema 也计入 Teacher prompt，额外预留 4K token。
    conservative_input_tokens = max_context_tokens + 4_096
    conservative_request_tokens = conservative_input_tokens + max_completion_tokens
    if token_limit is not None and token_limit < conservative_request_tokens:
        raise typer.BadParameter(
            "token_limit 小于单次请求的保守 Token 上界："
            f"{token_limit} < {conservative_request_tokens}"
        )
    if not has_usd_pricing:
        if cost_limit_usd is not None:
            raise typer.BadParameter("未设置 Token 单价时不能声明美元费用上限")
        return conservative_request_tokens
    if cost_limit_usd is None:
        raise typer.BadParameter("设置 Token 单价时必须同时设置累计费用上限")
    conservative_request_cost = (
        conservative_input_tokens * input_price_per_million
        + max_completion_tokens * output_price_per_million
    ) / 1_000_000
    if max_request_cost_usd + 1e-12 < conservative_request_cost:
        raise typer.BadParameter(
            "max_request_cost_usd 小于上下文、工具 schema 与最大输出的保守费用上界："
            f"{max_request_cost_usd:.6f} < {conservative_request_cost:.6f}"
        )
    if max_request_cost_usd > cost_limit_usd + 1e-12:
        raise typer.BadParameter("max_request_cost_usd 不能超过活动费用上限")
    return conservative_request_tokens


@app.command()
def version() -> None:
    """Print the package version."""
    typer.echo(__version__)


@app.command("cpu-e2e")
def cpu_e2e(
    output: Annotated[Path, typer.Option(file_okay=False)] = Path(
        "outputs/cpu_e2e/run"
    ),
    agent_loop_config: Annotated[
        Path, typer.Option(exists=True, dir_okay=False)
    ] = Path("configs/verl_agent_loop.yaml"),
) -> None:
    """运行不含 GPU、模型推理和外部 API 的端到端交付演练。"""
    report = run_cpu_e2e(output, agent_loop_config=agent_loop_config)
    typer.echo(json.dumps(report, indent=2, sort_keys=True))


@app.command("show-config")
def show_config(
    config: Annotated[Path, typer.Option(exists=True, dir_okay=False)] = Path(
        "configs/project.yaml"
    ),
) -> None:
    """Load, validate, resolve, and print project configuration."""
    loaded = load_project_config(config)
    typer.echo(json.dumps(loaded.model_dump(mode="json"), indent=2, sort_keys=True))


@data_app.command("inventory")
def data_inventory(
    config: Annotated[Path, typer.Option(exists=True, dir_okay=False)] = Path(
        "configs/project.yaml"
    ),
    output: Annotated[Path, typer.Option(dir_okay=False)] = Path(
        "data/manifests/train_inventory.json"
    ),
) -> None:
    """Validate and hash the immutable BIRD train source layout."""
    loaded = load_project_config(config)
    report = build_train_inventory(loaded.paths.train_root)
    write_json(output, report)
    typer.echo(json.dumps(report, indent=2, sort_keys=True))


@data_app.command("clean-train")
def data_clean_train(
    config: Annotated[Path, typer.Option(exists=True, dir_okay=False)] = Path(
        "configs/project.yaml"
    ),
    manifest_root: Annotated[Path, typer.Option(file_okay=False)] = Path("data/manifests"),
    limit: Annotated[int | None, typer.Option(min=1)] = None,
    resume: Annotated[bool, typer.Option("--resume/--no-resume")] = True,
) -> None:
    """Audit train Gold SQL twice in read-only isolated database workers."""
    loaded = load_project_config(config)
    summary = run_train_audit(
        loaded.paths.train_root,
        loaded.paths.output_root,
        manifest_root,
        timeout_seconds=loaded.runtime.gold_timeout_seconds,
        max_workers=loaded.runtime.max_database_workers,
        limit=limit,
        resume=resume,
    )
    typer.echo(json.dumps(summary, indent=2, sort_keys=True))


@data_app.command("download-dev")
def data_download_dev(
    config: Annotated[Path, typer.Option(exists=True, dir_okay=False)] = Path(
        "configs/project.yaml"
    ),
    manifest_root: Annotated[Path, typer.Option(file_okay=False)] = Path("data/manifests"),
    force: Annotated[bool, typer.Option("--force/--no-force")] = False,
    database_source: Annotated[
        str, typer.Option(help="数据库传输来源：mirror、drive 或 local")
    ] = "mirror",
    local_package_root: Annotated[
        Path,
        typer.Option(help="local 模式下已解压的 Mini-Dev 完整包目录"),
    ] = Path("dev500/raw/minidev"),
) -> None:
    """下载并校验固定版本的 BIRD Mini-Dev SQLite 500。"""
    loaded = load_project_config(config)
    report = download_dev500(
        loaded.paths.dev_root,
        force=force,
        database_source=database_source,
        local_package_root=local_package_root,
    )
    write_json(manifest_root / "dev_inventory.json", report)
    typer.echo(json.dumps(report, indent=2, sort_keys=True))


@data_app.command("clean-dev")
def data_clean_dev(
    config: Annotated[Path, typer.Option(exists=True, dir_okay=False)] = Path(
        "configs/project.yaml"
    ),
    manifest_root: Annotated[Path, typer.Option(file_okay=False)] = Path("data/manifests"),
    resume: Annotated[bool, typer.Option("--resume/--no-resume")] = True,
) -> None:
    """只读双执行审计 Mini-Dev 500，并生成 Final-N。"""
    loaded = load_project_config(config)
    summary = run_dev_audit(
        loaded.paths.dev_root,
        loaded.paths.output_root,
        manifest_root,
        timeout_seconds=loaded.runtime.gold_timeout_seconds,
        max_workers=loaded.runtime.max_database_workers,
        resume=resume,
    )
    typer.echo(json.dumps(summary, indent=2, sort_keys=True))


@data_app.command("split")
def data_split(
    config: Annotated[Path, typer.Option(exists=True, dir_okay=False)] = Path(
        "configs/project.yaml"
    ),
    manifest_root: Annotated[Path, typer.Option(file_okay=False)] = Path("data/manifests"),
) -> None:
    """按 db_id 切分 Train/Validation，并执行 Train/Dev 泄漏审计。"""
    loaded = load_project_config(config)
    manifest, leakage = build_splits_and_leakage_report(
        loaded.paths.output_root, manifest_root, seed=loaded.seed
    )
    typer.echo(json.dumps({"split": manifest, "leakage": leakage}, indent=2, sort_keys=True))


def _read_json_object(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise typer.BadParameter(f"文件必须包含 JSON 对象：{path}")
    return value


@agent_app.command("run")
def agent_run(
    task_path: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    answer_path: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    database: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    actions: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    output: Annotated[Path, typer.Option(dir_okay=False)] = Path(
        "outputs/episodes/episode.json"
    ),
    config: Annotated[Path, typer.Option(exists=True, dir_okay=False)] = Path(
        "configs/project.yaml"
    ),
) -> None:
    """使用预先生成的 Action JSONL 运行一个 CPU episode。"""
    loaded = load_project_config(config)
    task = TaskView.model_validate(_read_json_object(task_path))
    answer = HiddenAnswer.model_validate(_read_json_object(answer_path))
    action_rows = read_jsonl(actions)
    policy = ScriptedPolicy([AgentAction.model_validate(row) for row in action_rows])
    episode = run_episode(
        task,
        answer,
        database,
        policy,
        runtime=loaded.runtime,
        config_hash=sha256_file(config),
    )
    write_json(output, episode.model_dump(mode="json"))
    typer.echo(json.dumps(episode.model_dump(mode="json"), indent=2, sort_keys=True))


@agent_app.command("replay")
def agent_replay(
    episode_path: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    task_path: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    answer_path: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    database: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    output: Annotated[Path | None, typer.Option(dir_okay=False)] = None,
    config: Annotated[Path, typer.Option(exists=True, dir_okay=False)] = Path(
        "configs/project.yaml"
    ),
) -> None:
    """在同一数据库上重新执行 episode 中的规范化 Action。"""
    loaded = load_project_config(config)
    episode = EpisodeResult.model_validate(_read_json_object(episode_path))
    task = TaskView.model_validate(_read_json_object(task_path))
    answer = HiddenAnswer.model_validate(_read_json_object(answer_path))
    comparison = replay_episode(
        episode,
        task,
        answer,
        database,
        runtime=loaded.runtime,
    )
    payload = {
        "terminal_matches": comparison.terminal_matches,
        "submitted_sql_matches": comparison.submitted_sql_matches,
        "reward_matches": comparison.reward_matches,
        "events_match": comparison.events_match,
        "event_mismatches": list(comparison.event_mismatches),
        "database_sha_matches": comparison.database_sha_matches,
        "exact_terminal_outcome": comparison.exact_terminal_outcome,
        "exact_event_replay": comparison.exact_event_replay,
        "reproduced": comparison.reproduced.model_dump(mode="json"),
    }
    if output is not None:
        write_json(output, payload)
    typer.echo(json.dumps(payload, indent=2, sort_keys=True))


@eval_app.command("score")
def eval_score(
    tasks: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    answers: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    predictions: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    db_root: Annotated[Path, typer.Option(exists=True, file_okay=False)],
    output: Annotated[Path, typer.Option(dir_okay=False)] = Path(
        "outputs/evaluation/records.jsonl"
    ),
    summary_output: Annotated[Path, typer.Option(dir_okay=False)] = Path(
        "outputs/evaluation/summary.json"
    ),
    timeout_seconds: Annotated[float, typer.Option(min=0.01)] = 10.0,
    rves_iterations: Annotated[int, typer.Option(min=0)] = 0,
    official_count: Annotated[int, typer.Option(min=1)] = 500,
) -> None:
    """用只读 SQL 执行器计算 EX、Soft-F1 和可选 R-VES。"""
    task_rows = [TaskView.model_validate(row) for row in read_jsonl(tasks)]
    answer_rows = [HiddenAnswer.model_validate(row) for row in read_jsonl(answers)]
    prediction_rows = [
        PredictionRecord.model_validate(row) for row in read_jsonl(predictions)
    ]
    records, summary = score_dataset(
        task_rows,
        answer_rows,
        prediction_rows,
        db_root,
        timeout_seconds=timeout_seconds,
        rves_iterations=rves_iterations,
        official_count=official_count,
    )
    write_jsonl(output, [record.model_dump(mode="json") for record in records])
    write_json(summary_output, summary)
    typer.echo(json.dumps(summary, indent=2, sort_keys=True))


@eval_app.command("judge")
def eval_judge(
    tasks: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    episodes: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    output: Annotated[Path, typer.Option(dir_okay=False)] = Path(
        "outputs/evaluation/blind_judge_inputs.jsonl"
    ),
) -> None:
    """生成不含 Gold、Reward、EX 和模型名的盲评输入，不调用真实 API。"""
    task_by_id = {
        task.task_id: task
        for task in (TaskView.model_validate(row) for row in read_jsonl(tasks))
    }
    episode_rows = [EpisodeResult.model_validate(row) for row in read_jsonl(episodes)]
    payloads = []
    for episode in episode_rows:
        task = task_by_id.get(episode.task_id)
        if task is None:
            raise typer.BadParameter(f"缺少 TaskView：{episode.task_id}")
        payloads.append(
            {
                "task_id": episode.task_id,
                "payload": build_blind_payload(task, episode),
                "prompt": blind_prompt(task, episode),
            }
        )
    write_jsonl(output, payloads)
    typer.echo(json.dumps({"count": len(payloads), "output": str(output)}, indent=2))


@eval_app.command("report")
def eval_report(
    records: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    output: Annotated[Path, typer.Option(dir_okay=False)] = Path(
        "outputs/evaluation/report.md"
    ),
    episodes: Annotated[Path | None, typer.Option(dir_okay=False)] = None,
    inference_summary: Annotated[Path | None, typer.Option(dir_okay=False)] = None,
    official_count: Annotated[int, typer.Option(min=1)] = 500,
) -> None:
    """汇总确定性指标、错误分布、行为指标和复现信息。"""
    record_rows = [EvaluationRecord.model_validate(row) for row in read_jsonl(records)]
    episode_rows = (
        [EpisodeResult.model_validate(row) for row in read_jsonl(episodes)]
        if episodes is not None
        else None
    )
    try:
        git_sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        git_sha = "unknown"
    content = render_report(
        record_rows,
        episode_rows,
        official_count=official_count,
        project_git_sha=git_sha,
        inference_manifest=(
            _read_json_object(inference_summary) if inference_summary is not None else None
        ),
    )
    write_report(output, content)
    typer.echo(json.dumps({"records": len(record_rows), "output": str(output)}, indent=2))


@eval_app.command("rollout")
def eval_rollout(
    tasks: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    answers: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    db_root: Annotated[Path, typer.Option(exists=True, file_okay=False)],
    endpoint: Annotated[str, typer.Option()],
    model: Annotated[str, typer.Option()],
    model_label: Annotated[str, typer.Option()],
    cost_limit_usd: Annotated[float, typer.Option(min=0.000001)],
    output: Annotated[Path, typer.Option(dir_okay=False)] = Path(
        "outputs/evaluation/inference.jsonl"
    ),
    predictions_output: Annotated[Path, typer.Option(dir_okay=False)] = Path(
        "outputs/evaluation/predictions.jsonl"
    ),
    episodes_output: Annotated[Path, typer.Option(dir_okay=False)] = Path(
        "outputs/evaluation/episodes.jsonl"
    ),
    summary_output: Annotated[Path, typer.Option(dir_okay=False)] = Path(
        "outputs/evaluation/inference_summary.json"
    ),
    api_key_env: Annotated[str, typer.Option()] = "INFERENCE_API_KEY",
    confirm_real_api: Annotated[bool, typer.Option("--confirm-real-api")] = False,
    real_api: Annotated[bool, typer.Option("--real-api/--local-api")] = True,
    concurrency: Annotated[int, typer.Option(min=1, max=32)] = 4,
    temperature: Annotated[float, typer.Option(min=0)] = 0.0,
    input_price_per_million: Annotated[float, typer.Option(min=0)] = 0.0,
    output_price_per_million: Annotated[float, typer.Option(min=0)] = 0.0,
    max_request_cost_usd: Annotated[float, typer.Option(min=0.000001)] = 0.001,
    config: Annotated[Path, typer.Option(exists=True, dir_okay=False)] = Path(
        "configs/project.yaml"
    ),
) -> None:
    """用同一 harness 和 OpenAI-compatible backend 采集完整评测轨迹。"""
    project = load_project_config(config)
    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise typer.BadParameter(f"环境变量 {api_key_env} 未设置")
    task_rows = [TaskView.model_validate(row) for row in read_jsonl(tasks)]
    answer_rows = [HiddenAnswer.model_validate(row) for row in read_jsonl(answers)]
    client_config = LLMClientConfig(
        endpoint=endpoint,
        model=model,
        concurrency=concurrency,
        max_completion_tokens=project.runtime.max_action_tokens,
        temperature=temperature,
        input_price_per_million=input_price_per_million,
        output_price_per_million=output_price_per_million,
        max_request_cost_usd=max_request_cost_usd,
        real_api=real_api,
    )

    async def run() -> tuple[list[InferenceRecord], dict[str, Any]]:
        async with LLMClient(
            client_config,
            api_key=api_key,
            cost_limit_usd=cost_limit_usd,
        ) as client:
            records, summary = await collect_inference_episodes(
                task_rows,
                answer_rows,
                db_root,
                output,
                client,
                runtime=project.runtime,
                model_label=model_label,
                concurrency=concurrency,
                confirm_real_api=confirm_real_api,
            )
            return records, summary

    records, summary = asyncio.run(run())
    write_jsonl(
        predictions_output,
        [
            {
                "task_id": record.task_id,
                "prediction_sql": record.episode.submitted_sql,
                "usage": record.episode.usage,
            }
            for record in records
        ],
    )
    write_jsonl(
        episodes_output,
        [record.episode.model_dump(mode="json") for record in records],
    )
    write_json(summary_output, summary)
    typer.echo(json.dumps(summary, indent=2, sort_keys=True))


@eval_app.command("compare")
def eval_compare(
    base_records: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    sft_records: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    grpo_records: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    base_inference: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    sft_inference: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    grpo_inference: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    output: Annotated[Path, typer.Option(dir_okay=False)] = Path(
        "outputs/evaluation/comparison.md"
    ),
    official_count: Annotated[int, typer.Option(min=1)] = 500,
) -> None:
    """校验任务与推理条件一致后生成 Base/SFT/GRPO 对比报告。"""
    record_paths = {
        "base": base_records,
        "sft": sft_records,
        "grpo": grpo_records,
    }
    inference_paths = {
        "base": base_inference,
        "sft": sft_inference,
        "grpo": grpo_inference,
    }
    runs = {
        label: [EvaluationRecord.model_validate(row) for row in read_jsonl(path)]
        for label, path in record_paths.items()
    }
    inference_manifests = {
        label: _read_json_object(path) for label, path in inference_paths.items()
    }
    conditions = {
        label: str(manifest.get("comparison_condition_hash", ""))
        for label, manifest in inference_manifests.items()
    }
    try:
        git_sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        git_sha = "unknown"
    content = render_comparison_report(
        runs,
        conditions,
        official_count=official_count,
        project_git_sha=git_sha,
        inference_manifests=inference_manifests,
    )
    write_report(output, content)
    typer.echo(json.dumps({"output": str(output), "final_n": len(runs["base"])}, indent=2))


@teacher_app.command("probe")
def teacher_probe(
    endpoint: Annotated[str, typer.Option()],
    model: Annotated[str, typer.Option()],
    cost_limit_usd: Annotated[
        float | None, typer.Option(min=0.000001)
    ] = None,
    token_limit: Annotated[int | None, typer.Option(min=1)] = None,
    output: Annotated[Path, typer.Option(dir_okay=False)] = Path(
        "outputs/teacher/probe.json"
    ),
    campaign_state_output: Annotated[Path, typer.Option(dir_okay=False)] = Path(
        "outputs/teacher/campaign_state.json"
    ),
    api_key_env: str = "TEACHER_API_KEY",
    confirm_real_api: Annotated[bool, typer.Option("--confirm-real-api")] = False,
    input_price_per_million: Annotated[float, typer.Option(min=0)] = 0.0,
    output_price_per_million: Annotated[float, typer.Option(min=0)] = 0.0,
    max_request_cost_usd: Annotated[float, typer.Option(min=0.000001)] = 0.01,
    target_total: Annotated[int, typer.Option(min=1)] = 1_000,
    max_attempts: Annotated[int, typer.Option(min=1)] = 1_500,
    config: Annotated[Path, typer.Option(exists=True, dir_okay=False)] = Path(
        "configs/project.yaml"
    ),
) -> None:
    """执行一次受累计预算保护的 thinking + Action 兼容性探针。"""
    if not confirm_real_api:
        raise typer.BadParameter("真实 Teacher probe 必须显式设置 --confirm-real-api")
    if output.exists():
        raise typer.BadParameter(f"probe 输出已存在，拒绝重复请求：{output}")
    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise typer.BadParameter(f"环境变量 {api_key_env} 未设置")
    project = load_project_config(config)
    max_request_tokens = _validate_teacher_budget(
        input_price_per_million=input_price_per_million,
        output_price_per_million=output_price_per_million,
        max_context_tokens=project.runtime.max_context_tokens,
        max_completion_tokens=8_192,
        max_request_cost_usd=max_request_cost_usd,
        cost_limit_usd=cost_limit_usd,
        token_limit=token_limit,
    )
    client_config = LLMClientConfig(
        endpoint=endpoint,
        model=model,
        concurrency=1,
        max_retries=4,
        timeout_seconds=180.0,
        max_completion_tokens=8_192,
        normalized_action_token_limit=512,
        temperature=0.0,
        seed=42,
        enable_thinking=True,
        reasoning_effort="high",
        input_price_per_million=input_price_per_million,
        output_price_per_million=output_price_per_million,
        max_request_cost_usd=max_request_cost_usd,
        real_api=True,
    )
    campaign_state = prepare_campaign_state(
        campaign_state_output,
        attempt_paths=[],
        target_total=target_total,
        max_attempts=max_attempts,
        cost_limit_usd=cost_limit_usd,
        token_limit=token_limit,
        teacher_behavior_hash=client_config.behavior_fingerprint(),
        pricing_hash=client_config.pricing_fingerprint(),
    )

    async def run() -> tuple[dict[str, Any], float, int]:
        async with LLMClient(
            client_config,
            api_key=api_key,
            cost_limit_usd=cost_limit_usd,
            initial_spent_usd=campaign_state.spent_usd,
            token_limit=token_limit,
            initial_used_tokens=campaign_state.used_tokens,
            max_request_tokens=max_request_tokens,
        ) as client:
            try:
                report = await run_function_call_probe(client)
            except Exception as exc:
                error_code = str(getattr(exc, "error_code", "probe_error"))
                report = {
                    "schema_version": 1,
                    "ok": False,
                    "error_code": error_code,
                    "request_sent": getattr(exc, "request_sent", None),
                    "cost_usd": float(getattr(exc, "cost_usd", 0.0)),
                    "usage": {
                        "input_tokens": int(getattr(exc, "input_tokens", 0)),
                        "output_tokens": int(getattr(exc, "output_tokens", 0)),
                        "total_tokens": int(getattr(exc, "input_tokens", 0))
                        + int(getattr(exc, "output_tokens", 0)),
                    },
                    "reasoning_content_saved": False,
                }
            return report, client.spent_usd, client.used_tokens

    report, spent_usd, used_tokens = asyncio.run(run())
    report["campaign_usage"] = {
        "spent_usd": spent_usd,
        "cost_limit_usd": cost_limit_usd,
        "used_tokens": used_tokens,
        "token_limit": token_limit,
    }
    error_code = str(report.get("error_code", ""))
    budget_blocked_before_request = (
        error_code in {"cost_limit_exceeded", "token_limit_exceeded"}
        and report.get("request_sent") is False
    )
    if not budget_blocked_before_request:
        response_id = str(report.get("response_id") or "")
        episode_id = response_id or "probe_" + hashlib.sha256(
            stable_json(
                {
                    "client": client_config.behavior_fingerprint(),
                    "attempt": campaign_state.attempts,
                }
            ).encode("utf-8")
        ).hexdigest()[:20]
        register_campaign_attempt(
            campaign_state_output,
            campaign_state,
            episode_id=episode_id,
            spent_usd=spent_usd,
            used_tokens=used_tokens,
            harness_config_hash=harness_config_hash(project.runtime),
        )
    write_json(output, report)
    typer.echo(json.dumps(report, indent=2, sort_keys=True))
    if not report.get("ok"):
        raise typer.Exit(code=2)


@teacher_app.command("harness-preflight")
def teacher_harness_preflight(
    tasks: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    answers: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    db_root: Annotated[Path, typer.Option(exists=True, file_okay=False)],
    output: Annotated[Path, typer.Option(dir_okay=False)] = Path(
        "outputs/teacher/harness_preflight.json"
    ),
    limit: Annotated[int, typer.Option(min=1, max=1_000)] = 100,
    config: Annotated[Path, typer.Option(exists=True, dir_okay=False)] = Path(
        "configs/project.yaml"
    ),
) -> None:
    """用计划前缀的隐藏 Gold 离线验证 Harness，不保存 Gold SQL。"""
    project = load_project_config(config)
    task_rows = [TaskView.model_validate(row) for row in read_jsonl(tasks)]
    answer_rows = [HiddenAnswer.model_validate(row) for row in read_jsonl(answers)]
    report = preflight_scheduled_gold(
        task_rows,
        answer_rows,
        db_root,
        runtime=project.runtime,
        limit=limit,
        seed=project.seed,
    )
    write_json(output, report)
    # 完整逐任务记录保存在报告文件中，终端仅打印摘要，避免输出过长。
    summary = {key: value for key, value in report.items() if key != "records"}
    typer.echo(json.dumps(summary, indent=2, sort_keys=True))
    if report["failed"] or not report["database_hashes_unchanged"]:
        raise typer.Exit(code=2)


@teacher_app.command("plan")
def teacher_plan(
    tasks: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    answers: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    output: Annotated[Path, typer.Option(dir_okay=False)] = Path(
        "outputs/teacher/sampling_plan.json"
    ),
    target_total: Annotated[int, typer.Option(min=1)] = 1_000,
    train_quota: Annotated[int, typer.Option(min=0)] = 900,
    validation_quota: Annotated[int, typer.Option(min=0)] = 100,
    seed: int = 42,
    simple_ratio: Annotated[float, typer.Option(min=0)] = 0.30,
    moderate_ratio: Annotated[float, typer.Option(min=0)] = 0.50,
    challenging_ratio: Annotated[float, typer.Option(min=0)] = 0.20,
) -> None:
    """不调用 API，生成数据库与 SQL 复杂度交叉配额。"""
    collector = CollectorConfig(
        target_total=target_total,
        train_quota=train_quota,
        validation_quota=validation_quota,
        max_attempts=max(target_total, 1_500),
        seed=seed,
        simple_ratio=simple_ratio,
        moderate_ratio=moderate_ratio,
        challenging_ratio=challenging_ratio,
    )
    collector.validate_quotas()
    task_rows = [TaskView.model_validate(row) for row in read_jsonl(tasks)]
    answer_rows = [HiddenAnswer.model_validate(row) for row in read_jsonl(answers)]
    plan = build_sampling_plan(
        task_rows,
        answer_rows,
        split_targets={"train": train_quota, "validation": validation_quota},
        complexity_weights=collector.complexity_weights(),
        seed=seed,
    )
    manifest = plan.manifest()
    write_json(output, manifest)
    typer.echo(json.dumps(manifest, indent=2, sort_keys=True))


@teacher_app.command("collect")
def teacher_collect(
    tasks: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    answers: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    db_root: Annotated[Path, typer.Option(exists=True, file_okay=False)],
    endpoint: Annotated[str, typer.Option()],
    model: Annotated[str, typer.Option()],
    cost_limit_usd: Annotated[
        float | None, typer.Option(min=0.000001)
    ] = None,
    token_limit: Annotated[int | None, typer.Option(min=1)] = None,
    output: Annotated[Path, typer.Option(dir_okay=False)] = Path(
        "outputs/teacher/attempts.jsonl"
    ),
    summary_output: Annotated[Path, typer.Option(dir_okay=False)] = Path(
        "outputs/teacher/summary.json"
    ),
    diagnostics_dir: Annotated[Path, typer.Option(file_okay=False)] = Path(
        "outputs/teacher/diagnostics"
    ),
    campaign_state_output: Annotated[Path, typer.Option(dir_okay=False)] = Path(
        "outputs/teacher/campaign_state.json"
    ),
    migrate_from: Annotated[
        Path | None, typer.Option(exists=True, dir_okay=False)
    ] = None,
    api_key_env: Annotated[str, typer.Option()] = "TEACHER_API_KEY",
    confirm_real_api: Annotated[bool, typer.Option("--confirm-real-api")] = False,
    real_api: Annotated[bool, typer.Option("--real-api/--mock-api")] = True,
    target_total: Annotated[int, typer.Option(min=1)] = 1_000,
    train_quota: Annotated[int, typer.Option(min=0)] = 900,
    validation_quota: Annotated[int, typer.Option(min=0)] = 100,
    max_attempts: Annotated[int, typer.Option(min=1)] = 1_500,
    run_attempt_limit: Annotated[int, typer.Option(min=1)] = 100,
    concurrency: Annotated[int, typer.Option(min=1, max=32)] = 4,
    seed: int = 42,
    simple_ratio: Annotated[float, typer.Option(min=0)] = 0.30,
    moderate_ratio: Annotated[float, typer.Option(min=0)] = 0.50,
    challenging_ratio: Annotated[float, typer.Option(min=0)] = 0.20,
    enable_thinking: Annotated[
        bool, typer.Option("--enable-thinking/--disable-thinking")
    ] = True,
    reasoning_effort: str = "high",
    max_completion_tokens: Annotated[int, typer.Option(min=1)] = 8_192,
    normalized_action_token_limit: Annotated[int, typer.Option(min=16)] = 512,
    api_timeout_seconds: Annotated[float, typer.Option(min=0.001)] = 180.0,
    max_retries: Annotated[int, typer.Option(min=0, max=10)] = 4,
    tokenizer_json: Annotated[Path, typer.Option(exists=True, dir_okay=False)] = Path(
        "models/Qwen2.5-Coder-1.5B-Instruct-metadata/tokenizer.json"
    ),
    input_price_per_million: Annotated[float, typer.Option(min=0)] = 0.0,
    output_price_per_million: Annotated[float, typer.Option(min=0)] = 0.0,
    max_request_cost_usd: Annotated[float, typer.Option(min=0.000001)] = 1.0,
    config: Annotated[Path, typer.Option(exists=True, dir_okay=False)] = Path(
        "configs/project.yaml"
    ),
) -> None:
    """采集合格 Teacher 轨迹；真实 API 需要预算上限和显式确认。"""
    if real_api and not confirm_real_api:
        raise typer.BadParameter("真实 Teacher 采集必须显式设置 --confirm-real-api")
    project = load_project_config(config)
    if real_api:
        max_request_tokens = _validate_teacher_budget(
            input_price_per_million=input_price_per_million,
            output_price_per_million=output_price_per_million,
            max_context_tokens=project.runtime.max_context_tokens,
            max_completion_tokens=max_completion_tokens,
            max_request_cost_usd=max_request_cost_usd,
            cost_limit_usd=cost_limit_usd,
            token_limit=token_limit,
        )
    else:
        max_request_tokens = None
    if reasoning_effort not in {"low", "medium", "high"}:
        raise typer.BadParameter("reasoning_effort 必须是 low、medium 或 high")
    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise typer.BadParameter(f"环境变量 {api_key_env} 未设置")
    task_rows = [TaskView.model_validate(row) for row in read_jsonl(tasks)]
    answer_rows = [HiddenAnswer.model_validate(row) for row in read_jsonl(answers)]
    collector_config = CollectorConfig(
        target_total=target_total,
        train_quota=train_quota,
        validation_quota=validation_quota,
        max_attempts=max_attempts,
        run_attempt_limit=run_attempt_limit,
        concurrency=concurrency,
        seed=seed,
        simple_ratio=simple_ratio,
        moderate_ratio=moderate_ratio,
        challenging_ratio=challenging_ratio,
        confirm_real_api=confirm_real_api,
    )
    collector_config.validate_quotas()
    sampling_plan = build_sampling_plan(
        task_rows,
        answer_rows,
        split_targets={"train": train_quota, "validation": validation_quota},
        complexity_weights=collector_config.complexity_weights(),
        seed=seed,
    )
    client_config = LLMClientConfig(
        endpoint=endpoint,
        model=model,
        concurrency=concurrency,
        max_retries=max_retries,
        timeout_seconds=api_timeout_seconds,
        max_completion_tokens=max_completion_tokens,
        normalized_action_token_limit=normalized_action_token_limit,
        temperature=0.0,
        seed=seed,
        enable_thinking=enable_thinking,
        reasoning_effort=cast(Literal["low", "medium", "high"], reasoning_effort),
        input_price_per_million=input_price_per_million,
        output_price_per_million=output_price_per_million,
        max_request_cost_usd=max_request_cost_usd,
        real_api=real_api,
    )
    campaign_state = prepare_campaign_state(
        campaign_state_output,
        attempt_paths=[output, *([migrate_from] if migrate_from is not None else [])],
        target_total=target_total,
        max_attempts=max_attempts,
        cost_limit_usd=cost_limit_usd,
        token_limit=token_limit,
        sampling_manifest_hash=sampling_plan.manifest_hash,
        teacher_behavior_hash=client_config.behavior_fingerprint(),
        pricing_hash=client_config.pricing_fingerprint(),
    )
    async def run() -> dict[str, object]:
        async with LLMClient(
            client_config,
            api_key=api_key,
            cost_limit_usd=cost_limit_usd,
            initial_spent_usd=campaign_state.spent_usd,
            token_limit=token_limit,
            initial_used_tokens=campaign_state.used_tokens,
            max_request_tokens=max_request_tokens,
        ) as client:
            return await collect_trajectories(
                task_rows,
                answer_rows,
                db_root,
                output,
                client,
                runtime=project.runtime,
                config=collector_config,
                tokenizer=JSONTokenizer(tokenizer_json),
                diagnostics_dir=diagnostics_dir,
                campaign_state_path=campaign_state_output,
                campaign_state=campaign_state,
                migration_source=migrate_from,
            )

    summary = asyncio.run(run())
    write_json(summary_output, summary)
    typer.echo(json.dumps(summary, indent=2, sort_keys=True))


@teacher_app.command("build-sft")
def teacher_build_sft(
    tasks: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    attempts: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    train_output: Annotated[Path, typer.Option(dir_okay=False)] = Path(
        "outputs/sft/train.jsonl"
    ),
    validation_output: Annotated[Path, typer.Option(dir_okay=False)] = Path(
        "outputs/sft/validation.jsonl"
    ),
    require_complete: Annotated[
        bool, typer.Option("--require-complete/--allow-partial")
    ] = True,
) -> None:
    """只从 EX 正确且协议安全的轨迹构建无 Gold 泄漏 SFT 对话。"""
    task_rows = [TaskView.model_validate(row) for row in read_jsonl(tasks)]
    attempt_rows = [TeacherAttempt.model_validate(row) for row in read_jsonl(attempts)]
    accepted_attempts = [row for row in attempt_rows if row.accepted]
    if require_complete:
        split_counts = Counter(row.split for row in accepted_attempts)
        complexity_counts = Counter(row.complexity.value for row in accepted_attempts)
        expected_splits = {"train": 900, "validation": 100}
        expected_complexity = {"simple": 300, "moderate": 500, "challenging": 200}
        if dict(split_counts) != expected_splits:
            raise typer.BadParameter(
                f"SFT 合格轨迹 split 配额不完整：{dict(split_counts)} != {expected_splits}"
            )
        if dict(complexity_counts) != expected_complexity:
            raise typer.BadParameter(
                "SFT 合格轨迹复杂度配额不完整："
                f"{dict(complexity_counts)} != {expected_complexity}"
            )
        database_keys = {(row.split, row.db_id) for row in accepted_attempts}
        if len(database_keys) != 69:
            raise typer.BadParameter(f"SFT 数据库覆盖应为 69，实际为 {len(database_keys)}")
        if len({row.task_id for row in accepted_attempts}) != 1_000:
            raise typer.BadParameter("SFT 合格轨迹 task_id 必须恰好为 1,000 个")
    conversations = build_sft_conversations(task_rows, attempt_rows)
    train_rows = [
        row.model_dump(mode="json") for row in conversations if row.split == "train"
    ]
    validation_rows = [
        row.model_dump(mode="json")
        for row in conversations
        if row.split == "validation"
    ]
    write_jsonl(train_output, train_rows)
    write_jsonl(validation_output, validation_rows)
    typer.echo(
        json.dumps(
            {
                "train": len(train_rows),
                "validation": len(validation_rows),
                "source_config_hashes": sorted(
                    {row.source_config_hash for row in conversations}
                ),
                "source_harness_config_hashes": sorted(
                    {row.source_harness_config_hash for row in conversations}
                ),
            },
            indent=2,
        )
    )


@train_app.command("fetch-model-metadata")
def train_fetch_model_metadata(
    destination: Annotated[Path, typer.Option(file_okay=False)] = Path(
        "models/Qwen2.5-Coder-1.5B-Instruct-metadata"
    ),
    manifest: Annotated[Path, typer.Option(dir_okay=False)] = Path(
        "data/manifests/qwen_tokenizer_config.json"
    ),
    force: Annotated[bool, typer.Option("--force/--no-force")] = False,
) -> None:
    """下载固定 revision 的 tokenizer/config，不下载 1.5B 权重。"""
    report = download_model_metadata(destination, force=force)
    write_json(manifest, report)
    typer.echo(json.dumps(report, indent=2, sort_keys=True))


@train_app.command("sft")
def train_sft(
    sft_config: Annotated[Path, typer.Option(exists=True, dir_okay=False)] = Path(
        "configs/sft.yaml"
    ),
    dry_run: Annotated[bool, typer.Option("--dry-run/--run")] = True,
    resume_from_checkpoint: Annotated[Path | None, typer.Option(file_okay=False)] = None,
    preflight_output: Annotated[Path, typer.Option(dir_okay=False)] = Path(
        "outputs/sft/preflight.json"
    ),
) -> None:
    """检查或启动 Qwen2.5-Coder-1.5B Action-only LoRA SFT。"""
    config = load_sft_config(sft_config)
    report = preflight_sft(
        config,
        dry_run=dry_run,
        resume_from_checkpoint=resume_from_checkpoint,
    )
    if dry_run:
        result: dict[str, object] = report
    else:
        result = run_sft_training(config, resume_from_checkpoint=resume_from_checkpoint)
    write_json(preflight_output, result)
    typer.echo(json.dumps(result, indent=2, sort_keys=True))


@train_app.command("export")
def train_export(
    adapter: Annotated[Path, typer.Option(exists=True, file_okay=False)],
    output: Annotated[Path, typer.Option(file_okay=False)] = Path(
        "checkpoints/sft-merged"
    ),
    sft_config: Annotated[Path, typer.Option(exists=True, dir_okay=False)] = Path(
        "configs/sft.yaml"
    ),
    dry_run: Annotated[bool, typer.Option("--dry-run/--run")] = True,
    report_output: Annotated[Path, typer.Option(dir_okay=False)] = Path(
        "outputs/sft/export.json"
    ),
) -> None:
    """检查或执行 LoRA 合并，并生成 GRPO checkpoint handoff manifest。"""
    config = load_sft_config(sft_config)
    result = export_merged_checkpoint(
        config,
        adapter,
        output,
        dry_run=dry_run,
    )
    write_json(report_output, result)
    typer.echo(json.dumps(result, indent=2, sort_keys=True))


@train_app.command("grpo")
def train_grpo(
    grpo_config: Annotated[Path, typer.Option(exists=True, dir_okay=False)] = Path(
        "configs/grpo.yaml"
    ),
    dry_run: Annotated[bool, typer.Option("--dry-run/--run")] = True,
    prepare: Annotated[bool, typer.Option("--prepare/--no-prepare")] = False,
    preflight_output: Annotated[Path, typer.Option(dir_okay=False)] = Path(
        "outputs/grpo/preflight.json"
    ),
) -> None:
    """准备数据、校验 SFT handoff，或启动固定的 100-step Agentic GRPO。"""
    config = load_grpo_config(grpo_config)
    dataset_report = prepare_grpo_datasets(config) if prepare else None
    report = preflight_grpo(config, dry_run=dry_run)
    report["prepared_data"] = dataset_report
    report["command"] = build_verl_command(config)
    if not dry_run:
        report["return_code"] = run_grpo(config)
    write_json(preflight_output, report)
    typer.echo(json.dumps(report, indent=2, sort_keys=True))
