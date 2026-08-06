"""项目统一命令行入口。"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
from pathlib import Path
from typing import Annotated

import typer

from nl2sql_rl import __version__
from nl2sql_rl.agent.loop import ScriptedPolicy, run_episode
from nl2sql_rl.agent.replay import replay_episode
from nl2sql_rl.config import load_project_config
from nl2sql_rl.data.audit import run_train_audit
from nl2sql_rl.data.bird import build_train_inventory
from nl2sql_rl.data.dev import download_dev500, run_dev_audit
from nl2sql_rl.data.split import build_splits_and_leakage_report
from nl2sql_rl.eval.judge import blind_prompt, build_blind_payload
from nl2sql_rl.eval.pipeline import PredictionRecord, score_dataset
from nl2sql_rl.eval.report import render_report, write_report
from nl2sql_rl.io_utils import read_jsonl, sha256_file, write_json, write_jsonl
from nl2sql_rl.models import (
    AgentAction,
    EpisodeResult,
    EvaluationRecord,
    HiddenAnswer,
    TaskView,
)
from nl2sql_rl.teacher.client import LLMClient, LLMClientConfig
from nl2sql_rl.teacher.collector import CollectorConfig, TeacherAttempt, collect_trajectories
from nl2sql_rl.training.model_assets import download_model_metadata
from nl2sql_rl.training.sft import (
    export_merged_checkpoint,
    load_sft_config,
    preflight_sft,
    run_sft_training,
)
from nl2sql_rl.training.sft_data import build_sft_conversations

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


@app.command()
def version() -> None:
    """Print the package version."""
    typer.echo(__version__)


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
        str, typer.Option(help="数据库传输来源：mirror 或 drive")
    ] = "mirror",
) -> None:
    """下载并校验固定版本的 BIRD Mini-Dev SQLite 500。"""
    loaded = load_project_config(config)
    report = download_dev500(
        loaded.paths.dev_root,
        force=force,
        database_source=database_source,
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
        "exact_terminal_outcome": comparison.exact_terminal_outcome,
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
    )
    write_report(output, content)
    typer.echo(json.dumps({"records": len(record_rows), "output": str(output)}, indent=2))


@teacher_app.command("collect")
def teacher_collect(
    tasks: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    answers: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    db_root: Annotated[Path, typer.Option(exists=True, file_okay=False)],
    endpoint: Annotated[str, typer.Option()],
    model: Annotated[str, typer.Option()],
    cost_limit_usd: Annotated[float, typer.Option(min=0.000001)],
    output: Annotated[Path, typer.Option(dir_okay=False)] = Path(
        "outputs/teacher/attempts.jsonl"
    ),
    summary_output: Annotated[Path, typer.Option(dir_okay=False)] = Path(
        "outputs/teacher/summary.json"
    ),
    api_key_env: Annotated[str, typer.Option()] = "TEACHER_API_KEY",
    confirm_real_api: Annotated[bool, typer.Option("--confirm-real-api")] = False,
    real_api: Annotated[bool, typer.Option("--real-api/--mock-api")] = True,
    target_total: Annotated[int, typer.Option(min=1)] = 1_000,
    train_quota: Annotated[int, typer.Option(min=0)] = 900,
    validation_quota: Annotated[int, typer.Option(min=0)] = 100,
    max_attempts: Annotated[int, typer.Option(min=1)] = 1_500,
    concurrency: Annotated[int, typer.Option(min=1, max=32)] = 4,
    input_price_per_million: Annotated[float, typer.Option(min=0)] = 0.0,
    output_price_per_million: Annotated[float, typer.Option(min=0)] = 0.0,
    max_request_cost_usd: Annotated[float, typer.Option(min=0.000001)] = 1.0,
    config: Annotated[Path, typer.Option(exists=True, dir_okay=False)] = Path(
        "configs/project.yaml"
    ),
) -> None:
    """采集合格 Teacher 轨迹；真实 API 需要费用上限和显式确认。"""
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
        input_price_per_million=input_price_per_million,
        output_price_per_million=output_price_per_million,
        max_request_cost_usd=max_request_cost_usd,
        real_api=real_api,
    )
    collector_config = CollectorConfig(
        target_total=target_total,
        train_quota=train_quota,
        validation_quota=validation_quota,
        max_attempts=max_attempts,
        concurrency=concurrency,
        confirm_real_api=confirm_real_api,
    )

    async def run() -> dict[str, object]:
        async with LLMClient(
            client_config,
            api_key=api_key,
            cost_limit_usd=cost_limit_usd,
        ) as client:
            return await collect_trajectories(
                task_rows,
                answer_rows,
                db_root,
                output,
                client,
                runtime=project.runtime,
                config=collector_config,
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
) -> None:
    """只从 EX 正确且协议安全的轨迹构建无 Gold 泄漏 SFT 对话。"""
    task_rows = [TaskView.model_validate(row) for row in read_jsonl(tasks)]
    attempt_rows = [TeacherAttempt.model_validate(row) for row in read_jsonl(attempts)]
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
            {"train": len(train_rows), "validation": len(validation_rows)},
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
