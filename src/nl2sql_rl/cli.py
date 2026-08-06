"""Command-line entrypoint. Subsystems register commands as they are implemented."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from nl2sql_rl import __version__
from nl2sql_rl.config import load_project_config
from nl2sql_rl.data.audit import run_train_audit
from nl2sql_rl.data.bird import build_train_inventory
from nl2sql_rl.data.dev import download_dev500, run_dev_audit
from nl2sql_rl.data.split import build_splits_and_leakage_report
from nl2sql_rl.io_utils import write_json

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
