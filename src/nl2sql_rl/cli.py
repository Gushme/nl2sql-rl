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
    config: Annotated[
        Path, typer.Option(exists=True, dir_okay=False)
    ] = Path("configs/project.yaml"),
) -> None:
    """Load, validate, resolve, and print project configuration."""
    loaded = load_project_config(config)
    typer.echo(json.dumps(loaded.model_dump(mode="json"), indent=2, sort_keys=True))


@data_app.command("inventory")
def data_inventory(
    config: Annotated[
        Path, typer.Option(exists=True, dir_okay=False)
    ] = Path("configs/project.yaml"),
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
    config: Annotated[
        Path, typer.Option(exists=True, dir_okay=False)
    ] = Path("configs/project.yaml"),
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
