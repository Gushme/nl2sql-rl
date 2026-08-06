"""Command-line entrypoint. Subsystems register commands as they are implemented."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from nl2sql_rl import __version__
from nl2sql_rl.config import load_project_config

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
