"""Typed project configuration with repository-relative path resolution."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class ProjectPaths(BaseModel):
    model_config = ConfigDict(extra="forbid")

    root: Path
    train_root: Path
    dev_root: Path
    output_root: Path

    def resolved(self, config_dir: Path) -> ProjectPaths:
        root = self.root if self.root.is_absolute() else (config_dir / self.root).resolve()

        def under_root(value: Path) -> Path:
            return value.resolve() if value.is_absolute() else (root / value).resolve()

        return ProjectPaths(
            root=root,
            train_root=under_root(self.train_root),
            dev_root=under_root(self.dev_root),
            output_root=under_root(self.output_root),
        )


class RuntimeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    gold_timeout_seconds: float = Field(default=10.0, gt=0)
    exploration_timeout_seconds: float = Field(default=10.0, gt=0)
    max_database_workers: int = Field(default=4, ge=1, le=32)
    max_episode_actions: int = Field(default=10, ge=1, le=100)
    max_action_tokens: int = Field(default=512, ge=16)
    max_context_tokens: int = Field(default=16_384, ge=1_024)
    max_observation_bytes: int = Field(default=8_192, ge=256)


class ProjectConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int = Field(default=1, ge=1)
    base_model: str = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
    seed: int = 42
    paths: ProjectPaths
    runtime: RuntimeConfig = RuntimeConfig()

    @model_validator(mode="after")
    def validate_model(self) -> ProjectConfig:
        expected = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
        if self.base_model != expected:
            raise ValueError(f"base_model is frozen to {expected}")
        return self


def load_project_config(path: str | Path = "configs/project.yaml") -> ProjectConfig:
    config_path = Path(path).resolve()
    raw: Any = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"configuration must be a mapping: {config_path}")
    config = ProjectConfig.model_validate(raw)
    return config.model_copy(update={"paths": config.paths.resolved(config_path.parent)})
