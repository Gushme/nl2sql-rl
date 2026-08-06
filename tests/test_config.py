from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from nl2sql_rl.config import load_project_config


def test_project_config_resolves_paths() -> None:
    config = load_project_config("configs/project.yaml")
    assert config.base_model == "Qwen/Qwen2.5-Coder-1.5B-Instruct"
    assert config.paths.root == Path.cwd().resolve()
    assert config.paths.train_root == (Path.cwd() / "train").resolve()
    assert config.runtime.max_episode_actions == 10


def test_project_config_rejects_model_drift(tmp_path: Path) -> None:
    raw = yaml.safe_load(Path("configs/project.yaml").read_text(encoding="utf-8"))
    raw["paths"]["root"] = "."
    raw["base_model"] = "some/other-model"
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValidationError, match="base_model is frozen"):
        load_project_config(path)
