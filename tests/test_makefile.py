from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MAKEFILE = ROOT / "Makefile"
PUBLIC_TARGETS = {
    "help",
    "setup",
    "lint",
    "typecheck",
    "test",
    "test-full",
    "build",
    "check",
    "data-train",
    "data-dev-local",
    "data-split",
    "data-all",
    "cpu-e2e",
    "pipeline-cpu",
    "teacher-plan",
    "teacher-collect",
    "teacher-build-sft",
    "sft-preflight",
    "sft-run",
    "sft-export",
    "grpo-preflight",
    "grpo-run",
    "eval-base",
    "eval-sft",
    "eval-grpo",
    "eval-compare",
}


def _make(*arguments: str) -> subprocess.CompletedProcess[str]:
    if shutil.which("make") is None:
        pytest.skip("当前环境没有 make")
    return subprocess.run(
        ["make", *arguments],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def test_makefile_has_safe_default_and_all_public_targets() -> None:
    content = MAKEFILE.read_text(encoding="utf-8")
    assert ".DEFAULT_GOAL := help" in content
    assert ".NOTPARALLEL:" in content
    assert re.search(r"(?m)^all\s*:", content) is None
    for target in PUBLIC_TARGETS:
        assert re.search(rf"(?m)^{re.escape(target)}(?:\s|:)", content), target

    result = _make()
    assert result.returncode == 0
    assert "pipeline-cpu" in result.stdout
    assert "teacher-collect" in result.stdout


def test_cpu_pipeline_dry_run_contains_no_external_or_gpu_action() -> None:
    result = _make(
        "-n",
        "pipeline-cpu",
        "E2E_OUTPUT=/private/tmp/nl2sql-makefile-dry-run",
    )
    output = result.stdout + result.stderr
    assert result.returncode == 0
    assert "data clean-train" in output
    assert "data clean-dev" in output
    assert "cpu-e2e" in output
    assert "teacher collect" not in output
    assert "train sft --run" not in output
    assert "run_grpo_container.sh" not in output


@pytest.mark.parametrize(
    "target",
    ["teacher-collect", "sft-run", "sft-export", "grpo-run"],
)
def test_external_and_gpu_targets_require_explicit_confirmation(target: str) -> None:
    result = _make(target, "CONFIRM=0")
    output = result.stdout + result.stderr
    assert result.returncode != 0
    assert "CONFIRM=1" in output
    assert "teacher collect" not in output
    assert "train sft --run" not in output
    assert "train export" not in output
    assert "run_grpo_container.sh" not in output


def test_teacher_requires_api_key_and_explicit_cost_limit_after_confirmation() -> None:
    environment = dict(os.environ)
    environment.pop("TEACHER_API_KEY", None)
    missing_key = subprocess.run(
        [
            "make",
            "teacher-collect",
            "CONFIRM=1",
            "TEACHER_ENDPOINT=https://teacher.invalid/v1",
            "TEACHER_MODEL=fixture",
            "TEACHER_COST_LIMIT_USD=1",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
        env=environment,
    )
    assert missing_key.returncode != 0
    assert "TEACHER_API_KEY 未设置" in missing_key.stdout + missing_key.stderr
    assert "teacher collect" not in missing_key.stdout + missing_key.stderr

    environment["TEACHER_API_KEY"] = "fixture-secret"
    missing_cost = subprocess.run(
        [
            "make",
            "teacher-collect",
            "CONFIRM=1",
            "TEACHER_ENDPOINT=https://teacher.invalid/v1",
            "TEACHER_MODEL=fixture",
            "TEACHER_COST_LIMIT_USD=",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
        env=environment,
    )
    assert missing_cost.returncode != 0
    assert "TEACHER_COST_LIMIT_USD 必须显式设置" in (
        missing_cost.stdout + missing_cost.stderr
    )
    assert "fixture-secret" not in missing_cost.stdout + missing_cost.stderr
    assert "teacher collect" not in missing_cost.stdout + missing_cost.stderr
