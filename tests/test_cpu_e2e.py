from __future__ import annotations

from pathlib import Path

import pytest

from nl2sql_rl.e2e import run_cpu_e2e


def test_cpu_e2e_covers_harness_teacher_sft_eval_and_grpo(tmp_path: Path) -> None:
    agent_config = tmp_path / "agent_loop.yaml"
    agent_config.write_text(
        "- name: nl2sql_agent_loop\n"
        "  _target_: nl2sql_rl.training.verl_agent_loop.NL2SQLAgentLoop\n",
        encoding="utf-8",
    )
    output = tmp_path / "e2e"
    report = run_cpu_e2e(output, agent_loop_config=agent_config)
    assert report["fixture_only"] is True
    assert report["data"]["gold_deterministic"] is True
    assert report["data"]["database_hashes_unchanged"] is True
    assert report["harness"]["reward"] == 1.0
    assert report["harness"]["replay_exact"] is True
    assert report["teacher"]["accepted_total"] == 2
    assert report["sft"]["conversations"] == 2
    assert report["evaluation"]["metrics"]["ex"] == 1.0
    assert report["grpo"]["nominal_rollouts"] == 800
    assert report["grpo"]["assistant_mask_tokens"] > 0
    assert report["grpo"]["tool_mask_tokens"] > 0
    assert report["external_teacher_api_called"] is False
    assert (output / "report.json").is_file()

    with pytest.raises(FileExistsError):
        run_cpu_e2e(output, agent_loop_config=agent_config)
