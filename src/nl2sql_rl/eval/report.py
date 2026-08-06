"""生成可追溯的 Markdown 评测报告。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from nl2sql_rl.eval.behavior import behavior_metrics
from nl2sql_rl.eval.metrics import (
    OFFICIAL_EVALUATOR_SHA256,
    OFFICIAL_REPOSITORY_REVISION,
)
from nl2sql_rl.eval.pipeline import summarize_records
from nl2sql_rl.models import EpisodeResult, EvaluationRecord

BIRD_MINI_DEV_ANNOTATION_REVISION = "f65faf4"


def _percentage(value: float | None) -> str:
    return "N/A" if value is None else f"{value * 100:.2f}%"


def render_report(
    records: list[EvaluationRecord],
    episodes: list[EpisodeResult] | None = None,
    *,
    official_count: int = 500,
    project_git_sha: str = "unknown",
    inference_manifest: dict[str, Any] | None = None,
) -> str:
    summary = summarize_records(records, official_count=official_count)
    metrics = summary["metrics"]
    manifest = inference_manifest or {}
    inference_config_hash = manifest.get("config_hash", "not-provided")
    inference_condition = manifest.get("comparison_condition_hash", "not-provided")
    rves_display = "N/A" if metrics["r_ves"] is None else f"{metrics['r_ves']:.2f}"
    lines = [
        "# BIRD Mini-Dev 评测报告",
        "",
        "## 数据口径",
        "",
        f"- Official-{official_count}：完整性口径。",
        f"- Final-N：{summary['final_n']} 条可验证样本。",
        f"- 不可验证：{summary['unverifiable_count']} 条，不计入模型指标。",
        "",
        "## 确定性指标",
        "",
        "| 指标 | Final-N |",
        "|---|---:|",
        f"| EX | {_percentage(metrics['ex'])} |",
        f"| Soft-F1 | {_percentage(metrics['soft_f1'])} |",
        f"| R-VES | {rves_display} |",
        "",
        "SQL 执行器是唯一正确性裁判；LLM Judge 不参与 EX、Soft-F1、R-VES 或 Reward。",
        "",
        "## 错误分布",
        "",
    ]
    errors = summary["error_counts"]
    lines.extend(
        [f"- {name}: {count}" for name, count in errors.items()]
        or ["- 无可分类错误。"]
    )
    if episodes is not None:
        behavior = behavior_metrics(episodes)
        lines.extend(
            [
                "",
                "## Agent 行为指标",
                "",
                f"- 成功提交率：{behavior['successful_submission_rate']:.4f}",
                f"- 平均步数：{behavior['average_steps']:.4f}",
                f"- 执行错误率：{behavior['execution_error_rate']:.4f}",
                f"- 循环率：{behavior['loop_rate']:.4f}",
                f"- 纠错成功率：{behavior['correction_success_rate']:.4f}",
                f"- 工具分布：{behavior['tool_call_distribution']}",
            ]
        )
    lines.extend(
        [
            "",
            "## 复现信息",
            "",
            f"- 项目 Git SHA：`{project_git_sha}`",
            f"- Mini-Dev annotation revision：`{BIRD_MINI_DEV_ANNOTATION_REVISION}`",
            f"- BIRD evaluator revision：`{OFFICIAL_REPOSITORY_REVISION}`",
            f"- evaluator SHA256：`{OFFICIAL_EVALUATOR_SHA256}`",
            f"- 推理配置哈希：`{inference_config_hash}`",
            f"- 推理条件哈希：`{inference_condition}`",
            f"- 随机种子：`{manifest.get('seed', 42)}`",
            f"- Python：`{summary['rves_environment']['python_version']}`",
            f"- SQLite：`{summary['rves_environment']['sqlite_version']}`",
            f"- 平台：`{summary['rves_environment']['platform']}`",
            "",
        ]
    )
    return "\n".join(lines)


def write_report(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def render_comparison_report(
    runs: dict[str, list[EvaluationRecord]],
    condition_hashes: dict[str, str],
    *,
    official_count: int = 500,
    project_git_sha: str = "unknown",
    inference_manifests: dict[str, dict[str, Any]] | None = None,
) -> str:
    """在完全相同的 task 清单与推理条件下对比 Base/SFT/GRPO。"""
    expected_labels = {"base", "sft", "grpo"}
    if set(runs) != expected_labels or set(condition_hashes) != expected_labels:
        raise ValueError("三模型对比必须同时提供 base、sft、grpo")
    task_sets = {
        label: {record.task_id for record in records} for label, records in runs.items()
    }
    if any(len(task_sets[label]) != len(runs[label]) for label in runs):
        raise ValueError("三模型 EvaluationRecord 包含重复 task_id")
    if not task_sets["base"] or any(
        task_sets[label] != task_sets["base"] for label in ("sft", "grpo")
    ):
        raise ValueError("三模型 EvaluationRecord 的 task_id 清单不一致")
    if any(record.ex is None for records in runs.values() for record in records):
        raise ValueError("三模型比较只能读取清洗后的共同 Final-N")
    if any(len(value) != 64 for value in condition_hashes.values()) or len(
        set(condition_hashes.values())
    ) != 1:
        raise ValueError("三模型推理条件哈希不一致，拒绝生成漂移比较")

    summaries: dict[str, dict[str, Any]] = {
        label: summarize_records(records, official_count=official_count)
        for label, records in runs.items()
    }
    base_ex = summaries["base"]["metrics"]["ex"]
    lines = [
        "# Base / SFT / GRPO 对比报告",
        "",
        f"- Official-{official_count}；三组共同 Final-N：{len(task_sets['base'])}。",
        f"- 推理条件哈希：`{condition_hashes['base']}`。",
        "- SQL 执行器是唯一正确性裁判，LLM Judge 不修改任何指标。",
        "",
        "| 模型 | Final-N | EX | Soft-F1 | R-VES | 相对 Base EX |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for label in ("base", "sft", "grpo"):
        summary = summaries[label]
        metrics = summary["metrics"]
        ex = metrics["ex"]
        delta = None if ex is None or base_ex is None else ex - base_ex
        rves = "N/A" if metrics["r_ves"] is None else f"{metrics['r_ves']:.2f}"
        delta_text = "N/A" if delta is None else f"{delta * 100:+.2f} pp"
        lines.append(
            "| "
            + label.upper()
            + f" | {summary['final_n']} | {_percentage(ex)}"
            + f" | {_percentage(metrics['soft_f1'])}"
            + f" | {rves}"
            + f" | {delta_text} |"
        )
    lines.extend(
        [
            "",
            "## 复现信息",
            "",
            f"- 项目 Git SHA：`{project_git_sha}`",
            f"- Mini-Dev annotation revision：`{BIRD_MINI_DEV_ANNOTATION_REVISION}`",
            f"- BIRD evaluator revision：`{OFFICIAL_REPOSITORY_REVISION}`",
            f"- evaluator SHA256：`{OFFICIAL_EVALUATOR_SHA256}`",
            "- 随机种子：`42`",
            f"- Python：`{summaries['base']['rves_environment']['python_version']}`",
            f"- SQLite：`{summaries['base']['rves_environment']['sqlite_version']}`",
            f"- 平台：`{summaries['base']['rves_environment']['platform']}`",
            "",
        ]
    )
    if inference_manifests is not None:
        model_hash_lines = [
            (
                f"- {label.upper()}：模型 "
                f"`{inference_manifests[label].get('model', 'unknown')}`，"
                f"配置 `{inference_manifests[label].get('config_hash', 'unknown')}`"
            )
            for label in ("base", "sft", "grpo")
        ]
        lines.extend(
            [
                "## 模型运行哈希",
                "",
                *model_hash_lines,
                "",
            ]
        )
    return "\n".join(lines)
