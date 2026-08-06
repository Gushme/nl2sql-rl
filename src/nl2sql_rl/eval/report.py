"""生成可追溯的 Markdown 评测报告。"""

from __future__ import annotations

from pathlib import Path

from nl2sql_rl.eval.behavior import behavior_metrics
from nl2sql_rl.eval.metrics import (
    OFFICIAL_EVALUATOR_SHA256,
    OFFICIAL_REPOSITORY_REVISION,
)
from nl2sql_rl.eval.pipeline import summarize_records
from nl2sql_rl.models import EpisodeResult, EvaluationRecord


def _percentage(value: float | None) -> str:
    return "N/A" if value is None else f"{value * 100:.2f}%"


def render_report(
    records: list[EvaluationRecord],
    episodes: list[EpisodeResult] | None = None,
    *,
    official_count: int = 500,
    project_git_sha: str = "unknown",
) -> str:
    summary = summarize_records(records, official_count=official_count)
    metrics = summary["metrics"]
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
            f"- BIRD evaluator revision：`{OFFICIAL_REPOSITORY_REVISION}`",
            f"- evaluator SHA256：`{OFFICIAL_EVALUATOR_SHA256}`",
            "",
        ]
    )
    return "\n".join(lines)


def write_report(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
