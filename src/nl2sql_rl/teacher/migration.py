"""在新 Harness 下重放旧 Action，并只迁移语义完全一致的轨迹。"""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from nl2sql_rl.agent.fingerprint import visible_protocol_hash
from nl2sql_rl.agent.loop import build_actor_messages
from nl2sql_rl.agent.replay import replay_episode
from nl2sql_rl.config import RuntimeConfig
from nl2sql_rl.io_utils import read_jsonl, sha256_file, stable_json
from nl2sql_rl.models import HiddenAnswer, TaskView
from nl2sql_rl.training.sft_data import TokenizerLike

if TYPE_CHECKING:
    from nl2sql_rl.teacher.sampling import SamplingPlan


def migrate_accepted_attempts(
    source_path: Path,
    output_path: Path,
    *,
    tasks: Sequence[TaskView],
    answers: Sequence[HiddenAnswer],
    db_root: Path,
    runtime: RuntimeConfig,
    tokenizer: TokenizerLike,
    plan: SamplingPlan,
    new_config_hash: str,
    new_harness_config_hash: str,
) -> dict[str, object]:
    """Prompt/工具协议不变且逐事件 replay 一致时才写入新版本。"""
    if source_path.resolve() == output_path.resolve():
        raise ValueError("迁移来源和目标不能是同一个文件")
    # 延迟导入避免 collector 与迁移模块初始化时形成循环。
    from nl2sql_rl.teacher.collector import (
        TeacherAttempt,
        _accepted,
        _append_attempt,
    )

    source = [TeacherAttempt.model_validate(row) for row in read_jsonl(source_path)]
    existing = [TeacherAttempt.model_validate(row) for row in read_jsonl(output_path)]
    mismatched = [row.task_id for row in existing if row.config_hash != new_config_hash]
    if mismatched:
        raise ValueError("迁移目标包含不同 config_hash")
    completed = {row.task_id for row in existing}
    task_by_id = {task.task_id: task for task in tasks}
    answer_by_id = {answer.task_id: answer for answer in answers}
    current_visible_hash = visible_protocol_hash()
    rejected: Counter[str] = Counter()
    migrated = 0
    database_hashes: dict[str, str] = {}

    for old in source:
        if not old.accepted or old.task_id in completed:
            continue
        task = task_by_id.get(old.task_id)
        answer = answer_by_id.get(old.task_id)
        if task is None or answer is None:
            rejected["missing_task_or_answer"] += 1
            continue
        actor_hash = hashlib.sha256(
            stable_json(build_actor_messages(task)).encode("utf-8")
        ).hexdigest()
        if old.actor_message_hash != actor_hash:
            rejected["initial_actor_messages_changed"] += 1
            continue
        if old.visible_protocol_hash != current_visible_hash:
            rejected["visible_prompt_or_tools_changed"] += 1
            continue
        if task.db_ref not in database_hashes:
            database_hashes[task.db_ref] = sha256_file(db_root / task.db_ref)
        comparison = replay_episode(
            old.episode,
            task,
            answer,
            db_root / task.db_ref,
            runtime=runtime,
            config_hash=new_config_hash,
            current_database_sha256=database_hashes[task.db_ref],
        )
        if not comparison.exact_event_replay:
            rejected["event_replay_mismatch"] += 1
            continue
        accepted, reason, token_count = _accepted(
            task,
            comparison.reproduced,
            tokenizer=tokenizer,
            runtime=runtime,
        )
        if not accepted:
            rejected[f"new_acceptance:{reason or 'unknown'}"] += 1
            continue
        complexity = plan.task_complexity[task.task_id]
        migrated_attempt = TeacherAttempt(
            task_id=task.task_id,
            split=task.split,
            db_id=task.db_id,
            complexity=complexity.bucket,
            complexity_score=complexity.score,
            config_hash=new_config_hash,
            harness_config_hash=new_harness_config_hash,
            visible_protocol_hash=current_visible_hash,
            actor_message_hash=actor_hash,
            sampling_manifest_hash=plan.manifest_hash,
            accepted=True,
            episode=comparison.reproduced,
            sft_tokens=token_count,
            database_size_bytes=(db_root / task.db_ref).stat().st_size,
            migrated_from=f"{old.config_hash}:{old.episode.episode_id}",
        )
        _append_attempt(output_path, migrated_attempt)
        completed.add(task.task_id)
        migrated += 1
    return {
        "schema_version": 1,
        "source": str(source_path),
        "target": str(output_path),
        "eligible_accepted": sum(row.accepted for row in source),
        "migrated": migrated,
        "rejected": dict(sorted(rejected.items())),
        "new_config_hash": new_config_hash,
        "new_harness_config_hash": new_harness_config_hash,
    }
