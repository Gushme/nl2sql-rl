import json

import pytest
from pydantic import ValidationError

from nl2sql_rl.models import AuditStatus, HiddenAnswer, TaskView


def test_actor_payload_cannot_contain_gold() -> None:
    task = TaskView(
        task_id="fixture_001",
        split="train",
        db_id="fixture",
        question="How many rows are there?",
        evidence="",
        db_ref="fixture.sqlite",
    )
    payload = json.dumps(task.actor_payload())
    assert "gold_sql" not in payload
    assert "reward" not in payload


def test_task_rejects_hidden_fields() -> None:
    with pytest.raises(ValidationError):
        TaskView(
            task_id="fixture_001",
            split="train",
            db_id="fixture",
            question="question",
            db_ref="fixture.sqlite",
            gold_sql="SELECT 1",  # type: ignore[call-arg]
        )


def test_hidden_answer_is_separate_record() -> None:
    answer = HiddenAnswer(
        task_id="fixture_001",
        gold_sql="SELECT COUNT(*) FROM items",
        audit_status=AuditStatus.PASSED,
    )
    assert answer.audit_status is AuditStatus.PASSED
