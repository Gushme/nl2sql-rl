import json
from pathlib import Path

import pytest

from nl2sql_rl.data.dev import EXPECTED_DEV_ROWS, _download_verified_http, _load_annotation
from nl2sql_rl.io_utils import sha256_file


def _rows(count: int) -> list[dict[str, object]]:
    return [
        {
            "question_id": index,
            "db_id": f"db_{index % 11}",
            "question": f"question {index}",
            "evidence": "",
            "SQL": "SELECT 1",
            "difficulty": "simple",
        }
        for index in range(count)
    ]


def test_annotation_requires_exact_500_rows_and_11_databases(tmp_path: Path) -> None:
    valid = tmp_path / "valid.json"
    valid.write_text(json.dumps(_rows(EXPECTED_DEV_ROWS)), encoding="utf-8")
    assert len(_load_annotation(valid)) == EXPECTED_DEV_ROWS

    invalid = tmp_path / "invalid.json"
    invalid.write_text(json.dumps(_rows(EXPECTED_DEV_ROWS - 1)), encoding="utf-8")
    with pytest.raises(ValueError, match="500"):
        _load_annotation(invalid)


def test_verified_download_rejects_wrong_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"bird")
    destination = tmp_path / "destination.bin"

    class FakeResponse:
        status_code = 200

        def raise_for_status(self) -> None:
            return None

        def iter_bytes(self, chunk_size: int) -> list[bytes]:
            assert chunk_size > 0
            return [source.read_bytes()]

        def __enter__(self) -> "FakeResponse":
            return self

        def __exit__(self, *args: object) -> None:
            return None

    class FakeClient:
        def __init__(self, **kwargs: object) -> None:
            assert kwargs

        def stream(self, *args: object, **kwargs: object) -> FakeResponse:
            return FakeResponse()

        def __enter__(self) -> "FakeClient":
            return self

        def __exit__(self, *args: object) -> None:
            return None

    monkeypatch.setattr("nl2sql_rl.data.dev.httpx.Client", FakeClient)
    with pytest.raises(RuntimeError, match="SHA256"):
        _download_verified_http(
            "https://example.invalid/file",
            destination,
            expected_bytes=4,
            expected_sha256="0" * 64,
            force=False,
        )
    assert not destination.exists()
    assert sha256_file(source) != "0" * 64
