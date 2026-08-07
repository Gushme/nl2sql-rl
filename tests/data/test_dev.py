import json
from pathlib import Path
from typing import Any

import pytest

from nl2sql_rl.data import dev as dev_module
from nl2sql_rl.data.dev import (
    EXPECTED_DEV_ROWS,
    _cross_check_package_annotation,
    _download_verified_http,
    _load_annotation,
    _select_database_sources,
    _validate_canonical_annotation,
    download_dev500,
    load_dev_examples,
)
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


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _build_local_package(
    root: Path,
) -> tuple[Path, Path, list[dict[str, object]], list[dict[str, object]]]:
    dev_root = root / "dev500"
    package_root = dev_root / "raw/minidev"
    canonical_rows = _rows(EXPECTED_DEV_ROWS)
    package_rows = [dict(row) for row in canonical_rows]
    package_rows[194]["SQL"] = "SELECT 2"
    _write_json(dev_root / "mini_dev_sqlite.json", canonical_rows)
    _write_json(package_root / "MINIDEV/mini_dev_sqlite.json", package_rows)
    gold_lines = [f"{row['SQL']}\t{row['db_id']}" for row in package_rows]
    gold_path = package_root / "MINIDEV/mini_dev_sqlite_gold.sql"
    gold_path.write_text("\n".join(gold_lines) + "\n", encoding="utf-8")
    for db_id in sorted({str(row["db_id"]) for row in canonical_rows}):
        database = package_root / f"MINIDEV/dev_databases/{db_id}/{db_id}.sqlite"
        database.parent.mkdir(parents=True, exist_ok=True)
        database.write_bytes(f"fixture:{db_id}".encode())
    return dev_root, package_root, canonical_rows, package_rows


def test_local_import_is_offline_and_keeps_fixed_hf_gold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dev_root, package_root, canonical_rows, _ = _build_local_package(tmp_path)
    canonical_path = dev_root / "mini_dev_sqlite.json"
    monkeypatch.setattr(dev_module, "HF_ANNOTATION_SHA256", sha256_file(canonical_path))

    def reject_network(*args: object, **kwargs: object) -> None:
        raise AssertionError(f"本地导入不应联网：{args!r} {kwargs!r}")

    monkeypatch.setattr(dev_module, "_download_http", reject_network)
    source_hashes = {
        path.stem: sha256_file(path) for path in package_root.rglob("*.sqlite")
    }
    report = download_dev500(
        dev_root,
        database_source="local",
        local_package_root=package_root,
    )

    assert report["row_count"] == 500
    assert report["database_count"] == 11
    assert report["database_transport"]["source_hashes_unchanged"] is True
    assert report["package_annotation_cross_check"]["sql_mismatch_count"] == 1
    assert report["gold_cross_check"]["mismatch_count"] == 1
    assert load_dev_examples(dev_root)[194].gold_sql == canonical_rows[194]["SQL"]
    assert source_hashes == {
        path.stem: sha256_file(path) for path in package_root.rglob("*.sqlite")
    }
    for db_id, source_sha256 in source_hashes.items():
        destination = dev_root / f"dev_databases/{db_id}/{db_id}.sqlite"
        assert sha256_file(destination) == source_sha256


def test_fixed_annotation_rejects_unexpected_sha(tmp_path: Path) -> None:
    annotation = tmp_path / "mini_dev_sqlite.json"
    _write_json(annotation, _rows(EXPECTED_DEV_ROWS))
    with pytest.raises(ValueError, match="SHA256"):
        _validate_canonical_annotation(annotation)


def test_local_database_sources_must_exist_and_be_unique(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="缺少数据库"):
        _select_database_sources(tmp_path, {"db_0"}, require_unique=True)

    for parent in ("first", "second"):
        database = tmp_path / parent / "db_0.sqlite"
        database.parent.mkdir(parents=True, exist_ok=True)
        database.write_bytes(b"same")
    with pytest.raises(ValueError, match="必须唯一"):
        _select_database_sources(tmp_path, {"db_0"}, require_unique=True)


def test_package_annotation_rejects_non_sql_metadata_change(tmp_path: Path) -> None:
    canonical_rows = _rows(EXPECTED_DEV_ROWS)
    package_rows = [dict(row) for row in canonical_rows]
    package_rows[0]["question"] = "changed"
    annotation = tmp_path / "mini_dev_sqlite.json"
    _write_json(annotation, package_rows)
    with pytest.raises(ValueError, match="非 SQL 元数据"):
        _cross_check_package_annotation(canonical_rows, annotation)
