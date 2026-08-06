"""BIRD Mini-Dev 500 的下载、校验与 Gold SQL 审计。"""

from __future__ import annotations

import json
import os
import shutil
import tarfile
import zipfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import gdown  # type: ignore[import-untyped]
import httpx

from nl2sql_rl.data.audit import (
    DEFAULT_MAX_RESULT_BYTES,
    DEFAULT_MAX_ROWS,
    GoldAuditRecord,
    _base_record,
    _run_workers,
    _WorkerState,
    audit_fingerprint,
)
from nl2sql_rl.data.bird import BirdSourceExample
from nl2sql_rl.io_utils import read_jsonl, sha256_file, stable_json, write_json, write_jsonl
from nl2sql_rl.models import AuditStatus, HiddenAnswer, TaskView

EXPECTED_DEV_ROWS = 500
EXPECTED_DEV_DATABASES = 11
HF_REVISION = "f65faf4"
HF_ANNOTATION_URL = (
    "https://huggingface.co/datasets/birdsql/bird_mini_dev/resolve/"
    f"{HF_REVISION}/data/mini_dev_sqlite-00000-of-00001.json"
)
GDRIVE_FILE_ID = "13VLWIwpw5E3d5DUkMvzw7hvHE67a4XkG"
GDRIVE_URL = f"https://drive.google.com/file/d/{GDRIVE_FILE_ID}/view?usp=sharing"
OFFICIAL_REPOSITORY = "https://github.com/bird-bench/mini_dev"
OFFICIAL_REPOSITORY_REVISION = "b3d4bcbbae9a96934ad812551eb400c7a3b23c12"
MIRROR_REPOSITORY = "prem-research/birdbench"
MIRROR_REVISION = "03abfc646adfd2ff0ab33ef69df16579446d6572"
MIRROR_TREE_URL = (
    f"https://huggingface.co/api/datasets/{MIRROR_REPOSITORY}/tree/"
    f"{MIRROR_REVISION}/validation/dev_databases?recursive=true&expand=false"
)
MIRROR_RESOLVE_ROOT = (
    f"https://huggingface.co/datasets/{MIRROR_REPOSITORY}/resolve/{MIRROR_REVISION}"
)


def _load_annotation(path: Path) -> list[dict[str, Any]]:
    raw: Any = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"Mini-Dev annotation 必须是 JSON 数组：{path}")
    rows: list[dict[str, Any]] = []
    required = {"question_id", "db_id", "question", "evidence", "SQL", "difficulty"}
    for index, value in enumerate(raw):
        if not isinstance(value, dict):
            raise ValueError(f"Mini-Dev 第 {index} 项不是对象")
        missing = required.difference(value)
        if missing:
            raise ValueError(f"Mini-Dev 第 {index} 项缺少字段：{sorted(missing)}")
        rows.append(value)
    if len(rows) != EXPECTED_DEV_ROWS:
        raise ValueError(f"Mini-Dev 必须是 500 条，实际为 {len(rows)} 条")
    db_ids = {str(row["db_id"]) for row in rows}
    if len(db_ids) != EXPECTED_DEV_DATABASES:
        raise ValueError(f"Mini-Dev 必须包含 11 个数据库，实际为 {len(db_ids)} 个")
    question_ids = [str(row["question_id"]) for row in rows]
    if len(question_ids) != len(set(question_ids)):
        raise ValueError("Mini-Dev question_id 不唯一")
    return rows


def _download_http(url: str, destination: Path, *, force: bool) -> None:
    if destination.is_file() and not force:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(f"{destination.suffix}.part")
    with (
        httpx.Client(follow_redirects=True, timeout=120.0) as client,
        client.stream("GET", url) as response,
    ):
        response.raise_for_status()
        with partial.open("wb") as handle:
            for chunk in response.iter_bytes(chunk_size=1024 * 1024):
                handle.write(chunk)
    os.replace(partial, destination)


def _download_verified_http(
    url: str,
    destination: Path,
    *,
    expected_bytes: int,
    expected_sha256: str,
    force: bool,
) -> None:
    """支持断点续传，并在落盘前校验镜像声明的 LFS SHA256。"""
    if (
        destination.is_file()
        and not force
        and destination.stat().st_size == expected_bytes
        and sha256_file(destination) == expected_sha256
    ):
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(f"{destination.suffix}.part")
    if force:
        partial.unlink(missing_ok=True)
    if partial.is_file() and partial.stat().st_size > expected_bytes:
        partial.unlink()
    offset = partial.stat().st_size if partial.is_file() else 0
    headers = {"Range": f"bytes={offset}-"} if offset else {}
    timeout = httpx.Timeout(connect=60.0, read=300.0, write=60.0, pool=60.0)
    with (
        httpx.Client(follow_redirects=True, timeout=timeout) as client,
        client.stream("GET", url, headers=headers) as response,
    ):
        response.raise_for_status()
        append = offset > 0 and response.status_code == 206
        mode = "ab" if append else "wb"
        with partial.open(mode) as handle:
            for chunk in response.iter_bytes(chunk_size=1024 * 1024):
                handle.write(chunk)
    if partial.stat().st_size != expected_bytes:
        raise RuntimeError(
            f"镜像文件大小不符：{destination.name}，{partial.stat().st_size} != {expected_bytes}"
        )
    actual_sha256 = sha256_file(partial)
    if actual_sha256 != expected_sha256:
        raise RuntimeError(
            f"镜像文件 SHA256 不符：{destination.name}，{actual_sha256} != {expected_sha256}"
        )
    os.replace(partial, destination)


def _download_mirror_databases(
    rows: list[dict[str, Any]],
    dev_root: Path,
    *,
    force: bool,
    max_workers: int = 4,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """从固定 commit 的逐文件镜像下载与官方标注对应的 11 个数据库。"""
    metadata_path = dev_root / "raw" / "mirror_tree.json"
    _download_http(MIRROR_TREE_URL, metadata_path, force=force)
    raw_metadata: Any = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(raw_metadata, list):
        raise ValueError("数据库镜像文件清单不是 JSON 数组")
    expected_db_ids = {str(row["db_id"]) for row in rows}
    entries: dict[str, dict[str, Any]] = {}
    for value in raw_metadata:
        if not isinstance(value, dict) or value.get("type") != "file":
            continue
        path = str(value.get("path", ""))
        path_obj = Path(path)
        if path_obj.suffix != ".sqlite" or path_obj.stem not in expected_db_ids:
            continue
        lfs = value.get("lfs")
        if not isinstance(lfs, dict) or not isinstance(lfs.get("oid"), str):
            raise ValueError(f"镜像文件缺少 LFS SHA256：{path}")
        entries[path_obj.stem] = {
            "path": path,
            "bytes": int(value["size"]),
            "sha256": str(lfs["oid"]),
        }
    missing = expected_db_ids.difference(entries)
    if missing:
        raise FileNotFoundError(f"数据库镜像缺少：{sorted(missing)}")

    def download(db_id: str) -> tuple[str, Path]:
        entry = entries[db_id]
        destination = dev_root / "dev_databases" / db_id / f"{db_id}.sqlite"
        _download_verified_http(
            f"{MIRROR_RESOLVE_ROOT}/{entry['path']}",
            destination,
            expected_bytes=int(entry["bytes"]),
            expected_sha256=str(entry["sha256"]),
            force=force,
        )
        return db_id, destination

    downloaded: dict[str, Path] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(download, db_id): db_id for db_id in sorted(entries)}
        for future in as_completed(futures):
            db_id, destination = future.result()
            downloaded[db_id] = destination
    databases = {
        db_id: {
            "path": str(path.relative_to(dev_root)),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "task_count": sum(str(row["db_id"]) == db_id for row in rows),
        }
        for db_id, path in sorted(downloaded.items())
    }
    provenance = {
        "repository": f"https://huggingface.co/datasets/{MIRROR_REPOSITORY}",
        "revision": MIRROR_REVISION,
        "tree_url": MIRROR_TREE_URL,
        "tree_sha256": sha256_file(metadata_path),
        "max_workers": max_workers,
    }
    return databases, provenance


def _safe_destination(root: Path, member_name: str) -> Path:
    destination = (root / member_name).resolve()
    if destination != root.resolve() and root.resolve() not in destination.parents:
        raise ValueError(f"压缩包包含越界路径：{member_name}")
    return destination


def _extract_archive(archive: Path, destination: Path, *, force: bool) -> None:
    marker = destination / ".extracted"
    if marker.is_file() and not force:
        return
    if force and destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as source:
            for zip_member in source.infolist():
                _safe_destination(destination, zip_member.filename)
            source.extractall(destination)
    elif tarfile.is_tarfile(archive):
        with tarfile.open(archive) as source:
            members = source.getmembers()
            for tar_member in members:
                _safe_destination(destination, tar_member.name)
                if tar_member.issym() or tar_member.islnk():
                    raise ValueError(f"压缩包不允许符号链接：{tar_member.name}")
            source.extractall(destination, members=members)
    else:
        raise ValueError(f"无法识别 Mini-Dev 完整包格式：{archive}")
    marker.write_text("ok\n", encoding="utf-8")


def _select_database_sources(extracted: Path, expected_db_ids: set[str]) -> dict[str, Path]:
    candidates: dict[str, list[Path]] = {db_id: [] for db_id in expected_db_ids}
    for path in extracted.rglob("*.sqlite"):
        if path.stem in candidates:
            candidates[path.stem].append(path)
    selected: dict[str, Path] = {}
    for db_id, paths in candidates.items():
        if not paths:
            raise FileNotFoundError(f"完整包缺少数据库：{db_id}")
        ranked = sorted(
            paths,
            key=lambda path: ("dev_databases" not in path.parts, len(path.parts)),
        )
        first_hash = sha256_file(ranked[0])
        conflicting = [path for path in ranked[1:] if sha256_file(path) != first_hash]
        if conflicting:
            raise ValueError(f"完整包中 {db_id} 存在内容不同的重复 SQLite 文件")
        selected[db_id] = ranked[0]
    return selected


def _materialize_database(source: Path, destination: Path, *, force: bool) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        if not force and sha256_file(source) == sha256_file(destination):
            return
        destination.unlink()
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def _find_gold_file(extracted: Path) -> Path | None:
    matches = sorted(extracted.rglob("mini_dev_sqlite_gold.sql"), key=lambda path: len(path.parts))
    return matches[0] if matches else None


def _cross_check_gold(rows: list[dict[str, Any]], gold_path: Path | None) -> dict[str, Any]:
    if gold_path is None:
        return {"present": False, "mismatch_count": None}
    lines = gold_path.read_text(encoding="utf-8").splitlines()
    if len(lines) != len(rows):
        raise ValueError(f"完整包 Gold 行数与 annotation 不同：{len(lines)} != {len(rows)}")
    mismatches = 0
    for row, line in zip(rows, lines, strict=True):
        if "\t" not in line:
            mismatches += 1
            continue
        sql, db_id = line.rsplit("\t", maxsplit=1)
        if sql.strip() != str(row["SQL"]).strip() or db_id.strip() != str(row["db_id"]):
            mismatches += 1
    if mismatches:
        raise ValueError(f"HF annotation 与完整包 Gold 有 {mismatches} 条不一致")
    return {"present": True, "mismatch_count": 0, "sha256": sha256_file(gold_path)}


def download_dev500(
    dev_root: Path,
    *,
    force: bool = False,
    database_source: str = "mirror",
) -> dict[str, Any]:
    raw_root = dev_root / "raw"
    annotation_path = dev_root / "mini_dev_sqlite.json"
    _download_http(HF_ANNOTATION_URL, annotation_path, force=force)
    rows = _load_annotation(annotation_path)
    if database_source == "mirror":
        databases, transport = _download_mirror_databases(rows, dev_root, force=force)
        gold_check = {"present": False, "mismatch_count": None}
        complete_package: dict[str, Any] = {
            "url": GDRIVE_URL,
            "file_id": GDRIVE_FILE_ID,
            "downloaded": False,
        }
    elif database_source == "drive":
        archive_path = raw_root / "mini_dev_complete_package"
        extracted = raw_root / "package"
        if not archive_path.is_file() or force:
            raw_root.mkdir(parents=True, exist_ok=True)
            downloaded = gdown.download(
                id=GDRIVE_FILE_ID,
                output=str(archive_path),
                quiet=False,
                resume=True,
            )
            if downloaded is None or not archive_path.is_file():
                raise RuntimeError("Google Drive Mini-Dev 完整包下载失败")
        _extract_archive(archive_path, extracted, force=force)
        expected_db_ids = {str(row["db_id"]) for row in rows}
        sources = _select_database_sources(extracted, expected_db_ids)
        databases = {}
        for db_id, source in sorted(sources.items()):
            destination = dev_root / "dev_databases" / db_id / f"{db_id}.sqlite"
            _materialize_database(source, destination, force=force)
            databases[db_id] = {
                "path": str(destination.relative_to(dev_root)),
                "bytes": destination.stat().st_size,
                "sha256": sha256_file(destination),
                "task_count": sum(str(row["db_id"]) == db_id for row in rows),
            }
        gold_check = _cross_check_gold(rows, _find_gold_file(extracted))
        transport = {"kind": "official_complete_package"}
        complete_package = {
            "url": GDRIVE_URL,
            "file_id": GDRIVE_FILE_ID,
            "downloaded": True,
            "bytes": archive_path.stat().st_size,
            "sha256": sha256_file(archive_path),
        }
    else:
        raise ValueError("database_source 只允许 mirror 或 drive")
    report: dict[str, Any] = {
        "schema_version": 1,
        "dataset": "bird_mini_dev_sqlite_500",
        "row_count": len(rows),
        "database_count": len(databases),
        "annotation": {
            "url": HF_ANNOTATION_URL,
            "revision": HF_REVISION,
            "bytes": annotation_path.stat().st_size,
            "sha256": sha256_file(annotation_path),
        },
        "database_source": database_source,
        "database_transport": transport,
        "complete_package": complete_package,
        "official_repository": {
            "url": OFFICIAL_REPOSITORY,
            "revision": OFFICIAL_REPOSITORY_REVISION,
        },
        "gold_cross_check": gold_check,
        "difficulty_counts": dict(sorted(Counter(str(row["difficulty"]) for row in rows).items())),
        "databases": databases,
    }
    return report


def load_dev_examples(dev_root: Path) -> list[BirdSourceExample]:
    rows = _load_annotation(dev_root / "mini_dev_sqlite.json")
    examples: list[BirdSourceExample] = []
    for index, row in enumerate(rows):
        db_id = str(row["db_id"])
        question_id = str(row["question_id"])
        examples.append(
            BirdSourceExample(
                source_index=index,
                task_id=f"bird_minidev_{question_id}",
                db_id=db_id,
                question=str(row["question"]),
                evidence=str(row["evidence"]),
                difficulty=str(row["difficulty"]),
                gold_sql=str(row["SQL"]),
                gold_file_sql=str(row["SQL"]),
                gold_file_db_id=db_id,
                source_match=True,
                db_path=dev_root / "dev_databases" / db_id / f"{db_id}.sqlite",
            )
        )
    return examples


def run_dev_audit(
    dev_root: Path,
    output_root: Path,
    manifest_root: Path,
    *,
    timeout_seconds: float = 10.0,
    max_workers: int = 4,
    max_rows: int = DEFAULT_MAX_ROWS,
    max_result_bytes: int = DEFAULT_MAX_RESULT_BYTES,
    resume: bool = True,
) -> dict[str, Any]:
    inventory_path = manifest_root / "dev_inventory.json"
    if not inventory_path.is_file():
        raise FileNotFoundError("请先运行 nl2sql-rl data download-dev")
    inventory: dict[str, Any] = json.loads(inventory_path.read_text(encoding="utf-8"))
    examples = load_dev_examples(dev_root)
    db_hashes = {db_id: str(value["sha256"]) for db_id, value in inventory["databases"].items()}
    local_root = output_root / "data" / "dev"
    local_root.mkdir(parents=True, exist_ok=True)
    partial_path = local_root / "audit.partial.jsonl"
    previous: dict[str, GoldAuditRecord] = {}
    if resume:
        for raw in read_jsonl(partial_path):
            record = GoldAuditRecord.model_validate(raw)
            previous[record.task_id] = record

    records: dict[str, GoldAuditRecord] = {}
    pending_by_db: dict[str, list[BirdSourceExample]] = {}
    with partial_path.open("a" if resume else "w", encoding="utf-8") as partial:

        def accept(record: GoldAuditRecord) -> None:
            records[record.task_id] = record
            partial.write(stable_json(record.model_dump(mode="json")) + "\n")
            partial.flush()
            os.fsync(partial.fileno())

        for example in examples:
            db_sha = db_hashes.get(example.db_id)
            fingerprint = audit_fingerprint(
                example, db_sha, timeout_seconds, max_rows, max_result_bytes
            )
            cached = previous.get(example.task_id)
            if cached is not None and cached.fingerprint == fingerprint:
                records[example.task_id] = cached
            elif db_sha is None or not example.db_path.is_file():
                accept(
                    _base_record(
                        example,
                        db_sha,
                        timeout_seconds,
                        max_rows,
                        max_result_bytes,
                        status=AuditStatus.MISSING_DATABASE,
                        error="SQLite 数据库文件不存在",
                    )
                )
            else:
                pending_by_db.setdefault(example.db_id, []).append(example)

        states = [
            _WorkerState(
                db_id=db_id,
                db_path=items[0].db_path,
                examples=items,
                db_sha256=db_hashes[db_id],
            )
            for db_id, items in sorted(pending_by_db.items())
        ]
        _run_workers(
            states,
            timeout_seconds=timeout_seconds,
            max_workers=max_workers,
            max_rows=max_rows,
            max_result_bytes=max_result_bytes,
            on_record=accept,
        )

    if len(records) != EXPECTED_DEV_ROWS:
        raise RuntimeError(f"Dev 审计记录不完整：{len(records)} != {EXPECTED_DEV_ROWS}")
    sorted_records = sorted(records.values(), key=lambda record: record.source_index)
    audit_rows = [record.model_dump(mode="json") for record in sorted_records]
    write_jsonl(local_root / "audit.jsonl", audit_rows)
    write_jsonl(manifest_root / "dev_audit.jsonl", audit_rows)

    by_task = {example.task_id: example for example in examples}
    final_records = [
        record
        for record in sorted_records
        if record.executable and record.deterministic is True and not record.result_too_large
    ]
    tasks = []
    answers = []
    for record in final_records:
        example = by_task[record.task_id]
        tasks.append(
            TaskView(
                task_id=example.task_id,
                split="dev_final",
                db_id=example.db_id,
                question=example.question,
                evidence=example.evidence,
                db_ref=str(example.db_path.relative_to(dev_root)),
            ).model_dump(mode="json")
        )
        answers.append(
            HiddenAnswer(
                task_id=example.task_id,
                gold_sql=example.gold_sql,
                audit_status=record.status,
            ).model_dump(mode="json")
        )
    write_jsonl(local_root / "tasks" / "final.jsonl", tasks)
    write_jsonl(local_root / "answers" / "final.jsonl", answers)

    after_hashes = {
        db_id: sha256_file(dev_root / str(value["path"]))
        for db_id, value in inventory["databases"].items()
    }
    unchanged = all(after_hashes[db_id] == db_hashes[db_id] for db_id in db_hashes)
    if not unchanged:
        raise RuntimeError("Dev 审计期间数据库内容发生变化")
    status_counts = Counter(record.status.value for record in sorted_records)
    summary: dict[str, Any] = {
        "schema_version": 1,
        "dataset": "bird_mini_dev_sqlite_500",
        "official_count": EXPECTED_DEV_ROWS,
        "final_n": len(final_records),
        "unverifiable_count": EXPECTED_DEV_ROWS - len(final_records),
        "status_counts": dict(sorted(status_counts.items())),
        "empty_result_count": sum(record.empty_result for record in sorted_records),
        "result_too_large_count": sum(record.result_too_large for record in sorted_records),
        "database_hashes_unchanged": unchanged,
        "timeout_seconds": timeout_seconds,
        "max_workers": max_workers,
        "inventory_sha256": sha256_file(inventory_path),
    }
    write_json(local_root / "summary.json", summary)
    write_json(manifest_root / "dev_audit_summary.json", summary)
    return summary
