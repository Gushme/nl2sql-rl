"""按数据库切分训练集，并审计 Train/Validation/Dev 泄漏。"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from nl2sql_rl.io_utils import read_jsonl, write_json, write_jsonl


def choose_validation_databases(
    task_counts: dict[str, int], *, seed: int, target_ratio: float = 0.10
) -> list[str]:
    if not task_counts:
        raise ValueError("没有可切分的训练数据库")
    ordered = sorted(
        task_counts,
        key=lambda db_id: hashlib.sha256(f"{seed}:{db_id}".encode()).hexdigest(),
    )
    total = sum(task_counts.values())
    candidates: list[tuple[float, int, list[str]]] = []
    running = 0
    for index, db_id in enumerate(ordered[:-1], start=1):
        running += task_counts[db_id]
        if running >= 100:
            candidates.append((abs(running / total - target_ratio), index, ordered[:index]))
    if not candidates:
        raise ValueError("无法构造至少包含 100 条任务的 db_id validation 集")
    return min(candidates, key=lambda item: (item[0], item[1]))[2]


def normalize_question(question: str) -> str:
    normalized = unicodedata.normalize("NFKC", question).casefold()
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", normalized)).strip()


def _simhash(text: str) -> int:
    tokens = text.split()
    features = (
        tokens if len(tokens) < 3 else [" ".join(tokens[i : i + 3]) for i in range(len(tokens) - 2)]
    )
    weights = [0] * 64
    for feature in features:
        value = int.from_bytes(hashlib.blake2b(feature.encode(), digest_size=8).digest(), "big")
        for bit in range(64):
            weights[bit] += 1 if value & (1 << bit) else -1
    result = 0
    for bit, weight in enumerate(weights):
        if weight >= 0:
            result |= 1 << bit
    return result


def _near_question_pairs(
    train_tasks: list[dict[str, Any]], dev_tasks: list[dict[str, Any]], threshold: float = 0.95
) -> list[dict[str, Any]]:
    buckets: dict[tuple[int, int], list[tuple[str, str]]] = defaultdict(list)
    for task in train_tasks:
        normalized = normalize_question(str(task["question"]))
        fingerprint = _simhash(normalized)
        for chunk in range(4):
            buckets[(chunk, (fingerprint >> (chunk * 16)) & 0xFFFF)].append(
                (str(task["task_id"]), normalized)
            )
    pairs: dict[tuple[str, str], dict[str, Any]] = {}
    for task in dev_tasks:
        normalized = normalize_question(str(task["question"]))
        fingerprint = _simhash(normalized)
        candidates: set[tuple[str, str]] = set()
        for chunk in range(4):
            candidates.update(buckets[(chunk, (fingerprint >> (chunk * 16)) & 0xFFFF)])
        for train_id, train_question in candidates:
            ratio = SequenceMatcher(None, train_question, normalized).ratio()
            if ratio >= threshold and train_question != normalized:
                key = (train_id, str(task["task_id"]))
                pairs[key] = {
                    "train_task_id": train_id,
                    "dev_task_id": str(task["task_id"]),
                    "similarity": round(ratio, 6),
                }
    return sorted(pairs.values(), key=lambda row: (-row["similarity"], row["train_task_id"]))


def build_splits_and_leakage_report(
    output_root: Path,
    manifest_root: Path,
    *,
    seed: int = 42,
) -> tuple[dict[str, Any], dict[str, Any]]:
    train_root = output_root / "data" / "train"
    dev_root = output_root / "data" / "dev"
    train_tasks = read_jsonl(train_root / "tasks" / "rl_clean.jsonl")
    train_answers = read_jsonl(train_root / "answers" / "rl_clean.jsonl")
    dev_tasks = read_jsonl(dev_root / "tasks" / "final.jsonl")
    if not train_tasks or not dev_tasks:
        raise FileNotFoundError("请先完成 Train 和 Dev Gold 审计")

    counts = Counter(str(task["db_id"]) for task in train_tasks)
    validation_db_ids = choose_validation_databases(dict(counts), seed=seed)
    validation_set = set(validation_db_ids)
    split_train = [task for task in train_tasks if task["db_id"] not in validation_set]
    split_validation = [task for task in train_tasks if task["db_id"] in validation_set]
    split_by_id = {
        str(task["task_id"]): "validation" if task["db_id"] in validation_set else "train"
        for task in train_tasks
    }
    for task in split_train:
        task["split"] = "train"
    for task in split_validation:
        task["split"] = "validation"
    answers_by_split: dict[str, list[dict[str, Any]]] = {"train": [], "validation": []}
    for answer in train_answers:
        answers_by_split[split_by_id[str(answer["task_id"])]].append(answer)

    split_root = output_root / "data" / "splits"
    write_jsonl(split_root / "tasks" / "train.jsonl", split_train)
    write_jsonl(split_root / "tasks" / "validation.jsonl", split_validation)
    write_jsonl(split_root / "answers" / "train.jsonl", answers_by_split["train"])
    write_jsonl(split_root / "answers" / "validation.jsonl", answers_by_split["validation"])

    train_inventory = json.loads(
        (manifest_root / "train_inventory.json").read_text(encoding="utf-8")
    )
    dev_inventory = json.loads((manifest_root / "dev_inventory.json").read_text(encoding="utf-8"))
    train_db_ids = set(counts)
    dev_db_ids = {str(task["db_id"]) for task in dev_tasks}
    train_hashes = {value["sha256"] for value in train_inventory["databases"].values()}
    dev_hashes = {value["sha256"] for value in dev_inventory["databases"].values()}
    train_task_ids = {str(task["task_id"]) for task in train_tasks}
    dev_task_ids = {str(task["task_id"]) for task in dev_tasks}
    train_questions: dict[str, list[str]] = defaultdict(list)
    dev_questions: dict[str, list[str]] = defaultdict(list)
    for task in train_tasks:
        train_questions[normalize_question(str(task["question"]))].append(str(task["task_id"]))
    for task in dev_tasks:
        dev_questions[normalize_question(str(task["question"]))].append(str(task["task_id"]))
    exact_questions = sorted(set(train_questions).intersection(dev_questions))
    exact_pairs = [
        {"train_task_ids": train_questions[value], "dev_task_ids": dev_questions[value]}
        for value in exact_questions
    ]
    leakage: dict[str, Any] = {
        "schema_version": 1,
        "db_id_overlap": sorted(train_db_ids.intersection(dev_db_ids)),
        "database_sha256_overlap": sorted(train_hashes.intersection(dev_hashes)),
        "task_id_overlap": sorted(train_task_ids.intersection(dev_task_ids)),
        "exact_question_overlap": exact_pairs,
        "near_question_pairs": _near_question_pairs(train_tasks, dev_tasks),
    }
    leakage["hard_pass"] = not any(
        leakage[key]
        for key in (
            "db_id_overlap",
            "database_sha256_overlap",
            "task_id_overlap",
            "exact_question_overlap",
        )
    )
    if not leakage["hard_pass"]:
        raise RuntimeError("Train/Dev 泄漏硬检查失败")

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "seed": seed,
        "algorithm": "sha256(seed:db_id) 前缀最接近 10%，且 validation 至少 100 条",
        "train_count": len(split_train),
        "validation_count": len(split_validation),
        "dev_final_count": len(dev_tasks),
        "train_db_ids": sorted(train_db_ids - validation_set),
        "validation_db_ids": validation_db_ids,
        "dev_db_ids": sorted(dev_db_ids),
        "train_task_ids": [str(task["task_id"]) for task in split_train],
        "validation_task_ids": [str(task["task_id"]) for task in split_validation],
        "dev_task_ids": [str(task["task_id"]) for task in dev_tasks],
    }
    write_json(manifest_root / "split_manifest.json", manifest)
    write_json(manifest_root / "leakage_report.json", leakage)
    return manifest, leakage
