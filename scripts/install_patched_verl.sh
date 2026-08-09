#!/usr/bin/env bash
set -euo pipefail

# 只在固定源码身份上应用动态过滤补丁，防止补丁静默漂移。
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PATCH_FILE="${PROJECT_ROOT}/patches/verl-v0.8.0-dynamic-group-filter.patch"

: "${NL2SQL_VERL_COMMIT:?缺少 NL2SQL_VERL_COMMIT}"
: "${NL2SQL_VERL_PATCH_SHA256:?缺少 NL2SQL_VERL_PATCH_SHA256}"
: "${NL2SQL_VERL_UPSTREAM_TRAINER_SHA256:?缺少上游 trainer SHA256}"
: "${NL2SQL_VERL_PATCHED_TRAINER_SHA256:?缺少补丁后 trainer SHA256}"

sha256_file() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  else
    shasum -a 256 "$1" | awk '{print $1}'
  fi
}

if [[ ! -f "${PATCH_FILE}" ]]; then
  echo "缺少 veRL 动态过滤补丁：${PATCH_FILE}" >&2
  exit 2
fi

actual_patch_sha="$(sha256_file "${PATCH_FILE}")"
if [[ "${actual_patch_sha}" != "${NL2SQL_VERL_PATCH_SHA256}" ]]; then
  echo "veRL 动态过滤补丁 SHA256 不匹配" >&2
  exit 2
fi

VERL_SOURCE_ROOT="$(mktemp -d /tmp/nl2sql-verl-v080.XXXXXX)"
trap 'rm -rf -- "${VERL_SOURCE_ROOT}"' EXIT

git -C "${VERL_SOURCE_ROOT}" init --quiet
git -C "${VERL_SOURCE_ROOT}" remote add origin https://github.com/volcengine/verl.git
git -C "${VERL_SOURCE_ROOT}" fetch --quiet --depth 1 origin "${NL2SQL_VERL_COMMIT}"
git -C "${VERL_SOURCE_ROOT}" checkout --quiet --detach FETCH_HEAD

actual_commit="$(git -C "${VERL_SOURCE_ROOT}" rev-parse HEAD)"
if [[ "${actual_commit}" != "${NL2SQL_VERL_COMMIT}" ]]; then
  echo "veRL checkout commit 不匹配" >&2
  exit 2
fi

trainer_file="${VERL_SOURCE_ROOT}/verl/trainer/ppo/ray_trainer.py"
upstream_sha="$(sha256_file "${trainer_file}")"
if [[ "${upstream_sha}" != "${NL2SQL_VERL_UPSTREAM_TRAINER_SHA256}" ]]; then
  echo "veRL 上游 ray_trainer.py SHA256 不匹配" >&2
  exit 2
fi

git -C "${VERL_SOURCE_ROOT}" apply --unidiff-zero --check "${PATCH_FILE}"
git -C "${VERL_SOURCE_ROOT}" apply --unidiff-zero "${PATCH_FILE}"

patched_sha="$(sha256_file "${trainer_file}")"
if [[ "${patched_sha}" != "${NL2SQL_VERL_PATCHED_TRAINER_SHA256}" ]]; then
  echo "veRL 补丁后 ray_trainer.py SHA256 不匹配" >&2
  exit 2
fi

python -m pip install --no-deps --force-reinstall "${VERL_SOURCE_ROOT}"

# 不导入 CUDA trainer，只验证实际安装文件，确保检查可在安装阶段稳定执行。
python -c '
import hashlib
from pathlib import Path
import verl

path = Path(verl.__file__).resolve().parent / "trainer/ppo/ray_trainer.py"
actual = hashlib.sha256(path.read_bytes()).hexdigest()
expected = "'"${NL2SQL_VERL_PATCHED_TRAINER_SHA256}"'"
if actual != expected:
    raise SystemExit("已安装的 veRL ray_trainer.py SHA256 不匹配")
'
