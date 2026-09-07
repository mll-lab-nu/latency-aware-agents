#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO_ROOT}"
exec "${VLA_PY:-python}" -m scripts.starvla.train_task --task balance --model wanoft "$@"
