#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

python - <<'PY'
import sys

import torch

from training.flappy_sf.flappy_cuda_extension import load_flappy_cuda_extension

print(f"python={sys.executable}", flush=True)
print(f"torch={torch.__version__}", flush=True)
extension = load_flappy_cuda_extension()
torch.cuda.synchronize()
print(f"flappy_cuda_extension={extension.__name__}", flush=True)
PY
