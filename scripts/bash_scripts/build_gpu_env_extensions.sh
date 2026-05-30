#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

python - <<'PY'
import sys

import torch

from training.deadly_corridor_sf.deadly_corridor_cuda_extension import load_deadly_corridor_cuda_extension
from training.demon_attack_sf.cule_atari_extension import load_cule_atari_extension
from training.demon_attack_sf.demon_attack_cuda_extension import load_demon_attack_cuda_extension

print(f"python={sys.executable}", flush=True)
print(f"torch={torch.__version__}", flush=True)
demon_extension = load_demon_attack_cuda_extension()
demon_cule_extension = load_cule_atari_extension()
deadly_extension = load_deadly_corridor_cuda_extension()
torch.cuda.synchronize()
print(f"demon_attack_cuda_extension={demon_extension.__name__}", flush=True)
print(f"demon_attack_cule_atari_extension={demon_cule_extension.__name__}", flush=True)
print(f"deadly_corridor_cuda_extension={deadly_extension.__name__}", flush=True)
PY
