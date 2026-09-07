#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

export CUDA_HOME="${CONDA_PREFIX}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib:${CUDA_HOME}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export TORCH_CUDA_ARCH_LIST="8.6;9.0;12.0"

python - <<'PY'
import subprocess
import sys

import torch

from training.demon_attack_sf.cule_atari_extension import load_cule_atari_extension
from training.flappy_sf.flappy_cuda_extension import load_flappy_cuda_extension

print(f"python={sys.executable}", flush=True)
print(f"torch={torch.__version__}", flush=True)
print(f"cuda={torch.version.cuda}", flush=True)
flappy_extension = load_flappy_cuda_extension()
demon_cule_extension = load_cule_atari_extension()
torch.cuda.synchronize()
print(f"flappy_cuda_extension={flappy_extension.__name__}", flush=True)
print(f"demon_attack_cule_atari_extension={demon_cule_extension.__name__}", flush=True)
for extension in (flappy_extension, demon_cule_extension):
    code_objects = subprocess.check_output(
        ["cuobjdump", "--list-elf", extension.__file__],
        text=True,
    )
    assert "sm_86" in code_objects, code_objects
    assert "sm_90" in code_objects, code_objects
    assert "sm_120" in code_objects, code_objects
    print(code_objects, flush=True)
PY
