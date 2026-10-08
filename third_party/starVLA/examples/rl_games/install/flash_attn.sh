#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
FLASH_ATTN_VERSION="2.8.3"
FLASH_ATTN_WHEEL_URL="https://github.com/lesj0610/flash-attention/releases/download/v2.8.3-cu12-torch2.11/flash_attn-2.8.3%2Bcu12torch2.11cxx11abiTRUE-cp310-cp310-linux_x86_64.whl#sha256=9001c730642cdc1ea44ed8130b0dc80e763519d6efc01e4de44b0700a0dfa13d"
export CUDA_HOME="${CONDA_PREFIX}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib:${CUDA_HOME}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

usage() {
  cat <<'EOF'
Usage: bash examples/rl_games/install/flash_attn.sh [--check]

Installs the pinned community flash-attn 2.8.3 wheel for sm_80, sm_90, and
sm_120 against the fixed PyTorch 2.11.0/CUDA 12.8 environment. Installation and smoke-test
failures are fatal.

Options:
  --check       Run the import and BF16 forward/backward smoke test only.
  -h, --help    Show this help.
EOF
}

run_check() {
  "$PYTHON_BIN" - <<'PY'
import torch
from flash_attn import __version__, flash_attn_func
import flash_attn_2_cuda  # noqa: F401

q = torch.randn(2, 128, 8, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
k = torch.randn(2, 128, 8, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
v = torch.randn(2, 128, 8, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
output = flash_attn_func(q, k, v, causal=True)
output.float().square().mean().backward()
torch.cuda.synchronize()
assert torch.isfinite(output).all()
assert torch.isfinite(q.grad).all()
assert torch.isfinite(k.grad).all()
assert torch.isfinite(v.grad).all()
print(f"[flash_attn/check] OK {__version__}")
PY
}

run_arch_check() {
  local extension_path
  local code_objects
  extension_path="$("$PYTHON_BIN" - <<'PY'
import torch  # noqa: F401
import flash_attn_2_cuda

print(flash_attn_2_cuda.__file__)
PY
)"
  code_objects="$(cuobjdump --list-elf "${extension_path}")"
  grep -q "sm_80" <<<"${code_objects}"
  grep -q "sm_90" <<<"${code_objects}"
  grep -q "sm_120" <<<"${code_objects}"
}

case "${1:-}" in
  "")
    ;;
  --check)
    run_arch_check
    run_check
    exit 0
    ;;
  -h|--help)
    usage
    exit 0
    ;;
  *)
    echo "[install/flash_attn] unknown argument: $1" >&2
    usage >&2
    exit 1
    ;;
esac

echo "[install/flash_attn] Installing flash-attn==${FLASH_ATTN_VERSION} wheel"
"$PYTHON_BIN" -m pip install \
  "${FLASH_ATTN_WHEEL_URL}" \
  --force-reinstall \
  --no-deps
run_arch_check
run_check
echo "[install/flash_attn] done"
