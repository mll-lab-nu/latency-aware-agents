#!/usr/bin/env bash
set -euo pipefail

# Fresh cloud instance bootstrap for this repository branch.
#
# GitHub authentication on Vast.ai or similar rented GPU hosts:
#   1. On your local machine, make sure your GitHub SSH key is loaded:
#        ssh-add ~/.ssh/id_ed25519
#   2. SSH into the rented host with agent forwarding:
#        ssh -A -p <port> root@<host>
#   3. On the rented host, verify GitHub can see the forwarded agent:
#        ssh -T git@github.com
#
# Do not copy private SSH keys onto rented machines. If agent forwarding is not
# acceptable, use a GitHub deploy key or a short-lived token instead.
#
# If SSH auth is unnecessary, override REPO_URL when running this script:
#   REPO_URL=https://github.com/ZihanWang314/latency-sensitive-bench.git bash install.sh

REPO_URL="${REPO_URL:-git@github.com:ZihanWang314/latency-sensitive-bench.git}"
REPO_BRANCH="${REPO_BRANCH:-lixinyuan/bench_framework}"
REPO_DIR="${REPO_DIR:-latency-sensitive-bench}"
CONDA_ENV="${CONDA_ENV:-latency}"

git clone --branch "${REPO_BRANCH}" "${REPO_URL}" "${REPO_DIR}"
cd "${REPO_DIR}"

git config -f .gitmodules submodule.flappy-bird-gymnasium.url https://github.com/mindorigin150/flappy-bird-gymnasium.git
git submodule sync --recursive
git submodule update --init --recursive

conda create -n "${CONDA_ENV}" python=3.10 -y

# Enable `conda activate` when this script runs from a non-interactive shell.
if [[ -n "${CONDA_EXE:-}" ]]; then
    CONDA_BASE="$(dirname "$(dirname "${CONDA_EXE}")")"
elif command -v conda >/dev/null 2>&1; then
    CONDA_BASE="$(conda info --base)"
else
    echo "conda is required but was not found on PATH." >&2
    exit 1
fi

# shellcheck source=/dev/null
source "${CONDA_BASE}/etc/profile.d/conda.sh"

conda activate "${CONDA_ENV}"
python -m pip install --upgrade pip
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements.txt
wandb login
