#!/usr/bin/env bash
set -euo pipefail

cd "${WORKSPACE_DIR}"

if [[ ! -d "${WORKSPACE_DIR}/latency-sensitive-bench/.git" ]]; then
  git clone https://github.com/ZihanWang314/latency-sensitive-bench
fi

cd "${WORKSPACE_DIR}/latency-sensitive-bench"

git config --global url."https://github.com/".insteadOf "git@github.com:"
git config --global url."https://github.com/".insteadOf "ssh://git@github.com/"

git config -f .gitmodules submodule.flappy-bird-gymnasium.url https://github.com/mindorigin150/flappy-bird-gymnasium.git
git config -f .gitmodules submodule.sample-factory.url https://github.com/mindorigin150/sample-factory.git
git config -f .gitmodules submodule.starVLA.url https://github.com/talha1503/starVLA.git

git submodule sync --recursive
git submodule update --init --recursive


export PYTHONPATH="${WORKSPACE_DIR}/latency-sensitive-bench:${PYTHONPATH:-}"
