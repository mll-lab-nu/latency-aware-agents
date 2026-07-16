#!/usr/bin/env bash
set -euo pipefail

# Download the starVLA base VLM (Qwen3-VL-4B-Instruct) to the canonical absolute
# path that the trained checkpoints' config.yaml `framework.qwenvl.base_vlm`
# points to: /workspace/starVLA/playground/Pretrained_models/Qwen3-VL-4B-Instruct
#
# This base VLM is the architecture skeleton loaded at inference for every
# starVLA (OpenVLA-family) policy. The training-time bridge checkpoint
# (Qwen3VL-OFT-Bridge-RT-1) is NOT needed at eval time -- starVLA nulls it out
# and loads the fine-tuned weights on top of this base.

SESSION_NAME="${SESSION_NAME:-hfd_qwen3_vl_4b_instruct}"
LOG_DIR="${LOG_DIR:-/workspace/logs}"
LOG_FILE="${LOG_FILE:-$LOG_DIR/${SESSION_NAME}.log}"

MODEL_ID="${MODEL_ID:-Qwen/Qwen3-VL-4B-Instruct}"
# hfd downloads into $DOWNLOAD_DIR/<repo-basename>, i.e. this yields
# /workspace/starVLA/playground/Pretrained_models/Qwen3-VL-4B-Instruct
DOWNLOAD_DIR="${DOWNLOAD_DIR:-/workspace/starVLA/playground/Pretrained_models}"
HFD_SCRIPT="${HFD_SCRIPT:-/workspace/hfd.sh}"
DOWNLOAD_RETRY_DELAY_S="${DOWNLOAD_RETRY_DELAY_S:-10}"
MODEL_DIR="${DOWNLOAD_DIR}/$(basename "$MODEL_ID")"
MANIFEST_FILE="${MODEL_DIR}/.hfd/manifest"

SCRIPT_PATH="$(readlink -f "$0")"
REPO_ROOT="$(cd "$(dirname "$SCRIPT_PATH")/.." && pwd)"

install_packages() {
  local pkgs=("$@")
  export DEBIAN_FRONTEND=noninteractive

  if [[ "$(id -u)" -eq 0 ]]; then
    apt-get update
    apt-get install -y "${pkgs[@]}"
  else
    sudo apt-get update
    sudo apt-get install -y "${pkgs[@]}"
  fi
}

ensure_launcher_deps() {
  local pkgs=()

  command -v tmux >/dev/null 2>&1 || pkgs+=("tmux")
  command -v wget >/dev/null 2>&1 || pkgs+=("wget")

  if (( ${#pkgs[@]} > 0 )); then
    install_packages "${pkgs[@]}" ca-certificates
  fi
}

ensure_worker_deps() {
  local pkgs=()

  command -v curl >/dev/null 2>&1 || pkgs+=("curl")
  command -v wget >/dev/null 2>&1 || pkgs+=("wget")
  command -v aria2c >/dev/null 2>&1 || pkgs+=("aria2")

  if (( ${#pkgs[@]} > 0 )); then
    install_packages "${pkgs[@]}" ca-certificates
  fi
}

download_is_complete() {
  local expected_size
  local path
  local actual_size

  [[ -s "$MANIFEST_FILE" ]] || return 1
  while IFS=$'\t' read -r expected_size path; do
    [[ -e "${MODEL_DIR}/${path}" && ! -e "${MODEL_DIR}/${path}.aria2" ]] || return 1
    actual_size="$(stat -c %s "${MODEL_DIR}/${path}")" || return 1
    [[ "$actual_size" == "$expected_size" ]] || return 1
  done < "$MANIFEST_FILE"
}

download_once() {
  wget https://hf-mirror.com/hfd/hfd.sh -O "$HFD_SCRIPT" || return
  chmod +x "$HFD_SCRIPT" || return

  (
    cd "$DOWNLOAD_DIR"
    bash "$HFD_SCRIPT" "$MODEL_ID" \
      -x 10 \
      -j 10
  )
}

# 第一次执行：创建后台 tmux session，然后立刻退出
if [[ "${RUN_INSIDE_TMUX:-0}" != "1" ]]; then
  mkdir -p "$LOG_DIR"
  ensure_launcher_deps

  if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "[INFO] tmux session already exists: $SESSION_NAME"
    echo "[INFO] attach: tmux attach -t $SESSION_NAME"
    echo "[INFO] log: tail -f $LOG_FILE"
    exit 0
  fi

  q() { printf "%q" "$1"; }

  CMD="cd $(q "$REPO_ROOT") && RUN_INSIDE_TMUX=1 MODEL_ID=$(q "$MODEL_ID") DOWNLOAD_DIR=$(q "$DOWNLOAD_DIR") HFD_SCRIPT=$(q "$HFD_SCRIPT") DOWNLOAD_RETRY_DELAY_S=$(q "$DOWNLOAD_RETRY_DELAY_S") bash $(q "$SCRIPT_PATH") 2>&1 | tee -a $(q "$LOG_FILE")"

  tmux new-session -d -s "$SESSION_NAME" "$CMD"

  echo "[INFO] started background tmux session: $SESSION_NAME"
  echo "[INFO] attach: tmux attach -t $SESSION_NAME"
  echo "[INFO] detach: Ctrl-b then d"
  echo "[INFO] log: tail -f $LOG_FILE"
  echo "[INFO] stop: tmux kill-session -t $SESSION_NAME"
  exit 0
fi

# 第二次执行：真正跑下载逻辑，此时已经在 tmux 后台里
echo "[INFO] start at: $(date)"
echo "[INFO] model id: $MODEL_ID"
echo "[INFO] download dir: $DOWNLOAD_DIR"
echo "[INFO] hfd script: $HFD_SCRIPT"

ensure_worker_deps

mkdir -p "$DOWNLOAD_DIR"

until download_once && download_is_complete; do
  echo "[WARN] download incomplete; retrying in ${DOWNLOAD_RETRY_DELAY_S}s"
  sleep "$DOWNLOAD_RETRY_DELAY_S"
done

echo "[INFO] verified all manifest files: $MODEL_DIR"
echo "[INFO] base vlm at: $MODEL_DIR"
echo "[INFO] finished at: $(date)"
