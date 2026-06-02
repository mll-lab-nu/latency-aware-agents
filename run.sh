#!/usr/bin/env bash
set -euo pipefail

PORT="${HUMAN_EVAL_PORT:-}"
OUTPUT_DIR="${HUMAN_EVAL_OUTPUT_ROOT:-}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --port)
      PORT="${2:?missing value for --port}"
      shift 2
      ;;
    --output-dir|--output_dir)
      OUTPUT_DIR="${2:?missing value for --output-dir}"
      shift 2
      ;;
    -h|--help)
      echo "Usage: bash run.sh [--port PORT] [--output-dir DIR]"
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      echo "Usage: bash run.sh [--port PORT] [--output-dir DIR]" >&2
      exit 2
      ;;
  esac
done

if [[ -n "$OUTPUT_DIR" ]]; then
  export HUMAN_EVAL_OUTPUT_ROOT="$OUTPUT_DIR"
fi

if [[ -z "$PORT" ]]; then
  PORT="$(python -c 'from app_settings import DEFAULT_PORT; print(DEFAULT_PORT)')"
fi

python -m uvicorn server:app --reload --port "$PORT"
