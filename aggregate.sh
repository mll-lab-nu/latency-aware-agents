#!/usr/bin/env bash
set -euo pipefail

export HF_PROFILE_REPO=latency-sensitive-bench/profiles
export HF_PROFILE_REPO_TYPE=dataset

SPEC=configs/experiments/cross_device_protocol.yaml
ROOT=results/cross_device
TABLES_DIR=results/cross_device/tables
PAPER_ROOT=paper/latency-sensitive-vla/figures/4_1_cross_device_sim

python -m scripts.cross_device.protocol aggregate \
  --spec "$SPEC" \
  --root "$ROOT" \
  --out "$TABLES_DIR" \
  --only-policy openvla-flappy \
  --overwrite

python -m scripts.cross_device.build_figure_data \
  --tables-dir "$TABLES_DIR" \
  --profiles-root "$ROOT" \
  --out-fig "$PAPER_ROOT/fig6_cross_device/data.csv" \
  --out-table "$PAPER_ROOT/table3_latency_profiles/data.csv"

python "$PAPER_ROOT/fig6_cross_device/run.py"
python "$PAPER_ROOT/table3_latency_profiles/run.py"
