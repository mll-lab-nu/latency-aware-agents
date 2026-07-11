#!/usr/bin/env bash
set -euo pipefail

export HF_PROFILE_REPO=latency-sensitive-bench/profiles
export HF_PROFILE_REPO_TYPE=dataset

SPEC=configs/experiments/cross_device_10real_20sim.yaml
ROOT=results/cross_device/cross_device_10real_20sim_v1
TABLES_DIR="$ROOT/tables"
PAPER_ROOT=paper/latency-sensitive-vla/figures/4_1_cross_device_sim
FIG6_DIR="$PAPER_ROOT/fig6_cross_device"

python -m latency_bench.cross_device.protocol aggregate \
  --spec "$SPEC" \
  --root "$ROOT" \
  --out "$TABLES_DIR" \
  --only-policy openvla-flappy \
  --overwrite

python -m latency_bench.analysis.cross_device_figure_data \
  --tables-dir "$TABLES_DIR" \
  --out-fig "$FIG6_DIR/data.csv" \
  --out-table "$PAPER_ROOT/table3_latency_profiles/data.csv" \
  --out-sampler-ablation "$FIG6_DIR/sampler_ablation.csv" \
  --spec "$SPEC" \
  --root "$ROOT" \
  --raw-root "$ROOT/_hf_raw"

python "$FIG6_DIR/run.py" \
  --data "$FIG6_DIR/data.csv" \
  --sampler-ablation "$FIG6_DIR/sampler_ablation.csv" \
  --out-dir "$FIG6_DIR"
python "$PAPER_ROOT/table3_latency_profiles/run.py"
