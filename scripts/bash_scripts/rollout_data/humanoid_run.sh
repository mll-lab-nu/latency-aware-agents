MUJOCO_GL=egl uv run --project . python scripts/teachers/export_g1_sonic_gr00t.py \
    --output-root outputs/official_gr00t/teachers \
    --teacher g1_run_sonic_reference_playback_43d_v2 \
    --max-episodes 100 \
    --val-fraction 0.1 \
    --split-seed 0 \
    --force
