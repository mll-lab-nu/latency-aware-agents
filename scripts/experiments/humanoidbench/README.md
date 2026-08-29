# HumanoidBench experiments

Current G1 SONIC entry points:

```bash
python -m scripts.experiments.humanoidbench.alignment build-and-compare \
  --candidate-root results/humanoidbench_g1_official_asset_candidate

python -m scripts.experiments.humanoidbench.hurdle_teacher \
  --cache-dir results/humanoidbench_g1_sonic_assets \
  --jump-start-phase 30
```

`alignment` writes the generated G1 scene bundle and comparison artifacts.
`hurdle_teacher` downloads the pinned SONIC/BONES-SEED inputs, evaluates the
trigger sweep, and writes its summary JSON to stdout. `fasttd3` only reproduces
the retained gamma099 1M residual config/checkpoint; it is not the current
Gate 1 training path:

```bash
python -m scripts.experiments.humanoidbench.fasttd3 \
  --output-root results/humanoidbench_g1_sonic29d_fullres_anchor_h025_gamma099_1m \
  --gamma 0.99
```

The fixed Gate 1 teacher index is `configs/teachers/g1_sonic_gr00t.json` and is
the only teacher-job registry read by `scripts/teachers/export_g1_sonic_gr00t.py`.
Hurdle episodes require `teacher_accepted=true`; balance and run episodes must
last at least 18 seconds (900 raw frames at 50 Hz). The run entry is reference
playback from the generated `eval.yaml`, not a neural-network checkpoint. All
three exports use a latency-neutral base prompt, `head_camera`, one `ego_view`
RGB stream at 256x256 and 50 Hz, 93D `official_gr00t_proprio`, and 29D absolute
`action_target`; hand joints remain neutral and are not exported. Do not add
wrist, gripper, stereo, or external third-person cameras to these datasets;
deployment camera semantics must match `head_camera`.
