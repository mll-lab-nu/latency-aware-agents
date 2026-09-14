# HumanoidBench experiments

Generate and compare the G1 scene bundle:

```bash
python -m scripts.experiments.humanoidbench.alignment build-and-compare \
  --candidate-root results/humanoidbench_g1_official_asset_candidate
```

The Balance Simple teacher is registered in `configs/teachers/g1_sonic_gr00t.json`
and exported by `scripts/teachers/export_g1_sonic_gr00t.py`. Episodes must last
at least 18 seconds (900 raw frames at 50 Hz). The export uses `head_camera`,
one 256x256 `ego_view` RGB stream, the teacher's 198D pre-decision observation,
and 43D absolute joint targets including learned hand targets.

See [the Balance cookbook](../../../docs/humanoidbench/cookbook.md) for the
teacher export, GR00T training, and evaluation commands.
