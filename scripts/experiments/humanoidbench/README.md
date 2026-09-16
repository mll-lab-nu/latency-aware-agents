# HumanoidBench experiments

Generate and compare the G1 scene bundle:

```bash
python -m scripts.experiments.humanoidbench.alignment build-and-compare \
  --candidate-root results/humanoidbench_g1_official_asset_candidate
```

Balance Simple uses the unified StarVLA task pipeline in
`scripts/starvla/train_task.py`. Its H8 DAgger collector is
`scripts/starvla/balance_dagger.py`; rows contain one 256x256 `ego_view`, the
teacher's 198D pre-decision observation, and 43D absolute joint targets.

See [the Balance cookbook](../../../docs/humanoidbench/cookbook.md) for the
training and evaluation commands.
