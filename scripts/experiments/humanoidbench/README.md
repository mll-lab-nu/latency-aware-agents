# HumanoidBench experiments

Generate and compare the G1 scene bundle:

```bash
python -m scripts.experiments.humanoidbench.alignment build-and-compare \
  --candidate-root results/humanoidbench_g1_official_asset_candidate
```

Balance Simple uses `scripts/tasks/humanoidbench_balance_simple/vla_profile_dagger.py`
for its experiment schedule. The public `latency_bench collect` entrypoint selects
the H1 teacher → H8/H16 label protocol. Rows contain one 256x256 `ego_view`,
physical state and causal action context, and 43D absolute joint targets.

See [the Balance cookbook](../../../docs/cookbooks/balance.md) for the
training and evaluation commands.
