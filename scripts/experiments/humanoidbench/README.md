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
