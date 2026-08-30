---
base_model: nvidia/GR00T-N1.7-3B
tags:
  - robotics
  - humanoid
  - gr00t
---

# HumanoidBench GR00T models

GR00T models trained for the HumanoidBench tasks in
`latency-sensitive-bench`. The repository layout is:

```text
balance_simple/
├── checkpoint-10000/  # published
└── checkpoint-15000/  # published automatically after the future 15k run
```

The 15k run has not been started yet. Its fixed W&B project is
`finetune-gr00t-n1d7`, with run id and name
`g1_balance_simple_obs198_h40_repeat_fixedz0_onestep_dropout0_15k`.

## Balance Simple

`balance_simple/checkpoint-10000` is the completed 10k model from the formal
`g1_sonic_balance_simple` recipe. The future
`balance_simple/checkpoint-15000` will use the same recipe with a fresh 15k
schedule:

- input: 198D state and 256×256 head-camera RGB image;
- output: H40 repeated-current 43D absolute action;
- training: fixed Gaussian seed 0, one-step flow, action-head dropout 0,
  global batch 128, and 250 warmup steps; checkpoint-10000 used 10,000
  optimizer steps, while the future checkpoint-15000 uses 15,000 steps.

Download the completed 10k inference bundle before passing its local path to
GR00T:

```bash
third_party/Isaac-GR00T/.venv/bin/hf download \
  latency-sensitive-bench/humanoidbench \
  --include 'balance_simple/checkpoint-10000/**' \
  --local-dir outputs/official_gr00t/models/humanoidbench
```

After the future 15k run completes, replace `checkpoint-10000` with
`checkpoint-15000` to download its automatically published bundle.

The published bundle contains model safetensors, model/processor config,
embodiment mapping, and statistics. DeepSpeed optimizer
state, RNG state, scheduler state, and W&B artifacts are intentionally not
published; this bundle supports inference, not optimizer resume.

The base model is `nvidia/GR00T-N1.7-3B`. No additional license metadata is
asserted here.
