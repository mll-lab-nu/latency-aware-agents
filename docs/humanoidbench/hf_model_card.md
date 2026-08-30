# HumanoidBench GR00T models

GR00T models trained for the HumanoidBench tasks in
`latency-sensitive-bench`. The current repository layout is:

```text
balance_simple/
└── checkpoint-15000/
```

## Balance Simple

`balance_simple/checkpoint-15000` is the final model from the formal fresh
`g1_sonic_balance_simple` recipe:

- input: 198D state and 256×256 head-camera RGB image;
- output: H40 repeated-current 43D absolute action;
- training: fixed Gaussian seed 0, one-step flow, action-head dropout 0,
  global batch 128, 250 warmup steps, 15,000 optimizer steps.

Download the inference bundle before passing its local path to GR00T:

```bash
uv run --project third_party/Isaac-GR00T hf download \
  latency-sensitive-bench/humanoidbench \
  --include 'balance_simple/checkpoint-15000/**' \
  --local-dir outputs/official_gr00t/models/humanoidbench
```

The published bundle contains model safetensors, model/processor config,
embodiment mapping, and statistics. DeepSpeed optimizer
state, RNG state, scheduler state, and W&B artifacts are intentionally not
published; this bundle supports inference, not optimizer resume.

The base model is `nvidia/GR00T-N1.7-3B`. No additional license metadata is
asserted here.
