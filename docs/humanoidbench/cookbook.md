# HumanoidBench Balance Simple (G1)

This cookbook reproduces the zero-delay `g1_sonic_balance_simple` teacher to
official GR00T VLA pipeline. The [Chinese internal record](../../program/BALANCE.md)
contains the experiment history and unfinished 15k decision; this page keeps
the stable reader workflow.

## Fixed contract

- Environment: `g1_sonic_balance_simple`, 50 Hz, prompt `Keep the G1 humanoid
  balanced on the support board.`
- Teacher: FastTD3 small policy with the `raw_ramp_0p1s` recipe, 25M training
  frames, seed 1.
- Dataset: 100 accepted teacher episodes, split into 90 train and 10
  validation episodes.
- Schema: `observation.state` is 198D float32; `action` is 43D float32 native
  absolute joint target; `observation.image` is a 256x256 RGB head camera.
- VLA: official GR00T N1.7, H40 `repeat-current`; every 50 Hz frame replans and
  only chunk index 0 is applied. Gaussian seed 0, one-step flow, and action
  head dropout 0 are fixed.
- SFT: two GPUs, global batch 128, 15,000 optimizer steps, 250 warmup steps,
  periodic validation and closed-loop evaluation.


## Quick start

Install the pinned official GR00T environment and authenticate to Hugging Face
and W&B as described in the internal record:

```bash
bash install.sh --model official_gr00t
third_party/Isaac-GR00T/.venv/bin/hf auth login
third_party/Isaac-GR00T/.venv/bin/wandb login
```

Train the native FastTD3 teacher and run its acceptance gate:

```bash
bash scripts/tasks/humanoidbench_balance_simple/train_small_policy.sh
```

Pass the printed accepted checkpoint to the VLA entrypoint:

```bash
bash scripts/tasks/humanoidbench_balance_simple/train_vla.sh \
  /absolute/path/to/accepted_teacher.pt
```

The VLA entrypoint downloads the pinned assets and GR00T base model, collects
and converts the accepted teacher rollouts, runs the documented 15k SFT
recipe, and runs periodic simulated zero-delay evaluations. It does not upload
the checkpoint automatically.

## Profile the final VLA

After the fresh 15k run prints its final model path:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
  bash scripts/tasks/humanoidbench_balance_simple/profile.sh [MODEL_PATH]
```

All visible GPUs are inference workers. The registered remote is the exact
`zero-latency/humanoidbench-balance-simple/vla/gr00t-n1.7-15k-v1` bundle; it is
currently pending and the launcher never substitutes the historical 10k model.
Add `--resume RUN_ID` to continue a partial run or `--publish` to upload the
completed five-session profile.

## Manual workflow

The accepted dataset contains 90 train episodes (86,048 frames) and 10
validation episodes (9,736 frames). Keep the split and `split_seed=0`; do not
replace it with a random frame split. For a manual export:

```bash
MUJOCO_GL=egl conda run --no-capture-output -n latency \
  python scripts/teachers/export_g1_sonic_gr00t.py \
  --output-root outputs/official_gr00t/teachers \
  --teacher g1_balance_simple_raw_ramp_fasttd3_25m_43d_teacher_obs198 \
  --checkpoint /absolute/path/to/accepted_teacher.pth \
  --max-episodes 100 --val-fraction 0.1 --split-seed 0 \
  --num-envs 16 --replay-num-envs 16 --device cuda --deterministic \
  --image-writer-workers 16 --row-buffer-size 8192 --force \
  --balance-teacher-observation-state
```

The official GR00T runner uses the same `198D -> H40 repeat-current -> 43D`
contract. The training output should contain an inference bundle and a separate
training state; publish only the inference bundle. To evaluate a specific
checkpoint manually:

```bash
MUJOCO_GL=egl conda run --no-capture-output -n latency python -m latency_bench.run \
  --config configs/examples/gymnasium/g1_sonic_balance_simple/official_gr00t_eval_simulated.yaml

MUJOCO_GL=egl conda run --no-capture-output -n latency python -m latency_bench.run \
  --config configs/examples/gymnasium/g1_sonic_balance_simple/official_gr00t_eval.yaml
```

Set each config's `policy.model_path` to the exact inference directory being
evaluated. Do not use a `latest` symlink or resume an exhausted schedule.

## Reproduced result and status

The completed 10k reference run is the currently publishable baseline:

| Artifact | Result |
| --- | --- |
| Teacher data | 90 train / 10 validation episodes, fixed revision |
| GR00T SFT | checkpoint-10000 completed; 15k fresh schedule pending |
| Best 10-seed evaluation | return `353.532 +/- 133.571`; mean length `371.0` |
| Runtime integrity | 20 periodic evaluations; invalid, dropped, and underflow actions all 0 |

The 10k inference bundle is historically available in
`latency-sensitive-bench/humanoidbench`; the 15k directory must not be treated
as published until a fresh 15k run completes all validation and closed-loop
gates.

## Published and developer artifacts

Historical source repositories and revisions:

- dataset: [`latency-sensitive-bench/humanoidbench`](https://huggingface.co/datasets/latency-sensitive-bench/humanoidbench),
  revision `dbb954842942196e127c9acc0d776b62aa679990`;
- model: [`latency-sensitive-bench/humanoidbench`](https://huggingface.co/latency-sensitive-bench/humanoidbench),
  revision `734dcedcafbcbffab7f8665d09de42a9c8416d38`;
- scene assets: [`latency-sensitive-bench/g1-sonic-assets`](https://huggingface.co/datasets/latency-sensitive-bench/g1-sonic-assets),
  revision `7c0a6616c78106adf9fa54dfc40e56ea07dc7d5d`.

The accepted artifacts are also published in the central
[`benchmark-models`](https://huggingface.co/latency-sensitive-bench/benchmark-models)
repository at revision `204be41b83c892a1c5d0b37a1dd0d628dd2c6a46` and
[`benchmark-datasets`](https://huggingface.co/datasets/latency-sensitive-bench/benchmark-datasets)
at revision `82b99e2f238d2e1be46e435d5b1ca7a9b9a411a5`. The exact paths are in
the [artifact registry](../artifacts.md). The task runner never uploads
automatically.

## Troubleshooting

- Keep the first run online: GR00T may need the gated `nvidia/Cosmos-Reason2-2B`
  metadata and the fixed scene assets.
- If a checkpoint is resumed, its output directory must contain the prior
  checkpoint and the requested next step; do not resume checkpoint-10000 with
  its exhausted scheduler.
- A lower validation MAE is not a closed-loop gate. Compare the fixed-seed
  return and episode length from the evaluation output.
- Do not upload DeepSpeed optimizer state, W&B caches, raw rollouts, or local
  absolute paths.
