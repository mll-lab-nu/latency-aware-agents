# MIKASA Intercept Grab Fast (H1)

Install environments using [QUICK_START.md](../QUICK_START.md). The recipe below
covers the original task pipeline. For the four continuous-state H1 models,
use [SCALE_MODELS.md](../../program/SCALE_MODELS.md); their task inputs, training
budgets, and automatic publication behavior are specified there.

This cookbook reproduces the zero-delay `InterceptGrabFast-VLA-v0` baseline.
The detailed engineering record is in the [Chinese internal runbook](../../program/MIKASA.md);
this page is the reader-facing path.

## Fixed contract

- Environment: `InterceptGrabFast-VLA-v0`, H1, `pd_ee_delta_pose`, 7D action at
  20 Hz (50 ms control cadence).
- Teacher: privileged-state PPO with a 49D state. Rollouts and evaluation use
  deterministic mean actions; training samples actions.
- Dataset: 275 successful teacher episodes, split into 250 train and 25
  validation episodes.
- VLA: StarVLA QwenOFT (Qwen3-VL-4B-Instruct plus an MLP action head), predicting
  one current 7D action from the current frame; `include_state=false`.
- SFT: one GPU, DeepSpeed ZeRO-2, global batch 64, 5,000 optimizer steps.
- Gates: the teacher must pass the trainer `16/16` gate and an independent
  canonical `95/100` gate. The VLA baseline gate is at least `25/50`
  `success_once` episodes.

This is a zero-delay baseline. It does not train with fixed delay, device
profiles, action chunks, state/delay inputs, or online RL.

## Quick start

Install MIKASA, StarVLA, and both converters using
[QUICK_START.md](../QUICK_START.md#4-mikasa-与两个转换环境). The legacy wrappers
below re-run `uv sync`: on Blackwell hosts with a Torch override, use the
[runbook’s direct Python stages](../../program/MIKASA.md) to preserve it.
On hosts using the unmodified MIKASA lockfile, run from the repository root:

```bash
bash scripts/tasks/mikasa_intercept_grab_fast/train_small_policy.sh
```

The task entrypoint runs the native PPO recipe and prints the accepted teacher
checkpoint. Use that exact path for the VLA stages:

```bash
conda activate starvla_rl_games_openvla
bash scripts/tasks/mikasa_intercept_grab_fast/train_vla.sh \
  /absolute/path/to/final_success_ckpt.pt
```

The VLA entrypoint collects and audits the 275 demonstrations, creates the
250/25 LeRobot split, trains the 5,000-step QwenOFT model, starts the policy
server for the canonical evaluation, and writes the final metrics under its
output root. It does not publish artifacts automatically.

For a manual run, the underlying stages are:

```text
scripts/mikasa/evaluate.py
scripts/mikasa/prepare_starvla_dataset.py
MIKASA NPZ -> RLDS -> LeRobot converter
StarVLA starVLA/training/train_starvla.py
```

The task-owned entrypoint supplies the same fixed split and model settings.
Use the internal runbook for GPU/Vulkan setup, the finite tuning ladder, and
the exact environment variables required on the Blackwell host.

## Profile the final VLA

Expose one simulator GPU and at least one inference GPU, then run:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
  bash scripts/tasks/mikasa_intercept_grab_fast/profile.sh [MODEL_PATH]
```

The default `MODEL_PATH` is the 5k QwenOFT run printed by `train_vla.sh`. If that
root is absent, the launcher downloads the exact
`zero-latency/mikasa-intercept-grab-fast/vla/starvla-qwenoft-v1` artifact.
Logical `cuda:0` runs MIKASA; `cuda:1..` run inference. Add `--resume RUN_ID` to
continue a partial five-session run or `--publish` to upload the completed run
and profile to `MLL-Lab/LAGEN-profiles`.

## Manual stage gates

### Teacher

Run the PPO smoke first, then the formal teacher. Evaluate the selected
checkpoint independently on 100 fixed seeds before collecting data:

```bash
python scripts/mikasa/evaluate.py \
  --policy ppo \
  --checkpoint /absolute/path/to/final_success_ckpt.pt \
  --episodes 100 \
  --output-dir outputs/mikasa/ppo_teacher_strict_eval
```

The checkpoint is accepted only when both the trainer gate and the canonical
`95/100` gate pass. Do not collect demonstrations from a smoke or failed
candidate.

### Dataset

The accepted collector output must contain 275 successful trajectories. The
fixed split is:

```text
train: episodes 0..249   (250 episodes)
val:   episodes 250..274 (25 episodes)
fps:   20
```

Audit the NPZ metadata, unique seeds, success markers, action shape, and
`[-1, 1]` action range before conversion. The mixture file must reference the
exact train and validation dataset names used by the QwenOFT configuration.

### VLA and evaluation

Use the supplied Qwen3-VL and Bridge/RT-1 initialization checkpoints. Keep
global batch 64; the validated Blackwell setting is microbatch 8 and
accumulation 8 with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
After 5,000 optimizer steps, evaluate 50 fixed seeds at zero delay and judge
only the `success_once` JSON, not the held-out action loss.

## Reproduced result

The completed reference run produced:

| Artifact | Result |
| --- | --- |
| PPO teacher | `100/100` canonical episodes; mean return `36.819` |
| Rollouts | 275 episodes / 9,859 transitions; all successful |
| LeRobot split | 250 train episodes / 8,911 frames; 25 validation episodes / 948 frames |
| QwenOFT SFT | 5,000/5,000 steps; global batch 64 |
| VLA evaluation | `27/50` (`54%`) canonical success; mean return `22.326` |

## Published and developer artifacts

Historical assets use the central locations below. Original source revisions remain in the consolidation manifest:

- teacher/model: [central PPO teacher](https://huggingface.co/MLL-Lab/LAGEN-models/tree/e3fea81529fd98b5d5782db38803fe4062215dda/zero-latency/mikasa-intercept-grab-fast/small-policy/ppo-v1),
  central revision `e3fea81529fd98b5d5782db38803fe4062215dda`, original source revision `38327e16f9b33cf5f827b54a606cb4c22d4dfbfc`;
- dataset: [published rollouts](https://huggingface.co/datasets/MLL-Lab/LAGEN-datasets/tree/4d8be25dc6fccfe244c2fa6be4b771ee448f2c2a/zero-latency/mikasa-intercept-grab-fast/teacher-rollouts-v1),
  central revision `4d8be25dc6fccfe244c2fa6be4b771ee448f2c2a`, original source revision `28312021c634fa97f8068077dfd6eb0efe6d45ce`.

The accepted artifacts are also published in the central
[`benchmark-models`](https://huggingface.co/MLL-Lab/LAGEN-models)
repository at revision `204be41b83c892a1c5d0b37a1dd0d628dd2c6a46` and
[`benchmark-datasets`](https://huggingface.co/datasets/MLL-Lab/LAGEN-datasets)
at revision `82b99e2f238d2e1be46e435d5b1ca7a9b9a411a5`. Exact paths are listed in
the [artifact registry](../artifacts.md). The model copy excludes optimizer
state and W&B files; the dataset copy includes both raw NPZ and the fixed
LeRobot split.

## Troubleshooting

- On Blackwell, use the tested MIKASA Torch/Vulkan userspace documented in the
  internal runbook and complete the reset/step smoke before PPO.
- If PPO fails its gate, stop before collection and evaluate the next finite
  candidate only in the documented learning-rate/update-epoch order.
- If SFT runs out of memory, restart from step 0 with a new run id while
  preserving global batch 64; do not resume an incomplete checkpoint.
- A final model is valid only when its 5,000-step state, inference weights,
  resolved config, and 50-episode evaluation belong to the same run.
