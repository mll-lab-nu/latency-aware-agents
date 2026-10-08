# MIKASA Intercept Grab Fast

Install the simulator and model environments with [QUICK_START.md](../QUICK_START.md).
The [standard pipeline](../standard_pipeline.md) describes the shared training runner.

## Shared state-based pipeline

The `train_{qwenoft,qwenpi_v3,qwengr00t,wanoft}.sh` launchers select the
state-based H1 task. They use top and wrist RGB views, a 7D state, and one
7D action at 20 Hz. Qwen models start from Qwen3-VL-4B-Instruct.
WanOFT starts from Wan2.2-TI2V-5B-Diffusers and tiles the simultaneous views.

```bash
conda activate latency
CUDA_VISIBLE_DEVICES=0,1 \
  bash scripts/tasks/mikasa_intercept_grab_fast/train_qwenoft.sh \
  --smoke --run-id mikasa-qwenoft-smoke
```

Replace `--smoke` with `--no-publish` for a full local run, using a new run ID.
The full recipe uses 5,000 SFT updates, global batch 128, and 100 zero-latency
evaluation episodes. These launchers reuse the installed simulator interpreter.
They do not run `uv sync`. Omitting `--no-publish` enables artifact publication.

## H8 input contracts

H8 models predict eight 7D actions per request. They use 20/20 Hz and a
60-transition episode limit. One inference worker processes admitted requests.
The environment starts at reset and advances while the first request is pending.
Busy-worker observations are dropped. Returned chunks use observation-aligned heads.
The scheduler holds the final chunk action when the chunk expires.

| Task key | State | Dataset input |
| --- | --- | --- |
| `mikasa_h8` | 7D proprioception | Converted teacher suffix labels and valid-head masks |
| `mikasa_conditioned_h8` | 63D: 7D proprioception + 8×7 action prefix | Prepared dataset with prefix and loss masks |

The shared training module accepts these task keys through `--task`.
Conditioned H8 requires `--dataset-dir`. Do not supply an old 71D v2 dataset
to the 63D v3 task. A fixed training condition sets prefix length from its
declared delay. A profile condition uses the frozen profile mean, not the next
sampled delay. Prefix masks mark committed heads and loss masks mark trainable heads.

Published H8 SFT/DAgger bundles and the 63D transfer study use different protocols.
The paper's 63D transfer matrix uses a 3,500-update DAgger development snapshot
and 100 development episodes per cell. It is not a 200-episode held-out final
acceptance result. Preserve each experiment's checkpoint and input contract.
See the [artifact registry](../artifacts.md) for published references.

## Original image-only H1 reference

The published original baseline uses image-only inputs. Its fixed protocol and
artifacts are recorded below. The current launchers use the shared state-based
pipeline described above.

### Fixed contract

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

## Profile the final VLA

Expose one simulator GPU and at least one inference GPU, then run:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
  bash scripts/tasks/mikasa_intercept_grab_fast/profile.sh /ABSOLUTE/PATH/inference/checkpoints/model.pt
```

For a shared state-based run, pass the checkpoint path printed by its launcher.
Omitting the path selects the workload preset in `configs/profiles.yaml`, which
references a state-based H1 bundle. Select the original image-only recipe's own
configuration when evaluating its checkpoint.
Logical `cuda:0` runs MIKASA. Logical `cuda:1..` run inference.
Add `--resume RUN_ID` to continue a partial profiling run. Add `--publish` to
upload the completed run and profile to `MLL-Lab/LAGEN-profiles`.

## Original baseline result

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

- Complete the simulator check in the installation guide before PPO.
- If PPO fails its gate, select and independently evaluate a new teacher
  before collecting demonstrations. Record any training changes with that run.
- If SFT runs out of memory, restart from step 0 with a new run id while
  preserving global batch 64; do not resume an incomplete checkpoint.
- For the original baseline, keep its 5,000-step state, inference weights,
  resolved config, and 50-episode evaluation tied to the same run.
