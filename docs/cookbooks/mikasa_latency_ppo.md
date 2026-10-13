# MIKASA PPO with StarVLA RTX 3090 profiles

This recipe trains a 49D state / 7D action PPO policy from scratch for
`InterceptGrabFast-VLA-v0`, using a measured StarVLA RTX 3090 temporal profile.
The actor and critic each have three 256-unit tanh layers. Control runs at
20 Hz and physics at 100 Hz, with a 60-control-step episode horizon.

The [artifact guide](../artifacts.md) links the published teacher variants and fixed profile inputs.
Use each model's recorded evaluation protocol when comparing published results.
The current scratch recipe uses a 25M upper budget and disables learning-rate annealing.

## Training

Install MIKASA with `bash install.sh --env mikasa_intercept_grab_fast` and activate
`latency`. From the repository root:

```bash
CUDA_VISIBLE_DEVICES=0 python -m \
  scripts.tasks.mikasa_intercept_grab_fast.run_profile_policy \
  --config configs/training/latency/mikasa_qwenoft_rtx3090_scratch.yaml \
  --output-dir outputs/mikasa_profile_ppo/train
```

Use the corresponding `mikasa_qwenpi_v3_rtx3090_scratch.yaml`,
`mikasa_qwengr00t_rtx3090_scratch.yaml`, or
`mikasa_wanoft_rtx3090_scratch.yaml` for the other registered profiles.

The launcher downloads the exact profile revision, writes the resolved
`config.yaml` into the output directory, and replaces itself with the native
trainer. W&B uses the normal local login. Use a distinct output directory for
each experiment. The YAML owns training settings. The launcher also accepts an
explicit `--checkpoint` override for evaluating a selected local policy.

The scratch recipe sets `checkpoint: null` and `resume: null`. It matches the
successful zero-delay baseline's PPO settings: seed 123, 1,024 environments,
60 rollout steps, 32 minibatches, eight epochs, learning rate `1e-4`, and
target KL `0.2`.

The trainer computes actor loss, entropy and KL only for admitted commands.
The critic uses all control steps, including busy-worker steps used by GAE.
KL early stopping ends the current PPO optimization pass. Profile-mode training
itself continues to its budget or an explicit interruption. It does not stop
when a periodic evaluation reaches 100%.

Periodic `eval/success_at_end` selects `best_policy.pt`.
`best_evaluation.json` identifies its training step. `training.pt` contains
the latest policy, optimizer, random streams and progress for continuation.
To resume this checkpoint format, set `training.resume` in the YAML to that
bundle and use the same output directory so the best-policy files stay together.

## Independent evaluation

Download the selected policy, then evaluate it under the pinned profile:

```bash
hf download MLL-Lab/LAGEN-models \
  latency-aware/mikasa-intercept-grab-fast/small-policy/ppo-qwenoft-rtx3090-v1/checkpoint.pt \
  --revision fb727810a71a21e02c84635ce44e9c889763b236 \
  --local-dir outputs/published-models

CUDA_VISIBLE_DEVICES=0 python -m \
  scripts.tasks.mikasa_intercept_grab_fast.run_profile_policy \
  --config configs/training/latency/mikasa_qwenoft_rtx3090_eval.yaml \
  --output-dir outputs/mikasa_profile_ppo/eval
```

For a newly trained policy, add
`--checkpoint outputs/mikasa_profile_ppo/train/best_policy.pt`.
The evaluation config uses 256 episodes with deterministic actions, environment seed 20260910, and latency seed 271828 and
writes `evaluation.json`. It uses 64 environments with a 240-control-step cap. Acceptance is `metrics.success_at_end >= 0.9`.

Use `eval/success_at_end` as the task's primary training-monitor metric.
`success_once` means a success occurred at any step. `success_at_end` requires
success at the final step. `train/*` uses stochastic exploration, whereas
`eval/*` uses deterministic mean actions. The 16-episode periodic evaluation
is separate from the 256-episode acceptance evaluation.

## Checks

```bash
conda run -n latency python -m pytest tests/training/test_action_latency.py \
  tests/scripts/test_mikasa_profile_policy.py -q
PYTHONPATH="$PWD:$PWD/third_party/MIKASA-Robo" \
  python -m unittest \
  mikasa_robo_suite.vla.dataset_collectors.test_ppo_latency
```

The native PPO launcher saves local logs and checkpoints. Add `--wandb-project YOUR_PROJECT` to enable W&B.
