# MIKASA PPO with the QwenOFT RTX 3090 profile

This recipe trains a 49D state / 7D action PPO policy from scratch for
`InterceptGrabFast-VLA-v0`, using the measured QwenOFT RTX 3090 temporal profile.
The actor and critic each have three 256-unit tanh layers. Control runs at
20 Hz and physics at 100 Hz, with a 60-control-step episode horizon.

## Accepted experiment

- W&B: [ks6d837t](https://wandb.ai/saberrr-zju/MIKASA-Robo-dataset-collectors/runs/ks6d837t).
- Selected checkpoint: 13,824,000 environment steps, iteration 225.
- Independent evaluation: 255/256 success-once and **253/256 success-at-end
  (98.83%)**, mean return 33.6446; deterministic actions, 64 environments ×
  240 control steps, environment seed 20260910 and latency seed 271828.
- Published model: [`latency-aware/mikasa-intercept-grab-fast/small-policy/ppo-qwenoft-rtx3090-v1`](https://huggingface.co/latency-sensitive-bench/benchmark-models/tree/fb727810a71a21e02c84635ce44e9c889763b236/latency-aware/mikasa-intercept-grab-fast/small-policy/ppo-qwenoft-rtx3090-v1).
- Model publication revision: `fb727810a71a21e02c84635ce44e9c889763b236`.
- Profile revision: `66414463f2a609316da63935941a79abf0a73ce9`; exact dataset
  coordinates are recorded in the config's `profile_source`.

The successful run had a 150M upper budget. This recipe uses a
25M limit: 406 complete rollouts / 24,944,640 environment
steps. Learning-rate annealing is disabled, so the budget does not change the
PPO updates before the stopping point.

## Training

After the normal benchmark setup, initialize the MIKASA submodule and install
its runtime with `bash scripts/mikasa/install.sh`. From the repository root:

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m \
  scripts.tasks.mikasa_intercept_grab_fast.run_profile_policy \
  --config configs/training/latency/mikasa_qwenoft_rtx3090_scratch.yaml \
  --output-dir outputs/mikasa_profile_ppo/train
```

The launcher downloads the exact profile revision, writes the resolved
`config.yaml` into the output directory, and replaces itself with the native
trainer. W&B uses the normal local login. Use a distinct output directory for
each experiment. The YAML owns training settings; the launcher also accepts an
explicit `--checkpoint` override for evaluating a selected local policy.

The scratch recipe sets `checkpoint: null` and `resume: null`. It matches the
successful zero-delay baseline's PPO settings: seed 123, 1,024 environments,
60 rollout steps, 32 minibatches, eight epochs, learning rate `1e-4`, and
target KL `0.2`.

The trainer computes actor loss, entropy and KL only for admitted commands.
The critic uses all control steps, including busy-worker steps used by GAE.
KL early stopping ends the current PPO optimization pass. Profile-mode training
itself continues to its budget or an explicit interruption; it does not stop
when a periodic evaluation reaches 100%.

`best_policy.pt` is selected by periodic `eval/success_at_end`;
`best_evaluation.json` identifies its training step. `training.pt` contains
the latest policy, optimizer, random streams and progress for continuation.
To resume this checkpoint format, set `training.resume` in the YAML to that
bundle and use the same output directory so the best-policy files stay together.

## Independent evaluation

Download the accepted policy, then evaluate it under the pinned profile:

```bash
hf download latency-sensitive-bench/benchmark-models \
  latency-aware/mikasa-intercept-grab-fast/small-policy/ppo-qwenoft-rtx3090-v1/checkpoint.pt \
  --revision fb727810a71a21e02c84635ce44e9c889763b236 \
  --local-dir outputs/published-models

CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m \
  scripts.tasks.mikasa_intercept_grab_fast.run_profile_policy \
  --config configs/training/latency/mikasa_qwenoft_rtx3090_eval.yaml \
  --output-dir outputs/mikasa_profile_ppo/eval
```

For a newly trained policy, add
`--checkpoint outputs/mikasa_profile_ppo/train/best_policy.pt`.
The evaluation config uses the independent 256-episode protocol above and
writes `evaluation.json`. Acceptance is `metrics.success_at_end >= 0.9`.

Use `eval/success_at_end` as the task's primary training-monitor metric.
`success_once` means a success occurred at any step; `success_at_end` requires
success at the final step. `train/*` uses stochastic exploration, whereas
`eval/*` uses deterministic mean actions. The 16-episode periodic evaluation
is separate from the 256-episode acceptance evaluation.

## Checks

```bash
.venv/bin/python -m pytest tests/training/test_action_latency.py \
  tests/scripts/test_mikasa_latency_pipeline.py -q
PYTHONPATH="$PWD:$PWD/third_party/MIKASA-Robo" \
  third_party/MIKASA-Robo/.venv/bin/python -m unittest \
  mikasa_robo_suite.vla.dataset_collectors.test_ppo_latency
```
