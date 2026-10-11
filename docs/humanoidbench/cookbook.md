# HumanoidBench Balance Simple (G1)

Install the simulator and model environments with [QUICK_START.md](../QUICK_START.md).
The [standard pipeline](../standard_pipeline.md) describes the shared training workflow.

## Task and teacher inputs

The zero-latency student uses `g1_sonic_balance_simple` at 50 Hz, a 198D state,
256×256 head-camera RGB, and H8 chunks of 43D absolute joint targets.
The SONIC name identifies the scene assets and joint order. The policy acts
on joint targets and does not load a SONIC policy.

## Collect and train the student

The current recipe uses an H1 FastTD3 teacher to label H8 or H16 private
rollouts from student states. `zero_full` labels the full chunk.
`profile_conditioned` fixes the returned action prefix and labels its suffix.

For standalone collection, edit the model and asset paths in
`configs/examples/g1_sonic_balance_simple/collect.yaml`, then run:

```bash
python -m latency_bench collect --config my_collect.yaml
python -m latency_bench convert --config my_convert.yaml
```

The task experiment owns the SFT/DAgger schedule. Run a local smoke with
matching teacher weights, the resolved teacher config, and a downloaded backbone:

```bash
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl CUDA_VISIBLE_DEVICES=0,1 \
  python -m scripts.tasks.humanoidbench_balance_simple.vla_profile_dagger \
  --output outputs/balance-smoke \
  --vla-python /ABSOLUTE/VLA/bin/python \
  --backbone /ABSOLUTE/BACKBONE \
  --teacher /ABSOLUTE/TEACHER/checkpoint.pth \
  --teacher-config /ABSOLUTE/TEACHER/fasttd3_train.yaml \
  --model qwenoft --horizon 8 --label-mode zero_full \
  --gradient-checkpointing --smoke
```

Use `--label-mode profile_conditioned --profile-path /ABSOLUTE/PROFILE/profile.json`
for conditioned labels. Use `--horizon 16` for H16. The recipe stays local unless
`--publish` is supplied. `--resume` continues its recorded experiment.

## Profile a checkpoint

Inherit `configs/examples/g1_sonic_balance_simple/profile_qwenoft.yaml` in
`my_profile.yaml`. Select `policy.source: local` and set
`policy.root` to the training run's inference bundle.
The example uses `checkpoints/model.pt` inside that bundle. Configure the
matching backbone through `policy.backbone_path` when needed.

```bash
CUDA_VISIBLE_DEVICES=0 python -m latency_bench profile run --config my_profile.yaml
```

The ordinary config defines the workload and inference device. The
[profiling cookbook](../profiling/cookbook.md) describes measurement and fitting.

## Evaluate profile-conditioned bundles

Profile-conditioned H8 and H16 bundles use 550D and 902D inputs respectively.
These include the physical state and action context. Read the dimensions and
horizon from each bundle's `task_contract.json`.

The fixed-mean prefix protocol sets
`K = ceil(profile.summary.mean_ms * env_fps / 1000)` once per evaluation.
It constructs the prefix from actions already available when the request starts.
Actual action delays still come from the temporal profile. The evaluator applies
new results at their actual readiness time, not at an artificial K-frame deadline.

Run the two trained bundles against the same profile environment:

```bash
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
  python -m scripts.starvla.evaluate \
  --task balance --bundle /ABSOLUTE/BASELINE/bundle \
  --environment /ABSOLUTE/EXPERIMENT/environment.json \
  --output /ABSOLUTE/NEW_RESULTS/baseline \
  --prefix-mode none \
  --vla-python /ABSOLUTE/VLA/bin/python \
  --backbone /ABSOLUTE/BACKBONE
```

For the conditioned bundle, use a new output directory and
`--prefix-mode profile_mean --prefix-profile /ABSOLUTE/PROFILE/profile.json`.
The environment file must preserve the selected experiment's task and latency
settings. The evaluator requires a new output directory.

The published fixed-mean comparison uses 64 episodes, seeds 820001–820064,
50/50 Hz, capacity one, and a 1,000-frame cap. Profile seed is 271828 and
inference seed is 941731. Its OFT/PI/GR00T prefixes are 2/5/4 frames.
These are model-specific RTX 3090 profiles, not equal-delay architecture comparisons.
QwenOFT does not consume the explicit flow-model prefix. Its comparison measures
the difference between the two trained action-context policies.

These evaluations reuse existing checkpoints. They do not establish that a
new training run under the fixed-mean input protocol reproduces those results.
Use the [published model and experiment references](../artifacts.md) to identify
weights and evaluation evidence. Keep training, evaluation, acceptance, and
publication separate. A validation loss does not establish closed-loop performance.
