# HumanoidBench Balance Simple (G1)

Install the simulator and model environments with [QUICK_START.md](../QUICK_START.md).
The [standard pipeline](../standard_pipeline.md) describes the shared training workflow.

## Task and teacher inputs

The zero-latency student uses `g1_sonic_balance_simple` at 50 Hz, a 198D state,
256×256 head-camera RGB, and H8 chunks of 43D absolute joint targets.
The SONIC name identifies the scene assets and joint order. The policy acts
on joint targets and does not load a SONIC policy.

The shared runner accepts `qwenoft`, `qwenpi_v3`, `qwengr00t`, and `wanoft`.
Its current Balance recipe uses 5,000 SFT updates and ten DAgger rounds of
500 updates each, with global batch 128.

Supply an accepted H8 FastTD3 checkpoint and its resolved training config.
The runner's default `fasttd3-h8-v1/checkpoint.pth` path is unavailable in the
public model repository. The published `fasttd3-v1` teacher and the default
`train_small_policy.sh` recipe are H1 inputs, not replacements for this H8 recipe.

An H8 training recipe is available at
[`balance_fasttd3_chunk_h8_zero.yaml`](../../configs/training/latency/balance_fasttd3_chunk_h8_zero.yaml).
It has a six-million-frame budget. It does not reproduce the later
25-million-frame paper teacher without that experiment's resolved configuration.

## Train the student

Run from the repository root. Replace both teacher paths with matching inputs:

```bash
conda activate latency
CUDA_VISIBLE_DEVICES=0,1 \
  bash scripts/tasks/humanoidbench_balance_simple/train_qwenoft.sh \
  --teacher-checkpoint /ABSOLUTE/TEACHER/checkpoint.pth \
  --teacher-config /ABSOLUTE/TEACHER/resolved_config.yaml \
  --smoke --run-id balance-qwenoft-smoke
```

The smoke collects a small bootstrap dataset, runs two SFT updates and one
DAgger round with two updates, then evaluates two episodes. It stays local.

For a full local run, replace `--smoke` with `--no-publish` and use a new run ID.
Omitting `--no-publish` enables automatic artifact publication.
Use `--dagger-rounds 0` for SFT only. Use `--resume --run-id ID` to continue
an existing run with the same inputs. Model suffixes select the other three models.

## Profile a checkpoint

Inherit `configs/examples/profiling/g1_sonic_balance_simple/qwenoft.yaml` in
`my_profile.yaml`. Select `profiling.model_artifact.source: local` and set
`profiling.model_artifact.local_root` to the training run's inference bundle.
The example uses `checkpoints/model.pt` inside that bundle. Configure the
matching backbone through `policy.backbone_path` when needed.

```bash
CUDA_VISIBLE_DEVICES=0 python -m latency_bench.profiling.cli run --config my_profile.yaml
```

The ordinary config defines the workload and inference device. The
[profiling cookbook](../profiling/cookbook.md) describes resume and publication.

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
