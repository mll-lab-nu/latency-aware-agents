# Benchmark artifact registry

This is the human-readable index for the supported task pipelines, including
profile-based latency training. New artifacts use these central Hugging Face repositories:

- models: [`latency-sensitive-bench/benchmark-models`](https://huggingface.co/latency-sensitive-bench/benchmark-models)
- datasets: [`latency-sensitive-bench/benchmark-datasets`](https://huggingface.co/datasets/latency-sensitive-bench/benchmark-datasets)
- profiles: [`latency-sensitive-bench/profiles`](https://huggingface.co/datasets/latency-sensitive-bench/profiles)

Paths below are stable task prefixes. The task-owned scripts write local files;
publishing is an explicit developer operation. A status of `planned` means the
central path is reserved but must not be treated as downloadable.

The checked-in source for the central dataset card and its explicit Viewer
configs is [`benchmark_datasets_card.md`](benchmark_datasets_card.md).

## Model artifacts

| Task | Role | Local source | Central path | Status |
| --- | --- | --- | --- | --- |
| MIKASA Intercept Grab Fast | Small-policy PPO teacher | `outputs/tasks/mikasa_intercept_grab_fast/strict_teacher/oracle_checkpoints/InterceptGrabFast-VLA-v0/final_success_ckpt.pt` | `zero-latency/mikasa-intercept-grab-fast/small-policy/ppo-v1/` | Published |
| MIKASA Intercept Grab Fast | Small-policy PPO with QwenOFT RTX 3090 temporal latency | `outputs/latency_rl/mikasa/qwenoft_rtx3090_baseline_settings/20260910T070037Z/best_policy.pt` | `latency-aware/mikasa-intercept-grab-fast/small-policy/ppo-qwenoft-rtx3090-v1/` | Published; independent 256-episode final success 253/256 (98.83%) |
| MIKASA Intercept Grab Fast | Small-policy PPO with QwenPI v3 RTX 3090 temporal latency | `outputs/latency_rl/mikasa/qwenpi_v3_rtx3090_recipe_25m/20260910T093825Z/best_policy.pt` | `latency-aware/mikasa-intercept-grab-fast/small-policy/ppo-qwenpi_v3-rtx3090-v1/` | Published; independent 256-episode final success 248/256 (96.88%) |
| MIKASA Intercept Grab Fast | Small-policy PPO with QwenGR00T RTX 3090 temporal latency | `outputs/latency_rl/mikasa/qwengr00t_rtx3090_recipe_25m/20260910T093825Z/teacher.pt` | `latency-aware/mikasa-intercept-grab-fast/small-policy/ppo-qwengr00t-rtx3090-v1/` | Published; stopped at 20,951,040 steps; independent 256-episode final success 251/256 (98.05%) |
| MIKASA Intercept Grab Fast | Small-policy PPO with WanOFT RTX 3090 temporal latency | `outputs/latency_rl/mikasa/wanoft_rtx3090_recipe_25m/20260910T093825Z/best_policy.pt` | `latency-aware/mikasa-intercept-grab-fast/small-policy/ppo-wanoft-rtx3090-v1/` | Published; selected at 69,120,000 steps; independent 256-episode final success 122/256 (47.66%) |
| MIKASA Intercept Grab Fast | QwenOFT VLA inference bundle | `third_party/starVLA/results/Checkpoints/mikasa_grabfast_h1_gb64_mb8_ga8_expandable/` | `zero-latency/mikasa-intercept-grab-fast/vla/starvla-qwenoft-v1/` | Published; 5k SFT and eval accepted |
| Balance Simple | FastTD3 teacher | `results/humanoidbench_g1_balance_control_comparison_fasttd3_25m/checkpoints/g1_sonic_balance_simple_raw_ramp_0p1s_fasttd3_25000000steps_s1/checkpoint_p0/best_000325980_20864000_reward_702.976.pth` | `zero-latency/humanoidbench-balance-simple/small-policy/fasttd3-v1/` | Published |
| Flappy Bird | Sample Factory zero-latency teacher | Historical HF source recorded in `program/ARTIFACT_HANDOFF.md` | `zero-latency/flappy/small-policy/sample-factory-v1/` | Published; 100-episode mean 444.600; central readback passed |
| Demon Attack | Sample Factory zero-latency teacher | Historical HF source recorded in `program/ARTIFACT_HANDOFF.md` | `zero-latency/demon-attack/small-policy/sample-factory-v1/` | Published; soft-reset-corrected 14400-step 100-episode raw game-score mean 8727.150; central readback `7a6d7af88f921c6244aafe03078f1d30d94cc9b7` |
| Deadly Corridor | Sample Factory zero-latency teacher | Historical HF source recorded in `program/ARTIFACT_HANDOFF.md` | `zero-latency/deadly-corridor/small-policy/sample-factory-v1/` | Published; 100-episode mean 2116.794; central readback passed |
| Atlantis | Sample Factory zero-latency teacher | Existing central checkpoint/config | `zero-latency/atlantis/small-policy/sample-factory-v1/` | Published; 100-episode mean 29271.000; central readback passed |
| Asterix | Sample Factory zero-latency teacher | Existing central checkpoint/config | `zero-latency/asterix/small-policy/sample-factory-v1/` | Published; 100-episode mean 4099.000; central readback passed |
| Atlantis | Sample Factory zero-latency H8 CPU policy | [Published provenance](https://huggingface.co/latency-sensitive-bench/benchmark-models/blob/dfaeecb84cd6b5dfcc3adc0935d0ae86d0b504d2/zero-latency/atlantis/small-policy/sample-factory-hist8-cpu15m-s0-v1/provenance.json) | `zero-latency/atlantis/small-policy/sample-factory-hist8-cpu15m-s0-v1/` | Published at `dfaeecb`; 100-episode 14400-frame mean 88613.000; central readback passed |
| Asterix | Sample Factory zero-latency H8 CPU policy | [Published provenance](https://huggingface.co/latency-sensitive-bench/benchmark-models/blob/dfaeecb84cd6b5dfcc3adc0935d0ae86d0b504d2/zero-latency/asterix/small-policy/sample-factory-hist8-cpu15m-s0-v1/provenance.json) | `zero-latency/asterix/small-policy/sample-factory-hist8-cpu15m-s0-v1/` | Published at `dfaeecb`; 100-episode 14400-frame mean 6041.000; central readback passed |

The five historical Atari-like tasks additionally reserve three zero-latency
VLA paths per task, named
`zero-latency/<task>/vla/starvla-{openvla|gr00t|pi05}-v1/`.
The five zero-latency small-policy bundles remain valid. The nine old
latency-aware small-policy bundles for Flappy Bird, Demon Attack, and Deadly
Corridor were removed from the model repo's main branch at
[`a837166`](https://huggingface.co/latency-sensitive-bench/benchmark-models/commit/a837166929198f76092198ba1ef86efcad45d2ea):
their training did not enforce the worker-capacity admission contract used by
evaluation. Their exact paths and file hashes are in the
[deletion manifest](../outputs/tasks/decision_history_temporal_25m_seed42/legacy_small_policy_predelete.json),
and the bundles remain readable at the parent revision `3d24469718eaddc9db876796c102ec6e8184c777`.
Nine new temporal-H8 best checkpoints use distinct directories; see the
[first publication receipt](../outputs/tasks/decision_history_temporal_25m_seed42/hf_publish_receipt.json)
[second publication receipt](../outputs/tasks/decision_history_temporal_25m_seed42/hf_publish_receipt_three.json),
and [Demon Attack π0.5 receipt](../outputs/tasks/decision_history_temporal_25m_seed42/hf_publish_receipt_demon_pi05.json).
The 15 VLA paths remain pending until their corresponding evaluation and central
revision readback pass. Exact sources, checkpoint selections, results, exclusions,
and the parallel evaluation protocol are maintained in
[`ARTIFACT_HANDOFF`](../program/ARTIFACT_HANDOFF.md).

Do not replace an exact source path with a `latest` symlink in a publication
record.

The MIKASA profile-PPO training and evaluation commands are documented
in the [latency PPO guide](mikasa/latency_ppo.md).

## Hist8 VLA release: Flappy, Demon Attack, Deadly Corridor

The 18 step-5000 models are **Published** at central revision
`95aed229d8a90c0146885eb8f496518e98625d5f`: 9 zero-trained models under
`zero-latency/<game>/vla/starvla-<model>-h1/<run-id>/`, and 9 profile-trained
models under `latency-aware/<game>/vla/starvla-<model>-h1/<run-id>/`.
Models are `qwenoft`, `qwengr00t`, and `qwenpi_v3`; games are `flappy`,
`demon-attack`, and `deadly-corridor`. H1 is the action horizon; hist8 is the
eight-decision input history. Each receives one RGB frame and raw 0/1 history.

All 27 formal evaluation conditions contain 100 episodes. The final revision's
18 checkpoint LFS hashes match the original evaluated weights; 588 metadata
objects and 102 downloaded core files passed readback. The bundles contain
configuration, statistics, task contracts, provenance, exact profile sidecars,
model cards and episode-level results. Deadly uses sf-render-v4.

See the [published model index and scores](https://huggingface.co/latency-sensitive-bench/benchmark-models/blob/af8d7bfacbb03b59b5b5eabbed331577dd5fa690/README.md)
and [publication receipt](../outputs/tasks/decision_history_vla_hist8_v2/hf_publication/README.md).
These are distinct from the historical VLA paths still marked pending above.

## Standard-Pipeline import: seven Gymnasium tasks

The 65 selected bundles from `Standard-Pipeline@2208875f92b21e3f8ec2363a30a6705dfe56cc04`
are **Published**, with the index and verification report at central revision
`ad37dff2ec7ea0adea072024330d19b5e7099e21`.
AirRaid, Ant, HalfCheetah, Hopper, Humanoid, InvertedPendulum and Walker2d contribute
37 H1 VLAs and 28 Sample Factory APPO policies. The source archive labels
`OpenVLA`, `GR00T` and `Pi05` map to `qwenoft`, `qwengr00t` and `qwenpi_v3`.
Gymnasium Humanoid is distinct from HumanoidBench Balance Simple.

Each bundle contains one selected checkpoint. The 37 VLA weights retain their
source bytes; all 28 small-policy bundles passed tensor equality checks, with
23 exported to inference-only state and five copied directly. All 819 imported
files passed identity verification, and 269 core metadata files passed downloaded
readback. The source revision and all 1,773 pre-existing target files were
unchanged by the model import; the final publication appends the root index.

The source has no latency-aware HalfCheetah QwenGR00T/QwenPI v3 VLA or Humanoid
VLA. Seven zero-latency QwenOFT bundles lack standalone `manifest.json` and
`latency_prompt_map.json`; their task contracts are extracted from the original
full configurations, and those missing files remain explicitly recorded.
Evaluation and acceptance states remain those recorded by the source evidence.
Dataset repositories remain at their recorded source locations.

See the [complete model inventory and copy verification](https://huggingface.co/latency-sensitive-bench/benchmark-models/blob/ad37dff2ec7ea0adea072024330d19b5e7099e21/reports/standard-pipeline-2208875f92b2/README.md)
and [local publication receipt](../outputs/hf_migrations/standard-pipeline-2208875f92b2/publication.json).

## Dataset artifacts

| Task | Contents | Local source | Central path | Status |
| --- | --- | --- | --- | --- |
| MIKASA Intercept Grab Fast | 275 successful episodes; raw NPZ and LeRobot 250/25 split | `outputs/mikasa/grabfast_h1_dataset/` | `zero-latency/mikasa-intercept-grab-fast/teacher-rollouts-v1/` | Published |

Do not put optimizer state, W&B directories, temporary raw shards, or local
absolute-path manifests in either central repository. Keep large scene assets
in their existing dedicated repository.

## Deployment profiles

Completed profiles are published under
`profiles/<model>/<gpu-deployment>/<workload>/<physical-instance>/` in the
profiles dataset repository. The associated raw five-session run is published
under `runs/` only after all five 100-episode sessions succeed. Model provenance
records the central model path and resolved revision, not a machine-local
absolute path.

## Historical sources

These task-specific repositories remain unchanged and are listed so an older
result can be traced to its original source. They are not the new publication
layout.

| Task | Historical artifact | Source repository |
| --- | --- | --- |
| MIKASA | PPO teacher | [`latency-sensitive-bench/mikasa-robo`](https://huggingface.co/latency-sensitive-bench/mikasa-robo) |
| MIKASA | Teacher rollouts | [`latency-sensitive-bench/mikasa-robo`](https://huggingface.co/datasets/latency-sensitive-bench/mikasa-robo) |
| Balance Simple | Official GR00T 10k bundle | [`latency-sensitive-bench/humanoidbench`](https://huggingface.co/latency-sensitive-bench/humanoidbench) |
| Balance Simple | Teacher rollouts | [`latency-sensitive-bench/humanoidbench`](https://huggingface.co/datasets/latency-sensitive-bench/humanoidbench) |
| Balance Simple | Scene assets | [`latency-sensitive-bench/g1-sonic-assets`](https://huggingface.co/datasets/latency-sensitive-bench/g1-sonic-assets) |

## StarVLA models and shared data

The four StarVLA models (`qwenoft`, `qwenpi_v3`, `qwengr00t`, `wanoft`) use
`zero-latency/<task>/vla/starvla-<model>-h<horizon>/<run-id>/` in the central
model repository. Balance and MIKASA conditioned runs also have H8 contracts; the
original task pipelines use H1. Formal
pipelines upload after final checkpoint L0 evaluation; smoke runs stay local.
See [SCALE_MODELS](../program/SCALE_MODELS.md) for commands.

MIKASA H8 profile and fixed-2 models use
`latency-aware/mikasa-intercept-grab-fast/vla/starvla-<model>-h8/<run-id>/`.
The SFT run IDs end in `-profile` or `-fixed-2`; the DAgger run IDs are
`h8-conditioned-dagger-20260923-profile` and
`h8-conditioned-dagger-20260923-fixed-2`. The condition remains in each bundle's
provenance. The 12 migrated bundles are available at
[the verified layout revision](https://huggingface.co/latency-sensitive-bench/benchmark-models/tree/af8d7bfacbb03b59b5b5eabbed331577dd5fa690/latency-aware/mikasa-intercept-grab-fast/vla).

Each task shares `zero-latency/<task>/teacher-rollouts-h<horizon>-state-v1/`
across all four models. Balance additionally publishes
model-specific DAgger deltas at
`zero-latency/<task>/dagger/starvla-<model>-h<horizon>/<run-id>/round-NN/`.
The dataset README and task contract describe fields and fixed train/validation
normalization. Model provenance references the exact shared and delta revisions.

Published artifacts record the resolved publication revision where applicable.

## Prepare a backbone for evaluation

Select the exact model repository and revision from the checkpoint's recorded
provenance or its explicit `backbone` metadata. A directory named after a model
is not evidence of its identity. Prepare the model before starting inference:

```bash
hf download MODEL_REPO_ID --revision MODEL_COMMIT
```

The command prints the prepared snapshot directory in the Hugging Face cache.
Use that absolute path as `policy.backbone_path`, or supply an already prepared
local directory. Downloads do not need to live inside `third_party/starVLA`.
Cross-device profiling prepares the explicit `backbone.repo_id` from the resolved
model configuration before starting workers and passes any recorded `revision`
and `allow_patterns` to the download. A prepared `backbone_path` in a profile
spec takes precedence. The task training launchers already prepare their configured Qwen or Wan
backbone before training and evaluation; they use the standard Hugging Face
cache and record its resolved snapshot revision. Their current default-branch selection remains part of the task recipe;
for reproduction, select the recorded revision explicitly.

Create a local evaluation overlay rather than changing historical source
configs. Replace every uppercase path below with an absolute path to the exact
artifact for the selected experiment:

```yaml
__base__: /ABSOLUTE/CHECKOUT/configs/examples/gymnasium/inverted_pendulum/starvla_eval.yaml
policy:
  checkpoint_path: /ABSOLUTE/ARTIFACT/checkpoints/steps_2000_pytorch_model.pt
  model_config_path: /ABSOLUTE/ARTIFACT/config.full.yaml
  task_manifest_path: /ABSOLUTE/DATASET/manifest.json
  latency_prompt_map_path: /ABSOLUTE/DATASET/latency_prompt_map.json
  backbone_path: /ABSOLUTE/HF_CACHE/snapshots/MODEL_COMMIT
logging:
  output_dir: /ABSOLUTE/RESULTS/inverted_pendulum_eval
```

Run it in the installed benchmark environment:

```bash
python -m latency_bench.run --config /ABSOLUTE/PATH/local_eval.yaml
```

The overlay changes artifact locations only. Preserve the base recipe's model,
manifest, checkpoint step, normalization, latency, seed and FPS. The task StarVLA
adapter requires a prepared backbone path. A generic StarVLA model that needs no
external override can continue using its checkpoint's model configuration.
Historical configs under `configs/examples/.../source_contract/` and named
experiment snapshots retain their original paths as provenance.
