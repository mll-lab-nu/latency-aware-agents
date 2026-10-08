# Benchmark artifact registry

This is the human-readable index for the supported task pipelines, including
profile-based latency training. Benchmark release artifacts use these central Hugging Face repositories:

- models: [`MLL-Lab/LAGEN-models`](https://huggingface.co/MLL-Lab/LAGEN-models)
- datasets: [`MLL-Lab/LAGEN-datasets`](https://huggingface.co/datasets/MLL-Lab/LAGEN-datasets)
- profiles: [`MLL-Lab/LAGEN-profiles`](https://huggingface.co/datasets/MLL-Lab/LAGEN-profiles)

The [LAGEN collection](https://huggingface.co/collections/MLL-Lab/lagen-6ac750f21139008b8114fff5)
groups the three repositories under MLL-Lab.

Paths below are stable task prefixes. The task-owned scripts write local files;
publishing is an explicit developer operation. A status of `planned` means the
central path is reserved but must not be treated as downloadable.

The checked-in source for the central dataset card and its explicit Viewer
configs is [`benchmark_datasets_card.md`](benchmark_datasets_card.md).

## Model artifacts

Figure 2 source bindings and evaluation records are published in the
[latency-degradation catalogue](https://huggingface.co/datasets/MLL-Lab/LAGEN-datasets/tree/d1014aaaa1dd7f40ae21c60070dc0659fa265422/experiments/latency-degradation).
All 15 curve references, 477 evidence objects and 6000 MIKASA episode records
passed readback. The six confirmed game weights are published in
[benchmark-models](https://huggingface.co/MLL-Lab/LAGEN-models/tree/e3fea81529fd98b5d5782db38803fe4062215dda/zero-latency)
with the original SHA256, size and Xet identity. Available original configs and
statistics are stored with their canonical bundles. Six MIKASA weights reuse
existing fixed revisions. Flappy GR00T step 3000 remains missing. The two
historical game OpenVLA checkpoint bindings remain unconfirmed.
[Verification report](../outputs/hf_cleanup/latency-degradation-20261007/REPORT.md).

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
| Atlantis | Sample Factory zero-latency H8 CPU policy | [Published provenance](https://huggingface.co/MLL-Lab/LAGEN-models/blob/dfaeecb84cd6b5dfcc3adc0935d0ae86d0b504d2/zero-latency/atlantis/small-policy/sample-factory-hist8-cpu15m-s0-v1/provenance.json) | `zero-latency/atlantis/small-policy/sample-factory-hist8-cpu15m-s0-v1/` | Published at `dfaeecb`; 100-episode 14400-frame mean 88613.000; central readback passed |
| Asterix | Sample Factory zero-latency H8 CPU policy | [Published provenance](https://huggingface.co/MLL-Lab/LAGEN-models/blob/dfaeecb84cd6b5dfcc3adc0935d0ae86d0b504d2/zero-latency/asterix/small-policy/sample-factory-hist8-cpu15m-s0-v1/provenance.json) | `zero-latency/asterix/small-policy/sample-factory-hist8-cpu15m-s0-v1/` | Published at `dfaeecb`; 100-episode 14400-frame mean 6041.000; central readback passed |

The five historical Atari-like tasks additionally reserve three zero-latency
VLA paths per task, named
`zero-latency/<task>/vla/starvla-{openvla|gr00t|pi05}-v1/`.
The five zero-latency small-policy bundles remain valid. The nine old
latency-aware small-policy bundles for Flappy Bird, Demon Attack, and Deadly
Corridor were removed from the model repo's main branch at
[`a837166`](https://huggingface.co/MLL-Lab/LAGEN-models/commit/a837166929198f76092198ba1ef86efcad45d2ea):
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

See the [published model index and scores](https://huggingface.co/MLL-Lab/LAGEN-models/blob/af8d7bfacbb03b59b5b5eabbed331577dd5fa690/README.md)
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

See the [complete model inventory and copy verification](https://huggingface.co/MLL-Lab/LAGEN-models/blob/ad37dff2ec7ea0adea072024330d19b5e7099e21/reports/standard-pipeline-2208875f92b2/README.md)
and [local publication receipt](../outputs/hf_migrations/standard-pipeline-2208875f92b2/publication.json).

## Dataset artifacts

| Task | Contents | Local source | Central path | Status |
| --- | --- | --- | --- | --- |
| MIKASA Intercept Grab Fast | 275 successful episodes; raw NPZ and LeRobot 250/25 split | `outputs/mikasa/grabfast_h1_dataset/` | `zero-latency/mikasa-intercept-grab-fast/teacher-rollouts-v1/` | Published |

Model publication retains selected inference checkpoints and their source metadata.
Optimizer/trainer states and W&B binaries remain retired. Original training data,
evaluation records and logs are retained as artifacts. Scene assets use
`benchmark-datasets/assets/humanoidbench/g1-sonic/`.

The [storage retention report](../outputs/hf_cleanup/storage-prune-20261007/REPORT.md)
records the retirement of HAIC and Extreme Parkour, removal of recovery state,
and fixed reference repairs. Each run retains its selected evaluation checkpoint,
or its latest checkpoint when no evaluation selection exists. Latest-checkpoint
comparison metrics remain evidence even when the unselected weight is retired.
Model and dataset bodies are not downloaded for this cleanup. Git/LFS identities
verify retained bytes; framework loading and evaluation are not repeated.

## Deployment profiles

Completed profiles are published under
`profiles/<model>/<gpu-deployment>/<workload>/<physical-instance>/` in the
profiles dataset repository. The associated raw five-session run is published
under `runs/` only after all five 100-episode sessions succeed. Model provenance
records the central model path and resolved revision, not a machine-local
absolute path.

## Paper experiment catalogue

Paper assets use the three main repositories. Each shared checkpoint, dataset and profile has one canonical location. Task manifests pin exact revisions and retain original source identities.

| Experiment | Settings and asset references |
|---|---|
| Visual history | [visual-history](https://huggingface.co/datasets/MLL-Lab/LAGEN-datasets/tree/e0f4007f504978f93c32557ea4e9b97fdf62fcf5/experiments/visual-history) |
| Latency in prompt | [latency-in-prompt](https://huggingface.co/datasets/MLL-Lab/LAGEN-datasets/tree/e0f4007f504978f93c32557ea4e9b97fdf62fcf5/experiments/latency-in-prompt) |
| Latency transfer | [latency-transfer](https://huggingface.co/datasets/MLL-Lab/LAGEN-datasets/tree/e0f4007f504978f93c32557ea4e9b97fdf62fcf5/experiments/latency-transfer) |
| Mean vs. profile training | [mean-vs-profile](https://huggingface.co/datasets/MLL-Lab/LAGEN-datasets/tree/e0f4007f504978f93c32557ea4e9b97fdf62fcf5/experiments/mean-vs-profile) |

The root README pages provide navigation only. Experiment pages list tasks. Task pages contain detailed settings, results and manifests. Run directories retain model configuration and normalization statistics.

The original memory, latency-information, latency-transfer and fine-tuning-scope repositories were deleted after verified consolidation. Source records remain under the canonical experiments’ provenance paths. The original Flappy Ghost-7 model is removed from their current indices. Ghost-7 exp3 and its shared training data remain available. Identical Ghost-15 checkpoint bytes use one path while their source run metadata remain separate.

Factor-sweeps and fine-tuning-scope archives use the central experiment manifests. Their data consumers use benchmark-datasets.

## Consolidated historical sources

[Source archive index](https://huggingface.co/datasets/MLL-Lab/LAGEN-datasets/tree/a6e93d4a9a1277c5cd5a7145e0295e3e190374e1/reports/source-repository-retirement-20261008)
records the original repos, revisions, exact central paths and unresolved paper
bindings. The 65 selected Standard-Pipeline inference bundles reuse existing
central model paths. Hardware profiling directories use `profiles/runs/archive/standard-pipeline/`.

| Task | Historical asset | Central path |
|---|---|---|
| MIKASA | PPO teacher | `benchmark-models/zero-latency/mikasa-intercept-grab-fast/small-policy/ppo-v1/` |
| MIKASA | Published teacher rollouts | `benchmark-datasets/zero-latency/mikasa-intercept-grab-fast/teacher-rollouts-v1/` |
| Balance Simple | Official GR00T 10k bundle | `benchmark-models/zero-latency/humanoidbench-balance-simple/vla/gr00t-n1.7-10k-v1/` |
| Balance Simple | Published dataset export | `benchmark-datasets/zero-latency/humanoidbench-balance-simple/teacher-rollouts-v1/` |
| Balance Simple | Scene assets | `benchmark-datasets/assets/humanoidbench/g1-sonic/sonic_g1/` |

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
[the verified layout revision](https://huggingface.co/MLL-Lab/LAGEN-models/tree/af8d7bfacbb03b59b5b5eabbed331577dd5fa690/latency-aware/mikasa-intercept-grab-fast/vla).

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

The [experiment consolidation report](../outputs/hf_cleanup/experiment-repos-20261007/REPORT.md) records actual repository counts, immutable source revisions, native-loader checks and retained gaps. Remaining jitter, image-history, mixed-task and profile assets require separate ownership checks.

The [maintained experiment model and dataset cards](hf_cards/README.md) provide root YAML metadata, named dataset configurations, schema, row counts and immutable download examples for the five experiment archives.

## Appendix factor sweeps

The [observation-stride](https://huggingface.co/datasets/MLL-Lab/LAGEN-datasets/tree/main/experiments/observation-stride) and [context-window](https://huggingface.co/datasets/MLL-Lab/LAGEN-datasets/tree/main/experiments/context-window) experiments use the main repositories. Their task manifests pin 459 archived runs and 468 retained checkpoint candidates. Original evaluations, action traces, videos and training records live in benchmark-datasets.

Nine Deadly evaluations bind step 48840 through complete action traces. Four other Deadly paper weights at step 48836 and one Demon C=12 uniform-delay seed-14 weight remain missing. The task manifests preserve those gaps and the published measurements. Retained alternate weights are not substitutes.

The two factor-sweeps source repositories were retired after complete file identity, loading, reading and navigation checks. The cleanup also removes two verified Ghost-data duplicates and 23 unreferenced checkpoint versions. The 24 referenced candidate versions and other independent source assets remain available. See the [migration report](../outputs/hf_cleanup/factor-unification-20261007/REPORT.md).

## Task transfer release

The [task-transfer catalogue](https://huggingface.co/datasets/MLL-Lab/LAGEN-datasets/tree/e983a3a72e042c6fab5575b847cdf6f4432cddaa/experiments/task-transfer) contains two formal Demon Attack policies: zero-latency step 7000 and mixed-latency step 5000. Both are stored in benchmark-models. The former root task-transfer dataset archive was retired on 2026-10-08. Formal Demon Attack data and current four-task evaluations remain at their task-owned paths in benchmark-datasets.

The task manifest retains exact source revisions and checkpoint identities. It records the 5812 versus 5832.25 paper/evaluation difference, the raw-frame versus latency-label protocol difference, and missing mixed-policy training metadata. These gaps do not change the user's formal/archive classification. Migration and source-retirement evidence is in [the task-transfer report](../outputs/hf_cleanup/task-transfer-20261007/REPORT.md).
