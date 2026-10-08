# Models, datasets, and latency profiles

Public artifacts use three Hugging Face repositories:

| Resource | Repository | Contents |
| --- | --- | --- |
| Models | [LAGEN-models](https://huggingface.co/MLL-Lab/LAGEN-models) | Inference weights, model configuration, normalization, task contracts, provenance |
| Data | [LAGEN-datasets](https://huggingface.co/datasets/MLL-Lab/LAGEN-datasets) | Training data, evaluation records, experiment manifests, paper inputs |
| Profiles | [LAGEN-profiles](https://huggingface.co/datasets/MLL-Lab/LAGEN-profiles) | Measured traces, fitted profiles, calibration records |

The [LAGEN collection](https://huggingface.co/collections/MLL-Lab/lagen-6ac750f21139008b8114fff5)
groups these repositories. The [maintained cards](hf_cards/README.md) define
repository navigation and dataset selections.

## Select an artifact

Use an experiment manifest to select the exact repository, revision, and path.
A task prefix alone does not identify the evaluated checkpoint.
Preserve the model's observation, action, normalization, latency, seed, and FPS
settings. Training completion, evaluation, acceptance, and publication are separate states.

| Artifact | Path pattern within its repository |
| --- | --- |
| Teacher | `zero-latency/<task>/small-policy/<teacher-version>/` |
| Profile-trained teacher | `latency-aware/<task>/small-policy/<teacher-version>/` |
| Shared task dataset | `<condition>/<task>/teacher-rollouts-h<horizon>-state-<version>/` |
| StarVLA bundle | `<condition>/<task>/vla/starvla-<model>-h<horizon>/<run-id>/` |
| Balance DAgger data | `<condition>/<task>/dagger/starvla-<model>-h<horizon>/<run-id>/round-NN/` |
| Deployment profile | `profiles/<model>/<gpu-deployment>/<workload>/<physical-instance>/` |

Conditions include `zero-latency` and `latency-aware`.
Model keys are `qwenoft`, `qwenpi_v3`, `qwengr00t`, and `wanoft`.
Current Balance H8 datasets use `v2`. Other shared task datasets use `v1`.
Some original bundles use older path names. Follow their manifests rather than
constructing a path from this table.

The [model inventory](https://huggingface.co/MLL-Lab/LAGEN-models/blob/main/reports/benchmark-release/README.md)
and [dataset inventory](https://huggingface.co/datasets/MLL-Lab/LAGEN-datasets/blob/main/reports/benchmark-release/README.md)
provide task-level entries. The
[source index](https://huggingface.co/datasets/MLL-Lab/LAGEN-datasets/tree/a6e93d4a9a1277c5cd5a7145e0295e3e190374e1/reports/source-repository-retirement-20261008)
records original repositories, revisions, and canonical paths.

Model releases contain inference weights and necessary metadata. Optimizer states
and training recovery files are not part of the published inference bundles.
Original configurations and evaluation records remain experiment evidence.
Use `--no-publish` for local task training. Formal task runs otherwise publish
automatically. Profiling uploads only with `--publish`.

## Paper experiments

| Experiment | Settings, results, and fixed inputs |
| --- | --- |
| Latency degradation | [Figure 2 catalogue](https://huggingface.co/datasets/MLL-Lab/LAGEN-datasets/tree/d1014aaaa1dd7f40ae21c60070dc0659fa265422/experiments/latency-degradation) |
| Sim2Real calibration | [30-case release](https://huggingface.co/datasets/MLL-Lab/LAGEN-datasets/tree/50d8f0aef64749a849760fc580ce3efb3c5c5e76/experiments/sim2real-calibration) |
| Visual history | [visual-history](https://huggingface.co/datasets/MLL-Lab/LAGEN-datasets/tree/e0f4007f504978f93c32557ea4e9b97fdf62fcf5/experiments/visual-history) |
| Latency in prompt | [latency-in-prompt](https://huggingface.co/datasets/MLL-Lab/LAGEN-datasets/tree/e0f4007f504978f93c32557ea4e9b97fdf62fcf5/experiments/latency-in-prompt) |
| Latency transfer | [latency-transfer](https://huggingface.co/datasets/MLL-Lab/LAGEN-datasets/tree/e0f4007f504978f93c32557ea4e9b97fdf62fcf5/experiments/latency-transfer) |
| Mean vs. profile training | [mean-vs-profile](https://huggingface.co/datasets/MLL-Lab/LAGEN-datasets/tree/e0f4007f504978f93c32557ea4e9b97fdf62fcf5/experiments/mean-vs-profile) |
| Task transfer | [task-transfer](https://huggingface.co/datasets/MLL-Lab/LAGEN-datasets/tree/e983a3a72e042c6fab5575b847cdf6f4432cddaa/experiments/task-transfer) |
| Observation stride | [observation-stride](https://huggingface.co/datasets/MLL-Lab/LAGEN-datasets/tree/main/experiments/observation-stride) |
| Context window | [context-window](https://huggingface.co/datasets/MLL-Lab/LAGEN-datasets/tree/main/experiments/context-window) |
| Fine-tuning scope | [vla-finetuning-scope](https://huggingface.co/datasets/MLL-Lab/LAGEN-datasets/tree/main/experiments/vla-finetuning-scope) |

Root pages provide navigation. Task manifests identify detailed settings and
immutable inputs. Historical source labels OpenVLA, Pi05, and GR00T can identify
StarVLA implementations `qwenoft`, `qwenpi_v3`, and `qwengr00t`. Check each
manifest's model configuration before comparing architectures.

## Task protocols and source limits

- [Balance](humanoidbench/cookbook.md): the current shared runner needs an H8
  teacher and its training configuration. The default H8 teacher download path
  is unavailable. Published H1 teachers do not satisfy that contract.
- [MIKASA](mikasa/cookbook.md): distinguish image-only H1, state-based H1,
  unconditioned H8, and conditioned H8. Published 71D conditioned bundles do not
  use the current 63D v3 input. The 63D transfer matrix is a development snapshot.
- [Profile PPO](mikasa/latency_ppo.md): fixed model and profile revisions identify
  the four MIKASA teacher experiments.
- [Demon Attack](demon_attack.md): score the full game across life losses with
  the declared frame budget. A first-life return is not the full-game return.

Model revision
`95aed229d8a90c0146885eb8f496518e98625d5f` contains the 18 hist8 game VLAs.
They use eight-decision input history
and H1 actions. Their 27 evaluation conditions each contain 100 episodes.
This hist8 input is distinct from an H8 output chunk.
Nine earlier profile-trained small policies did not enforce the evaluation
worker-capacity contract. Their old scores are not valid matched-protocol baselines.

The Standard-Pipeline import contains 37 H1 VLAs and 28 small policies across
seven Gymnasium tasks at model revision `ad37dff2ec7ea0adea072024330d19b5e7099e21`.
It has no profile-trained HalfCheetah GR00T/Pi0.5 VLA or profile-trained Humanoid VLA.
Seven zero-trained QwenOFT bundles lack standalone manifests and latency prompt maps.
Their available contracts come from the archived full configurations.

Figure 2 source records identify six game weights and six reused MIKASA weights.
Flappy GR00T step 3000 is missing. Two historical OpenVLA curve-to-checkpoint
bindings remain unresolved. Original training-data revisions for those game runs
are also missing. Keep these limits with the
[published source bindings](https://huggingface.co/MLL-Lab/LAGEN-models/blob/main/reports/latency-degradation-20261007/README.md).
Other selected checkpoints cannot substitute for these weights.

The factor-sweep manifests retain four missing Deadly Corridor step-48836 weights
and one missing Demon C=12 uniform-delay seed-14 weight as explicit gaps.
Nine Deadly evaluations instead bind step 48840 through complete action traces.
Task-transfer manifests retain the 5812 versus 5832.25 paper/evaluation difference,
the raw-frame versus latency-label difference, and missing mixed-policy training metadata.

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
backbone before training and evaluation. They use the standard Hugging Face
cache and record its resolved snapshot revision. Some task recipes select a default branch.
For reproduction, select the recorded revision explicitly.

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
