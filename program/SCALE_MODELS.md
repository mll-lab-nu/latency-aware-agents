# StarVLA H1 task models

> 2026-09-14 核查：当前产物、具体评测数值和未核验事项见 [PROGRESS](PROGRESS.md)；下文旧运行状态按原日期阅读。

Four models share each task's teacher bootstrap: `qwenoft`, `qwenpi_v3`,
`qwengr00t`, and `wanoft`. All receive RGB plus continuous state and train
one current-step action. Qwen models start from Qwen3-VL-4B-Instruct; WanOFT
starts from Wan2.2-TI2V-5B-Diffusers. Task heads start fresh.

| Task directory | State | Action | VLA Hz | Formal training |
| --- | ---: | ---: | ---: | --- |
| humanoidbench_balance_simple | 198 | 43 absolute joint targets | 50 | 15k SFT updates |
| extreme_parkour_go2 | 53 | 32 terrain latent + 2 yaw | 10 | 5k SFT + 10 × 500 DAgger |
| mikasa_intercept_grab_fast | 7 | 7 manipulation controls | 20 | 5k SFT updates |
| haic_pull_cart | 605 | 256 privileged latent | 50 | 5k SFT + 10 × 500 DAgger |

Parkour holds one command for five control steps; its frozen actor receives
fresh observations at every 50 Hz step. HAIC produces a new latent every
50 Hz step. The native regression/flow loss is combined with the existing
frozen actor action objective for Parkour and HAIC. MIKASA uses top and wrist
views; Wan tiles the two simultaneous views into one image.

## Run

Use the benchmark Python environment to launch the pipeline. By default it
resolves the `starvla_rl_games_openvla` conda environment for training and
inference, and each task's existing simulator environment. Those environments
and simulator assets must be installed using [QUICK_START.md](../docs/QUICK_START.md),
which covers all interpreter environments and the separate MIKASA converters.

```bash
# Substitute any task directory and one of the four model names.
bash scripts/tasks/humanoidbench_balance_simple/train_qwenoft.sh --smoke
bash scripts/tasks/haic_pull_cart/train_wanoft.sh --run-id experiment-01
```

Each of the four task directories contains `train_qwenoft.sh`,
`train_qwenpi_v3.sh`, `train_qwengr00t.sh`, and `train_wanoft.sh`. The shared
entry point is `python -m scripts.starvla.train_task`.

Useful overrides: `--teacher-checkpoint`, `--dataset-dir` (containing
`lerobot/` and `lerobot__val/`), `--output-dir`, `--vla-python`, `--task-python`,
`--num-gpus`, `--micro-batch`, `--simulator-device`, and `--inference-device`.
Defaults use GPU 0 for simulation and GPU 1 for inference; training uses
Accelerate on two visible GPUs. Formal global batch sizes are 128 for all four
tasks. Choose GPU count and microbatch whose
product divides the global batch size.

Training starts at microbatch 64 per GPU. On an explicit CUDA OOM the launcher
restarts at 32, 16, 8, 4, 2, then 1, increasing gradient accumulation to preserve
the global batch. Other errors propagate unchanged. Retries retain their OOM
logs and restore the latest complete training state when available. The resolved
microbatch and accumulation are recorded in the training YAML, W&B, and run metadata.
For online tracking, launch with `WANDB_MODE=online`.

The pipeline downloads teachers and backbones from their repositories' default
branches, reuses published bootstrap data or collects it once, trains, saves a reloadable inference
bundle, and evaluates the final checkpoint for 100 simulated zero-latency
episodes. Simulation cadence does not imply real-time inference throughput.
`--smoke` uses two optimizer updates, two final evaluation episodes, and one
short DAgger round for Parkour/HAIC; smoke artifacts remain local.

Balance and MIKASA bootstrap preparation reuses the published canonical
`teacher-rollouts-v1` train/validation split from the dataset repository's
default branch, adding the H1 task contract and normalization metadata. HAIC
and Parkour use their native collectors. HAIC
uses different seeds for training (0) and validation (1).

For DAgger, half the sampling weight remains on bootstrap and half is split
across all accumulated model-specific rounds. The validation split stays fixed
and separate. Each new round saves only its new rows. Optimizer state continues
across DAgger rounds with one cosine schedule over the complete DAgger phase.

## Publication

Formal runs automatically publish bootstrap/DAgger data to
`placeholder/benchmark-datasets`, then the final inference bundle
to `placeholder/benchmark-models` after evaluation completes.
Authenticate through the standard Hugging Face login or `HF_TOKEN`.

- Shared data: `zero-latency/<task>/teacher-rollouts-h1-state-v1/`.
- DAgger delta: `zero-latency/<task>/dagger/starvla-<model>-h1/<run-id>/round-NN/`.
- Model: `zero-latency/<task>/vla/starvla-<model>-h1/<run-id>/`.

Task components in Hub paths use hyphens. Model bundles include weights,
portable model config, task contract, normalization statistics, final L0
metrics, model card, and provenance with source references and resolved
publication revisions where applicable. Optimizer states and raw collection
intermediates remain local.
Use a new run ID for new training. Dataset contract changes require a new
dataset version, rather than replacing a shared version.
Publication refuses to replace an existing central path; rerun after an
interrupted upload only when the failed upload left no committed target.
Publication pins the repository head while checking and uploading, so a
concurrent repository change fails and must be retried.

If upload fails after training, rerun publication without retraining:

```bash
bash scripts/tasks/haic_pull_cart/train_wanoft.sh \
  --output-dir outputs/starvla/runs/haic_pull_cart/wanoft/experiment-01 \
  --publish-only
```

## Smoke acceptance — acceptance-v1

All 16 combinations completed two real bootstrap optimizer updates, checkpoint
save/reload, and two simulated L0 episodes. Each HAIC/Parkour model also collected
one four-row DAgger delta and completed two additional updates: 48 optimizer
updates, 32 evaluation episodes, and eight DAgger rounds in total. Every final
bundle uses H1 with zero future window and the task's actual state/action dimensions.

Loss pairs below are update 1 → update 2 on different sampled batches. These are
pipeline checks, not convergence results. Model links open the final L0 metrics.

| Task | Model | Global batch | Bootstrap loss | DAgger loss | Mean return | Mean control steps |
| --- | --- | ---: | --- | --- | ---: | ---: |
| Balance | [qwenoft](../outputs/starvla/smoke/humanoidbench_balance_simple/qwenoft/acceptance-v1/inference/eval_l0.json) | 1 | 0.8594 → 1.5730 | — | 8.333708 | 18.0 |
| Balance | [qwenpi_v3](../outputs/starvla/smoke/humanoidbench_balance_simple/qwenpi_v3/acceptance-v1/inference/eval_l0.json) | 2 | 1.9882 → 2.5331 | — | 9.407859 | 15.5 |
| Balance | [qwengr00t](../outputs/starvla/smoke/humanoidbench_balance_simple/qwengr00t/acceptance-v1/inference/eval_l0.json) | 1 | 1.4549 → 1.4063 | — | 13.156083 | 20.5 |
| Balance | [wanoft](../outputs/starvla/smoke/humanoidbench_balance_simple/wanoft/acceptance-v1/inference/eval_l0.json) | 2 | 0.7306 → 2.3201 | — | 8.792190 | 14.0 |
| Parkour | [qwenoft](../outputs/starvla/smoke/extreme_parkour_go2/qwenoft/acceptance-v1/inference/eval_l0.json) | 2 | 2.8331 → 4.4347 | 3.2652 → 4.1622 | 0.175204 | 125.0 |
| Parkour | [qwenpi_v3](../outputs/starvla/smoke/extreme_parkour_go2/qwenpi_v3/acceptance-v1/inference/eval_l0.json) | 2 | 4.0751 → 6.9355 | 4.3338 → 6.9008 | 2.374324 | 785.5 |
| Parkour | [qwengr00t](../outputs/starvla/smoke/extreme_parkour_go2/qwengr00t/acceptance-v1/inference/eval_l0.json) | 2 | 3.1394 → 3.0817 | 3.3233 → 3.4163 | 4.429395 | 916.0 |
| Parkour | [wanoft](../outputs/starvla/smoke/extreme_parkour_go2/wanoft/acceptance-v1/inference/eval_l0.json) | 2 | 4.0900 → 5.2002 | 4.3198 → 5.9520 | 0.000108 | 1500.0 |
| MIKASA | [qwenoft](../outputs/starvla/smoke/mikasa_intercept_grab_fast/qwenoft/acceptance-v1/inference/eval_l0.json) | 2 | 0.6603 → 3.8269 | — | 0.086313 | 60.0 |
| MIKASA | [qwenpi_v3](../outputs/starvla/smoke/mikasa_intercept_grab_fast/qwenpi_v3/acceptance-v1/inference/eval_l0.json) | 2 | 2.5481 → 21.5782 | — | 0.043818 | 60.0 |
| MIKASA | [qwengr00t](../outputs/starvla/smoke/mikasa_intercept_grab_fast/qwengr00t/acceptance-v1/inference/eval_l0.json) | 2 | 1.6387 → 1.5250 | — | 0.214943 | 60.0 |
| MIKASA | [wanoft](../outputs/starvla/smoke/mikasa_intercept_grab_fast/wanoft/acceptance-v1/inference/eval_l0.json) | 2 | 0.4896 → 3.5766 | — | 0.119566 | 60.0 |
| HAIC | [qwenoft](../outputs/starvla/smoke/haic_pull_cart/qwenoft/acceptance-v1/inference/eval_l0.json) | 2 | 0.9108 → 0.9136 | 0.6358 → 0.9066 | 4.174161 | 60.0 |
| HAIC | [qwenpi_v3](../outputs/starvla/smoke/haic_pull_cart/qwenpi_v3/acceptance-v1/inference/eval_l0.json) | 2 | 1.3454 → 1.8257 | 0.9890 → 2.2237 | 2.607474 | 49.5 |
| HAIC | [qwengr00t](../outputs/starvla/smoke/haic_pull_cart/qwengr00t/acceptance-v1/inference/eval_l0.json) | 2 | 1.2720 → 1.2583 | 1.1536 → 1.1946 | 3.675855 | 58.5 |
| HAIC | [wanoft](../outputs/starvla/smoke/haic_pull_cart/wanoft/acceptance-v1/inference/eval_l0.json) | 2 | 0.8587 → 1.2328 | 0.6185 → 1.7559 | 3.033967 | 55.0 |

All 32 episodes reported zero simulated latency and zero invalid actions.
Parkour issued one VLA command per five control steps, including the final partial
block; HAIC issued one command every control step. Each model used the same fixed
bootstrap train/validation split for its task:

| Task | Train / validation episodes | Train / validation rows |
| --- | ---: | ---: |
| Balance | 90 / 10 | 86,048 / 9,736 |
| Parkour | 2 / 1 | 161 / 94 |
| MIKASA | 2 / 1 | 70 / 38 |
| HAIC | 2 / 1 | 259 / 76 |

Source episode/shard identities and HAIC seeds 0/1 were checked for disjointness.
The HAIC QwenOFT final checkpoint was also rechecked on validation seed 1;
its held-out action loss was 1.391366, recorded in
[`validation_seed1_recheck.json`](../outputs/starvla/smoke/haic_pull_cart/qwenoft/acceptance-v1/validation_seed1_recheck.json).
The published canonical Balance/MIKASA bootstrap download-and-prepare path also
passed in separate directories.

Run outputs are under `outputs/starvla/smoke/<task-directory>/<model>/acceptance-v1/`.
Each contains `bootstrap.log`, the exact training YAML, `inference/checkpoints/model.pt`,
and `inference/eval_l0.json`; DAgger tasks additionally contain `dagger.log` and
`dagger/round-00/lerobot/`. The [machine-readable report](../outputs/starvla/smoke/acceptance-report.json)
lists every checkpoint, metrics path, update count, and unrounded loss.

The smoke command for each model was:

```bash
python -m scripts.starvla.train_task --task parkour --model qwenoft \
  --smoke --dataset-dir outputs/starvla/smoke_data/parkour --run-id acceptance-v1
```

Substitute `balance`, `parkour`, `mikasa`, or `haic` and any of the four model names.
Use a fresh run ID when rerunning. Balance QwenOFT/QwenGR00T used `--num-gpus 1`;
all other rows used two GPUs, with microbatch 1.

Final verification passed 91 task/model/data/trainer/publication regression tests,
18 HAIC tests, three MIKASA adapter tests, and four inference-device tests.
All 16 wrappers passed `bash -n`. QwenOFT and WanOFT inference files were copied
to fresh directories on another filesystem, loaded with the real model loader
and downloaded backbones, and produced finite `(1, 43)` predictions.

Publication filtering, revision references, no-overwrite behavior, upload failures,
and publication-only retry were tested with mocked Hub responses. No smoke
artifacts were uploaded, and no formal long training was started.
