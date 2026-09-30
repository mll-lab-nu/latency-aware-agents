# Flappy sim2real fit-budget study

This pilot measures how OFT and GR00T return calibration changes with the number
of independent fitting sessions. Real collection runs on the profiling dual
RTX 3090 in `servers.md`. Formal simulation runs on a different host;
the profiling instance stops after collection, profile fitting and verified backup.
The code entry is `scripts.experiments.sim2real.budget`.

## Frozen protocol

Each model has 55 real runs, two episodes/run. Reuse original runs 0–9; collect
runs 10–54 with fresh evaluator/model processes and the original warmup. Keep the
original checkpoint, calibrated clocks, frame/action conventions and 3,600-step
horizon. Environment seed is `2000000 + 20 * run_id + episode_in_run`.

| Model | Physical GPU | Frozen env/obs FPS | Zero-latency reference |
| --- | --- | ---: | ---: |
| OFT | 0, UUID `GPU-9091e431-9a0b-168d-a5d2-b5053628d519` | 10.490762184435336 | 439.13 |
| GR00T | 1, UUID `GPU-6a1c5dc6-6a38-ff30-cdec-1121a23df467` | 10.783693488119345 | 428.48 |

The source refs and immutable HF revisions live in `configs/experiments/sim2real/budget.yaml`; the
normalizers are the frozen `paper/exps/sim2real_heldout/references.csv` values.
The campaign-root `budget.json` captures those inputs and the complete split lists before new
results are observed. Source traces are retained under each model's `source/`.

Fit sizes: **5, 10, 20, 30, 40, 50**. Methods: Fixed, Normal, IID, Temporal.

- **Original**: valid runs 5–9; fit order 0–4 followed by 10–54.
- **Random**: shuffle IDs 0–54 with `random.Random(20260929)`; first five are
  validation, remaining fifty form the nested fit order. Validation IDs are
  **2, 12, 16, 26, 48**. One partition, shared across the two models.
- Each partition holds validation fixed across all six fit sizes. Runs remain
  intact: no latency-record or episode-level shuffling. Each profile sees only
  its partition's fitting runs. Each split is a separate sensitivity analysis;
  original validation data can participate in the random branch's training.

New real work is **180 episodes**. Simulation uses a fixed **5 runs × 2 episodes**
per model, split, fit size and method, with environment seeds matching that split's
validation runs. Total: **960 episodes** (480/model). Each fitted profile produces
one simulation mean, compared with both the mean of all its real fitting runs
and the mean of its five real validation runs. The simulation sample size stays
fixed while fitting data increases. The random analysis uses the complete
55-run collection pool, even at fit size 5.

## Parallel replay and completion

The profiling stage order is verification → real → fit → verified backup → stop.
Formal sim runs on the independent RTX PRO host, OFT on GPU 2 and GR00T on GPU 3.
The profiling runner still ends after fitting. The sim CLI accepts an explicit
`--gpu-uuid`, records a fresh execution contract and requires GPU verification.
Each collection stage finishes on both GPUs before proceeding.
A resident model occupies each card during sim; up to eight run slots share
batched inference (batch cap eight). Each formal evaluation has five active run
slots. Simulated capacity remains **one per slot**.

Each slot runs a complete two-episode session in order. Temporal keeps the same
sampler across those episodes and gets a fresh sampler for the next run. IID
and Normal keep their existing episode-seeded semantics. Original validation
runs use latency seeds 0–4; all other runs use `1000 + run_id`. Temporal retains
the original replica-zero seed derivation. Physical slot assignment never
selects a logical seed. GR00T simulated inference explicitly seeds each observation's
initial diffusion noise with `derive_seed(episode_env_seed, stream=raw_env_step)`;
`policy.action_noise_seeded` enables that path only for this sim. Real collection
keeps the archived global run RNG. The old sim adapter did not forward action
noise seeds to the GR00T head, so its global RNG depended on batch scheduling.
GPU environment rewards retain the core's float64 precision, matching the CPU
reward rather than rounding every step to float32. Policy history, environment, action queue, and worker
capacity reset at episode boundaries as in the original evaluator.

Explicit logical episode IDs are `20 * run_id + episode_in_run`. Raw simulation
traces retain these IDs, and episode metadata records both `run_id` and
`episode_in_run`. Run means, rather than individual episodes, are the statistical
unit. Changing the number of slots does not change this grouping.

A complete run has its two episode metrics and raw traces written before `DONE`.
Each new real or simulated run is published before it is considered delivered.
On the RTX PRO replay host, `--defer-uploads` keeps computation independent of
HF commit rate limits: full runs persist immediately and queue for batched
publication. Pending stage manifests are retained until actual upload receipts
are saved; queued computation is not reported as uploaded.
Publication and computation completion are separate; resume republishes completed
local runs if needed and reruns only incomplete runs. Profiles are partitioned
by split/fit count. Result publication uses a new
`sim2real-budget/flappy/<campaign>/<model>/` namespace; historical artifacts remain
available at their original revisions.

The actual-GPU verification replays eight two-episode sessions for all four
methods at a 64-step cap with one and eight slots. It compares actions, sampled
latencies and returns. A mismatch writes `verification/result.json` and fails;
formal sim requires the verification marker. This is a short numerical check,
not proof that every possible long episode is bitwise invariant to GPU batching.
OOM propagates; it never silently lowers parallelism.

## Execution and recovery

Run from the repository root. Invoke the selected model and stage explicitly:

```bash
python -m scripts.experiments.sim2real.budget --stage verify --model qwenoft \
  --config configs/experiments/sim2real/budget.yaml \
  --root /path/to/campaign --worker-python /path/to/model/python --gpu-uuid GPU_UUID
python -m scripts.experiments.sim2real.budget --stage real --model qwenoft \
  --config configs/experiments/sim2real/budget.yaml \
  --root /path/to/campaign --worker-python /path/to/model/python --gpu-uuid GPU_UUID
python -m scripts.experiments.sim2real.budget --stage fit --model qwenoft \
  --config configs/experiments/sim2real/budget.yaml \
  --root /path/to/campaign --gpu-uuid GPU_UUID
python -m scripts.experiments.sim2real.budget --stage sim --model qwenoft \
  --config configs/experiments/sim2real/budget.yaml \
  --root /path/to/campaign --worker-python /path/to/model/python --gpu-uuid GPU_UUID
```

Run the same stages for `qwengr00t` with its allocated device. Both models must
finish real acquisition before simulation. Reuse the same campaign root and
frozen protocol to resume; restore published artifacts before resuming a lost
local directory. Each stage uses its completion marker and exact publication
revision. `PARALLEL_VERIFIED`, `REAL_DONE`, `SIM_DONE`, and `REPORT_DONE` are
separate states. Incomplete runs are replayed; completed runs are reused.

## Analysis and interpretation

For each case, normalized absolute error is
`100 * abs(mean(sim run means) - mean(real run means)) / reference`.
The two-case NMAE weights OFT and GR00T equally. Both curves use the same sim
mean: the fit-reference curve uses all real fitting runs at that size, while
the validation-reference curve uses the five held-out real runs. The fit curve
is a mean-calibration diagnostic, not a paired training-rollout error or a
return loss optimized during fitting.

Bootstrap uses 2,000 resamples of run means. Validation resamples matching
real/sim run pairs. The fit-reference comparison independently resamples its N
real fitting runs and the five simulated validation-seed runs. The same simulation
resamples feed both curves. Draws match across the two models because they share
environment seeds and partitions, and across methods and fit sizes. Intervals
condition on the fitted profile and the frozen reference; they exclude profile
refitting and reference uncertainty.

`report/` contains per-case signed/absolute errors, two-case NMAE, per-run real
latency tails/quantiles/burst dwell lengths, measured costs, and descriptive
stability decisions. After both models complete, export exactly two curve figures
in PNG and PDF: `valid_original` for the fixed original validation runs, and
`valid_random` for the single fixed random validation partition. Each figure has
fit-run count (5, 10, 20, 30, 40, 50) on the x-axis and validation-reference NMAE
(pp) on the y-axis, with four curves: Fixed, Normal, IID, Temporal. OFT and GR00T
receive equal weights. The user confirmed NMAE; their initial NMSE wording was
a typo. `cumulative_costs.csv` separates the full
55-run collection cost, original-prefix equivalent cost, and cumulative replay
cost. Random-split results require the complete collected pool. The latency diagnostics preserve session
boundaries and distinguish old/new acquisition runs.

A descriptive plateau candidate is the first size below 50 for which every
remaining point's absolute error and normalized sim mean stay within 2 pp of
the 50-run point. A validation signed-error interval half-width above 2 pp
prevents treating that candidate as a supported budget. This heuristic is not a
significance test. Five validation runs and one random partition limit precision;
Flappy estimates do not establish the required budget on other tasks.

Tests are retained in the existing cross-device and executor suites. They cover
split isolation, seed/session invariance across slot counts, input counts for all
profiles, the fixed 960-episode schedule, complete-run recovery and reuse of the
same sim mean in both NMAE curves. GPU verification supplements
those CPU contracts.


## Five random validation draws

The follow-up keeps seed 20260929 as draw one and adds seeds 20260930, 20261001,
20261002, and 20261003. Each seed chooses five validation runs. The original
20260929 shuffled priority is fixed, with each draw's valid runs removed before
selecting nested fit prefixes. The first draw therefore stays exactly reusable.
New profiles/sims use a separate campaign and require 1,920 new episodes.
GPU 2/3 use ten run slots per model, combining settings to fill all ten slots.
The updated second figure shows four method means across five splits with
± one sample SD; OFT/GR00T retain equal weight within each split.
See [the frozen draw list and execution record](logs/sim2real-budget-random5-20260930.md).

## Timing 记录与报告

单 setting 成本记录在 timing JSON 中写明 model、stage 及该阶段的 run_ids、split、
fit_runs、method。成本报告读取这些字段，文件路径仅作为来源位置记录。
报告直接读取 campaign 根目录的 `budget.json`，不重新读取当前实验 YAML 或代码常量。
已有独立 timing 产物通过 `--timing-root` 显式传入：

```bash
python -m scripts.experiments.sim2real.budget_report --root /path/to/campaign \
  --timing-root /path/to/timing-artifacts --defer-uploads
```

成本定义继续区分新采集成本、原始 prefix 等价成本与逐预算累计 replay 成本。

## Repeated splits and 28/27 holdout

The extension recipes are `configs/experiments/sim2real/repeats.yaml` and
`configs/experiments/sim2real/holdout.yaml`. Both prepare stages read the frozen
base campaign's `budget.json`, retain its source revisions, seeds and reference
returns, and record a complete `repeats.json` or `holdout.json` in the new root.
Subsequent fit, simulation and reporting use that frozen manifest.

```bash
python -m scripts.experiments.sim2real.budget_repeats --stage prepare \
  --config configs/experiments/sim2real/repeats.yaml \
  --root /path/to/repeats --base-root /path/to/campaign
python -m scripts.experiments.sim2real.budget_repeats --stage fit \
  --config configs/experiments/sim2real/repeats.yaml \
  --root /path/to/repeats --base-root /path/to/campaign --model qwenoft
python -m scripts.experiments.sim2real.budget_repeats --stage sim \
  --config configs/experiments/sim2real/repeats.yaml \
  --root /path/to/repeats --base-root /path/to/campaign --model qwenoft --gpu-uuid GPU_UUID
python -m scripts.experiments.sim2real.budget_repeats --stage report \
  --config configs/experiments/sim2real/repeats.yaml \
  --root /path/to/repeats --base-root /path/to/campaign

python -m scripts.experiments.sim2real.budget_holdout --stage prepare \
  --config configs/experiments/sim2real/holdout.yaml \
  --root /path/to/holdout --base-root /path/to/campaign
python -m scripts.experiments.sim2real.budget_holdout --stage fit \
  --config configs/experiments/sim2real/holdout.yaml \
  --root /path/to/holdout --model qwenoft
python -m scripts.experiments.sim2real.budget_holdout --stage sim \
  --config configs/experiments/sim2real/holdout.yaml \
  --root /path/to/holdout --model qwenoft --gpu-uuid GPU_UUID
python -m scripts.experiments.sim2real.budget_holdout --stage report \
  --config configs/experiments/sim2real/holdout.yaml --root /path/to/holdout
```

Run fit/sim for each model recorded by the base campaign. Extension stages queue
publication receipts using the existing deferred-publication protocol.
The `--config` recipe initializes a new experiment; existing manifests remain
authoritative on resume. Original historical manifests and reports remain
unchanged and can be reproduced with their original source snapshot; these
entrypoints require the explicit frozen fields documented above.
