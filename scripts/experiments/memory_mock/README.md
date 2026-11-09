# CUDA memory mock

This standalone background workload holds two equally sized CUDA `uint8`
buffers. It can keep them idle or copy them back and forth to create GPU memory
bandwidth contention alongside an inference process. Both buffers remain
resident during OFF phases. It does not simulate model KV memory.

Use a CUDA-enabled PyTorch environment and run the commands from this checkout.
Set `MEMORY_MOCK_PY` to its Python executable if needed. The tool uses logical
`cuda:0`; bind the same physical GPU for inference and the load process with
`CUDA_VISIBLE_DEVICES`. It does not change GPU clocks, drivers or other processes.

## Full live profiling and comparison

For the Temporal/AR(1)/ARMA experiment, use **`profile.sh`**, which extends the
same live profiling recipe previously used for Temporal/HMM/block:

```bash
GPU=0 MIN_MEASUREMENT_SECONDS=1200 OUTPUT_DIR=/workspace/outputs/irregular_profiles_20min_run1 \
  bash scripts/experiments/memory_mock/profile.sh
```

One terminal is sufficient. The script installs/activates the existing StarVLA
environment, downloads the same pinned zero-latency-trained Flappy QwenOFT
checkpoint, and profiles inference at **30/30 FPS with one worker**. The GPU
copy process runs on that same physical GPU during each measured session.
Model warmup finishes before workload allocation; measured observations start
only after the workload reports ready. Both processes are supervised together.
The irregular timeline continues through episode resets and restarts with a
different load seed for each independent profiling session.

The default contention protocol runs five independent processes for **at least
20 minutes each** (`MIN_MEASUREMENT_SECONDS=1200`). Every session completes both
the time budget and at least `EPISODES=10` episodes, then stops at an episode
boundary. The clock starts after model warmup and workload readiness; it includes
episode resets and excludes final reporting. Short episodes therefore do not
end collection early. Expect at least 100 minutes of measurement across five
sessions, plus startup and fitting. The supervisor prints status every minute.

The 3,600-frame episode cap, fitting sessions 0/1/2, held-out sessions 3/4, load
seeds 42–46, checkpoint, FPS and episode-seed rule follow the original recipe.
Actual episode counts can differ across sessions. Set `MIN_MEASUREMENT_SECONDS=0`
to reproduce or resume the previous fixed-episode protocol. Use a **new output
directory** when switching from the short pilot to the longer protocol; changing
the budget cannot extend an already completed recording. Inspect actual ON
counts and short/long event coverage before interpreting burst fidelity. The comparison
warns when a session has fewer than ten observed ON events; this is a collection
diagnostic, not a statistical acceptance threshold.

After recording, it fits **Temporal, log-AR(1), and log-ARMA(1,1)** using the same
training sessions and exports normal reusable `profile.json` bundles. It then
compares ten independent generated sequences per method and held-out session.
The GPU workload runs during the live measurements; CPU fitting follows them.
No synthetic delay is injected into the measured inference timings.

For the allocation-matched idle control, use a separate output:

```bash
GPU=0 MEMORY_MOCK_CONDITION=idle \
  OUTPUT_DIR=/workspace/outputs/idle_profiles_run1 \
  bash scripts/experiments/memory_mock/profile.sh
```

Other controls follow the original recipe: `REPEATS`, `EPISODES`,
`MAX_ENV_STEPS`, `FIT_SESSION_IDS`, `EXISTING_RUN_DIR` and `SKIP_INSTALL`.
`METHODS=temporal,ar1,arma,hmm,block` includes the previous alternatives too.
Rerun with the same output and settings to resume completed sessions. A failed
session is restarted; incomplete sessions are never marked complete. Reusing an
uncontended recording as an irregular recording is rejected.

The load controls are `LOAD_SEED`, `BUFFER_MIB`, `OFF_MIN_SECONDS`,
`OFF_MAX_SECONDS`, `SHORT_ON_MIN_SECONDS`, `SHORT_ON_MAX_SECONDS`,
`LONG_ON_MIN_SECONDS`, `LONG_ON_MAX_SECONDS`, and `LONG_PROBABILITY`.
Defaults are two 512 MiB buffers, OFF 5–15 s, short ON 0.2–0.8 s, long ON 2–5 s,
and 20% long events. `LOAD_COVERAGE_SECONDS=10800` exceeds the default
`RUN_TIMEOUT_SECONDS=7200`; `STARTUP_TIMEOUT_SECONDS=1800` covers startup.
Keep coverage above the run timeout. These are declared synthetic workload
parameters; hardware slowdown must be measured.

After successful profiling, fitting and comparison, this entrypoint uploads the
complete bundle to the **dataset** repo
`latency-sensitive-bench/irregular-temporal-profiles`, under
`experiments/irregular-memory/`. It creates the repo as private if absent and
preserves an existing repo's visibility. Each run uses its own folder; existing
experiment folders are never overwritten. `upload_receipt.json` records the exact
HF commit and folder URL. The upload contains all fitted profiles, measurement
traces, workload schedules/logs, protocols and comparison results, without model
weights. Authenticate with a token that can create/write in the chosen namespace.

Set `HF_REPO` and `HF_PATH_PREFIX` to change the destination, `HF_CREATE_REPO=0`
to require an existing repo, or `SKIP_UPLOAD=1` to retain results locally. Rerun
with the same settings to retry an upload without repeating completed profiling.
No repository creation or upload happens before the local bundle is complete.
On the first run after upgrading, allow installation, or install
`.[profiling,analysis]` in the model
environment before using `SKIP_INSTALL=1`; fitting now requires statsmodels 0.14.6.

Outputs include:

- `profiles/{temporal,ar1,arma}/`: portable fitted models, using only sessions 0–2.
- `comparison/report.html`, `traces.png`: held-out timing diagnostics and sample traces.
- `comparison/metrics.csv`, `summary.json`, `sequences.npz`: per-session metrics,
  training-derived thresholds, generation variability and generated sequences.
- The raw run named in `measurement.json`: ordinary profiling sessions plus each
  session's `contention/` schedule, logs and ON/OFF report.
- `measurement_duration.json` in each timed session: requested duration, actual
  elapsed measurement seconds and number of completed episodes.
- `bundle/`: profiles, raw measurement evidence, protocol and comparison for transfer.

The comparison reports action and worker-service distributions, p95/p99 errors,
autocorrelation, slow-request frequency/persistence, and burst lengths in admitted
requests. Thresholds come exclusively from fitting data. ON/OFF labels enter only
the workload diagnostics, never a timing model. No automatic winner or policy
return comparison is produced. Generation SD describes repeated synthetic draws,
not uncertainty across new hardware deployments. Two held-out sessions support
an exploratory comparison only.

AR fitting uses standardized log busy time and log overhead (action latency minus
busy time). Each component has AR(1) or ARMA(1,1) coefficients shared across sessions.
The objective sums independent-session stationary Gaussian log-likelihoods with
three starts per component; failed optimizations are reported. Correlated Gaussian
innovations use paired filtering residuals to retain contemporaneous dependence.
Generated state begins in its joint stationary distribution and persists across
ordinary episodes. This avoids negative delays without clipping high latencies.
These are fixed-order baselines, not a search over all ARMA orders or a claim that
ARMA cannot represent bursts. The likelihood implementation uses
[statsmodels SARIMAX](https://www.statsmodels.org/v0.14.6/generated/statsmodels.tsa.statespace.sarimax.SARIMAX.html).

## Run a load

```bash
# Reserve two 512 MiB buffers, without continuous copy work.
CUDA_VISIBLE_DEVICES=0 bash scripts/experiments/memory_mock/run.sh run \
  --mode idle --output outputs/memory_mock/idle

# Keep the same allocation; alternate 5 s OFF / 5 s ON, starting OFF.
CUDA_VISIBLE_DEVICES=0 bash scripts/experiments/memory_mock/run.sh run \
  --mode periodic --output outputs/memory_mock/periodic
```

These commands run in the foreground until Ctrl-C or SIGTERM. Select a fresh
output directory for every run. `--buffer-mib` sets the size of **each** buffer
(default 512, total tensor storage 1 GiB); the CUDA context and allocator use
additional memory. Allocation errors fail the run instead of silently changing
its intensity. Use the same explicit size for compared conditions.

ON repeatedly submits two device-to-device copies followed by a CUDA
synchronization. Only one pair is in flight at a time; an already submitted pair
may complete past an ON boundary. OFF submits no copy work. The load timeline
continues independently of model episode resets. This is a concrete co-located
workload, not a calibrated claim about a particular percentage of GPU bandwidth.

## Measure irregular contention during a real evaluation

Use the same CUDA/StarVLA environment in which your ordinary realtime evaluation
already works. The launcher starts two processes on the selected physical GPU:
the evaluator and the memory workload. It waits for model warmup before
allocating the workload buffers, then waits for the workload to be ready before
starting measured episodes. It stops both process trees on failure or timeout.

Run these two conditions sequentially on an otherwise idle GPU, from the repo
root. Use a fresh output directory every time:

```bash
# Control: identical buffer allocation, no repeated copies.
CUDA_VISIBLE_DEVICES=0 MEMORY_MOCK_PY=python \
  bash scripts/experiments/memory_mock/measure.sh \
  --config configs/examples/flappy/profile_openvla.yaml \
  --condition idle --episodes 300 --episode-seed 42 --load-seed 42 \
  --buffer-mib 512 --output outputs/memory_mock/idle_s42

# Irregular, alternating quiet intervals and short/long copy events.
CUDA_VISIBLE_DEVICES=0 MEMORY_MOCK_PY=python \
  bash scripts/experiments/memory_mock/measure.sh \
  --config configs/examples/flappy/profile_openvla.yaml \
  --condition irregular --episodes 300 --episode-seed 42 --load-seed 42 \
  --buffer-mib 512 --output outputs/memory_mock/irregular_s42
```

`MEMORY_MOCK_PY` can be an absolute path to your environment's Python. The example
config selects the pinned **zero-latency-trained QwenOFT Flappy checkpoint at
30 FPS**; it is a profiling example, not the paper's Target/Real checkpoint.
Pass your prepared realtime evaluation YAML to measure a different checkpoint.
The launcher preserves the checkpoint, prompt, action history, environment FPS,
observation FPS and per-episode step limit in that YAML. It requires
`executor.mode: realtime` and exactly one inference device, logical `cuda:0`.
It overrides episode count/seed, disables suites and calibration sample stopping,
sets one environment, enables timing logs and disables video. The fully resolved
configuration is saved. Simulated execution cannot measure this workload's
effect on inference latency.

The irregular condition uses these declared synthetic assumptions:

| Phase | Duration | Probability per ON event |
| --- | --- | --- |
| OFF | Uniform 5–15 seconds | — |
| Short ON | Uniform 0.2–0.8 seconds | 80% |
| Long ON | Uniform 2–5 seconds | 20% |

OFF and ON alternate, starting OFF. The expected ON share is about 9.9% over a
long run: mean ON duration 1.1 s, mean OFF duration 10 s. The frozen seed-specific
schedule and measured duty cycle will differ. ON repeatedly copies the same two
buffers; OFF keeps them allocated. These timings represent intermittent shared
GPU activity but have not been calibrated against a deployment trace. They are
workload durations, not prescribed inference delays. Actual latency bursts must
be established from the GPU measurements.

Override the schedule with `--off-seconds MIN MAX`,
`--short-on-seconds MIN MAX`, `--long-on-seconds MIN MAX`, and
`--long-probability P`. `--condition periodic` retains the original 5 s OFF / 5 s
ON control. `--load-seed` affects only the background timeline;
`--episode-seed` controls the evaluation seed. Repeat full sessions with fresh
seeds and vary condition order when collecting evidence. Keep the allocation,
checkpoint and evaluation configuration fixed between conditions.

The default startup timeout is 1,800 s and measurement timeout is 7,200 s.
`--coverage-seconds` generates at least 7,200 s of schedule by default. If using
a longer `--run-timeout-seconds`, also increase schedule coverage beyond it.
An exhausted schedule fails the run. The load starts just before measured
episodes and continues through episode resets and final evaluator reporting;
allow some coverage beyond the expected total runtime. The standalone
`measure.sh` remains episode-count based by default. Add
`--min-measurement-seconds 1200 --coverage-seconds 10800` to require at least
20 minutes as well as the configured episode count. The minimum duration must
be below the run timeout, and schedule coverage must cover that timeout.

### Outputs and interpretation

- `evaluation_process.log`, `load_process.log`: live process output.
- `schedule.json`, `protocol.json`, `hardware.json`, `evaluation.yaml`, `code/`:
  frozen schedule, seeds, GPU identity, resolved settings and source snapshot.
- `load/load.jsonl`: actual phase boundaries, copy counts and boundary overruns.
- `evaluation/observation_attempts.jsonl`: observation timestamps, completed
  requests and capacity drops. Normal evaluation summaries are also retained.
- `timing_by_load.csv`: per-request action latency, worker service time, raw-frame
  delay and ON/OFF/MIXED classification.
- `report.json`: latency summaries by phase, observed ON share/event count,
  dropped observations, and consecutive slow-request statistics.
- `DONE`: evaluation and reporting succeeded. `FAILED.json`: failure reason;
  inspect the process logs for the underlying error.

Phase classification uses the **actual worker-service interval**: OFF means no
overlap with an ON window, ON means entirely inside ON, and MIXED means crossing
a boundary. Both observation-to-action and worker-service latencies are reported.
Worker service here includes worker-to-parent result delivery. The ON share is
measured over the full observed span, including inter-episode gaps. Raw-frame
delays use the configured environment FPS and the executor's ceiling convention.

First inspect the number of observed ON events and completed requests in each
phase. Then compare ON vs OFF latency distributions, the separate idle control,
and slow-run lengths. Slow thresholds are this session's OFF p95 and p99, so
these diagnostics are exploratory. No ON requests is expected for the idle
control. For irregular runs, it means the current run cannot establish an ON
effect. A copy event is not itself evidence of a latency burst.

The single-session `measure.sh` command collects and checks the hardware trace.
Use `profile.sh` above for the complete live profiling, fitting and held-out
comparison workflow.

To regenerate a report from an existing run:

```bash
python -m scripts.experiments.memory_mock.report \
  --output outputs/memory_mock/irregular_s42
```

### Generate an irregular schedule without a GPU

```bash
bash scripts/experiments/memory_mock/run.sh irregular-schedule \
  --seed 42 --output outputs/memory_mock/irregular_schedule.json
```

This command only writes JSON. Use it with the standalone `run --mode schedule`
entrypoint below if you already have your own evaluator coordination.

## Frozen random timing

```bash
# CPU-only: generate at least two hours of alternating OFF/ON phases.
bash scripts/experiments/memory_mock/run.sh schedule \
  --seed 42 --output outputs/memory_mock/schedule.json

CUDA_VISIBLE_DEVICES=0 bash scripts/experiments/memory_mock/run.sh run \
  --mode schedule --schedule outputs/memory_mock/schedule.json \
  --output outputs/memory_mock/random
```

Each duration is drawn independently from continuous Uniform(0, 10) seconds
using a local NumPy PCG64 generator, including the first OFF phase. The saved
JSON records the seed, NumPy version and complete numeric schedule. Generation
does not touch NumPy's global RNG; runtime reads the frozen file without drawing
random numbers. Keep the file with the workload logs.

Phase boundaries use absolute cumulative deadlines from the load's start,
avoiding drift from repeated relative sleeps. Zero or already expired short
phases are skipped without forcing any copy work. A completed schedule raises
an error; stop the process before its saved coverage ends.

## Coordinate with inference

Pass `--wait-for /absolute/path/to/warmup.done` to defer Torch import, CUDA
initialization and allocation until the inference process creates that file.
The load writes `load.ready` in its output directory only after allocation,
initialization and the first phase switch. Have the evaluator wait for that
marker before starting measured episodes. The caller owns both processes and
must terminate the load after evaluation or stop evaluation if the load fails.

`load.jsonl` records start/stop, actual and planned phase boundaries, skipped
phases, completed copy counts/bytes and batches crossing an ON boundary.
Timestamps are host `perf_counter` milliseconds, shared with same-host pipeline
timing; they are not Unix timestamps. Copy bytes count payload, not total DRAM
traffic. Use actual ON windows intersected with the measured intervals to
calculate workload duty cycle.

## Check the implementation

```bash
python -m pytest -q tests/scripts/test_memory_mock.py \
  tests/scripts/test_memory_mock_measure.py tests/eval/test_eval_logging.py \
  tests/executors/test_realtime_executor.py
bash -n scripts/experiments/memory_mock/run.sh scripts/experiments/memory_mock/measure.sh
```

The CPU suite covers frozen schedules, phase boundaries, load-loop accounting,
startup/shutdown and output preservation. CUDA behavior requires a GPU-enabled
PyTorch runtime; CPU tests do not establish bandwidth or inference slowdown.

The standalone workload was imported from `origin/memory-mock` commit
`0f28f11b9db3843b5bd74e6cedb3d9db3a6fdef7`. The irregular schedule, coordinated
measurement and aligned timing report extend that workload without changing its
CUDA copy loop.
