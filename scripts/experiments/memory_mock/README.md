# CUDA memory mock

This standalone background workload holds two equally sized CUDA `uint8`
buffers. It can keep them idle or copy them back and forth to create GPU memory
bandwidth contention alongside an inference process. Both buffers remain
resident during OFF phases. It does not simulate model KV memory.

Use a CUDA-enabled PyTorch environment and run the commands from this checkout.
Set `MEMORY_MOCK_PY` to its Python executable if needed. The tool uses logical
`cuda:0`; bind the same physical GPU for inference and the load process with
`CUDA_VISIBLE_DEVICES`. It does not change GPU clocks, drivers or other processes.

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
allow some coverage beyond the expected total runtime. Episode count is not a
minimum measurement duration: short episodes may observe too few ON events.

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

This experiment collects and checks the hardware trace; it does not fit temporal,
AR or ARMA profiles. For that comparison, fit every model on the same training
sessions and evaluate burst statistics and downstream behavior on separate held-out
sessions. The comparison should determine which model works better, rather than
assuming AR/ARMA cannot capture bursts.

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
