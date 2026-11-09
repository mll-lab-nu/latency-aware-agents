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
uv run --extra dev pytest tests/scripts/test_memory_mock.py -q
bash -n scripts/experiments/memory_mock/run.sh
```

The CPU suite covers frozen schedules, phase boundaries, load-loop accounting,
startup/shutdown and output preservation. CUDA behavior requires a GPU-enabled
PyTorch runtime; CPU tests do not establish bandwidth or inference slowdown.
