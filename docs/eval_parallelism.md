# Evaluation Parallelism

This project uses `evaluation.eval_parallel_envs` to run multiple evaluation episode slots inside the same eval latency stage. It does not reduce the eval workload. For a full Sample Factory eval sweep, every available checkpoint kind still evaluates the configured eval runners, all eval latencies, and the configured number of episodes per latency.

## Config

```yaml
evaluation:
  eval_episodes: 100
  eval_parallel_envs: 32
```

`eval_parallel_envs` defaults to `1`. Values greater than `1` mean:

- start at most `min(eval_parallel_envs, eval_episodes)` active episode slots;
- assign deterministic `episode_id` values `0..eval_episodes-1`;
- reset only the slot that finished;
- stop only when exactly `eval_episodes` episodes completed.

This is episode-slot parallelism, not a workload reduction.

## Supported Paths

### Sample Factory Queue Eval: Flappy

Normal command:

```bash
bash scripts/bash_scripts/eval_flappy.sh
```

For Flappy queue eval, `eval_parallel_envs > 1` uses the existing GPU-rendered batched Flappy env during eval. It maps the active eval slots to the env's agent/render batch count. Training behavior is unchanged. `env.use_gpu_render: false` remains the serial path and does not support Flappy queue eval batching.

Recommended starting values:

- `32` on smaller GPUs or when checking correctness;
- `64` on A100-class machines;
- `100` when `eval_episodes=100` and GPU memory is not the bottleneck.

### Sample Factory Queue Eval: DemonAttack

Normal command:

```bash
bash scripts/bash_scripts/eval_demon_attack.sh
```

For DemonAttack queue eval, `eval_parallel_envs <= 1` uses the serial backend. `eval_parallel_envs > 1` automatically uses subprocess env stepping. Each worker owns a normal DemonAttack env created through the existing env factory and wrapper stack; the main process keeps policy inference, episode accounting, and output writing.

Recommended starting values:

- `16` for correctness and moderate CPU machines;
- `32` for most multi-core servers;
- `64` only when the machine has enough CPU cores and memory bandwidth.

Avoid setting this much higher than the available physical CPU cores. ALE stepping is CPU-bound, and too many workers can spend more time context switching and moving observations over IPC than stepping the env.

### latency_bench Simulated Eval

Normal command:

```bash
python -m latency_bench.run --config configs/train/demon_attack/latency_0/seed_0/latency_bench_eval.yaml
```

`latency_bench` batched eval is only for:

```yaml
executor:
  mode: simulated
evaluation:
  eval_parallel_envs: N
```

The current batched simulated executor supports fixed/zero simulated latency. Simulated replay/profile latency is not wall-clock realtime measurement and is not rejected, but the current factory keeps unsupported latency-source combinations on the old single-executor path instead of silently changing semantics.

Optional env stepping backend:

```yaml
evaluation:
  latency_bench_env_backend: serial  # serial | subproc | auto
```

- `serial`: N env instances are stepped in a Python loop; policy inference can still be batched.
- `subproc`: one worker process per active env slot; useful for CPU-bound envs.
- `auto`: keeps Flappy GPU-rendered eval serial, chooses subproc for DemonAttack and Deadly Corridor.

Recommended starting values:

- Flappy GPU-rendered eval: `32-100`, backend `serial` or `auto`.
- DemonAttack: `16-64`, backend `auto` or `subproc`.
- Deadly Corridor: `8-32`, backend `auto` or `subproc`.

Current Sample Factory policy batching supports Discrete, flat Tuple(Discrete, ...), and one-dimensional MultiDiscrete action spaces. Recurrent Sample Factory policies still fail fast for batched `latency_bench` eval; keep `eval_parallel_envs: 1` for those checkpoints until per-slot recurrent state batching is implemented.

## Unsupported Paths

These paths must not be parallelized with `eval_parallel_envs > 1`:

- `executor.mode: realtime`;
- `executor.mode: paused`.

Realtime and measured wall-clock execution depend on scheduling and wall time. Running multiple envs in parallel would contaminate the measurement. Use `eval_parallel_envs: 1` for those modes.

## What Gets Faster

`eval_parallel_envs` can help at three different layers:

- Episode slots: multiple episodes are active inside one eval latency stage, so one long episode does not block the whole stage.
- Policy batching: the main process can send one larger batch through the policy instead of many single-observation forwards.
- Subprocess env stepping: CPU-bound envs can step in parallel across worker processes.

These are separate. If CPU env stepping dominates and the backend is `serial`, GPU utilization can remain low even with many active episode slots. If IPC dominates, increasing `eval_parallel_envs` past the useful CPU core count may slow the run down.

## Benchmarking

Use the lightweight benchmark script to compare parallel settings without changing eval workload:

```bash
python scripts/bench_latency_bench_env_backend.py \
  --config configs/train/demon_attack/latency_0/seed_0/latency_bench_eval.yaml \
  --checkpoint-kind best \
  --parallel-envs 1,4,16,32,64 \
  --backends auto \
  --output-root outputs/env_backend_bench/demon_attack
```

Flappy:

```bash
python scripts/bench_latency_bench_env_backend.py \
  --config configs/train/flappy/latency_0/seed_0/latency_bench_eval.yaml \
  --checkpoint-kind best \
  --parallel-envs 1,4,16,32,64 \
  --backends auto \
  --output-root outputs/env_backend_bench/flappy
```

Deadly Corridor:

```bash
python scripts/bench_latency_bench_env_backend.py \
  --config configs/train/deadly_corridor/latency_0/seed_0/latency_bench_eval.yaml \
  --checkpoint-kind best \
  --parallel-envs 1,4,16,32 \
  --backends auto \
  --output-root outputs/env_backend_bench/deadly_corridor
```

The script prints CSV with wall-clock time, episodes/sec, cumulative CPU seconds, CPU utilization, expected episodes, completed episodes, and an output correctness flag. `cpu_util` is `cpu_sec / wall_sec`; a value near `16.0` means the run used about sixteen CPU cores on average.

The CSV is written to stdout and eval progress logs are redirected to stderr, so results can be saved with shell redirection:

```bash
python scripts/bench_latency_bench_env_backend.py \
  --config configs/train/demon_attack/latency_0/seed_0/latency_bench_eval.yaml \
  --checkpoint-kind best \
  --parallel-envs 1,4,16,32,64 \
  --backends auto \
  --output-root outputs/env_backend_bench/demon_attack \
  > outputs/env_backend_bench/demon_attack.csv
```

For GPU utilization, sample in another terminal:

```bash
nvidia-smi dmon -s pucvmt -d 1 -o DT
```

For CPU utilization, use `htop` interactively or:

```bash
pidstat -durh 1
```

Check `completed_ok=1` and `episodes == expected_episodes` for every CSV row before comparing speed.

## Debug Overrides

Normal eval should be config-only. These environment variables are debug overrides, not the normal interface:

- `LATENCY_BENCH_ENV_BACKEND=serial|subproc|auto`
- `LATENCY_BENCH_ENV_BACKEND_START_METHOD=spawn|forkserver|fork`
- `LATENCY_BENCH_ENV_WORKER_TIMEOUT_SEC=<seconds>`

The subprocess backend sets `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, and `NUMEXPR_NUM_THREADS` internally with `os.environ.setdefault(...)` before spawning workers. Users do not need to set those for normal eval.
