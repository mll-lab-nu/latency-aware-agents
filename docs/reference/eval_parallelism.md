# Run parallel simulated episodes

Set `executor.parallel_envs` to run multiple simulated episode slots.
Parallelism changes resource use, not the requested episode count.
Paused and realtime execution require one environment.

## Select slots and inference resources

```yaml
executor:
  mode: simulated
  parallel_envs: 32
evaluation:
  episodes: 100
```

`parallel_envs` defaults to 1.
The executor starts at most `min(parallel_envs, episodes)` active slots.
It assigns episode IDs from `0` through `episodes - 1`.
Only a completed slot resets. Execution ends after the configured episodes complete.
Each suite and retained checkpoint still receives its full configured budget.

Run your prepared evaluation configuration:

```bash
python -m latency_bench eval --config my_eval.yaml
```

Batched simulation supports zero, fixed, normal, uniform, IID, and temporal delay.
IID and parametric samplers derive per-episode streams.
Temporal replay retains one machine-state timeline per replica across episode boundaries.
Each replica owns its virtual worker capacity.
Physical inference batches do not create contention between virtual replicas.

For pooled inference, set `executor.inference_pool: true` and declare `executor.inference_batch_size`.
The batch size limits physical inference batches. It does not set simulated worker capacity.
Stateful policies use `Observation.metadata["slot_id"]` because batch positions can change.
Episode and soft resets clear the affected slot's state on every inference worker.

## Select an environment backend

```yaml
executor:
  env_backend: serial
```

| Backend | Behavior |
| --- | --- |
| `serial` | Steps environment instances in the parent process |
| `subproc` | Steps each active environment in a worker process |
| `flappy_gpu_batched` | Steps Flappy GPU slots in one backend call |
| `demon_attack_gpu_batched` | Explicitly selects the Demon Attack CUDA backend |
| `auto` | Selects supported Flappy GPU batching, CPU subprocess stepping where applicable, and the default Demon Attack CPU path |

Task entrypoints can provide their existing native vector backend.
For CPU-bound environments, increase slots within the available CPU capacity.
For supported GPU environments, batching can reduce step overhead.
If communication dominates, more slots can reduce throughput.
Select resources using the same task, model, and episode protocol when comparing performance.

Demon Attack CUDA stepping preserves the CPU adapter's policy observation contract.
The same seed and action trace must preserve observations, simulation time, and raw RGB history.
Current Sample Factory policy batching supports Discrete, flat Tuple(Discrete), and one-dimensional MultiDiscrete actions.
Recurrent policies keep separate state for each slot.

## Use one environment for measurement

Profiling and realtime evaluation use `parallel_envs: 1`.
Select concurrent model workers with `executor.inference_devices`.
See [profiling](../cookbooks/profiling.md) for device allocation and [execution semantics](latency_execution_semantics_en.md) for virtual capacity.

## Diagnose backend selection

Use normal YAML settings for ordinary runs.
These existing environment variables override backend behavior for diagnosis:

- `LATENCY_BENCH_ENV_BACKEND`
- `LATENCY_BENCH_ENV_BACKEND_START_METHOD`
- `LATENCY_BENCH_ENV_WORKER_TIMEOUT_SEC`

The subprocess backend supplies its thread-limit environment variables before worker creation.
You do not need to set them for ordinary evaluation.
