# Evaluation Parallelism

This project uses `evaluation.eval_parallel_envs` to run multiple evaluation episode slots inside the same `latency_bench` simulated eval latency stage. It does not reduce the eval workload. For a generated Sample Factory checkpoint sweep, every available checkpoint kind still evaluates the configured eval runners, all eval latencies, and the configured number of episodes per latency.

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

### latency_bench Simulated Eval

Normal command:

```bash
python -m latency_bench.run --config configs/train/demon_attack/latency_0/seed_0/latency_bench_eval.yaml
```

Generated Sample Factory sweep wrappers also call this entrypoint:

```bash
bash scripts/bash_scripts/eval_flappy.sh
bash scripts/bash_scripts/eval_demon_attack.sh
bash scripts/bash_scripts/eval_deadly_corridor.sh
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
  latency_bench_env_backend: serial  # serial | subproc | flappy_gpu_batched | demon_attack_gpu_batched | auto
```

- `serial`: N env instances are stepped in a Python loop; policy inference can still be batched.
- `subproc`: one worker process per active env slot; useful for CPU-bound envs.
- `flappy_gpu_batched`: steps Flappy GPU env slots in one backend call.
- `demon_attack_gpu_batched`: steps DemonAttack CUDA env slots in one backend call. This is explicit opt-in so CPU Atari behavior is not silently replaced.
- `auto`: uses Flappy's GPU-batched backend for supported Flappy policies, chooses subproc for CPU-bound envs such as Deadly Corridor, and keeps DemonAttack on the existing CPU path unless explicitly configured.

Recommended starting values:

- Flappy GPU-rendered eval: `32-100`, backend `serial` or `auto`.
- DemonAttack CPU env: `16-64`, backend `auto` or `subproc`.
- DemonAttack CUDA env: `4-32`, backend `demon_attack_gpu_batched`.
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
- GPU env-step batching: supported GPU envs can advance all active slots in one backend call, and slots that finish are immediately refilled while pending episodes remain.

These are separate. If CPU env stepping dominates and the backend is `serial`, GPU utilization can remain low even with many active episode slots. If IPC dominates, increasing `eval_parallel_envs` past the useful CPU core count may slow the run down.

For DemonAttack StarVLA simulated eval, `demon_attack_gpu_batched` keeps the policy-facing observation contract locked to the CPU adapter: `Observation.data`, `env_step`, `sim_time_ms`, and `metadata["env_raw_rgb_frame_stack"]` must match the CPU path for the same seed/action trace. The GPU backend only changes how active slots are stepped.

## Profiling

When `logging.simulated_pipeline_profile: true`, batched simulated eval writes `simulated_pipeline_summary.json` in the run output directory with `observe_slots_ms`, policy timing stages, `env_step_ms`, active slot count, and policy batch size. StarVLA reports separate `policy_build_example_ms`, `policy_wrapper_predict_action_ms`, and `policy_decode_ms` values, so the summary distinguishes model inference from image/example preparation and env stepping.

## Debug Overrides

Normal eval should be config-only. These environment variables are debug overrides, not the normal interface:

- `LATENCY_BENCH_ENV_BACKEND=serial|subproc|flappy_gpu_batched|demon_attack_gpu_batched|auto`
- `LATENCY_BENCH_ENV_BACKEND_START_METHOD=spawn|forkserver|fork`
- `LATENCY_BENCH_ENV_WORKER_TIMEOUT_SEC=<seconds>`

The subprocess backend sets `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, and `NUMEXPR_NUM_THREADS` internally with `os.environ.setdefault(...)` before spawning workers. Users do not need to set those for normal eval.
