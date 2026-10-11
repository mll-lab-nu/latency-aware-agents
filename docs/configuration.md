# Run configuration

Each section owns one decision. Evaluation, profiling and teacher collection use
these same runtime fields.

| Section | Responsibility | Main fields |
| --- | --- | --- |
| `env` | Concrete task, observation clock and task parameters | `name`, `env_fps`, `obs_fps`, `make_kwargs` |
| `policy` | Model implementation, model files and action selection | `type`, `source`, `root`, `checkpoint`, `model_config`, `deterministic` |
| `executor` | Execution mode and runtime resources | `mode`, `inference_devices`, `parallel_envs`, `inference_pool`, `simulated_worker_capacity`, `sync_cuda`, `env_backend` |
| `latency` | Simulated action delay | `method` and sampler parameters |
| `evaluation` | Episode budget and named simulated conditions | `episodes`, `max_env_steps`, `suites` |
| `logging` | Saved records, output location and video | `output_dir`, `save_step_records`, `save_action_records`, `save_latency_records`, `video` |
| `profiling` | Measurement repetitions and publication coordinates | `repeats`, `model_id`, `workload_id`, `calibration`, `publish_repo` |

## Execution and timing

`executor.mode: realtime` advances the environment on its wall clock and measures
actual timing. It does not read `latency`. Basic request timestamps, inference
latency, worker service time, environment lag and deadlines remain measured.
Saving those records depends on `logging`.

`executor.mode: paused` waits for each action and applies it with zero simulated
delay. It does not read `latency`.

`executor.mode: simulated` reads `latency`. Supported methods are `zero`, `fixed`,
`normal`, `uniform`, `iid` and `temporal`. Durations use milliseconds. `iid` and
`temporal` consume a measured profile.

`inference_devices` lists actual model workers. Repeated device names create
separate workers on one device. Realtime and pooled inference require an
explicit list. In-process Sample Factory can use its native `auto` device
selection. `simulated_worker_capacity` independently sets virtual worker
capacity. It does not select physical devices or the number of model workers.

`parallel_envs` defaults to 1. Parallel episode execution requires simulated mode.
`max_env_steps` counts raw environment steps, including steps between observations.
`episodes` counts complete episodes for each evaluation condition.

## Task identity and policy implementation

`env.name` selects a concrete registered task, such as `air_raid` or `flappy`.
The registration owns its native integration, environment ID and simulator
registration imports. These derived fields appear in resolved configurations.
An external task uses `env.registration_imports` to load its registration module.
See [custom components](custom_components.md).

All Sample Factory policies use `policy.type: sample_factory`. The task's
registered native integration selects the appropriate adapter. StarVLA uses
`policy.type: starvla`. `policy.deterministic` controls Sample Factory action
selection in evaluation and collection.

## Model files

Both local and HF models use the same flat fields:

```yaml
policy:
  type: starvla
  source: hf
  hf_repo: org/models
  revision: optional-pinned-revision
  root: models/air_raid
  checkpoint: checkpoints/model.pt
  model_config: config.full.yaml
```

For local files, change `source` to `local`, set `root` to the local directory,
and omit `hf_repo` and `revision`. File names are relative to `root`.
Relative task metadata paths also use this root. An absolute metadata path can
refer to a separate local dataset or contract.

HF resolution downloads the selected directory before model workers start.
The recorded provenance retains the repository, snapshot revision and filenames.
A local source must provide its files locally. Sample Factory can omit
`checkpoint` to use its native checkpoint selection.

Filesystem roots and output paths use the caller's working directory.
`__base__` paths use the containing YAML file's directory.
CLI file overrides produce the same flat local declaration. A partial override
retains the locations of the other model inputs.

## Evaluation conditions

Each suite has a unique name and a complete ordinary latency declaration:

```yaml
evaluation:
  episodes: 10
  max_env_steps: 3600
  suites:
    - name: zero
      latency: {method: zero}
    - name: slow
      latency: {method: fixed, fixed_latency_ms: 100}
```

Each condition runs the same episode and seed protocol. Results are saved under
`<output_dir>/<name>/`. An empty `suites` list runs the current condition once.
Realtime and paused mode use their own timing semantics.

## Persistence and profiling

Logging controls persistence. It does not enable realtime timing. Simulation
video enables the step records needed for replay. See
[realtime timing](realtime_pipeline_logging.md) for measurement boundaries and
sample populations.

`profiling.repeats` counts independent measurement processes. Each process loads
the model, performs warmup, and measures `evaluation.episodes` episodes.
`model_id` and `workload_id` label artifacts. They do not choose implementations.
`publish_repo` selects the publication repository. See the
[profiling guide](profiling/cookbook.md).

Native trainer arguments, checkpoint contracts, dataset fields and stored
profile formats retain their own schemas. Historical run snapshots are converted
only by artifact readers. New run YAML uses the fields documented above.
