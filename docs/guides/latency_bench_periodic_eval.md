# Evaluate during Sample Factory training

Use periodic evaluation to compare a training checkpoint across named latency conditions.
Run the Flappy example from the repository root:

```bash
python -m training.train --config configs/examples/flappy_periodic_latency_bench_eval.yaml
```

The base Flappy, Demon Attack, and Deadly Corridor configurations provide the reference evaluation settings.
Keep reusable recipes in `configs/examples/` or inherit the corresponding `configs/base/` file.
Generated `configs/train/` files are local run artifacts.

## Select conditions and budgets

```yaml
executor:
  parallel_envs: 100
policy:
  deterministic: true
evaluation:
  eval_interval_steps: 1000000
  episodes: 20
  suites:
    - name: zero
      latency: {method: zero}
    - name: fixed_100ms
      latency: {method: fixed, fixed_latency_ms: 100}
    - name: normal_100ms
      latency:
        method: normal
        mean_latency_ms: 100
        std_latency_ms: 20
        min_latency_ms: 0
        max_latency_ms: 500
        seed: 42
```

Each suite has a unique name and a complete latency declaration.
Time parameters use milliseconds.
`episodes` counts episodes per condition. `parallel_envs` sets the maximum active episode slots.
`policy.deterministic` selects deterministic actions.

`eval_interval_steps` sets the training-step interval between evaluations.
Set it to `null` to disable periodic submission.
Periodic evaluations run in a background process in submission order.
The final evaluation waits for submitted periodic evaluations to finish.
The runner also evaluates at training completion by default.

## Read the results

Each evaluation writes results under the training output directory:

```text
latency_bench_train_eval/step_<env_steps>/
latency_bench_train_eval/step_<env_steps>_final/
```

Each suite writes a subdirectory with its configured name, such as `fixed_100ms`.
The saved configuration identifies the checkpoint, task, resources, and evaluation conditions.

W&B records ordinary scalar metrics.
Each payload includes `global_step` and `eval/global_steps`.
Use `global_step` as the W&B X axis to compare against Sample Factory environment steps.
The runner merges the base task configuration with the current training configuration before evaluation.
See [configuration](configuration.md) for field ownership and [parallel evaluation](../reference/eval_parallelism.md) for resource settings.
