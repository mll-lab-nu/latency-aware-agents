# Connect an external environment or policy

Custom components live in your own Python module. LAGEN constructs them through
the existing environment and policy factories. Each side can independently use
a built-in component or a custom factory.

`env.name` selects one concrete task registered with `register_env_adapter`.
The registration owns the Gymnasium ID and native framework integration.
`env.registration_imports` loads a user module that registers that task.

## Factory contracts

| Factory | Arguments | Return value |
| --- | --- | --- |
| Environment | `config, noop_action` | `EnvAdapter` |
| Policy | `config, action_resolver` | `PolicyRunner` |

`config` is the complete run configuration. `noop_action` is the resolved native
`Action`. `action_resolver` maps configured action names to native `Action`
objects. A model adapter can also construct native actions directly.

An environment adapter advances one raw step per `step`, preserves the declared
clock, and supplies observations through the existing `Observation` interface.
A policy implements `predict(observation) -> PolicyOutput`. It owns model input
preprocessing, output decoding, and any recurrent state. Implement
`reset_state(slot_id=None)` to reset that state at episode boundaries.

For batched evaluation, use `observation.metadata["slot_id"]` to identify each
environment's state. Batch positions can change as episodes finish.
`predict_batch` returns outputs in the same order as its input observations.
`reset_state(slot_id=...)` clears the addressed slot's state; single-environment
execution uses slot 0. For a stateful realtime policy, start with one inference
worker. Separate worker processes do not share policy state.

## Example from your own project

Save the following module as `my_components.py` in your project. The constant
policy demonstrates construction; replace its factory with your model adapter.

```python
from latency_bench.envs.adapters.gymnasium import GymnasiumEnvAdapter
from latency_bench.policy.base import ConstantPolicyRunner
from latency_bench.envs.factory import register_env_adapter


@register_env_adapter("mountain_car", env_id="MountainCar-v0")
def make_env(config, noop_action):
    env = config["env"]
    return GymnasiumEnvAdapter(
        env_id=env["env_id"],
        registration_imports=[],
        make_kwargs={},
        env_fps=env["env_fps"],
        noop_action=noop_action,
        frame_stack=1,
        export_env_raw_rgb_frames=False,
    )


def make_policy(config, action_resolver):
    action = action_resolver.resolve(config["policy"]["action"])
    return ConstantPolicyRunner(action)
```

Save this configuration as `eval.yaml`:

```yaml
experiment:
  name: custom_mountain_car
  seed: 7
env:
  name: mountain_car
  registration_imports: [my_components]
  env_fps: 50
  obs_fps: 10
  noop_action: coast
  action_map: {left: 0, coast: 1, right: 2}
policy:
  type: custom
  factory: my_components:make_policy
  action: right
executor:
  mode: simulated
latency:
  method: fixed
  fixed_latency_ms: 40
scheduler:
  hold_policy: hold
  ordering_policy: latest_ready
evaluation:
  episodes: 2
  max_env_steps: 100
logging:
  output_dir: outputs/custom_mountain_car
  save_step_records: true
  save_action_records: true
  save_latency_records: true
```

With LAGEN installed, run from your project directory:

```bash
PYTHONPATH="$PWD" python -m latency_bench.run --config eval.yaml
```

For realtime execution, set `executor.mode: realtime` and
`executor.inference_devices: [cpu]` for this CPU example. Realtime measures actual
latency automatically. The `latency` section only supplies simulated delays. For a GPU model, select the intended devices
explicitly. Inference workers receive their assigned device in
`config["executor"]["inference_devices"][0]` before calling the factory.

In realtime execution, the environment factory runs in the environment process.
The policy factory runs in each inference process. Install your package in each
interpreter that constructs it, including a separately configured
`policy.worker_python_executable`. Import paths must work in those interpreters;
LAGEN does not search source directories for a missing module.

Factory strings remain in `resolved_config.yaml`. Invalid references, import
errors and constructor errors propagate to the caller. Executors close an
environment they have created if subsequent policy construction fails.

## Python construction

Use the same functions directly when constructing components in your process:

```python
from latency_bench.core.actions import ActionResolver
from latency_bench.envs.factory import build_env_adapter
from latency_bench.policy.factory import build_policy

actions = ActionResolver(config["env"])
env = build_env_adapter(config, actions.noop_action, factory=make_env)
policy = build_policy(config, actions, factory=make_policy)
```

The explicit `factory=` argument overrides factory selection from the config.
Pass constructed components to `build_executor(config, env=env, policy=policy)`
for single-environment simulated or paused execution. The executor closes its
environment when it closes. Use environment registration modules and importable policy factory strings for
realtime or pooled inference. Each process constructs its own components.

## Collect teacher demonstrations

The same external factories support [canonical demonstration collection](rollout_collection.md).
Select student RGB and optional state separately from teacher input. The output
pairs each source observation with its teacher command and can be passed directly
to the existing training-format converter.
