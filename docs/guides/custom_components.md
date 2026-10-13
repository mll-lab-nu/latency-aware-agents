# Connect your environment or policy

Keep your components in your own Python package. LAGEN constructs them through importable factories.
You can combine a built-in environment with your policy, or your environment with a built-in policy.
Install the [Python package](../installation.md#use-a-cpu-environment-or-your-own-components) first.

## Run the CPU example

The example registers MountainCar and constructs a constant policy that selects the right action.
It demonstrates the interfaces. It does not provide a trained MountainCar policy.
Run these commands from the checkout:

```bash
export PYTHONPATH="$PWD/configs/examples/custom_env${PYTHONPATH:+:$PYTHONPATH}"
python -m latency_bench eval --config configs/examples/custom_env/eval.yaml
```

The configuration runs three episodes with seed 7 and a limit of 50 raw steps per episode.
The environment runs at 50 FPS. The policy receives observations at 10 FPS.
A fixed 100 ms delay becomes five raw steps.
Results and the resolved configuration appear under `outputs/custom_mountain_car/eval/`.

Use the same files from a project outside the checkout:

```bash
export LAGEN_CHECKOUT="$PWD"
mkdir -p /tmp/my-lagen-project
cp "$LAGEN_CHECKOUT"/configs/examples/custom_env/*.yaml /tmp/my-lagen-project/
cp "$LAGEN_CHECKOUT"/configs/examples/custom_env/components.py /tmp/my-lagen-project/
cd /tmp/my-lagen-project
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
python -m latency_bench eval --config eval.yaml
```

Use the interpreter where you installed LAGEN. Output paths are relative to your project directory.
YAML `__base__` paths are relative to the YAML file that contains them.

## Implement the factories

[`components.py`](../../configs/examples/custom_env/components.py) contains the complete example.
Replace its constant policy with your model adapter.

| Factory | Arguments | Result |
| --- | --- | --- |
| Environment | `config, noop_action` | `EnvAdapter` |
| Policy | `config, action_resolver` | `PolicyRunner` |

`config` contains the full run configuration.
`noop_action` is the resolved native `Action` used while no prediction is available.
`action_resolver` maps configured names to native actions.
A policy can construct native actions directly.

Register a concrete task with `register_env_adapter`.
The registration owns its Gymnasium ID and native integration.
Set `env.name` to that task name.
Set `env.registration_imports` to your registration module.
For a custom policy, set `policy.type: custom` and `policy.factory: module:function`.

An `EnvAdapter` advances one raw step per `step()` and supplies an `Observation`.
A `PolicyRunner` implements `predict(observation) -> PolicyOutput`.
The policy owns preprocessing, inference, action decoding, and recurrent state.
See the [Gymnasium guide](gymnasium_adapter.md) for task and action contracts.

## Select execution and manage policy state

`eval.yaml` uses simulated execution. For synchronous execution, set `executor.mode: paused`.
For realtime execution, use this executor block:

```yaml
executor:
  mode: realtime
  inference_devices: [cpu]
  parallel_envs: 1
  sync_cuda: false
```

Realtime measures actual latency. Simulated execution uses the `latency` section.
See [execution semantics](../reference/latency_execution_semantics_en.md) for scheduling rules.

Implement `reset_state(slot_id=None)` for a stateful policy.
Implement `close()` to release policy-owned subprocesses or other resources;
executors call it when evaluation ends, including during error cleanup.
Realtime executors call `prepare_episode(episode_id)` before starting the
episode clock so external runtimes can initialize separately from action latency.
Each observation identifies its episode slot through `observation.metadata["slot_id"]`.
Batch positions can change when episodes finish.
`predict_batch()` must return results in input order.
An episode reset clears the addressed slot's state. Single-environment execution uses slot 0.

Start a stateful realtime policy with one inference worker.
Each worker process constructs its own policy and retains its own state.
Workers receive their device in `config["executor"]["inference_devices"][0]`.
Install your package in every interpreter that constructs a component.
This includes `policy.worker_python_executable` when configured.

## Collect and convert demonstrations

Install LeRobot as described in the [installation guide](../installation.md).
From the external example directory, run:

```bash
python -m latency_bench collect --config collect.yaml
python -m latency_bench convert --config convert.yaml
```

Collection writes RGB observations paired with the teacher's issued commands.
The example keeps student state out of the dataset.
It writes canonical data under `outputs/custom_mountain_car/dataset/`.
Conversion writes LeRobot data under `outputs/custom_mountain_car/lerobot/` and its validation companion.
See [collection](rollout_collection.md) for alignment and training metadata.

## Measure and replay latency

Run the example measurement:

```bash
python -m latency_bench profile run --config profile.yaml
```

The example measures two repetitions of three episodes with the same clocks and seed protocol.
The command prints `RUN_DIR`, `PROFILE_PATH`, `MEASUREMENT_STATUS`, and `FIT_STATUS`.
Replace `/ABSOLUTE/PROFILE/profile.json` in `replay.yaml` with the returned `PROFILE_PATH`.
Run the replay:

```bash
python -m latency_bench eval --config replay.yaml
```

Replay uses the fitted temporal profile with latency seed 7.
Keep the profile's companion files beside `profile.json`.
The [profiling cookbook](../cookbooks/profiling.md) explains measurement, fitting, and recovery.

## Construct components in Python

Use explicit factories when your application constructs components in one process:

```python
from latency_bench.core.actions import ActionResolver
from latency_bench.envs.factory import build_env_adapter
from latency_bench.policy.factory import build_policy

from components import make_env, make_policy

# config is your resolved run configuration.
actions = ActionResolver(config["env"])
env = build_env_adapter(config, actions.noop_action, factory=make_env)
policy = build_policy(config, actions, factory=make_policy)
```

Pass the components to `build_executor(config, env=env, policy=policy)` for single-environment simulated or paused execution.
The executor closes its environment when it closes.
Use registration modules and importable policy factories for realtime or pooled inference.
Import and constructor errors propagate to the caller.
