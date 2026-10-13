# Adapt a Gymnasium environment

Use `GymnasiumEnvAdapter` to connect a Gymnasium task to evaluation, demonstration collection, and profiling.
Use `ActionLatencyWrapper` to add latency to your own Gymnasium training loop.
You can keep task code in your own package.
The [external component example](custom_components.md) provides complete runnable configurations.

## Define the task contract

Choose the environment ID and install its pinned package in each interpreter that constructs it.
Check `reset()`, `step()`, action and observation spaces, rendering, and the native step duration.
Define these settings before collecting data or training a model:

| Setting | Meaning |
| --- | --- |
| `env.name` | Concrete registered task name |
| `env.env_fps` | Raw environment steps per second |
| `env.obs_fps` | Policy observations per second |
| `env.noop_action` | Native action used while waiting for a prediction |
| Action contract | Action order, values, dimensions, bounds, units, and dtype |
| Observation contract | Model inputs, RGB views, optional state, and preprocessing |
| Episode contract | Reward, success, termination, truncation, and step limit |
| `env.base_prompt` | Student task instruction |

`env_fps / obs_fps` must be a positive integer.
The configured clock defines simulated time. Simulator wall time does not advance that clock.
Keep the task contract consistent across training, collection, conversion, and evaluation.

## Register the environment

Follow [`components.py`](../../configs/examples/custom_env/components.py) to register a task through `register_env_adapter`.
The registration binds the task name to its Gymnasium ID.
Its factory receives `config` and `noop_action` and returns an adapter.

In your YAML, select the registered task and module:

```yaml
env:
  name: custom_mountain_car
  registration_imports: [components]
  env_fps: 50
  obs_fps: 10
  noop_action: coast
  action_map: {left: 0, coast: 1, right: 2}
```

Use `make_kwargs` for native constructor arguments.
For collection, select `render_mode: rgb_array` when the environment requires it for RGB rendering.
Add a task wrapper only when the native interface cannot provide the required action, observation, or episode contract.
Keep task-specific behavior in that wrapper.

## Declare native actions and student inputs

For Discrete actions, provide `action_map` and `action_order`:

```yaml
env:
  action_map: {left: 0, coast: 1, right: 2}
  action_order: [left, coast, right]
  noop_action: coast
  base_prompt: Drive the car up the hill.
```

For a bounded Box vector, declare its ordered dimensions and native bounds:

```yaml
env:
  action_space:
    type: box
    shape: [2]
    labels: [joint_0, joint_1]
    low: [-1.0, -0.5]
    high: [1.0, 0.5]
    dtype: float32
  noop_action: [0.0, 0.0]
```

Select a no-op that is valid for your task. Zero is not automatically a safe no-op.
LAGEN preserves the declared native action representation.
Structured actions require an explicit compatible representation in your task adapter.

Student RGB must come from the same source observation as its teacher label.
Select student state separately from privileged teacher input.
For a Gymnasium state vector, declare `env.state_space.labels` and use `collection.student_observation.state_key: gymnasium_state`.
See [collection](rollout_collection.md) for the complete settings.

## Train with the public latency wrapper

The wrapper implements the Gymnasium `reset()` and `step()` interface.
Save this example in your Python project:

```python
import gymnasium as gym
from latency_bench.envs.latency_wrapper import ActionLatencyWrapper

env = ActionLatencyWrapper(
    gym.make("MountainCar-v0"),
    noop_action=1,
    env_fps=50,
    obs_fps=10,
    fixed_latency_raw_frames=2,
    hold_policy="one_frame_then_noop",
    ordering_policy="latest_ready",
)
try:
    observation, info = env.reset(seed=7)
    observation, reward, terminated, truncated, info = env.step(2)
finally:
    env.close()
```

One wrapper step submits a decision and advances up to five raw steps.
The action becomes ready after two raw steps.
The wrapper sums rewards and stops immediately at termination or truncation.
Your training code owns the algorithm and training budget.
See [execution semantics](../reference/latency_execution_semantics_en.md) for ordering, capacity, hold, and chunk rules.

## Use the Sample Factory reference path

The reference integration supports native Discrete and bounded Box tasks.
Existing task configurations are under `configs/examples/gymnasium/`.
The InvertedPendulum recipe uses a task-owned finite action catalog.
Other recipes can use native continuous actions.
Copy only settings that match your task.

Run an existing training recipe from the repository root:

```bash
python -m training.train \
  --config configs/examples/gymnasium/inverted_pendulum/small_model_train.yaml
```

Evaluate its selected checkpoint:

```bash
python -m latency_bench eval \
  --config configs/examples/gymnasium/inverted_pendulum/small_model_eval.yaml \
  --checkpoint-path /ABSOLUTE/TEACHER/checkpoint.pth
```

Use [collection](rollout_collection.md) to create canonical demonstrations and convert them for your training framework.
Use the [artifact guide](../artifacts.md) to evaluate existing VLA bundles.
The [standard pipeline](../workflow.md) links the task-specific training recipes.

## Check your integration

Check native observations, rewards, and episode endings with a fixed seed and known action trace.
Check the configured no-op and action mapping.
Compare zero-delay execution with the native environment.
Check that fixed delay changes application frames by the declared raw-step delay.
Check RGB and state shapes before producing a dataset.

Save configurations, traces, and experiment reports under `outputs/` or a published artifact repository.
Record the checkpoint revision, episode count, clocks, latency settings, and seed protocol with each result.
The [artifact guide](../artifacts.md) links published Gymnasium experiment evidence.
