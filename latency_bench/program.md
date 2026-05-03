# Task: Implement a Modular Latency-Sensitive Control Framework

## Goal

Implement a clean modular framework for latency-sensitive real-time game control.

Core semantics:

- The environment is frame-based.
- One `env.step(action)` advances exactly one environment frame.
- `fps_env` defines the environment clock.
- `frame_ms = 1000 / fps_env`.
- The agent observes the environment and launches policy inference at a configurable control frequency.
- A policy action is not applied immediately. It becomes available only after a latency.
- Latency can be fixed, sampled from a distribution, replayed from a trace, or measured from real model inference.
- While waiting for a new action, the system follows a configurable action persistence policy:
  - hold / repeat previous action
  - or apply a one-shot action and then fall back to no-op
- The first executor to implement is `SimulatedLatencyExecutor`.
- `SimulatedLatencyExecutor` must not sleep in wall-clock time. It should advance the environment in virtual time as fast as possible.
- The code should be designed so that `PausedExecutor` and `RealTimeExecutor` can be added later.

## Target Directory Structure

Create or update the following structure:

```text
latency_bench/
  core/
    __init__.py
    types.py
    clock.py
    config.py

  envs/
    __init__.py
    base.py

  observation/
    __init__.py
    builder.py
    history.py

  policy/
    __init__.py
    base.py

  latency/
    __init__.py
    sources.py
    profiler.py
    trace.py

  scheduler/
    __init__.py
    decision.py
    persistence.py
    action_queue.py

  executors/
    __init__.py
    simulated.py

  logging/
    __init__.py
    records.py
    trajectory_logger.py
    metrics.py

  experiments/
    run_simulated_latency.py
    run_cross_device_latency.py

configs/
  base/
    flappy.yaml
  experiments/
    fixed_latency_curve.yaml
    trace_replay.yaml
    cross_device_replay.yaml

tests/
  test_clock.py
  test_config.py
  test_latency_sources.py
  test_decision_scheduler.py
  test_action_queue.py
  test_persistence.py
  test_simulated_executor.py
````

Do not rewrite existing environment implementation unless necessary. Add adapters around existing environments.

---

## 1. `latency_bench/core/types.py`

Implement shared dataclasses and enums.

Required enums:

```python
class ExecutorMode(str, Enum):
    PAUSED = "paused"
    SIMULATED = "simulated"
    REALTIME = "realtime"

class PersistenceMode(str, Enum):
    HOLD_PREVIOUS = "hold_previous"
    NOOP_AFTER_ONESHOT = "noop_after_oneshot"

class StaleActionMode(str, Enum):
    FIFO = "fifo"
    LATEST_READY = "latest_ready"
    DROP_EXPIRED = "drop_expired"

class LatencySourceType(str, Enum):
    ZERO = "zero"
    FIXED = "fixed"
    NORMAL_DISTRIBUTION = "normal_distribution"
    TRACE_REPLAY = "trace_replay"
    MEASURED = "measured"
```

Required dataclasses:

```python
@dataclass
class Action:
    value: Any
    name: str | None = None
    is_noop: bool = False
    is_oneshot: bool = False

@dataclass
class Observation:
    data: Any
    env_step: int
    sim_time_ms: float
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass
class StepResult:
    observation: Observation | None
    reward: float
    done: bool
    truncated: bool = False
    info: dict[str, Any] = field(default_factory=dict)

@dataclass
class PolicyOutput:
    action: Action
    raw_output: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass
class LatencyRecord:
    obs_id: int
    latency_ms: float
    source: str
    start_wall_time: float | None = None
    end_wall_time: float | None = None
    preprocess_ms: float | None = None
    inference_ms: float | None = None
    postprocess_ms: float | None = None
    end_to_end_ms: float | None = None
    device: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass
class ActionEvent:
    action_id: int
    obs_id: int
    action: Action
    obs_env_step: int
    obs_time_ms: float
    latency_ms: float
    ready_time_ms: float
    ready_step_float: float
    applied: bool = False
    applied_env_step: int | None = None
    applied_time_ms: float | None = None
    dropped: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass
class StepRecord:
    episode_id: int
    env_step: int
    sim_time_ms: float
    active_action: Action
    reward: float
    done: bool
    info: dict[str, Any] = field(default_factory=dict)

@dataclass
class EpisodeMetrics:
    episode_id: int
    episode_return: float
    survival_steps: int
    game_score: float | None = None
    mean_latency_ms: float | None = None
    p90_latency_ms: float | None = None
    p99_latency_ms: float | None = None
    num_actions: int = 0
    num_dropped_actions: int = 0
    num_invalid_actions: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
```

---

## 2. `latency_bench/core/clock.py`

Implement:

```python
class EnvClock:
    def __init__(self, fps_env: float):
        ...

    @property
    def frame_ms(self) -> float:
        ...

    def step_to_time_ms(self, env_step: int) -> float:
        ...

    def time_ms_to_step_float(self, time_ms: float) -> float:
        ...

    def latency_ms_to_frames_float(self, latency_ms: float) -> float:
        ...
```

Implement a residual accumulator:

```python
class FrameAccumulator:
    def __init__(self):
        self.residual_frames = 0.0

    def consume(self, frames_float: float) -> int:
        """
        Add fractional frames and return the number of whole frames to advance.
        Keep the remaining fractional frames internally.
        """
        ...
```

Tests:

* `fps_env=30` gives `frame_ms≈33.333`.
* `latency_ms=100` gives approximately `3.0` frames.
* Accumulator should preserve fractional frame residuals across calls.

---

## 3. `latency_bench/core/config.py`

Implement YAML config loading.

Requirements:

* Load YAML configs from `configs/`.
* Support recursive `__base__`.
* Current config overrides base config.
* Support nested dictionary merge.
* Validate required fields.
* Return a plain dictionary or a dataclass config object.

Example:

```yaml
__base__: ../base/flappy.yaml

experiment:
  name: fixed_latency_curve

latency:
  type: fixed
  fixed_latency_ms: 100
```

If `__base__` exists:

1. Load base config first.
2. Recursively resolve base config.
3. Merge current config into base config.
4. Current config values override base config values.

Implement:

```python
def load_config(path: str | Path) -> dict:
    ...

def merge_dicts(base: dict, override: dict) -> dict:
    ...

def validate_config(config: dict) -> None:
    ...
```

Tests:

* Load config without base.
* Load config with one base.
* Load config with nested base.
* Override nested fields correctly.

---

## 4. `configs/base/flappy.yaml`

Create a base config:

```yaml
experiment:
  name: flappy_base
  seed: 0
  num_episodes: 10

evaluation:
  eval_max_steps: 10000

env:
  name: flappy
  fps_env: 30
  noop_action: noop

control:
  frequency_hz: 30
  observe_every_n_frames: 1

latency:
  type: zero

scheduler:
  persistence_mode: noop_after_oneshot
  stale_action_mode: latest_ready

policy:
  type: random

logging:
  output_dir: outputs/flappy_base
  save_step_records: true
  save_action_records: true
  save_latency_records: true
```

---

## 5. Example experiment configs

### `configs/experiments/fixed_latency_curve.yaml`

```yaml
__base__: ../base/flappy.yaml

experiment:
  name: fixed_latency_curve
  num_episodes: 20

latency:
  type: fixed
  fixed_latency_ms: 100

logging:
  output_dir: outputs/fixed_latency_curve
```

### `configs/experiments/trace_replay.yaml`

```yaml
__base__: ../base/flappy.yaml

experiment:
  name: trace_replay
  num_episodes: 20

latency:
  type: trace_replay
  trace_path: traces/small_device_latency.jsonl
  loop_trace: true

logging:
  output_dir: outputs/trace_replay
```

### `configs/experiments/cross_device_replay.yaml`

```yaml
__base__: ../base/flappy.yaml

experiment:
  name: cross_device_replay
  num_episodes: 20

latency:
  type: trace_replay
  trace_path: traces/profile_device_latency.jsonl
  loop_trace: true

cross_device:
  profile_device: small_gpu
  simulation_device: real_gpu
  target_relation: real_replay_profile_should_match_profile_native

logging:
  output_dir: outputs/cross_device_replay
```

---

## 6. `latency_bench/envs/base.py`

Implement abstract environment adapter:

```python
class EnvAdapter(ABC):
    fps_env: float
    noop_action: Action

    @abstractmethod
    def reset(self, seed: int | None = None) -> Observation:
        ...

    @abstractmethod
    def step(self, action: Action) -> StepResult:
        ...

    def render_frame(self) -> Any:
        ...

    def get_state_info(self) -> dict[str, Any]:
        return {}
```

Also implement a simple `DummyEnvAdapter` for tests:

* Has integer state.
* Each step increments state by 1.
* Reward = 1.0 per step.
* Done after configurable max steps.
* Records applied actions.

---

## 7. `latency_bench/observation/builder.py`

Implement:

```python
class ObservationBuilder:
    def build(self, env: EnvAdapter, history: HistoryBuffer) -> Observation:
        ...
```

Initial implementation can return the latest observation or rendered frame.

---

## 8. `latency_bench/observation/history.py`

Implement:

```python
class HistoryBuffer:
    def __init__(self, max_len: int = 16):
        ...

    def append(self, observation: Observation) -> None:
        ...

    def last(self) -> Observation | None:
        ...

    def last_k(self, k: int) -> list[Observation]:
        ...
```

---

## 9. `latency_bench/policy/base.py`

Implement base policy interface:

```python
class PolicyRunner(ABC):
    @abstractmethod
    def predict(self, observation: Observation) -> PolicyOutput:
        ...
```

Implement simple policies:

```python
class RandomPolicyRunner(PolicyRunner):
    ...

class ConstantPolicyRunner(PolicyRunner):
    ...
```

---

## 10. `latency_bench/latency/sources.py`

Implement latency source interface:

```python
class LatencySource(ABC):
    @abstractmethod
    def next_latency_ms(self, context: dict[str, Any] | None = None) -> float:
        ...
```

Implement:

```python
class ZeroLatencySource(LatencySource):
    ...

class FixedLatencySource(LatencySource):
    ...

class NormalDistributionLatencySource(LatencySource):
    ...

class TraceReplayLatencySource(LatencySource):
    ...

class MeasuredLatencySource(LatencySource):
    ...
```

Behavior:

* `ZeroLatencySource` always returns `0.0`.
* `FixedLatencySource` always returns `fixed_latency_ms`.
* `NormalDistributionLatencySource` samples from a clamped normal distribution.
* `TraceReplayLatencySource` reads a JSONL file with records containing `latency_ms`.
* `MeasuredLatencySource` wraps a `LatencyProfiler`.

Trace JSONL format:

```json
{"obs_id": 0, "latency_ms": 35.2}
{"obs_id": 1, "latency_ms": 41.8}
{"obs_id": 2, "latency_ms": 97.5}
```

Tests:

* Fixed latency returns constant value.
* Trace replay returns values in order.
* Trace replay loops if `loop_trace=True`.
* Distribution source samples only from given values if using empirical mode.

---

## 11. `latency_bench/latency/profiler.py`

Implement:

```python
class LatencyProfiler:
    def run_and_measure(self, fn: Callable, *args, **kwargs) -> tuple[Any, LatencyRecord]:
        ...
```

Use `time.perf_counter()`.

For GPU workloads, leave a hook:

```python
def synchronize_if_needed(self) -> None:
    ...
```

Do not require PyTorch as a hard dependency.

---

## 12. `latency_bench/latency/trace.py`

Implement helpers:

```python
def load_latency_trace(path: str | Path) -> list[float]:
    ...

def save_latency_trace(records: list[LatencyRecord], path: str | Path) -> None:
    ...
```

---

## 13. `latency_bench/scheduler/decision.py`

Implement decision scheduler.

```python
class DecisionScheduler(ABC):
    @abstractmethod
    def should_observe(self, env_step: int, sim_time_ms: float) -> bool:
        ...
```

Implement:

```python
class EveryFrameDecisionScheduler(DecisionScheduler):
    ...

class FixedFrameIntervalDecisionScheduler(DecisionScheduler):
    ...

class FixedFrequencyDecisionScheduler(DecisionScheduler):
    ...
```

Behavior:

* Every frame: observe at every env step.
* Fixed frame interval: observe every `n` frames.
* Fixed frequency: derive interval from `fps_env / frequency_hz`.

Tests:

* `fps_env=30`, `frequency_hz=10` should observe every 3 frames.

---

## 14. `latency_bench/scheduler/persistence.py`

Implement action persistence policies.

```python
class ActionPersistencePolicy(ABC):
    @abstractmethod
    def on_action_applied(self, action: Action) -> None:
        ...

    @abstractmethod
    def get_action_for_step(self, current_action: Action, noop_action: Action) -> Action:
        ...
```

Implement:

```python
class HoldPreviousActionPolicy(ActionPersistencePolicy):
    """
    Keep executing the latest applied action until a new action is applied.
    """
    ...

class NoopAfterOneShotPolicy(ActionPersistencePolicy):
    """
    If an action is one-shot, apply it for one frame and then fall back to noop.
    Non-one-shot actions may be held.
    """
    ...
```

Tests:

* Hold previous keeps action across frames.
* Noop-after-one-shot applies one-shot action once, then returns noop.

---

## 15. `latency_bench/scheduler/action_queue.py`

Implement action scheduler.

```python
class ActionScheduler:
    def __init__(
        self,
        clock: EnvClock,
        noop_action: Action,
        persistence_policy: ActionPersistencePolicy,
        stale_action_mode: StaleActionMode = StaleActionMode.LATEST_READY,
    ):
        ...

    def enqueue(self, event: ActionEvent) -> None:
        ...

    def update(self, current_step: int, current_time_ms: float) -> Action:
        ...

    def get_pending_events(self) -> list[ActionEvent]:
        ...

    def get_applied_events(self) -> list[ActionEvent]:
        ...
```

Semantics:

* Each action event has `ready_time_ms`.
* At each env step, all events with `ready_time_ms <= current_time_ms` are ready.
* If stale mode is `FIFO`, apply the earliest ready event.
* If stale mode is `LATEST_READY`, apply the latest ready event and mark older ready events as dropped.
* If stale mode is `DROP_EXPIRED`, drop events older than a configurable expiration window.
* After applying a new action, delegate to the persistence policy for later frames.

Tests:

* Fixed 2-frame latency action applies at the correct step.
* FIFO mode applies actions in order.
* LATEST_READY mode drops older ready actions if multiple are ready.
* Persistence policy affects active action correctly.

---

## 16. `latency_bench/executors/simulated.py`

Implement the main executor.

```python
class SimulatedLatencyExecutor:
    def __init__(
        self,
        env: EnvAdapter,
        policy: PolicyRunner,
        observation_builder: ObservationBuilder,
        history: HistoryBuffer,
        decision_scheduler: DecisionScheduler,
        latency_source: LatencySource,
        action_scheduler: ActionScheduler,
        clock: EnvClock,
        logger: TrajectoryLogger | None = None,
    ):
        ...

    def run_episode(self, episode_id: int, seed: int | None = None, eval_max_steps: int = 10000) -> EpisodeMetrics:
        ...
```

Execution semantics:

For each env frame:

1. Compute `current_time_ms = clock.step_to_time_ms(env_step)`.
2. If `decision_scheduler.should_observe(...)` is true:

   * Build observation.
   * Run policy to get predicted action.
   * Get latency from latency source.
   * Create `ActionEvent`.
   * `ready_time_ms = current_time_ms + latency_ms`.
   * Enqueue action event.
3. Call `action_scheduler.update(current_step, current_time_ms)` to get active action.
4. Step environment once with active action.
5. Append observation/result to history.
6. Log step/action/latency records.
7. Stop on done/truncated or max steps.

Important:

* Do not call `time.sleep`.
* Do not depend on wall-clock time in this executor.
* This executor uses virtual time only.

Tests:

* With zero latency, action from obs at step t can affect step t or the next step depending on implementation choice; document the chosen convention and test it.
* With fixed latency equal to 2 frames, action from obs at step t is applied at step t+2.
* With trace replay latency, applied action times match trace values.
* Episode metrics are computed.

---

## 17. `latency_bench/logging/trajectory_logger.py`

Implement logger.

```python
class TrajectoryLogger:
    def __init__(self, output_dir: str | Path):
        ...

    def log_step(self, record: StepRecord) -> None:
        ...

    def log_action_event(self, event: ActionEvent) -> None:
        ...

    def log_latency(self, record: LatencyRecord) -> None:
        ...

    def flush(self) -> None:
        ...
```

Save JSONL files:

```text
steps.jsonl
actions.jsonl
latencies.jsonl
episode_metrics.jsonl
```

---

## 18. `latency_bench/logging/metrics.py`

Implement:

```python
def compute_episode_metrics(
    episode_id: int,
    step_records: list[StepRecord],
    action_events: list[ActionEvent],
    latency_values_ms: list[float],
    metadata: dict[str, Any] | None = None,
) -> EpisodeMetrics:
    ...
```

Compute:

* episode return
* survival steps
* mean latency
* p90 latency
* p99 latency
* number of actions
* number of dropped actions

---

## 19. `latency_bench/experiments/run_simulated_latency.py`

Implement CLI:

```bash
python -m latency_bench.experiments.run_simulated_latency \
  --config configs/experiments/fixed_latency_curve.yaml
```

Behavior:

1. Load config.
2. Construct env adapter.
3. Construct policy.
4. Construct clock.
5. Construct decision scheduler.
6. Construct latency source.
7. Construct persistence policy.
8. Construct action scheduler.
9. Construct simulated executor.
10. Run episodes.
11. Save logs and metrics.

For now, use `DummyEnvAdapter` and simple policies if real env adapters are not available.

---

## 20. `latency_bench/experiments/run_cross_device_latency.py`

Implement CLI skeleton:

```bash
python -m latency_bench.experiments.run_cross_device_latency \
  --config configs/experiments/cross_device_replay.yaml
```

Initial behavior:

* Load latency trace from profile device.
* Replay or sample latency on current machine using `SimulatedLatencyExecutor`.
* Save metrics.
* Include config metadata in output.

Do not implement full GPU profiling yet unless existing model code is available.

---

## 21. Required Tests

Implement pytest tests:

```text
tests/test_clock.py
tests/test_config.py
tests/test_latency_sources.py
tests/test_decision_scheduler.py
tests/test_action_queue.py
tests/test_persistence.py
tests/test_simulated_executor.py
```

Important test cases:

1. Time conversion:

   * fps=30
   * frame_ms≈33.333
   * 100ms≈3 frames

2. Config inheritance:

   * `__base__` loads and merges correctly
   * current config overrides base config

3. Trace replay:

   * latencies are returned in order
   * loop mode works

4. Decision frequency:

   * fps=30 and control frequency=10Hz gives observe every 3 frames

5. Action delay:

   * action with latency 2 frames is applied exactly 2 frames later

6. Persistence:

   * hold previous action persists
   * one-shot action falls back to noop

7. Simulated executor:

   * no wall-clock sleep
   * correct number of env steps
   * logs step/action/latency records
   * returns episode metrics

---

## 22. Design Constraints

* Keep modules small and testable.
* Do not hard-code Flappy Bird details into the core framework.
* Do not mix environment frequency with control frequency.
* Do not mix latency with frame skip.
* Do not use wall-clock sleeping in `SimulatedLatencyExecutor`.
* Store experiment configs in YAML under `configs/`.
* Put config loading logic in `latency_bench/core/config.py`.
* Use dataclasses for records and events.
* Make it easy to add:

  * `PausedExecutor`
  * `RealTimeExecutor`
  * real Flappy Bird adapter
  * real VLM/VLA policy runner
  * cross-device latency profiling
