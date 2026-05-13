from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ExecutorMode(str, Enum):
    PAUSED = "paused"
    SIMULATED = "simulated"
    REALTIME = "realtime"


class ActionHoldPolicy(str, Enum):
    ONE_FRAME_THEN_NOOP = "one_frame_then_noop"
    HOLD = "hold"


class OrderingPolicy(str, Enum):
    ISSUE_ORDER_FIFO = "issue_order_fifo"
    LATEST_READY = "latest_ready"


class LatencySourceType(str, Enum):
    ZERO = "zero"
    FIXED = "fixed"
    NORMAL_DISTRIBUTION = "normal_distribution"
    UNIFORM_DISTRIBUTION = "uniform_distribution"
    TRACE_REPLAY = "trace_replay"
    EMPIRICAL_SAMPLE = "empirical_sample"
    EMPIRICAL_TRACE_REPLAY = "empirical_trace_replay"
    MEASURED = "measured"


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
    start_wall_time: float | None = None
    end_wall_time: float | None = None
    inference_ms: float | None = None
    device: str | None = None
    latency_type: str | None = None
    profile_id: str | None = None
    profile_type: str | None = None
    target_device: str | None = None
    profile_path: str | None = None
    sample_index: int | None = None
    trace_index: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ActionEvent:
    action_id: int
    action: Action
    obs_env_step: int
    ready_time_ms: float
    ready_step_float: float
    latency: LatencyRecord
    latency_raw_frames: int | None = None
    ready_raw_frame: int | None = None
    apply_raw_frame: int | None = None
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
    episode_return_env: float | None = None
    return_raw: float | None = None
    game_score: float | None = None
    mean_latency_ms: float | None = None
    p90_latency_ms: float | None = None
    p99_latency_ms: float | None = None
    num_actions: int = 0
    num_dropped_actions: int = 0
    num_invalid_actions: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
