"""Build scheduler components from the ``scheduler`` config section."""

from __future__ import annotations

from typing import Any

from latency_bench.core.clock import EnvClock
from latency_bench.core.types import Action, ActionHoldPolicy, OrderingPolicy
from latency_bench.scheduler.action_queue import ActionScheduler
from latency_bench.scheduler.persistence import (
    ActionPersistencePolicy,
    HoldPolicy,
    OneFrameThenNoopPolicy,
)

_SCHEDULER_ALLOWED_KEYS = frozenset({"ordering_policy", "hold_policy"})


def build_action_scheduler(
    scheduler_cfg: dict[str, Any],
    *,
    clock: EnvClock,
    noop_action: Action,
) -> ActionScheduler:
    return ActionScheduler(
        clock=clock,
        noop_action=noop_action,
        persistence_policy=build_persistence_policy(scheduler_cfg, clock=clock),
        ordering_policy=build_ordering_policy(scheduler_cfg),
    )


def build_ordering_policy(scheduler_cfg: dict[str, Any]) -> OrderingPolicy:
    _raise_if_unsupported_scheduler_fields(scheduler_cfg)
    ordering_policy = scheduler_cfg.get("ordering_policy", OrderingPolicy.ISSUE_ORDER_FIFO.value)
    return _normalize_ordering_policy(ordering_policy, field_name="scheduler.ordering_policy")


def build_persistence_policy(
    scheduler_cfg: dict[str, Any],
    *,
    clock: EnvClock | None = None,
) -> ActionPersistencePolicy:
    mode = _resolve_hold_policy_mode(scheduler_cfg)
    if mode == ActionHoldPolicy.ONE_FRAME_THEN_NOOP:
        return OneFrameThenNoopPolicy()
    if mode == ActionHoldPolicy.HOLD:
        if clock is None:
            raise ValueError("clock is required for hold policy")
        return HoldPolicy(duration_raw_frames=clock.obs_stride_raw_frames)
    raise ValueError(f"Unsupported hold policy: {mode.value}")


def _resolve_hold_policy_mode(scheduler_cfg: dict[str, Any]) -> ActionHoldPolicy:
    _raise_if_unsupported_scheduler_fields(scheduler_cfg)
    hold_policy = scheduler_cfg.get("hold_policy", ActionHoldPolicy.ONE_FRAME_THEN_NOOP.value)
    return _normalize_hold_policy(hold_policy, field_name="scheduler.hold_policy")


def _raise_if_unsupported_scheduler_fields(scheduler_cfg: dict[str, Any]) -> None:
    if set(scheduler_cfg) - _SCHEDULER_ALLOWED_KEYS:
        raise ValueError(
            "scheduler contains unsupported fields; use only "
            "['hold_policy', 'ordering_policy']"
        )


def _normalize_ordering_policy(value: Any, *, field_name: str) -> OrderingPolicy:
    normalized = str(value).strip().lower()
    try:
        return OrderingPolicy(normalized)
    except ValueError as exc:
        raise ValueError(f"{field_name} must be one of ['issue_order_fifo', 'latest_ready']") from exc


def _normalize_hold_policy(value: Any, *, field_name: str) -> ActionHoldPolicy:
    normalized = str(value).strip().lower()
    try:
        return ActionHoldPolicy(normalized)
    except ValueError as exc:
        raise ValueError(
            f"{field_name} must be one of "
            "['one_frame_then_noop', 'hold']"
        ) from exc
