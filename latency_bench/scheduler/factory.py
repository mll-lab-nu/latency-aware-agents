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

_SCHEDULER_ALLOWED_KEYS = frozenset({"ordering_policy", "hold_policy", "hold_last_chunk_action"})


def build_action_scheduler(
    scheduler_cfg: dict[str, Any],
    *,
    clock: EnvClock,
    noop_action: Action,
) -> ActionScheduler:
    _raise_if_unsupported_scheduler_fields(scheduler_cfg)
    return ActionScheduler(
        clock=clock,
        noop_action=noop_action,
        persistence_policy=build_persistence_policy(
            ActionHoldPolicy(scheduler_cfg["hold_policy"]),
            clock=clock,
        ),
        ordering_policy=OrderingPolicy(scheduler_cfg["ordering_policy"]),
        hold_last_chunk_action=scheduler_cfg.get("hold_last_chunk_action", False),
    )


def build_persistence_policy(
    hold_policy: ActionHoldPolicy,
    *,
    clock: EnvClock,
) -> ActionPersistencePolicy:
    if hold_policy == ActionHoldPolicy.HOLD:
        return HoldPolicy(duration_raw_frames=clock.obs_stride_raw_frames)
    return OneFrameThenNoopPolicy()


def _raise_if_unsupported_scheduler_fields(scheduler_cfg: dict[str, Any]) -> None:
    if set(scheduler_cfg) - _SCHEDULER_ALLOWED_KEYS:
        raise ValueError(
            "scheduler contains unsupported fields; use only "
            "['hold_policy', 'ordering_policy', 'hold_last_chunk_action']"
        )
