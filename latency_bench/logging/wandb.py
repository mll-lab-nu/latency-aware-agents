from __future__ import annotations

import sys
from collections.abc import Sequence
from typing import Any

try:
    from stable_baselines3.common.logger import KVWriter
except Exception:  # pragma: no cover - SB3 is optional for non-SB3 tests/imports
    KVWriter = object  # type: ignore[misc,assignment]


def parse_wandb_tags(value: str | Sequence[str] | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        items = value.split(",")
    else:
        items = value
    return [str(item).strip() for item in items if str(item).strip()]


def to_wandb_scalar(value: Any) -> int | float | bool | None:
    shape = getattr(value, "shape", None)
    if shape not in (None, (), []):
        return None

    if hasattr(value, "item"):
        try:
            value = value.item()
        except Exception:
            return None

    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return float(value)
    return None


def wandb_config_from_args(args: Any) -> dict[str, Any]:
    return {key: value for key, value in vars(args).items() if key != "wandb"}


def init_wandb(args: Any, default_run_name: str) -> Any:
    missing = [
        name
        for name in ("wandb_project", "wandb_entity", "wandb_run_name", "wandb_tags")
        if not hasattr(args, name)
    ]
    if missing:
        raise AttributeError(f"args missing required W&B fields: {', '.join(missing)}")

    import wandb

    run_name = getattr(args, "wandb_run_name", "") or default_run_name
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=run_name,
        tags=parse_wandb_tags(args.wandb_tags),
        config=wandb_config_from_args(args),
        save_code=True,
    )
    return wandb


class WandbMetricLogger:
    def __init__(self, wandb_module: Any) -> None:
        self.wandb = wandb_module

    def add_scalar(self, tag: str, value: Any, global_step: int) -> None:
        scalar = to_wandb_scalar(value)
        if scalar is None:
            return
        self.wandb.log({tag: scalar}, step=int(global_step))

    def close(self) -> None:
        return None


class WandbKVWriter(KVWriter):  # type: ignore[misc]
    def __init__(self, wandb_module: Any) -> None:
        self.wandb = wandb_module

    def write(
        self,
        key_values: dict[str, Any],
        key_excluded: dict[str, tuple[str, ...]],
        step: int = 0,
    ) -> None:
        payload: dict[str, int | float | bool] = {}
        for key, value in key_values.items():
            excluded = key_excluded.get(key, ())
            if excluded is not None and "wandb" in excluded:
                continue
            scalar = to_wandb_scalar(value)
            if scalar is not None:
                payload[key] = scalar
        if payload:
            self.wandb.log(payload, step=int(step))

    def close(self) -> None:
        return None


def build_sb3_logger(wandb_module: Any):
    from stable_baselines3.common.logger import HumanOutputFormat, Logger

    return Logger(
        folder=None,
        output_formats=[
            HumanOutputFormat(sys.stdout),
            WandbKVWriter(wandb_module),
        ],
    )
