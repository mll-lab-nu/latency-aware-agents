from __future__ import annotations

import json
import math
import os
import time
from datetime import datetime
from typing import Any

from latency_bench.logging.wandb import WandbMetricLogger, init_wandb


class MetricLogger(WandbMetricLogger):
    def __init__(self, wandb_module: Any) -> None:
        super().__init__(wandb_module)


def setup_wandb(args: Any) -> Any:
    return init_wandb(args, default_run_name=f"demon_attack_cleanrl_{int(time.time())}")


def current_time_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, bool)) or value is None:
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return float(value) if math.isfinite(value) else None

    shape = getattr(value, "shape", None)
    is_scalar_shape = shape in (None, (), [])
    if not is_scalar_shape and hasattr(shape, "__len__"):
        is_scalar_shape = len(shape) == 0
    if is_scalar_shape:
        item = getattr(value, "item", None)
        if callable(item):
            try:
                return _json_safe(item())
            except Exception:
                return str(value)

    return str(value)


class JsonlEventLogger:
    def __init__(self, path: str, run_id: str) -> None:
        self.path = path
        self.run_id = run_id
        self._file = None
        if not path:
            return

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._file = open(path, "w", encoding="utf-8")

    def emit(self, event: dict[str, Any]) -> None:
        if self._file is None:
            return
        payload = dict(event)
        payload.setdefault("run_id", self.run_id)
        payload.setdefault("time", current_time_iso())
        self._file.write(json.dumps(_json_safe(payload), separators=(",", ":")) + "\n")
        self._file.flush()

    def close(self) -> None:
        if self._file is None:
            return
        self._file.close()
        self._file = None


def _fmt_float(value: Any, precision: int = 3) -> str:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "nan"
    if not math.isfinite(value):
        return "nan"
    return f"{value:.{precision}f}"


def _fmt_int(value: Any) -> str:
    try:
        return str(int(value))
    except (TypeError, ValueError):
        return "0"


def _fmt_general(value: Any, precision: int = 6) -> str:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "nan"
    if not math.isfinite(value):
        return "nan"
    return f"{value:.{precision}g}"


def format_train_start(event: dict[str, Any]) -> str:
    return (
        f"[train] run={event.get('run_id', '')} device={event.get('device', '')} "
        f"envs={event.get('n_envs', '')} steps={event.get('n_steps', '')} "
        f"updates={event.get('num_updates', '')} obs={event.get('obs_mode', '')} "
        f"save={event.get('save_path', '')}"
    )


def format_rollout_progress(event: dict[str, Any]) -> str:
    return (
        f"[rollout] u={event.get('update')}/{event.get('num_updates')} "
        f"step={event.get('rollout_step')}/{event.get('n_steps')} "
        f"gstep={event.get('global_step')} sps={_fmt_int(event.get('sps'))} "
        f"eps={event.get('episode_count')} "
        f"last={_fmt_float(event.get('last_ep_rew'))} "
        f"mean={_fmt_float(event.get('ep_rew_mean'))} "
        f"len={_fmt_float(event.get('ep_len_mean'), 1)}"
    )


def format_update(event: dict[str, Any]) -> str:
    return (
        f"[update] u={event.get('update')}/{event.get('num_updates')} "
        f"gstep={event.get('global_step')} sps={_fmt_int(event.get('sps'))} "
        f"eps={event.get('episode_count')} rew={_fmt_float(event.get('ep_rew_mean'))} "
        f"len={_fmt_float(event.get('ep_len_mean'), 1)} "
        f"loss={_fmt_float(event.get('loss'))} "
        f"kl={_fmt_float(event.get('approx_kl'), 4)} "
        f"clip={_fmt_float(event.get('clipfrac'))} "
        f"lr={_fmt_general(event.get('learning_rate'))}"
    )


def format_checkpoint_save(event: dict[str, Any]) -> str:
    return (
        f"[ckpt] gstep={event.get('global_step')} path={event.get('path', '')} "
        f"reason={event.get('reason', '')}"
    )


def format_train_end(event: dict[str, Any]) -> str:
    status = event.get("status", "")
    base = (
        f"[train] done run={event.get('run_id', '')} gstep={event.get('global_step')} "
        f"updates={event.get('update')} eps={event.get('episode_count')} "
        f"rew={_fmt_float(event.get('ep_rew_mean'))} save={event.get('save_path', '')}"
    )
    if status == "success":
        return base
    return (
        f"{base} status={status} error={event.get('exception_type', '')}:"
        f"{event.get('exception_message', '')}"
    )


def emit_event(
    jsonl_logger: JsonlEventLogger,
    event: dict[str, Any],
    console_formatter=None,
) -> None:
    event.setdefault("time", current_time_iso())
    event.setdefault("run_id", jsonl_logger.run_id)
    jsonl_logger.emit(event)
    if console_formatter is not None:
        print(console_formatter(event), flush=True)
