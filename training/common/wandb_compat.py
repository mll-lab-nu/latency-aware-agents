from __future__ import annotations

import os
from netrc import netrc
from pathlib import Path
from typing import Any


_PATCHED = False
_FILTERED_TENSORBOARD_PREFIXES = ("len/", "policy_stats/", "reward/", "stats/")


class _FilteredSummaryWriter:
    def __init__(self, wrapped_writer: Any):
        self._wrapped_writer = wrapped_writer

    def add_scalar(self, tag: str, *args: Any, **kwargs: Any):
        if tag.startswith(_FILTERED_TENSORBOARD_PREFIXES):
            return None
        return self._wrapped_writer.add_scalar(tag, *args, **kwargs)

    def __getattr__(self, name: str):
        return getattr(self._wrapped_writer, name)


def _filtered_summary_writer_class(summary_writer_class: Any):
    class FilteredSummaryWriter(_FilteredSummaryWriter):
        def __init__(self, *args: Any, **kwargs: Any):
            super().__init__(summary_writer_class(*args, **kwargs))

    return FilteredSummaryWriter


def _normalize_tags(value: Any) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        tags = [item.strip() for item in value.split(",") if item.strip()]
        return tags or None
    tags = [
        tag
        for item in value
        for tag in (part.strip() for part in str(item).split(","))
        if tag
    ]
    return tags or None


def _wandb_key_from_env_or_netrc() -> str | None:
    env_key = os.environ.get("WANDB_API_KEY")
    if env_key:
        stripped = env_key.strip()
        if stripped:
            return stripped

    netrc_path = Path.home() / ".netrc"
    if not netrc_path.exists():
        return None

    try:
        auth = netrc(str(netrc_path)).authenticators("api.wandb.ai")
    except Exception:
        return None

    if not auth:
        return None

    _, _, password = auth
    if password and str(password).strip():
        return str(password).strip()
    return None


def patch_sample_factory_wandb() -> None:
    global _PATCHED
    if _PATCHED:
        return

    import sample_factory.utils.wandb_utils as sf_wandb_utils

    def init_wandb(cfg):
        """
        Sample Factory 2.1.1 forces an explicit run id and retries wandb.init().
        Use the configured unique id with W&B resume support so scheduler
        recovery continues the same W&B run instead of creating same-name runs.
        """

        if not getattr(cfg, "with_wandb", False):
            return

        import wandb

        key = _wandb_key_from_env_or_netrc()
        if key:
            wandb.login(key=key, verify=True, relogin=True)

        run_name = str(getattr(cfg, "wandb_unique_id", None) or getattr(cfg, "experiment"))
        cfg.wandb_unique_id = run_name
        os.environ["WANDB_RUN_ID"] = run_name
        os.environ["WANDB_NAME"] = run_name
        os.environ["WANDB_RESUME"] = "allow"
        cfg.wandb_tags = _normalize_tags(getattr(cfg, "wandb_tags", None))

        wandb_group = cfg.env if getattr(cfg, "wandb_group", None) is None else cfg.wandb_group
        init_kwargs = {
            "project": getattr(cfg, "wandb_project", None),
            "entity": getattr(cfg, "wandb_user", None),
            "sync_tensorboard": os.environ.get("LATENCY_BENCH_DISABLE_WANDB_TB_SYNC") != "1",
            "id": run_name,
            "name": run_name,
            "resume": "allow",
            "group": wandb_group,
            "job_type": getattr(cfg, "wandb_job_type", None),
            "tags": cfg.wandb_tags,
        }

        train_dir = getattr(cfg, "train_dir", None)
        experiment = getattr(cfg, "experiment", None)
        if train_dir and experiment:
            run_dir = Path(str(train_dir)).expanduser() / str(experiment)
            run_dir.mkdir(parents=True, exist_ok=True)
            init_kwargs["dir"] = str(run_dir)

        wandb.init(**init_kwargs)
        wandb.config.update(cfg, allow_val_change=True)

        wandb.define_metric("global_step")
        wandb.define_metric("train/env_steps")
        wandb.define_metric("train/*", step_metric="train/env_steps")
        wandb.define_metric("perf/*", step_metric="train/env_steps")
        wandb.define_metric("eval/global_steps")
        wandb.define_metric("eval/*", step_metric="global_step")

    def finish_wandb(cfg):
        if not getattr(cfg, "with_wandb", False):
            return
        import wandb

        if wandb.run is not None:
            wandb.run.finish()

    sf_wandb_utils.init_wandb = init_wandb
    sf_wandb_utils.finish_wandb = finish_wandb

    try:
        import sample_factory.algo.runners.runner as sf_runner

        sf_runner.SummaryWriter = _filtered_summary_writer_class(sf_runner.SummaryWriter)
        sf_runner.init_wandb = init_wandb
    except Exception:
        pass

    _PATCHED = True
