from __future__ import annotations

import sys
import types
from argparse import Namespace


def _install_fake_sample_factory(monkeypatch):
    sample_factory = types.ModuleType("sample_factory")
    sample_factory.__path__ = []
    utils = types.ModuleType("sample_factory.utils")
    utils.__path__ = []
    wandb_utils = types.ModuleType("sample_factory.utils.wandb_utils")
    algo = types.ModuleType("sample_factory.algo")
    algo.__path__ = []
    runners = types.ModuleType("sample_factory.algo.runners")
    runners.__path__ = []
    runner = types.ModuleType("sample_factory.algo.runners.runner")

    class BaseSummaryWriter:
        instances = []

        def __init__(self, *args, **kwargs):
            self.scalars = []
            BaseSummaryWriter.instances.append(self)

        def add_scalar(self, tag, scalar_value, global_step=None, *args, **kwargs):
            self.scalars.append((tag, scalar_value, global_step, args, kwargs))
            return "logged"

        def flush(self):
            return "flushed"

    runner.SummaryWriter = BaseSummaryWriter

    sample_factory.utils = utils
    utils.wandb_utils = wandb_utils
    sample_factory.algo = algo
    algo.runners = runners
    runners.runner = runner

    monkeypatch.setitem(sys.modules, "sample_factory", sample_factory)
    monkeypatch.setitem(sys.modules, "sample_factory.utils", utils)
    monkeypatch.setitem(sys.modules, "sample_factory.utils.wandb_utils", wandb_utils)
    monkeypatch.setitem(sys.modules, "sample_factory.algo", algo)
    monkeypatch.setitem(sys.modules, "sample_factory.algo.runners", runners)
    monkeypatch.setitem(sys.modules, "sample_factory.algo.runners.runner", runner)
    return wandb_utils, runner, BaseSummaryWriter


def test_sample_factory_wandb_patch_filters_noisy_tensorboard_tags(monkeypatch, tmp_path):
    from training.common import wandb_compat

    wandb_utils, runner, base_writer_cls = _install_fake_sample_factory(monkeypatch)
    monkeypatch.setattr(wandb_compat, "_PATCHED", False)

    defined_metrics = []
    fake_wandb = types.ModuleType("wandb")
    fake_wandb.run = None
    fake_wandb.init = lambda **kwargs: None
    fake_wandb.define_metric = lambda *args, **kwargs: defined_metrics.append((args, kwargs))
    fake_wandb.config = types.SimpleNamespace(update=lambda cfg, allow_val_change: None)
    fake_wandb.login = lambda **kwargs: None
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

    wandb_compat.patch_sample_factory_wandb()

    cfg = Namespace(
        with_wandb=True,
        train_dir=str(tmp_path),
        experiment="exp",
        env="latency_flappy",
        wandb_group=None,
        wandb_project="project",
        wandb_user=None,
        wandb_job_type=None,
        wandb_tags=None,
        wandb_unique_id="run",
    )
    wandb_utils.init_wandb(cfg)

    assert [args[0] for args, _ in defined_metrics] == [
        "global_step",
        "train/env_steps",
        "train/*",
        "perf/*",
        "eval/global_steps",
        "eval/*",
    ]
    assert defined_metrics[-1][1] == {"step_metric": "global_step"}

    writer = runner.SummaryWriter(str(tmp_path / "tb"))
    for tag in (
        "len/len",
        "policy_stats/avg_min_raw_reward",
        "reward/reward",
        "stats/master_process_memory_mb",
    ):
        writer.add_scalar(tag, 1.0, 5)
    for tag in (
        "perf/_fps",
        "train/loss",
        "episode/return",
        "rollout/ep_rew_mean",
    ):
        writer.add_scalar(tag, 2.0, 5)

    assert [tag for tag, *_ in base_writer_cls.instances[-1].scalars] == [
        "perf/_fps",
        "train/loss",
        "episode/return",
        "rollout/ep_rew_mean",
    ]
