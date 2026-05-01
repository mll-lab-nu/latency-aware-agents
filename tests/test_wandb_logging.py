from __future__ import annotations

import sys
import types
from argparse import Namespace

import numpy as np
import pytest
import torch

from latency_bench.logging.wandb import (
    WandbKVWriter,
    WandbMetricLogger,
    init_wandb,
    parse_wandb_tags,
    to_wandb_scalar,
    wandb_config_from_args,
)


class FakeWandb:
    def __init__(self) -> None:
        self.logged: list[tuple[dict[str, object], int]] = []

    def log(self, payload, step):
        self.logged.append((dict(payload), int(step)))


def test_parse_wandb_tags_trims_and_drops_empty_values():
    assert parse_wandb_tags("a, b,, ") == ["a", "b"]
    assert parse_wandb_tags([" x ", "", "y"]) == ["x", "y"]
    assert parse_wandb_tags(None) == []


def test_to_wandb_scalar_accepts_python_numpy_and_torch_scalars_only():
    assert to_wandb_scalar(1) == 1
    assert to_wandb_scalar(1.5) == 1.5
    assert to_wandb_scalar(False) is False
    assert to_wandb_scalar(np.float32(2.25)) == pytest.approx(2.25)
    assert to_wandb_scalar(torch.tensor(3.0)) == 3.0
    assert to_wandb_scalar(np.asarray([1.0])) is None
    assert to_wandb_scalar(torch.tensor([3.0])) is None
    assert to_wandb_scalar({"not": "scalar"}) is None


def test_wandb_config_excludes_legacy_wandb_noop_flag():
    args = Namespace(wandb=True, train_steps=16, wandb_project="project")

    assert wandb_config_from_args(args) == {
        "train_steps": 16,
        "wandb_project": "project",
    }


def test_init_wandb_uses_contract_and_propagates_init_errors(monkeypatch):
    calls: list[dict[str, object]] = []
    fake_module = types.ModuleType("wandb")

    def fake_init(**kwargs):
        calls.append(kwargs)

    fake_module.init = fake_init  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "wandb", fake_module)
    args = Namespace(
        wandb=True,
        wandb_project="project",
        wandb_entity=None,
        wandb_run_name="",
        wandb_tags="a, b,,",
    )

    returned = init_wandb(args, default_run_name="default-run")

    assert returned is fake_module
    assert calls == [
        {
            "project": "project",
            "entity": None,
            "name": "default-run",
            "tags": ["a", "b"],
            "config": {
                "wandb_project": "project",
                "wandb_entity": None,
                "wandb_run_name": "",
                "wandb_tags": "a, b,,",
            },
            "save_code": True,
        }
    ]

    def failing_init(**kwargs):
        raise RuntimeError("init failed")

    fake_module.init = failing_init  # type: ignore[attr-defined]
    with pytest.raises(RuntimeError, match="init failed"):
        init_wandb(args, default_run_name="default-run")


def test_metric_logger_logs_scalars_and_skips_non_scalars():
    fake_wandb = FakeWandb()
    logger = WandbMetricLogger(fake_wandb)

    logger.add_scalar("charts/SPS", np.int64(12), global_step=5)
    logger.add_scalar("charts/vector", np.asarray([1, 2]), global_step=6)

    assert fake_wandb.logged == [({"charts/SPS": 12}, 5)]


def test_sb3_wandb_writer_preserves_keys_and_exclusions():
    fake_wandb = FakeWandb()
    writer = WandbKVWriter(fake_wandb)

    writer.write(
        {
            "train/loss": np.float32(0.5),
            "rollout/flag": torch.tensor(True),
            "train/vector": np.asarray([1, 2]),
            "time/fps": 99,
        },
        {
            "time/fps": ("stdout", "wandb"),
        },
        step=123,
    )

    assert fake_wandb.logged == [
        (
            {
                "train/loss": pytest.approx(0.5),
                "rollout/flag": True,
            },
            123,
        )
    ]
