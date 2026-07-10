from __future__ import annotations

from types import SimpleNamespace

import pytest

from latency_bench.cross_device import planning


def _args(*, stage: str) -> SimpleNamespace:
    return SimpleNamespace(
        checkpoint_kind=None,
        dry_run=True,
        latency_seed_set=None,
        n_episodes=None,
        only_device=None,
        only_env=None,
        only_policy=None,
        seed_set=None,
        smoke=False,
        smoke_max_steps=None,
        target="a100/instance_target" if stage == "sim_cross" else None,
    )


def _spec() -> dict:
    device = {
        "id": "a100",
        "display_name": "A100",
        "profile_id": "L_A100",
        "aliases": [],
    }
    return {
        "checkpoint_kind": "best",
        "n_episodes": 100,
        "seed_set": [5, 7],
        "latency_seed_set": [11, 13, 17],
        "devices": [device],
        "rows": [
            {
                "policy": "openvla-flappy",
                "env": "Flappy",
                "model_type": "openvla",
                "env_name": "flappy",
            }
        ],
    }


@pytest.mark.parametrize("stage", ["sim_same", "sim_cross"])
def test_sim_plan_expands_eval_and_latency_seed_product(tmp_path, stage):
    spec = _spec()
    current_device = spec["devices"][0]

    plan = planning._build_plan(
        _args(stage=stage),
        spec,
        stage,
        tmp_path / "results",
        tmp_path / "profiles",
        current_device,
        instance_id="instance_local",
        instance_origin="manual_override",
    )

    assert [(item.task.seed, item.task.latency_seed) for item in plan] == [
        (5, 11),
        (5, 13),
        (5, 17),
        (7, 11),
        (7, 13),
        (7, 17),
    ]
    assert [item.task.output_dir.name for item in plan] == [
        "best_seed_5_latency_seed_11_n100",
        "best_seed_5_latency_seed_13_n100",
        "best_seed_5_latency_seed_17_n100",
        "best_seed_7_latency_seed_11_n100",
        "best_seed_7_latency_seed_13_n100",
        "best_seed_7_latency_seed_17_n100",
    ]


def test_measure_plan_expands_only_eval_seeds(tmp_path):
    spec = _spec()
    current_device = spec["devices"][0]

    plan = planning._build_plan(
        _args(stage="measure"),
        spec,
        "measure",
        tmp_path / "results",
        tmp_path / "profiles",
        current_device,
        instance_id="instance_local",
        instance_origin="manual_override",
    )

    assert [(item.task.seed, item.task.latency_seed) for item in plan] == [(5, None), (7, None)]
    assert [item.task.output_dir.name for item in plan] == [
        "best_seed_5_n100",
        "best_seed_7_n100",
    ]


def test_sim_plan_uses_latency_seed_cli_override(tmp_path):
    spec = _spec()
    current_device = spec["devices"][0]
    args = _args(stage="sim_same")
    args.latency_seed_set = "19,23"

    plan = planning._build_plan(
        args,
        spec,
        "sim_same",
        tmp_path / "results",
        tmp_path / "profiles",
        current_device,
        instance_id="instance_local",
        instance_origin="manual_override",
    )

    assert [(item.task.seed, item.task.latency_seed) for item in plan] == [
        (5, 19),
        (5, 23),
        (7, 19),
        (7, 23),
    ]
