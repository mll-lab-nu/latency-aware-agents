from __future__ import annotations

import torch

from training.common.checkpoints import (
    checkpoint_dir,
    experiment_dir,
    find_checkpoint,
    format_checkpoint_name,
    read_checkpoint_backend_type,
)


def test_format_checkpoint_name_zero_pads_sort_fields():
    assert format_checkpoint_name(train_step=7, env_steps=42) == "checkpoint_000000007_000000000042.pth"


def test_find_latest_checkpoint_ignores_best_files(tmp_path):
    ckpt_dir = checkpoint_dir(tmp_path, "run_a")
    ckpt_dir.mkdir(parents=True)
    older = ckpt_dir / format_checkpoint_name(train_step=1, env_steps=128)
    newer = ckpt_dir / format_checkpoint_name(train_step=2, env_steps=256)
    best = ckpt_dir / "best_999999999_999999999999_reward_999.000.pth"
    for path in (older, newer, best):
        path.write_bytes(b"checkpoint")

    assert find_checkpoint(tmp_path, "run_a", kind="latest") == newer


def test_find_best_checkpoint_uses_only_best_files(tmp_path):
    ckpt_dir = checkpoint_dir(tmp_path, "run_a")
    ckpt_dir.mkdir(parents=True)
    latest = ckpt_dir / format_checkpoint_name(train_step=9, env_steps=900)
    best = ckpt_dir / "best_000000003_000000000300_reward_3.000.pth"
    for path in (latest, best):
        path.write_bytes(b"checkpoint")

    assert find_checkpoint(tmp_path, "run_a", kind="best") == best


def test_experiment_and_checkpoint_dirs_match_sample_factory_contract(tmp_path):
    assert experiment_dir(tmp_path, "run_a") == tmp_path / "run_a"
    assert checkpoint_dir(tmp_path, "run_a", policy_id=2) == tmp_path / "run_a" / "checkpoint_p2"


def test_read_checkpoint_backend_type_accepts_missing_type_for_legacy_sample_factory(tmp_path):
    flappy_path = tmp_path / "flappy.pth"
    legacy_path = tmp_path / "legacy_sf.pth"
    torch.save({"backend_type": "flappy_ppo"}, flappy_path)
    torch.save({"model": {}}, legacy_path)

    assert read_checkpoint_backend_type(flappy_path) == "flappy_ppo"
    assert read_checkpoint_backend_type(legacy_path) is None
