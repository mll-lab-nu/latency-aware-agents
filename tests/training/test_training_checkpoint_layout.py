from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from training.common.checkpoints import (
    atomic_torch_save,
    capture_rng_state,
    restore_rng_state,
    checkpoint_dir,
    experiment_dir,
    find_checkpoint,
    format_checkpoint_name,
)


def test_checkpoint_rng_restores_next_native_learner_draws(tmp_path):
    """Long-lived checkpoint contract for native learner resume, including CPU runs."""
    original = capture_rng_state()
    try:
        checkpoint = tmp_path / "rng.pt"
        atomic_torch_save(capture_rng_state(), checkpoint)
        expected = (torch.rand(3), np.random.random(3), random.random())
        restore_rng_state(torch.load(checkpoint, weights_only=False))
        torch.testing.assert_close(torch.rand(3), expected[0])
        np.testing.assert_array_equal(np.random.random(3), expected[1])
        assert random.random() == expected[2]
    finally:
        restore_rng_state(original)


def test_atomic_checkpoint_failure_preserves_previous_training_state(tmp_path, monkeypatch):
    checkpoint = tmp_path / "training.pt"
    atomic_torch_save({"iteration": 2}, checkpoint)

    def interrupted_save(payload, path):
        path.write_bytes(b"partial checkpoint")
        raise OSError("interrupted write")

    with monkeypatch.context() as patch:
        patch.setattr(torch, "save", interrupted_save)
        with pytest.raises(OSError, match="interrupted"):
            atomic_torch_save({"iteration": 3}, checkpoint)
    assert torch.load(checkpoint, weights_only=True) == {"iteration": 2}
    atomic_torch_save({"iteration": 3}, checkpoint)
    assert torch.load(checkpoint, weights_only=True) == {"iteration": 3}


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
