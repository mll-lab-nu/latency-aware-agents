"""Native ViZDoom regression for the teacher-to-VLA raw image contract."""

import json

import numpy as np
import pytest

pytest.importorskip("vizdoom", minversion="1.3.0")
pytest.importorskip("sample_factory")

from latency_bench.envs.adapters.deadly_corridor import DeadlyCorridorEnvAdapter, DeadlyCorridorVlaEnvAdapter
from latency_bench.envs.base import ENV_RAW_RGB_FRAME_STACK_INFO_KEY
from scripts.tasks.decision_history.eval_vla_hist8 import evaluation_config


def test_hist8_deadly_vla_uses_the_teacher_resolution_and_hud(tmp_path):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "provenance.json").write_text(json.dumps({
        "training_run_id": "renderer-test", "training_step": 5000,
    }))
    config = evaluation_config(
        "deadly_corridor", "qwenoft", bundle, tmp_path / "profile.json",
        tmp_path / "evaluation", tmp_path / "backbone", tmp_path / "python",
        device="cpu", episodes=1, env_condition="zero",
    )
    teacher = DeadlyCorridorEnvAdapter(config=config, export_env_raw_rgb_frames=True)
    student = DeadlyCorridorVlaEnvAdapter(config=config, export_env_raw_rgb_frames=True)
    try:
        teacher_frame = teacher.reset(seed=1_000_000).metadata[ENV_RAW_RGB_FRAME_STACK_INFO_KEY][-1]
        student_frame = student.reset(seed=1_000_000).metadata[ENV_RAW_RGB_FRAME_STACK_INFO_KEY][-1]
        assert teacher_frame.shape == student_frame.shape == (120, 160, 3)
        # The health/ammo panel is stable across the two engine reset paths;
        # the animated face and enemies can differ with their RNG streams.
        np.testing.assert_array_equal(teacher_frame[-20:, :64], student_frame[-20:, :64])
    finally:
        teacher.close()
        student.close()
