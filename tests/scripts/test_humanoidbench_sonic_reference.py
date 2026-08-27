from __future__ import annotations

import csv
import tarfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from latency_bench.envs import humanoidbench_sonic_assets as assets
from latency_bench.envs.humanoidbench_sonic_contract import MUJOCO_INDEX_FOR_POLICY
from latency_bench.envs.humanoidbench_sonic_reference import (
    SONIC_ENCODER_INPUT_DIM,
    build_encoder_observation,
    canonicalize_reference_heading,
    default_initial_qpos,
    generate_planner_qpos,
    load_bones_seed_reference_qpos,
    load_bones_seed_qpos,
    resample_qpos_to_50hz,
    straighten_reference_heading,
)


class _PlannerSession:
    def __init__(self):
        self.contexts = []
        self.target_velocities = []
        self._outputs = [
            SimpleNamespace(name="mujoco_qpos"),
            SimpleNamespace(name="num_pred_frames"),
        ]

    def get_outputs(self):
        return self._outputs

    def run(self, _output_names, inputs):
        context = inputs["context_mujoco_qpos"]
        self.target_velocities.append(inputs["target_vel"].copy())
        self.contexts.append(context.copy())
        start = context[0, -1, 0] + 1.0
        qpos = np.repeat(context[:, -1:, :], 8, axis=1)
        qpos[0, :, 0] = start + np.arange(8, dtype=np.float32)
        return [qpos, np.asarray(8, dtype=np.int64)]


def test_prepare_g1_sonic_encoder_uses_pinned_path(tmp_path, monkeypatch):
    calls = []

    def fake_download(**kwargs):
        calls.append(kwargs)
        path = tmp_path / kwargs["filename"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"onnx")
        return path

    monkeypatch.setattr(assets, "hf_hub_download", fake_download)

    encoder = assets.prepare_g1_sonic_encoder(tmp_path / "cache")

    assert encoder == tmp_path / "sonic_v1_1/model_encoder.onnx"
    assert [call["filename"] for call in calls] == [assets.SONIC_ENCODER_FILENAME]
    assert {call["revision"] for call in calls} == {assets.SONIC_REVISION}


def test_prepare_g1_sonic_planner_uses_pinned_path(tmp_path, monkeypatch):
    calls = []

    def fake_download(**kwargs):
        calls.append(kwargs)
        path = tmp_path / kwargs["filename"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"onnx")
        return path

    monkeypatch.setattr(assets, "hf_hub_download", fake_download)

    planner = assets.prepare_g1_sonic_planner(tmp_path / "cache")

    assert planner == tmp_path / assets.SONIC_PLANNER_FILENAME
    assert [call["filename"] for call in calls] == [assets.SONIC_PLANNER_FILENAME]
    assert {call["revision"] for call in calls} == {assets.SONIC_REVISION}


def test_prepare_g1_sonic_skill_references_extracts_pinned_csvs(tmp_path, monkeypatch):
    calls = []
    archive_root = tmp_path / "archive_root"
    csv_paths = {
        name: archive_root / member
        for name, member in assets.BONES_SEED_SKILL_CSVS.items()
    }
    for name, csv_path in csv_paths.items():
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        csv_path.write_text(f"Frame,skill\n0,{name}\n", encoding="utf-8")

    def fake_download(**kwargs):
        calls.append(kwargs)
        archive = Path(kwargs["local_dir"]) / kwargs["filename"]
        archive.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive, mode="w:gz") as bundle:
            for name, csv_path in csv_paths.items():
                bundle.add(csv_path, arcname=assets.BONES_SEED_SKILL_CSVS[name])
        return archive

    monkeypatch.setattr(assets, "hf_hub_download", fake_download)
    extracted = assets.prepare_g1_sonic_skill_references(tmp_path / "cache")

    assert set(extracted) == set(assets.BONES_SEED_SKILL_CSVS)
    for name, path in extracted.items():
        assert path == tmp_path / "cache" / "bones_seed" / assets.BONES_SEED_SKILL_CSVS[name]
        assert path.read_text(encoding="utf-8") == f"Frame,skill\n0,{name}\n"
    assert calls[0]["repo_id"] == assets.BONES_SEED_REPO_ID
    assert calls[0]["repo_type"] == "dataset"
    assert calls[0]["filename"] == assets.BONES_SEED_ARCHIVE_FILENAME
    assert calls[0]["revision"] == assets.BONES_SEED_REVISION


def test_load_bones_seed_qpos_converts_units_rotation_and_joint_order(tmp_path):
    joint_columns = [f"joint_{index}_dof" for index in range(29)]
    columns = [
        "Frame",
        "root_translateX",
        "root_translateY",
        "root_translateZ",
        "root_rotateX",
        "root_rotateY",
        "root_rotateZ",
        *joint_columns,
    ]
    path = tmp_path / "jump.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        writer.writerow([0, 100, 200, 300, 0, 0, 90, 180, -90, *([0] * 27)])

    qpos = load_bones_seed_qpos(path)

    np.testing.assert_allclose(qpos[0, :3], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(qpos[0, 3:7], [np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)])
    np.testing.assert_allclose(qpos[0, 7:9], [np.pi, -np.pi / 2])


def test_generate_planner_qpos_repeats_context_and_returns_30hz_frames():
    planner = _PlannerSession()
    result = generate_planner_qpos(planner, horizon_steps=5)

    assert result.shape[1:] == (36,)
    assert len(result) >= 8
    np.testing.assert_array_equal(
        planner.contexts[0][0, :, :],
        np.repeat(default_initial_qpos()[None, :], 4, axis=0),
    )
    np.testing.assert_array_equal(planner.contexts[1][0], result[4:8])
    np.testing.assert_array_equal(planner.target_velocities, np.full((len(planner.contexts), 1), 3.0))


def test_resample_and_encoder_pack_follow_v1_1_g1_layout():
    source = np.repeat(default_initial_qpos()[None, :], 4, axis=0)
    source[:, 0] = np.arange(4, dtype=np.float32)
    resampled = resample_qpos_to_50hz(source, 6)
    assert resampled.shape == (6, 36)
    np.testing.assert_allclose(resampled[[0, 5], 0], [0.0, 3.0])

    observation = build_encoder_observation(resampled, 0)
    assert observation.shape == (SONIC_ENCODER_INPUT_DIM,)
    np.testing.assert_allclose(observation[:4], 0.0)
    np.testing.assert_allclose(
        observation[4 : 4 + 29],
        default_initial_qpos()[7:][MUJOCO_INDEX_FOR_POLICY],
    )
    np.testing.assert_allclose(
        observation[4 + 20 * 29 : 4 + 20 * 29 + 60],
        np.tile([1.0, 0.0, 0.0, 1.0, 0.0, 0.0], 10),
    )
    np.testing.assert_allclose(observation[644:], 0.0)

    turned = np.repeat(default_initial_qpos()[None, :], 47, axis=0)
    turned[1:, 3:7] = np.sqrt(0.5) * np.asarray([1.0, 0.0, 0.0, 1.0])
    turned_observation = build_encoder_observation(turned, 1)
    np.testing.assert_allclose(
        turned_observation[4 + 20 * 29 : 4 + 20 * 29 + 6],
        [1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        atol=1e-6,
    )
    actual_heading_observation = build_encoder_observation(
        turned,
        1,
        actual_root_quaternion=np.asarray([1.0, 0.0, 0.0, 0.0]),
    )
    np.testing.assert_allclose(
        actual_heading_observation[4 + 20 * 29 : 4 + 20 * 29 + 6],
        [0.0, -1.0, 1.0, 0.0, 0.0, 0.0],
        atol=1e-6,
    )


def test_resample_accepts_bones_seed_source_rate():
    source = np.repeat(default_initial_qpos()[None, :], 6, axis=0)
    source[:, 0] = np.arange(6, dtype=np.float32)
    resampled = resample_qpos_to_50hz(source, 3, source_hz=120.0)
    np.testing.assert_allclose(resampled[:, 0], [0.0, 2.4, 4.8])


def test_canonicalize_reference_heading_starts_at_origin_facing_forward():
    qpos = np.repeat(default_initial_qpos()[None, :], 2, axis=0)
    qpos[:, :2] = [[2.0, 3.0], [2.0, 4.0]]
    qpos[:, 3:7] = [np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)]

    canonical = canonicalize_reference_heading(qpos)

    np.testing.assert_allclose(canonical[0, :2], 0.0, atol=1e-6)
    np.testing.assert_allclose(canonical[0, 3:7], [1.0, 0.0, 0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(canonical[1, :2], [1.0, 0.0], atol=1e-6)


def test_straighten_reference_heading_removes_yaw_and_lateral_drift():
    qpos = np.repeat(default_initial_qpos()[None, :], 2, axis=0)
    qpos[:, 1] = [0.0, 1.0]
    qpos[1, 3:7] = [np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)]

    straight = straighten_reference_heading(qpos)

    np.testing.assert_allclose(straight[:, 1], 0.0)
    np.testing.assert_allclose(
        straight[:, 3:7], [[1.0, 0.0, 0.0, 0.0]] * 2, atol=1e-7
    )


def test_bones_seed_reference_keeps_the_exact_clip_duration(tmp_path):
    columns = [
        "Frame",
        "root_translateX",
        "root_translateY",
        "root_translateZ",
        "root_rotateX",
        "root_rotateY",
        "root_rotateZ",
        *[f"joint_{index}_dof" for index in range(29)],
    ]
    path = tmp_path / "jump.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        writer.writerows(
            [[frame, 0, 0, 76, 0, 0, 0, *([0] * 29)] for frame in range(12)]
        )
    qpos, num_frames = load_bones_seed_reference_qpos(path)

    assert num_frames == 5
    assert qpos.shape == (50, 36)
