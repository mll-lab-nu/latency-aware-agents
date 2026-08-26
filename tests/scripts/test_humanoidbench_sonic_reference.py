from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from latency_bench.envs import humanoidbench_sonic_assets as assets
from latency_bench.envs.humanoidbench_sonic_reference import (
    _MUJOCO_INDEX_FOR_POLICY,
    SONIC_ENCODER_INPUT_DIM,
    build_encoder_observation,
    default_initial_qpos,
    encode_reference_tokens,
    generate_planner_qpos,
    resample_qpos_to_50hz,
)


class _PlannerSession:
    def __init__(self):
        self.contexts = []
        self._outputs = [
            SimpleNamespace(name="mujoco_qpos"),
            SimpleNamespace(name="num_pred_frames"),
        ]

    def get_outputs(self):
        return self._outputs

    def run(self, _output_names, inputs):
        context = inputs["context_mujoco_qpos"]
        self.contexts.append(context.copy())
        start = context[0, -1, 0] + 1.0
        qpos = np.repeat(context[:, -1:, :], 8, axis=1)
        qpos[0, :, 0] = start + np.arange(8, dtype=np.float32)
        return [qpos, np.asarray(8, dtype=np.int64)]


class _EncoderSession:
    def __init__(self):
        self.inputs = []
        self._outputs = [SimpleNamespace(name="encoded_tokens")]

    def get_outputs(self):
        return self._outputs

    def run(self, _output_names, inputs):
        value = inputs["obs_dict"]
        self.inputs.append(value.copy())
        return [np.repeat(value[:, :1], 64, axis=1).astype(np.float32)]


def test_prepare_g1_sonic_reference_assets_uses_pinned_paths(tmp_path, monkeypatch):
    calls = []

    def fake_download(**kwargs):
        calls.append(kwargs)
        path = tmp_path / kwargs["filename"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"onnx")
        return path

    monkeypatch.setattr(assets, "hf_hub_download", fake_download)

    planner, encoder = assets.prepare_g1_sonic_reference_assets(tmp_path / "cache")

    assert planner == tmp_path / "planner_sonic.onnx"
    assert encoder == tmp_path / "sonic_v1_1/model_encoder.onnx"
    assert [call["filename"] for call in calls] == [
        assets.SONIC_PLANNER_FILENAME,
        assets.SONIC_ENCODER_FILENAME,
    ]
    assert {call["revision"] for call in calls} == {assets.SONIC_REVISION}


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
        default_initial_qpos()[7:][_MUJOCO_INDEX_FOR_POLICY],
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


def test_encode_reference_tokens_saves_horizon_plus_one_tokens():
    planner = _PlannerSession()
    encoder = _EncoderSession()
    tokens = encode_reference_tokens(planner, encoder, horizon_steps=5)

    assert tokens.shape == (6, 64)
    assert len(encoder.inputs) == 6
    assert all(item.shape == (1, SONIC_ENCODER_INPUT_DIM) for item in encoder.inputs)
