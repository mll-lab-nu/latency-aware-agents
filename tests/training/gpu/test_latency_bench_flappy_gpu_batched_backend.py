from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np
import pytest

from latency_bench.core.types import Action, Observation, StepResult
from latency_bench.envs.adapters.flappy import FlappyBirdEnvAdapter
from latency_bench.executors.env_step_backend import SerialEnvStepBackend
from latency_bench.executors.factory import build_executor
from latency_bench.executors.backend_selection import auto_latency_bench_env_backend
from latency_bench.envs.base import ENV_RAW_RGB_FRAME_STACK_INFO_KEY


def _noop() -> Action:
    return Action(value=0, name="noop", is_noop=True)


def _flap() -> Action:
    return Action(value=1, name="flap", is_oneshot=True)


def _config(output_dir: str, *, num_episodes: int = 3, score_limit: int | None = None) -> dict[str, Any]:
    env_cfg: dict[str, Any] = {
        "name": "flappy",
        "gym_id": "FlappyBird-v0",
        "env_fps": 30,
        "obs_fps": 30,
        "noop_action": "noop",
        "action_map": {
            "noop": 0,
            "flap": 1,
        },
        "oneshot_actions": ["flap"],
        "pipe_gap": 100,
        "bird_color": "yellow",
        "pipe_color": "green",
        "background": "day",
        "obs_resize": [84, 84],
        "frame_stack": 4,
        "use_gpu_render": True,
        "simulator": "cpu",
        "gpu_render_device": "cpu",
        "gpu_render_profile": False,
        "gpu_render_profile_interval": 1000,
    }
    if score_limit is not None:
        env_cfg["score_limit"] = score_limit
    return {
        "experiment": {
            "name": "flappy_gpu_batched_backend_test",
            "seed": 0,
        },
        "executor": {
            "mode": "simulated",
        },
        "env": env_cfg,
        "latency": {
            "method": "zero",
        },
        "scheduler": {
            "hold_policy": "one_frame_then_noop",
            "ordering_policy": "latest_ready",
        },
        "policy": {
            "type": "constant",
            "action": "flap",
            "actions": ["noop", "flap"],
        },
        "evaluation": {
            "episodes": int(num_episodes),
            "max_env_steps": 6,
            "parallel_envs": 3,
        },
        "logging": {
            "output_dir": output_dir,
            "save_step_records": True,
            "save_action_records": True,
            "save_latency_records": True,
        },
    }


def _serial_backend(num_slots: int, *, score_limit: int | None = None) -> SerialEnvStepBackend:
    return SerialEnvStepBackend(
        [
            FlappyBirdEnvAdapter(
                gym_id="FlappyBird-v0",
                env_fps=30,
                noop_action=_noop(),
                pipe_gap=100,
                bird_color="yellow",
                pipe_color="green",
                background="day",
                score_limit=score_limit,
                obs_resize=(84, 84),
                frame_stack=4,
                use_gpu_render=True,
                gpu_render_device="cpu",
                gpu_render_profile=False,
                gpu_render_profile_interval=1000,
                policy_observation_layout="chw",
            )
            for _slot in range(num_slots)
        ]
    )


def _serial_starvla_backend(num_slots: int) -> SerialEnvStepBackend:
    return SerialEnvStepBackend(
        [
            FlappyBirdEnvAdapter(
                gym_id="FlappyBird-v0",
                env_fps=30,
                noop_action=_noop(),
                pipe_gap=100,
                bird_color="yellow",
                pipe_color="green",
                background="day",
                obs_resize=(84, 84),
                frame_stack=4,
                use_gpu_render=True,
                gpu_render_device="cpu",
                gpu_render_profile=False,
                gpu_render_profile_interval=1000,
                policy_observation_layout="hwc",
                export_env_raw_rgb_frames=True,
            )
            for _slot in range(num_slots)
        ]
    )


def _array(value: Any) -> np.ndarray:
    if hasattr(value, "detach") and callable(value.detach):
        value = value.detach().cpu().numpy()
    elif hasattr(value, "cpu") and callable(value.cpu):
        value = value.cpu().numpy()
    return np.asarray(value)


def _assert_info_equal(actual: dict[str, Any], expected: dict[str, Any]) -> None:
    assert set(actual) == set(expected)
    for key, expected_value in expected.items():
        actual_value = actual[key]
        if isinstance(expected_value, dict):
            _assert_info_equal(actual_value, expected_value)
        elif isinstance(expected_value, np.ndarray):
            np.testing.assert_allclose(_array(actual_value), expected_value, rtol=0, atol=0)
        else:
            assert actual_value == expected_value


def _assert_observation_equal(actual: Observation, expected: Observation) -> None:
    assert actual.env_step == expected.env_step
    assert actual.sim_time_ms == expected.sim_time_ms
    np.testing.assert_array_equal(_array(actual.data), _array(expected.data))
    _assert_info_equal(actual.metadata, expected.metadata)


def _assert_step_equal(actual: StepResult, expected: StepResult) -> None:
    assert expected.observation is not None
    assert actual.reward == expected.reward
    assert actual.done is expected.done
    assert actual.truncated is expected.truncated
    _assert_info_equal(actual.info, expected.info)
    if actual.observation is not None:
        _assert_observation_equal(actual.observation, expected.observation)


def _assert_backend_observations_equal(
    actual: dict[int, Observation],
    expected: dict[int, Observation],
) -> None:
    assert set(actual) == set(expected)
    for slot_id in expected:
        _assert_observation_equal(actual[slot_id], expected[slot_id])


def test_flappy_gpu_batched_backend_matches_serial_gpu_backend_step_for_step(tmp_path):
    pytest.importorskip("torch")
    from latency_bench.executors.flappy_gpu_batched_backend import FlappyGpuBatchedEnvStepBackend

    config = _config(str(tmp_path), num_episodes=3)
    config["policy"]["type"] = "flappy_sf"
    serial = _serial_backend(3)
    batched = FlappyGpuBatchedEnvStepBackend(config=config, noop_action=_noop(), num_slots=3)
    try:
        for slot_id, seed in enumerate([101, 202, 303]):
            serial_reset = serial.reset_slot(slot_id, episode_id=slot_id, seed=seed)
            batched_reset = batched.reset_slot(slot_id, episode_id=slot_id, seed=seed)
            _assert_observation_equal(batched_reset, serial_reset)

        _assert_backend_observations_equal(
            batched.observe_slots([0, 1, 2]),
            serial.observe_slots([0, 1, 2]),
        )

        action_plan = [
            {0: _noop(), 1: _flap(), 2: _noop()},
            {0: _flap(), 2: _flap()},
            {1: _noop()},
            {0: _noop(), 1: _flap(), 2: _flap()},
        ]
        for actions_by_slot in action_plan:
            serial_responses = serial.step_slots(actions_by_slot)
            batched_responses = batched.step_slots(actions_by_slot)
            assert set(batched_responses) == set(serial_responses)
            for slot_id in serial_responses:
                _assert_step_equal(
                    batched_responses[slot_id].result,
                    serial_responses[slot_id].result,
                )
            _assert_backend_observations_equal(
                batched.observe_slots([0, 1, 2]),
                serial.observe_slots([0, 1, 2]),
            )
    finally:
        batched.close()
        serial.close()


def test_flappy_gpu_simulator_batched_backend_matches_serial_gpu_backend_step_for_step(tmp_path):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    from latency_bench.executors.flappy_gpu_batched_backend import FlappyGpuBatchedEnvStepBackend

    config = _config(str(tmp_path), num_episodes=2)
    config["env"]["simulator"] = "gpu"
    config["env"]["gpu_render_device"] = "cuda"
    config["policy"]["type"] = "flappy_sf"
    serial = _serial_backend(2)
    batched = FlappyGpuBatchedEnvStepBackend(config=config, noop_action=_noop(), num_slots=2)
    try:
        for slot_id, seed in enumerate([404, 505]):
            _assert_observation_equal(
                batched.reset_slot(slot_id, episode_id=slot_id, seed=seed),
                serial.reset_slot(slot_id, episode_id=slot_id, seed=seed),
            )
        _assert_backend_observations_equal(
            batched.observe_slots([0, 1]),
            serial.observe_slots([0, 1]),
        )

        action_plan = [
            {0: _noop(), 1: _flap()},
            {0: _flap(), 1: _noop()},
            {0: _flap()},
            {1: _flap()},
        ]
        for actions_by_slot in action_plan:
            serial_responses = serial.step_slots(actions_by_slot)
            batched_responses = batched.step_slots(actions_by_slot)
            assert set(batched_responses) == set(serial_responses)
            for slot_id in serial_responses:
                _assert_step_equal(
                    batched_responses[slot_id].result,
                    serial_responses[slot_id].result,
                )
            _assert_backend_observations_equal(
                batched.observe_slots([0, 1]),
                serial.observe_slots([0, 1]),
            )
    finally:
        batched.close()
        serial.close()


def test_flappy_gpu_batched_backend_matches_serial_done_auto_reset_contract(tmp_path):
    pytest.importorskip("torch")
    from latency_bench.executors.flappy_gpu_batched_backend import FlappyGpuBatchedEnvStepBackend

    config = _config(str(tmp_path), num_episodes=2, score_limit=0)
    config["policy"]["type"] = "flappy_sf"
    serial = _serial_backend(2, score_limit=0)
    batched = FlappyGpuBatchedEnvStepBackend(config=config, noop_action=_noop(), num_slots=2)
    try:
        for slot_id, seed in enumerate([111, 222]):
            _assert_observation_equal(
                batched.reset_slot(slot_id, episode_id=slot_id, seed=seed),
                serial.reset_slot(slot_id, episode_id=slot_id, seed=seed),
            )

        serial_responses = serial.step_slots({0: _noop(), 1: _flap()})
        batched_responses = batched.step_slots({0: _noop(), 1: _flap()})

        for slot_id in serial_responses:
            _assert_step_equal(
                batched_responses[slot_id].result,
                serial_responses[slot_id].result,
            )
            assert "reset_info" in batched_responses[slot_id].result.info
    finally:
        batched.close()
        serial.close()


def test_flappy_gpu_batched_backend_exports_raw_rgb_stack_for_starvla(tmp_path):
    pytest.importorskip("torch")
    from latency_bench.executors.flappy_gpu_batched_backend import FlappyGpuBatchedEnvStepBackend

    config = _config(str(tmp_path), num_episodes=1)
    config["policy"]["type"] = "starvla"
    batched = FlappyGpuBatchedEnvStepBackend(config=config, noop_action=_noop(), num_slots=1)
    try:
        batched.reset_slot(0, episode_id=0, seed=123)
        observation = batched.observe_slots([0])[0]
        raw_stack = observation.metadata[ENV_RAW_RGB_FRAME_STACK_INFO_KEY]

        assert observation.data.shape == (84, 84, 12)
        assert raw_stack.shape[0] == 4
        assert raw_stack.shape[-1] == 3
        assert raw_stack.shape[:3] != (4, 84, 84)
        assert raw_stack.dtype == np.uint8
    finally:
        batched.close()


def test_flappy_gpu_batched_backend_preserves_starvla_raw_rgb_stack_between_observations(tmp_path):
    pytest.importorskip("torch")
    from latency_bench.executors.flappy_gpu_batched_backend import FlappyGpuBatchedEnvStepBackend

    config = _config(str(tmp_path), num_episodes=1)
    config["policy"]["type"] = "starvla"
    config["env"]["obs_fps"] = 10
    serial = _serial_starvla_backend(1)
    batched = FlappyGpuBatchedEnvStepBackend(config=config, noop_action=_noop(), num_slots=1)
    try:
        serial.reset_slot(0, episode_id=0, seed=123)
        batched.reset_slot(0, episode_id=0, seed=123)
        serial.observe_slots([0])
        batched.observe_slots([0])

        for action in [_noop(), _flap(), _noop()]:
            serial.step_slots({0: action})
            batched.step_slots({0: action})

        serial_observation = serial.observe_slots([0])[0]
        batched_observation = batched.observe_slots([0])[0]

        np.testing.assert_array_equal(
            batched_observation.metadata[ENV_RAW_RGB_FRAME_STACK_INFO_KEY],
            serial_observation.metadata[ENV_RAW_RGB_FRAME_STACK_INFO_KEY],
        )
    finally:
        batched.close()
        serial.close()


def test_flappy_gpu_batched_backend_does_not_materialize_policy_observation_during_step(tmp_path):
    pytest.importorskip("torch")
    from latency_bench.executors.flappy_gpu_batched_backend import FlappyGpuBatchedEnvStepBackend

    config = _config(str(tmp_path), num_episodes=1)
    config["policy"]["type"] = "flappy_sf"
    batched = FlappyGpuBatchedEnvStepBackend(config=config, noop_action=_noop(), num_slots=1)
    try:
        batched.reset_slot(0, episode_id=0, seed=123)
        reset_observation = batched.slot_handles[0].observe()

        response = batched.step_slots({0: _noop()})[0]

        assert response.result.observation is None
        assert batched.slot_handles[0].observe() is reset_observation
    finally:
        batched.close()


def test_flappy_sf_frame_stack_advances_through_dropped_observation_frames(tmp_path):
    torch = pytest.importorskip("torch")
    from latency_bench.executors.flappy_gpu_batched_backend import FlappyGpuBatchedEnvStepBackend

    config = _config(str(tmp_path), num_episodes=1)
    config["policy"]["type"] = "flappy_sf"
    batched = FlappyGpuBatchedEnvStepBackend(config=config, noop_action=_noop(), num_slots=1)
    try:
        batched.reset_slot(0, episode_id=0, seed=123)
        frames = [batched.observe_slots([0])[0].data[-3:].clone()]
        for _ in range(3):
            batched.step_slots({0: _noop()})
            frames.append(batched.pipeline.current()[0, -3:].clone())

        observation = batched.observe_slots([0])[0].data
        for index, frame in enumerate(frames):
            torch.testing.assert_close(observation[index * 3 : (index + 1) * 3], frame)
        assert any(not torch.equal(frame, frames[0]) for frame in frames[1:])
    finally:
        batched.close()


def _metrics_contract(metrics: list[Any]) -> list[tuple[Any, ...]]:
    return [
        (
            metric.episode_id,
            metric.episode_return,
            metric.survival_steps,
            metric.episode_return_env,
            metric.return_raw,
            metric.game_score,
            metric.num_actions,
            metric.num_dropped_actions,
            metric.num_invalid_actions,
            metric.metadata,
        )
        for metric in metrics
    ]


def test_flappy_gpu_batched_executor_preserves_serial_episode_metrics(tmp_path):
    pytest.importorskip("torch")

    serial_config = _config(str(tmp_path / "serial"), num_episodes=3)
    serial_config["executor"]["env_backend"] = "serial"
    batched_config = deepcopy(serial_config)
    batched_config["logging"]["output_dir"] = str(tmp_path / "batched")
    batched_config["executor"]["env_backend"] = "flappy_gpu_batched"

    serial_executor = build_executor(serial_config)
    batched_executor = build_executor(batched_config)
    try:
        episode_ids = [0, 1, 2]
        seeds = [11, 22, 33]
        serial_metrics = serial_executor.run_episodes(
            episode_ids=episode_ids,
            seeds=seeds,
            max_env_steps=6,
        )
        batched_metrics = batched_executor.run_episodes(
            episode_ids=episode_ids,
            seeds=seeds,
            max_env_steps=6,
        )
    finally:
        batched_executor.close()
        serial_executor.close()

    assert _metrics_contract(batched_metrics) == _metrics_contract(serial_metrics)
