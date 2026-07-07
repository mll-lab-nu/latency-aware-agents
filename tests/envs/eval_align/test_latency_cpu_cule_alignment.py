from __future__ import annotations

import numpy as np
import pytest

from latency_bench.core.types import Action
from training.common.raw_rgb_export import ENV_RAW_RGB_FRAME_STACK_INFO_KEY


FRAME_STACK = 4
SEED = 12345
ACTIONS = [0, 1, 2, 3, 4, 5, 0, 2]


def test_cule_core_and_latency_backend_are_byte_identical():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    core_trace = _cule_core_trace()
    backend_trace = _cule_latency_backend_trace()

    assert len(core_trace) == len(backend_trace)
    for core_point, backend_point in zip(core_trace, backend_trace):
        assert set(core_point) == set(backend_point)
        for key, core_value in core_point.items():
            backend_value = backend_point[key]
            if isinstance(core_value, np.ndarray):
                assert np.array_equal(core_value, backend_value), key
            else:
                assert core_value == backend_value


def test_cule_reset_matches_latency_cpu_ram_but_not_reset_frame_or_obs_yet():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    pytest.importorskip("cv2")

    cpu = _make_latency_cpu_env()
    core = _make_cule_core()
    try:
        cpu.reset(seed=SEED)
        cule_obs, _infos = core.reset_batch(seeds=[SEED])
        torch.cuda.synchronize()

        cpu_raw = cpu.render_game_frame().copy()
        cule_raw = core.native_rgb_frames()[0].detach().cpu().numpy().copy()
        cpu_obs84 = _latest_latency_cpu_frame(cpu.observe().data)
        cule_obs84 = cule_obs[0, 0].detach().cpu().numpy().copy()

        assert np.array_equal(cpu.gym_env.unwrapped.ale.getRAM(), core.ram[0].detach().cpu().numpy())
        assert _diff_stats(cpu_raw, cule_raw) == (30, 252)
        assert _diff_stats(cpu_obs84, cule_obs84) == (6, 91)

        cpu.step(Action(value=0, name="0", is_noop=True))
        cule_step = core.step_batch([0])
        torch.cuda.synchronize()

        cpu_step_obs = _latest_latency_cpu_frame(cpu.observe().data)
        cule_step_obs = cule_step.observations[0, 0].detach().cpu().numpy().copy()
        assert np.array_equal(cpu.gym_env.unwrapped.ale.getRAM(), core.ram[0].detach().cpu().numpy())
        assert np.array_equal(cpu.render_game_frame(), core.native_rgb_frames()[0].detach().cpu().numpy())
        assert np.array_equal(cpu_step_obs, cule_step_obs)
    finally:
        cpu.close()


def test_obs84_rounding_is_the_remaining_post_reset_cpu_vs_cule_obs_difference():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    pytest.importorskip("cv2")

    from training.common.atari_cv2_obs import native_rgb_to_obs84

    cpu = _make_latency_cpu_env()
    core = _make_cule_core()
    try:
        cpu.reset(seed=SEED)
        core.reset_batch(seeds=[SEED])
        torch.cuda.synchronize()

        for action_value in ACTIONS[:4]:
            cpu.step(Action(value=action_value, name=str(action_value), is_noop=action_value == 0))
            cule_step = core.step_batch([action_value])
            torch.cuda.synchronize()

        cpu_raw = cpu.render_game_frame().copy()
        cule_raw = core.native_rgb_frames()[0].detach().cpu().numpy().copy()
        cpu_obs84 = _latest_latency_cpu_frame(cpu.observe().data)
        cule_obs84 = cule_step.observations[0, 0].detach().cpu().numpy().copy()
        obs84_from_raw = native_rgb_to_obs84(torch.from_numpy(cule_raw)).cpu().numpy()

        assert np.array_equal(cpu_raw, cule_raw)
        assert np.array_equal(cule_obs84, obs84_from_raw)
        assert _diff_stats(cpu_obs84, cule_obs84) == (2, 1)
    finally:
        cpu.close()


def test_latency_cule_backend_rejects_non_fire_reset_mode():
    # CULE currently runs the hard-coded noop+fire reset sequence. Until raw/noop
    # are implemented there, non-fire latency_bench configs must fail loudly.
    from latency_bench.executors.demon_attack_gpu_batched_backend import DemonAttackGpuBatchedEnvStepBackend

    with pytest.raises(ValueError, match="reset_mode='fire'"):
        DemonAttackGpuBatchedEnvStepBackend(
            config={
                "env": {
                    "name": "demon_attack",
                    "env_fps": 60.0,
                    "obs_fps": 15.0,
                    "noop_max": 30,
                    "reset_mode": "noop",
                    "terminal_on_life_loss": False,
                    "clip_reward": False,
                    "frame_stack": FRAME_STACK,
                    "gpu_env_device": "cuda",
                    "max_episode_steps": 25000,
                },
                "policy": {"type": "starvla"},
            },
            noop_action=Action(value=0, name="NOOP", is_noop=True),
            num_slots=1,
        )


def _make_latency_cpu_env():
    from latency_bench.envs.demon_attack import DemonAttackEnvAdapter

    return DemonAttackEnvAdapter(
        env_fps=60.0,
        noop_action=Action(value=0, name="NOOP", is_noop=True),
        screen_size=84,
        noop_max=30,
        reset_mode="fire",
        terminal_on_life_loss=False,
        clip_reward=False,
        frame_stack=FRAME_STACK,
        export_env_raw_rgb_frames=True,
    )


def _latest_latency_cpu_frame(observation_data) -> np.ndarray:
    return np.asarray(observation_data, dtype=np.uint8)[-1].copy()


def _make_cule_core():
    from training.demon_attack_sf.gpu_core import CudaDemonAttackCore

    return CudaDemonAttackCore(
        batch_size=1,
        device="cuda",
        noop_max=30,
        terminal_on_life_loss=False,
        obs_stride=1,
        frame_stack=1,
    )


def _cule_core_trace() -> list[dict[str, object]]:
    torch = pytest.importorskip("torch")

    core = _make_cule_core()
    obs, infos = core.reset_batch(seeds=[SEED])
    torch.cuda.synchronize()
    rgb = core.native_rgb_frames()[0].detach().cpu().numpy().copy()
    obs84 = obs[0, 0].detach().cpu().numpy().copy()
    raw_stack = [rgb.copy() for _slot in range(FRAME_STACK)]
    obs_stack = [obs84.copy() for _slot in range(FRAME_STACK)]
    trace = [
        _trace_point(
            label="reset",
            action=-1,
            env_step=0,
            observation=np.stack(obs_stack, axis=0),
            raw_rgb_stack=np.stack(raw_stack, axis=0),
            reward=0.0,
            done=False,
            truncated=False,
            score=infos[0]["score"],
            lives=infos[0]["ale.lives"],
        )
    ]
    for env_step, action_value in enumerate(ACTIONS, start=1):
        result = core.step_batch([action_value])
        torch.cuda.synchronize()
        rgb = core.native_rgb_frames()[0].detach().cpu().numpy().copy()
        obs84 = result.observations[0, 0].detach().cpu().numpy().copy()
        done = bool(result.terminated[0].detach().cpu().item())
        truncated = bool(result.truncated[0].detach().cpu().item())
        if not bool(done or truncated):
            raw_stack = raw_stack[1:] + [rgb.copy()]
            obs_stack = obs_stack[1:] + [obs84.copy()]
        trace.append(
            _trace_point(
                label=f"step_{env_step}",
                action=action_value,
                env_step=env_step,
                observation=np.stack(obs_stack, axis=0),
                raw_rgb_stack=np.stack(raw_stack, axis=0),
                reward=float(result.rewards[0].detach().cpu().item()),
                done=done,
                truncated=truncated,
                score=int(result.scores[0].detach().cpu().item()),
                lives=int(result.lives[0].detach().cpu().item()),
            )
        )
    return trace


def _cule_latency_backend_trace() -> list[dict[str, object]]:
    from latency_bench.executors.demon_attack_gpu_batched_backend import DemonAttackGpuBatchedEnvStepBackend

    backend = DemonAttackGpuBatchedEnvStepBackend(
        config={
            "env": {
                "name": "demon_attack",
                "env_fps": 60.0,
                "obs_fps": 15.0,
                "noop_max": 30,
                "reset_mode": "fire",
                "terminal_on_life_loss": False,
                "clip_reward": False,
                "frame_stack": FRAME_STACK,
                "gpu_env_device": "cuda",
                "max_episode_steps": 25000,
            },
            "policy": {"type": "starvla"},
        },
        noop_action=Action(value=0, name="NOOP", is_noop=True),
        num_slots=1,
    )
    try:
        reset_observation = backend.reset_slot(0, episode_id=0, seed=SEED)
        trace = [_trace_point_from_backend_observation("reset", -1, reset_observation, 0.0, False, False)]
        for env_step, action_value in enumerate(ACTIONS, start=1):
            result = backend.step_slots(
                {0: Action(value=action_value, name=str(action_value), is_noop=action_value == 0)}
            )[0].result
            observation = backend.observe_slots([0])[0]
            trace.append(
                _trace_point_from_backend_observation(
                    f"step_{env_step}",
                    action_value,
                    observation,
                    result.reward,
                    result.done,
                    result.truncated,
                )
            )
        return trace
    finally:
        backend.close()


def _trace_point_from_backend_observation(
    label: str,
    action: int,
    observation,
    reward: float,
    done: bool,
    truncated: bool,
) -> dict[str, object]:
    metadata = observation.metadata
    return _trace_point(
        label=label,
        action=action,
        env_step=int(observation.env_step),
        observation=np.asarray(observation.data, dtype=np.uint8).copy(),
        raw_rgb_stack=np.asarray(metadata[ENV_RAW_RGB_FRAME_STACK_INFO_KEY], dtype=np.uint8).copy(),
        reward=reward,
        done=done,
        truncated=truncated,
        score=metadata["score"],
        lives=metadata["ale.lives"],
    )


def _trace_point(
    *,
    label: str,
    action: int,
    env_step: int,
    observation: np.ndarray,
    raw_rgb_stack: np.ndarray,
    reward: float,
    done: bool,
    truncated: bool,
    score: int,
    lives: int,
) -> dict[str, object]:
    return {
        "label": label,
        "action": int(action),
        "env_step": np.asarray([int(env_step)], dtype=np.int64),
        "sim_time_ms": np.asarray([float(env_step) * (1000.0 / 60.0)], dtype=np.float64),
        "observation": np.asarray(observation, dtype=np.uint8).copy(),
        "raw_rgb_stack": np.asarray(raw_rgb_stack, dtype=np.uint8).copy(),
        "reward": np.asarray([float(reward)], dtype=np.float32),
        "done": np.asarray([bool(done)], dtype=np.bool_),
        "truncated": np.asarray([bool(truncated)], dtype=np.bool_),
        "score": np.asarray([int(score)], dtype=np.int64),
        "lives": np.asarray([int(lives)], dtype=np.int64),
    }


def _diff_stats(left, right) -> tuple[int, int]:
    left_arr = np.asarray(left)
    right_arr = np.asarray(right)
    diff = np.abs(left_arr.astype(np.int16) - right_arr.astype(np.int16))
    return int(np.count_nonzero(left_arr != right_arr)), int(diff.max())
