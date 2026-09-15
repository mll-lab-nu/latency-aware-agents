from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

import gymnasium as gym

from flappy_bird_gymnasium.envs.batched_core import BatchedFlappyBirdCore
from flappy_bird_gymnasium.envs.render_state import BatchedFlappyRenderState


def _make_scalar_env(**overrides):
    kwargs = {
        "use_lidar": False,
        "normalize_obs": True,
        "audio_on": False,
        "pipe_gap": 100,
        "bird_color": "yellow",
        "pipe_color": "green",
        "background": "day",
        "debug": False,
        "render_mode": None,
    }
    kwargs.update(overrides)
    return gym.make("FlappyBird-v0", **kwargs)


def _assert_render_batch_equal(actual, expected):
    np.testing.assert_array_equal(actual.player_xy, expected.player_xy)
    np.testing.assert_array_equal(actual.player_indices, expected.player_indices)
    np.testing.assert_array_equal(actual.visible_rotations, expected.visible_rotations)
    np.testing.assert_array_equal(actual.upper_pipe_xy, expected.upper_pipe_xy)
    np.testing.assert_array_equal(actual.lower_pipe_xy, expected.lower_pipe_xy)
    np.testing.assert_array_equal(actual.ground_xy, expected.ground_xy)
    assert actual.screen_width == expected.screen_width
    assert actual.screen_height == expected.screen_height
    assert actual.bird_color == expected.bird_color
    assert actual.pipe_color == expected.pipe_color
    assert actual.background == expected.background


def _array(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return value


def _assert_render_state_equal(actual, expected) -> None:
    np.testing.assert_array_equal(_array(actual.player_xy), _array(expected.player_xy))
    np.testing.assert_array_equal(_array(actual.player_indices), _array(expected.player_indices))
    np.testing.assert_array_equal(_array(actual.visible_rotations), _array(expected.visible_rotations))
    np.testing.assert_array_equal(_array(actual.upper_pipe_xy), _array(expected.upper_pipe_xy))
    np.testing.assert_array_equal(_array(actual.lower_pipe_xy), _array(expected.lower_pipe_xy))
    np.testing.assert_array_equal(_array(actual.ground_xy), _array(expected.ground_xy))
    assert actual.screen_width == expected.screen_width
    assert actual.screen_height == expected.screen_height
    assert actual.bird_color == expected.bird_color
    assert actual.pipe_color == expected.pipe_color
    assert actual.background == expected.background


def _cfg(**overrides):
    values = {
        "gym_id": "FlappyBird-v0",
        "use_lidar": False,
        "normalize_obs": True,
        "audio_on": False,
        "pipe_gap": 100,
        "bird_color": "yellow",
        "pipe_color": "green",
        "background": "day",
        "debug": False,
        "debug_timelimit_diagnostics": False,
        "screen_size": "",
        "score_limit": -1,
        "obs_resize": "84,84",
        "frame_stack": 4,
        "max_episode_steps": 0,
        "latency_type": "fixed",
        "fixed_latency_ms": 0.0,
        "mean_latency_ms": None,
        "std_latency_ms": None,
        "min_latency_ms": None,
        "max_latency_ms": None,
        "latency_seed": None,
        "add_latency_info": True,
        "seed": 0,
        "env_fps": 30,
        "obs_fps": 30,
        "episode_metrics_path": "",
        "use_gpu_render": True,
        "simulator": "cpu",
        "gpu_render_device": "cpu",
        "gpu_render_batch_size": 1,
        "gpu_render_profile": False,
        "gpu_render_profile_interval": 1000,
        "hold_policy": "one_frame_then_noop",
        "ordering_policy": "latest_ready",
        "max_pending_actions": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _env_config():
    return SimpleNamespace(worker_index=0, vector_index=0, env_id=0)


RESET_INFO_KEYS = (
    "score",
    "episode_idx",
    "decision_step_idx",
    "env_step_idx",
    "raw_frame",
    "latency_raw_frames",
    "latency_ms",
    "applied_action",
    "pending_action_count",
    "obs_stride_raw_frames",
    "env_fps",
    "obs_fps",
    "hold_policy",
    "ordering_policy",
)

STEP_INFO_KEYS = RESET_INFO_KEYS + (
    "queued_action",
    "ready_raw_frame",
    "raw_frames_advanced",
)


def _select_info(info, keys):
    selected = {key: info[key] for key in keys}
    if "episode_extra_stats" in info:
        selected["episode_extra_stats"] = dict(info["episode_extra_stats"])
    if "reset_info" in info:
        selected["reset_info"] = _select_info(info["reset_info"], RESET_INFO_KEYS)
    return selected


class _ScalarGpuOracle:
    def __init__(self, cfg, env_config, batch_size):
        from latency_bench.core.clock import EnvClock
        from latency_bench.envs.flappy import FlappyObservationConfig, FlappyTorchObservationPipeline
        from training.common.latency_config import (
            sample_factory_latency_params_from_flat_config,
            sample_factory_latency_seed_from_flat_config,
        )
        from training.common.latency_wrapper import (
            ActionLatencyWrapper,
            EpisodeMetricsWrapper,
            RewardTraceWrapper,
        )
        from training.flappy_sf.batched_gpu_env import FlappyBatchedGpuRenderedEnv

        self.cfg = cfg
        self.env_config = env_config
        self.batch_size = int(batch_size)
        self.clock = EnvClock(env_fps=float(cfg.env_fps), obs_fps=float(cfg.obs_fps))
        self.envs = []
        for agent_idx in range(self.batch_size):
            kwargs = {
                "use_lidar": bool(cfg.use_lidar),
                "normalize_obs": bool(cfg.normalize_obs),
                "audio_on": bool(cfg.audio_on),
                "pipe_gap": int(cfg.pipe_gap),
                "bird_color": cfg.bird_color,
                "pipe_color": cfg.pipe_color,
                "background": cfg.background,
                "debug": bool(cfg.debug),
                "render_mode": None,
            }
            if cfg.score_limit not in (None, "", 0, -1):
                kwargs["score_limit"] = int(cfg.score_limit)
            env = gym.make(cfg.gym_id, **kwargs)
            env = RewardTraceWrapper(env)
            sub_seed = int(cfg.seed) + 100000 * int(env_config.worker_index) + 1000 * int(env_config.vector_index) + agent_idx
            latency_seed = sample_factory_latency_seed_from_flat_config(
                cfg,
                fallback_seed=sub_seed,
                env_id=agent_idx,
            )
            latency_params = sample_factory_latency_params_from_flat_config(cfg, env_fps=self.clock.env_fps)
            env = ActionLatencyWrapper(
                env,
                fixed_latency_raw_frames=latency_params.fixed_latency_raw_frames,
                env_fps=self.clock.env_fps,
                obs_fps=self.clock.obs_fps,
                hold_policy=cfg.hold_policy,
                ordering_policy=cfg.ordering_policy,
                latency_distribution_spec=latency_params.latency_distribution_spec,
                seed=latency_seed,
                env_reset_seed_base=sub_seed + 104729,
                add_latency_info=bool(cfg.add_latency_info),
                max_pending_actions=cfg.max_pending_actions,
            )
            env = EpisodeMetricsWrapper(
                env,
                frames_per_decision=self.clock.obs_stride_raw_frames,
                metrics_path="",
                worker_index=int(env_config.worker_index),
                vector_index=int(env_config.vector_index),
                env_id=agent_idx,
                issued_action_labels={0: "noop", 1: "flap"},
            )
            self.envs.append(env)

        self.pipeline = FlappyTorchObservationPipeline(
            FlappyObservationConfig(image_size=84, grayscale=False, stack_frames=int(cfg.frame_stack)),
            n_envs=self.batch_size,
            device="cpu",
        )
        self.renderer = None
        self._env_cls = FlappyBatchedGpuRenderedEnv

    def reset(self, seed):
        infos = []
        for agent_idx, env in enumerate(self.envs):
            numeric_obs, info = env.reset(seed=int(seed) + agent_idx)
            infos.append(self._env_cls._with_numeric_obs(info, numeric_obs))
        obs = self.pipeline.reset(self._render())
        return {"obs": obs}, infos

    def step(self, actions):
        rewards = np.zeros(self.batch_size, dtype=np.float32)
        terminated = np.zeros(self.batch_size, dtype=np.bool_)
        truncated = np.zeros(self.batch_size, dtype=np.bool_)
        infos = []
        for agent_idx, (env, action) in enumerate(zip(self.envs, actions)):
            numeric_obs, reward, term, trunc, info = env.step(int(action))
            info = self._env_cls._with_numeric_obs(info, numeric_obs)
            rewards[agent_idx] = float(reward)
            terminated[agent_idx] = bool(term)
            truncated[agent_idx] = bool(trunc)
            if bool(term or trunc):
                reset_numeric_obs, reset_info = env.reset()
                reset_info = self._env_cls._with_numeric_obs(reset_info, reset_numeric_obs)
                info["reset_info"] = reset_info
            infos.append(info)
        obs = self.pipeline.step(
            self._render(),
            dones=np.logical_or(terminated, truncated),
        )
        return {"obs": obs}, rewards, terminated, truncated, infos

    def _render(self):
        from flappy_bird_gymnasium.gpu_render import FlappyRenderAssets, TorchFlappyRenderer

        states = [env.unwrapped.get_render_state() for env in self.envs]
        if self.renderer is None:
            self.renderer = TorchFlappyRenderer(FlappyRenderAssets.for_state(states[0]), device="cpu")
        batch = BatchedFlappyRenderState.from_states(states)
        return self.renderer.render_into(batch, None)

    def close(self):
        for env in self.envs:
            env.close()


def test_batched_flappy_core_matches_scalar_env_for_mixed_actions_and_render_state():
    seed = 123
    actions = [0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0]
    scalar = _make_scalar_env()
    core = BatchedFlappyBirdCore(batch_size=1)
    try:
        scalar_obs, scalar_info = scalar.reset(seed=seed)
        core_obs, core_infos = core.reset_batch(seeds=[seed])

        np.testing.assert_allclose(core_obs[0], scalar_obs, rtol=0, atol=0)
        assert core_infos[0]["score"] == scalar_info["score"]
        _assert_render_batch_equal(
            core.render_state_batch(),
            BatchedFlappyRenderState.from_states([scalar.unwrapped.get_render_state()]),
        )

        for action in actions:
            scalar_obs, scalar_reward, scalar_term, scalar_trunc, scalar_info = scalar.step(action)
            result = core.step_batch(np.asarray([action], dtype=np.int64))

            np.testing.assert_allclose(result.observations[0], scalar_obs, rtol=0, atol=0)
            assert float(result.rewards[0]) == float(scalar_reward)
            assert bool(result.terminated[0]) is bool(scalar_term)
            assert bool(result.truncated[0]) is bool(scalar_trunc)
            assert int(result.scores[0]) == scalar_info["score"]
            _assert_render_batch_equal(
                core.render_state_batch(),
                BatchedFlappyRenderState.from_states([scalar.unwrapped.get_render_state()]),
            )
    finally:
        scalar.close()


def test_batched_flappy_core_keeps_agents_independent_inside_larger_batch():
    seeds = [101, 202, 303, 404]
    action_rows = np.asarray(
        [
            [0, 1, 0, 1],
            [1, 0, 0, 1],
            [0, 0, 1, 1],
            [1, 1, 0, 0],
            [0, 1, 1, 0],
            [0, 0, 0, 1],
        ],
        dtype=np.int64,
    )
    scalars = [_make_scalar_env() for _ in seeds]
    core = BatchedFlappyBirdCore(batch_size=len(seeds))
    try:
        scalar_obs = []
        scalar_infos = []
        for env, seed in zip(scalars, seeds):
            obs, info = env.reset(seed=seed)
            scalar_obs.append(obs)
            scalar_infos.append(info)
        core_obs, core_infos = core.reset_batch(seeds=seeds)
        np.testing.assert_allclose(core_obs, np.asarray(scalar_obs), rtol=0, atol=0)
        assert [info["score"] for info in core_infos] == [info["score"] for info in scalar_infos]

        for actions in action_rows:
            expected = [env.step(int(action)) for env, action in zip(scalars, actions)]
            result = core.step_batch(actions)
            np.testing.assert_allclose(
                result.observations,
                np.asarray([item[0] for item in expected]),
                rtol=0,
                atol=0,
            )
            np.testing.assert_array_equal(result.rewards, np.asarray([item[1] for item in expected], dtype=np.float64))
            np.testing.assert_array_equal(result.terminated, np.asarray([item[2] for item in expected], dtype=np.bool_))
            np.testing.assert_array_equal(result.truncated, np.asarray([item[3] for item in expected], dtype=np.bool_))
            assert [int(score) for score in result.scores] == [item[4]["score"] for item in expected]
    finally:
        for env in scalars:
            env.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_cuda_flappy_core_matches_cpu_core_for_seeded_action_stream():
    from training.flappy_sf.gpu_core import CudaFlappyBirdCore

    seeds = [101, 202, 303, 404]
    action_rows = np.asarray(
        [
            [0, 1, 0, 1],
            [1, 0, 0, 1],
            [0, 0, 1, 1],
            [1, 1, 0, 0],
            [0, 1, 1, 0],
            [0, 0, 0, 1],
            [1, 0, 1, 0],
            [0, 1, 0, 0],
        ],
        dtype=np.int64,
    )
    active_rows = np.asarray(
        [
            [True, True, True, True],
            [True, False, True, True],
            [True, True, False, True],
            [False, True, True, True],
            [True, True, True, False],
            [True, True, True, True],
            [True, False, True, False],
            [True, True, True, True],
        ],
        dtype=np.bool_,
    )

    cpu = BatchedFlappyBirdCore(batch_size=len(seeds))
    cuda = CudaFlappyBirdCore(batch_size=len(seeds), device=torch.device("cuda"))

    cpu_obs, cpu_infos = cpu.reset_batch(seeds=seeds)
    cuda_obs, cuda_infos = cuda.reset_batch(seeds=seeds)

    np.testing.assert_allclose(cuda_obs, cpu_obs, rtol=0, atol=0)
    assert cuda_infos == cpu_infos
    _assert_render_state_equal(cuda.render_state_batch(), cpu.render_state_batch())

    for row_idx, (actions, active) in enumerate(zip(action_rows, active_rows)):
        cpu_result = cpu.step_batch(actions, active_mask=active)
        if row_idx % 2 == 0:
            cuda_result = cuda.step_batch_numpy(actions, active_mask=active)
            torch.cuda.synchronize()

            np.testing.assert_allclose(cuda.observations(), cpu_result.observations, rtol=0, atol=0)
            np.testing.assert_array_equal(cuda_result.rewards, cpu_result.rewards)
            np.testing.assert_array_equal(cuda_result.terminated, cpu_result.terminated)
            np.testing.assert_array_equal(cuda_result.truncated, cpu_result.truncated)
            np.testing.assert_array_equal(cuda_result.scores, cpu_result.scores)
            _assert_render_state_equal(cuda.render_state_batch(), cpu.render_state_batch())
            continue

        cuda_result = cuda.step_batch_tensors(
            torch.as_tensor(actions, dtype=torch.int64, device="cuda"),
            active_mask=active,
        )
        torch.cuda.synchronize()

        np.testing.assert_allclose(cuda.observations(), cpu_result.observations, rtol=0, atol=0)
        np.testing.assert_array_equal(cuda_result.rewards.detach().cpu().numpy(), cpu_result.rewards)
        np.testing.assert_array_equal(cuda_result.terminated.detach().cpu().numpy(), cpu_result.terminated)
        np.testing.assert_array_equal(cuda_result.truncated.detach().cpu().numpy(), cpu_result.truncated)
        np.testing.assert_array_equal(cuda_result.scores.detach().cpu().numpy(), cpu_result.scores)
        _assert_render_state_equal(cuda.render_state_batch(), cpu.render_state_batch())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_cuda_flappy_render_matches_torch_renderer_for_seeded_core_state():
    from flappy_bird_gymnasium.gpu_render import FlappyRenderAssets, TorchFlappyRenderer
    from training.flappy_sf.flappy_cuda_extension import CudaFlappyRenderer
    from training.flappy_sf.gpu_core import CudaFlappyBirdCore

    seeds = [7, 8, 9]
    actions = np.asarray(
        [
            [0, 1, 0],
            [1, 0, 1],
            [0, 0, 1],
            [1, 1, 0],
        ],
        dtype=np.int64,
    )
    cpu = BatchedFlappyBirdCore(batch_size=len(seeds))
    cuda = CudaFlappyBirdCore(batch_size=len(seeds), device=torch.device("cuda"))
    cpu.reset_batch(seeds=seeds)
    cuda.reset_batch(seeds=seeds)
    for step_actions in actions:
        cpu.step_batch(step_actions)
        cuda.step_batch_tensors(torch.as_tensor(step_actions, dtype=torch.int64, device="cuda"))
    torch.cuda.synchronize()

    assets = FlappyRenderAssets.load()
    expected = TorchFlappyRenderer(assets, device="cpu").render(cpu.render_state_batch()).numpy()
    renderer = CudaFlappyRenderer(assets, device=torch.device("cuda"))
    out = torch.empty((len(seeds), cuda.screen_height, cuda.screen_width, 3), dtype=torch.uint8, device="cuda")
    actual = renderer.render_core_into(cuda, out).cpu().numpy()

    np.testing.assert_array_equal(actual, expected)


def test_gpu_render_batched_env_batch_size_one_matches_scalar_wrapped_oracle():
    torch_module = pytest.importorskip("torch")
    from training.flappy_sf.integration import make_flappy_sf_env

    del torch_module
    cfg = _cfg(gpu_render_batch_size=1, fixed_latency_ms=1000.0 / 30.0)
    actions = np.asarray([[0], [1], [0], [1], [1], [0]], dtype=np.int64)
    scalar = _ScalarGpuOracle(cfg, _env_config(), batch_size=1)
    env = make_flappy_sf_env("latency_flappy", cfg=cfg, env_config=_env_config())
    try:
        scalar_obs, scalar_infos = scalar.reset(seed=123)
        obs, infos = env.reset(seed=123)
        assert env.uses_batched_core is True
        assert obs["obs"].shape == scalar_obs["obs"].shape
        np.testing.assert_array_equal(obs["obs"].detach().cpu().numpy(), scalar_obs["obs"].detach().cpu().numpy())
        assert _select_info(infos[0], RESET_INFO_KEYS) == _select_info(scalar_infos[0], RESET_INFO_KEYS)

        for action in actions:
            scalar_step = scalar.step(action)
            step = env.step(action)
            np.testing.assert_array_equal(step[0]["obs"].detach().cpu().numpy(), scalar_step[0]["obs"].detach().cpu().numpy())
            np.testing.assert_array_equal(step[1], scalar_step[1])
            np.testing.assert_array_equal(step[2], scalar_step[2])
            np.testing.assert_array_equal(step[3], scalar_step[3])
            assert [_select_info(info, STEP_INFO_KEYS) for info in step[4]] == [
                _select_info(info, STEP_INFO_KEYS) for info in scalar_step[4]
            ]
    finally:
        scalar.close()
        env.close()


def test_gpu_render_batched_env_multi_agent_matches_independent_scalar_oracles():
    pytest.importorskip("torch")
    from training.flappy_sf.integration import make_flappy_sf_env

    cfg = _cfg(gpu_render_batch_size=4, latency_type="fixed", fixed_latency_ms=0.0)
    actions = np.asarray(
        [
            [0, 1, 0, 1],
            [1, 0, 0, 1],
            [0, 0, 1, 1],
            [1, 1, 0, 0],
        ],
        dtype=np.int64,
    )
    scalar = _ScalarGpuOracle(cfg, _env_config(), batch_size=4)
    env = make_flappy_sf_env("latency_flappy", cfg=cfg, env_config=_env_config())
    try:
        scalar_obs, scalar_infos = scalar.reset(seed=55)
        obs, infos = env.reset(seed=55)
        np.testing.assert_array_equal(obs["obs"].detach().cpu().numpy(), scalar_obs["obs"].detach().cpu().numpy())
        assert [_select_info(info, RESET_INFO_KEYS) for info in infos] == [
            _select_info(info, RESET_INFO_KEYS) for info in scalar_infos
        ]

        for action_row in actions:
            scalar_step = scalar.step(action_row)
            step = env.step(action_row)
            np.testing.assert_array_equal(step[0]["obs"].detach().cpu().numpy(), scalar_step[0]["obs"].detach().cpu().numpy())
            np.testing.assert_array_equal(step[1], scalar_step[1])
            np.testing.assert_array_equal(step[2], scalar_step[2])
            np.testing.assert_array_equal(step[3], scalar_step[3])
            assert [_select_info(info, STEP_INFO_KEYS) for info in step[4]] == [
                _select_info(info, STEP_INFO_KEYS) for info in scalar_step[4]
            ]
    finally:
        scalar.close()
        env.close()
