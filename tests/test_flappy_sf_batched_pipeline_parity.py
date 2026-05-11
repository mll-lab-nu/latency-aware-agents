from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from training.flappy_ppo.paths import ensure_flappy_gymnasium_on_path

ensure_flappy_gymnasium_on_path()


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
        "gpu_render_device": "cpu",
        "gpu_render_batch_size": 1,
        "gpu_render_profile": False,
        "gpu_render_profile_interval": 1000,
        "hold_policy": "one_frame_then_noop",
        "ordering_policy": "latest_ready",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _env_config():
    return SimpleNamespace(worker_index=0, vector_index=0, env_id=0)


def _select_info(info):
    selected = {
        "score": info.get("score"),
        "latency_raw_frames": info.get("latency_raw_frames"),
        "queued_action": info.get("queued_action"),
        "applied_action": info.get("applied_action"),
        "ready_raw_frame": info.get("ready_raw_frame"),
        "raw_frames_advanced": info.get("raw_frames_advanced"),
        "obs_stride_raw_frames": info.get("obs_stride_raw_frames"),
    }
    if "episode_extra_stats" in info:
        selected["episode_extra_stats"] = dict(info["episode_extra_stats"])
    if "reset_info" in info:
        selected["reset_info"] = _select_info(info["reset_info"])
    return selected


class _ScalarGpuOracle:
    def __init__(self, cfg, env_config, batch_size):
        import gymnasium as gym

        from latency_bench.core.clock import EnvClock
        from modeling.flappy_ppo.observation import FlappyPpoObservationConfig
        from modeling.flappy_ppo.preprocessing import FlappyPpoTorchObservationPipeline
        from training.flappy_sf.batched_gpu_env import FlappyBatchedGpuRenderedEnv
        from training.sample_factory_common.latency_config import (
            sample_factory_latency_params_from_flat_config,
            sample_factory_latency_seed_from_flat_config,
        )
        from training.sample_factory_common.latency_wrapper import (
            ActionLatencyWrapper,
            EpisodeMetricsWrapper,
            RewardTraceWrapper,
        )

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
            score_limit = getattr(cfg, "score_limit", None)
            if score_limit not in (None, "", 0, -1):
                kwargs["score_limit"] = int(score_limit)
            env = gym.make(cfg.gym_id, **kwargs)
            env = RewardTraceWrapper(env)
            sub_seed = int(cfg.seed) + 100000 * int(env_config.worker_index) + 1000 * int(env_config.vector_index) + agent_idx
            latency_seed = sample_factory_latency_seed_from_flat_config(
                cfg,
                fallback_seed=sub_seed,
                worker_index=int(env_config.worker_index),
                vector_index=int(env_config.vector_index),
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
                max_pending_actions=getattr(cfg, "max_pending_actions", None),
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

        self.pipeline = FlappyPpoTorchObservationPipeline(
            FlappyPpoObservationConfig(image_size=84, grayscale=False, stack_frames=int(cfg.frame_stack)),
            n_envs=self.batch_size,
            device="cpu",
        )
        self.renderer = None
        self._raw_rgb_buffer = None
        self._env_cls = FlappyBatchedGpuRenderedEnv

    def reset(self, seed):
        infos = []
        for agent_idx, env in enumerate(self.envs):
            numeric_obs, info = env.reset(seed=int(seed) + agent_idx)
            infos.append(self._env_cls._with_numeric_obs(info, numeric_obs))
        obs = self.pipeline.reset(self._render(), frame_skips=self._env_cls._latency_raw_frames(infos))
        return {"obs": obs}, infos

    def step(self, actions):
        rewards = np.zeros(self.batch_size, dtype=np.float32)
        terminated = np.zeros(self.batch_size, dtype=np.bool_)
        truncated = np.zeros(self.batch_size, dtype=np.bool_)
        infos = []
        observation_infos = []
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
                observation_infos.append(reset_info)
            else:
                observation_infos.append(info)
            infos.append(info)
        obs = self.pipeline.step(
            self._render(),
            dones=np.logical_or(terminated, truncated),
            frame_skips=self._env_cls._latency_raw_frames(observation_infos),
        )
        return {"obs": obs}, rewards, terminated, truncated, infos

    def _render(self):
        from flappy_bird_gymnasium.envs.render_state import BatchedFlappyRenderState
        from flappy_bird_gymnasium.gpu_render import FlappyRenderAssets, TorchFlappyRenderer

        states = [env.unwrapped.get_render_state() for env in self.envs]
        if self.renderer is None:
            self.renderer = TorchFlappyRenderer(FlappyRenderAssets.for_state(states[0]), device="cpu")
        batch = BatchedFlappyRenderState.from_states(states)
        if self._raw_rgb_buffer is None:
            self._raw_rgb_buffer = np.empty((batch.batch_size, batch.screen_height, batch.screen_width, 3), dtype=np.uint8)
        return self.renderer.render_into(batch, None)

    def close(self):
        for env in self.envs:
            env.close()


def test_gpu_render_batched_env_batch_size_one_matches_scalar_wrapped_oracle():
    torch = pytest.importorskip("torch")
    from training.flappy_sf.integration import make_flappy_sf_env

    del torch
    cfg = _cfg(gpu_render_batch_size=1, fixed_latency_ms=1000.0 / 30.0)
    actions = np.asarray([[0], [1], [0], [1], [1], [0]], dtype=np.int64)
    scalar = _ScalarGpuOracle(cfg, _env_config(), batch_size=1)
    env = make_flappy_sf_env("latency_flappy", cfg=cfg, env_config=_env_config())
    try:
        scalar_obs, scalar_infos = scalar.reset(seed=123)
        obs, infos = env.reset(seed=123)
        assert getattr(env, "uses_batched_core") is True
        assert obs["obs"].shape == scalar_obs["obs"].shape
        np.testing.assert_array_equal(obs["obs"].detach().cpu().numpy(), scalar_obs["obs"].detach().cpu().numpy())
        assert _select_info(infos[0]) == _select_info(scalar_infos[0])

        for action in actions:
            scalar_step = scalar.step(action)
            step = env.step(action)
            np.testing.assert_array_equal(step[0]["obs"].detach().cpu().numpy(), scalar_step[0]["obs"].detach().cpu().numpy())
            np.testing.assert_array_equal(step[1], scalar_step[1])
            np.testing.assert_array_equal(step[2], scalar_step[2])
            np.testing.assert_array_equal(step[3], scalar_step[3])
            assert [_select_info(info) for info in step[4]] == [_select_info(info) for info in scalar_step[4]]
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
        assert [_select_info(info) for info in infos] == [_select_info(info) for info in scalar_infos]

        for action_row in actions:
            scalar_step = scalar.step(action_row)
            step = env.step(action_row)
            np.testing.assert_array_equal(step[0]["obs"].detach().cpu().numpy(), scalar_step[0]["obs"].detach().cpu().numpy())
            np.testing.assert_array_equal(step[1], scalar_step[1])
            np.testing.assert_array_equal(step[2], scalar_step[2])
            np.testing.assert_array_equal(step[3], scalar_step[3])
            assert [_select_info(info) for info in step[4]] == [_select_info(info) for info in scalar_step[4]]
    finally:
        scalar.close()
        env.close()

