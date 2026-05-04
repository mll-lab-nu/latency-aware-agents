from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
from PIL import Image

from latency_bench.core.types import Action, Observation, StepResult
from latency_bench.envs.base import EnvAdapter
from modeling.flappy_ppo.preprocessing import resize_frame_pil


_FLAPPY_INTERNAL_LATENCY_INFO_KEYS = ("queue_latency", "frame_skip", "latency_frame_skip")


class FlappyBirdEnvAdapter(EnvAdapter):
    """Image-observation adapter for the local FlappyBird-v0 Gymnasium env."""

    def __init__(
        self,
        gym_id: str = "FlappyBird-v0",
        fps_env: float = 30.0,
        noop_action: Action | None = None,
        render_mode: str = "rgb_array",
        observation_mode: str = "image",
        use_lidar: bool = False,
        normalize_obs: bool = True,
        audio_on: bool = False,
        screen_size: tuple[int, int] | None = None,
        pipe_gap: int = 100,
        bird_color: str = "yellow",
        pipe_color: str = "green",
        background: str | None = "day",
        score_limit: int | None = None,
        debug: bool = False,
        obs_resize: tuple[int, int] | None = None,
    ):
        if observation_mode != "image":
            raise ValueError("FlappyBirdEnvAdapter currently supports observation_mode='image' only")
        if render_mode != "rgb_array":
            raise ValueError("FlappyBirdEnvAdapter image mode requires render_mode='rgb_array'")
        if fps_env <= 0:
            raise ValueError("fps_env must be positive")

        gymnasium = _import_flappy_gymnasium()
        kwargs: dict[str, Any] = {
            "audio_on": bool(audio_on),
            "normalize_obs": bool(normalize_obs),
            "use_lidar": bool(use_lidar),
            "pipe_gap": int(pipe_gap),
            "bird_color": bird_color,
            "pipe_color": pipe_color,
            "render_mode": render_mode,
            "background": background,
            "score_limit": score_limit,
            "debug": bool(debug),
        }
        if screen_size is not None:
            kwargs["screen_size"] = tuple(screen_size)

        self.gym_env = gymnasium.make(gym_id, **kwargs)
        self.gym_id = gym_id
        self.fps_env = float(fps_env)
        self.noop_action = noop_action or Action(value=0, name="noop", is_noop=True)
        self.render_mode = render_mode
        self.observation_mode = observation_mode
        self.obs_resize = obs_resize
        self.env_step = 0
        self._last_info: dict[str, Any] = {}
        self._last_numeric_obs: Any = None
        self._last_frame: Any = None

    @property
    def frame_ms(self) -> float:
        return 1000.0 / self.fps_env

    def reset(self, seed: int | None = None) -> Observation:
        self.env_step = 0
        numeric_obs, info = self.gym_env.reset(seed=seed)
        self._last_numeric_obs = numeric_obs
        self._last_info = dict(info or {})
        self._last_frame = self._render_rgb_frame()
        return self._make_observation(numeric_obs=numeric_obs, info=self._last_info)

    def step(self, action: Action) -> StepResult:
        gym_action = int(action.value)
        numeric_obs, reward, terminated, truncated, info = self.gym_env.step(gym_action)
        self.env_step += 1
        self._last_numeric_obs = numeric_obs
        self._last_info = dict(info or {})
        self._last_frame = self._render_rgb_frame()
        observation = self._make_observation(numeric_obs=numeric_obs, info=self._last_info)
        step_info = dict(self._last_info)
        step_info.update(
            {
                "env_step": self.env_step,
                "sim_time_ms": self.env_step * self.frame_ms,
                "applied_action": gym_action,
                "applied_action_name": action.name,
                "observation_mode": self.observation_mode,
            }
        )
        return StepResult(
            observation=observation,
            reward=float(reward),
            done=bool(terminated),
            truncated=bool(truncated),
            info=step_info,
        )

    def render_frame(self) -> Any:
        if self._last_frame is None:
            self._last_frame = self._render_rgb_frame()
        return self._last_frame

    def get_state_info(self) -> dict[str, Any]:
        info = dict(self._last_info)
        info.update(
            {
                "env_step": self.env_step,
                "sim_time_ms": self.env_step * self.frame_ms,
                "score": info.get("score"),
                "observation_mode": self.observation_mode,
            }
        )
        return info

    def close(self) -> None:
        self.gym_env.close()

    def _render_rgb_frame(self) -> Any:
        frame = self.gym_env.render()
        if frame is None:
            raise RuntimeError("FlappyBird-v0 render() returned None; use render_mode='rgb_array'")
        if self.obs_resize is None:
            return frame
        arr = np.asarray(frame, dtype=np.uint8)
        if arr.ndim != 3 or arr.shape[-1] != 3:
            raise RuntimeError(f"Expected Flappy RGB frame with shape (H, W, 3), got {arr.shape!r}")
        height, width = self.obs_resize
        if height == width:
            arr = resize_frame_pil(arr, image_size=height, grayscale=False)
        else:
            image = Image.fromarray(arr).convert("RGB")
            image = image.resize((width, height), Image.BILINEAR)
            arr = np.asarray(image, dtype=np.uint8)
        if arr.shape[:2] != (height, width):
            raise RuntimeError(f"obs_resize expected HWC frame {(height, width, 3)}, got {arr.shape!r}")
        return np.transpose(arr, (2, 0, 1))

    def _make_observation(self, numeric_obs: Any, info: dict[str, Any]) -> Observation:
        metadata = dict(info)
        metadata.update(
            {
                "env_step": self.env_step,
                "sim_time_ms": self.env_step * self.frame_ms,
                "numeric_obs": numeric_obs,
                "observation_mode": self.observation_mode,
            }
        )
        return Observation(
            data=self._last_frame,
            env_step=self.env_step,
            sim_time_ms=self.env_step * self.frame_ms,
            metadata=metadata,
        )


class FlappySampleFactoryGpuEnvAdapter(EnvAdapter):
    """Latency-bench adapter that reuses the Sample Factory Flappy GPU renderer."""

    def __init__(
        self,
        *,
        config: dict[str, Any],
        noop_action: Action | None = None,
    ):
        env_cfg = config["env"]
        fps_env = float(env_cfg["fps_env"])
        if fps_env <= 0:
            raise ValueError("fps_env must be positive")

        obs_resize = _parse_obs_resize(env_cfg.get("obs_resize"))
        if obs_resize is None:
            raise ValueError("env.use_gpu_render=true requires env.obs_resize")

        from training.flappy_sf.integration import make_flappy_sf_env

        self.gym_id = env_cfg.get("gym_id", "FlappyBird-v0")
        self.fps_env = fps_env
        self.noop_action = noop_action or Action(value=0, name="noop", is_noop=True)
        self.env_step = 0
        self._last_info: dict[str, Any] = {}
        self._last_frame: np.ndarray | None = None

        experiment_cfg = config.get("experiment", {})
        latency_cfg = config.get("latency", {})
        # The Sample Factory env always contains a QueueLatencyWrapper. In latency_bench
        # the scheduler owns latency, so keep the internal wrapper at zero latency.
        sf_cfg = SimpleNamespace(
            gym_id=self.gym_id,
            use_lidar=bool(env_cfg.get("use_lidar", False)),
            normalize_obs=bool(env_cfg.get("normalize_obs", True)),
            audio_on=bool(env_cfg.get("audio_on", False)),
            pipe_gap=int(env_cfg.get("pipe_gap", 100)),
            bird_color=env_cfg.get("bird_color", "yellow"),
            pipe_color=env_cfg.get("pipe_color", "green"),
            background=env_cfg.get("background", "day"),
            debug=bool(env_cfg.get("debug", False)),
            debug_timelimit_diagnostics=bool(env_cfg.get("debug_timelimit_diagnostics", False)),
            screen_size="",
            score_limit=env_cfg.get("score_limit", -1),
            obs_resize=f"{obs_resize[0]},{obs_resize[1]}",
            obs_mode="image_only",
            frame_stack=int(env_cfg.get("frame_stack", 1) or 1),
            max_episode_steps=int(env_cfg.get("max_episode_steps", 0) or 0),
            queue_latency_values="0",
            latency_mode=str(latency_cfg.get("train_mode", "episode_fixed")),
            latency_sampling_mode=str(latency_cfg.get("train_sampling_mode", "uniform_all")),
            episodes_per_stage=int(latency_cfg.get("episodes_per_stage", 10)),
            randomize_within_stage=bool(latency_cfg.get("randomize_within_stage", False)),
            within_episode_change_interval=int(latency_cfg.get("within_episode_change_interval", 1)),
            within_episode_change_prob=float(latency_cfg.get("within_episode_change_prob", 1.0)),
            add_latency_info=False,
            seed=int(experiment_cfg.get("seed", 0) or 0),
            episode_metrics_path="",
            use_gpu_render=True,
            gpu_render_device=env_cfg.get("gpu_render_device", "auto"),
            gpu_render_batch_size=1,
            gpu_render_profile=bool(env_cfg.get("gpu_render_profile", False)),
            gpu_render_profile_interval=int(env_cfg.get("gpu_render_profile_interval", 1000) or 1000),
        )
        self.sf_cfg = sf_cfg
        self.sf_env = make_flappy_sf_env(
            "latency_flappy",
            cfg=sf_cfg,
            env_config=SimpleNamespace(worker_index=0, vector_index=0, env_id=0),
        )

    @property
    def frame_ms(self) -> float:
        return 1000.0 / self.fps_env

    def reset(self, seed: int | None = None) -> Observation:
        self.env_step = 0
        obs, infos = self.sf_env.reset(seed=seed)
        self._last_frame = _single_agent_obs(obs)
        self._last_info = _clean_sf_info(_single_agent_info(infos))
        return self._make_observation()

    def step(self, action: Action) -> StepResult:
        gym_action = int(action.value)
        obs, reward, terminated, truncated, infos = self.sf_env.step(np.asarray([gym_action], dtype=np.int64))
        self.env_step += 1
        self._last_frame = _single_agent_obs(obs)
        self._last_info = _clean_sf_info(_single_agent_info(infos))
        observation = self._make_observation()
        step_info = dict(self._last_info)
        step_info.update(
            {
                "env_step": self.env_step,
                "sim_time_ms": self.env_step * self.frame_ms,
                "applied_action": gym_action,
                "applied_action_name": action.name,
                "observation_mode": "sample_factory_gpu",
            }
        )
        return StepResult(
            observation=observation,
            reward=float(_single_agent_scalar(reward)),
            done=bool(_single_agent_scalar(terminated)),
            truncated=bool(_single_agent_scalar(truncated)),
            info=step_info,
        )

    def render_frame(self) -> Any:
        return self._last_frame

    def get_state_info(self) -> dict[str, Any]:
        info = dict(self._last_info)
        info.update(
            {
                "env_step": self.env_step,
                "sim_time_ms": self.env_step * self.frame_ms,
                "observation_mode": "sample_factory_gpu",
            }
        )
        return info

    def close(self) -> None:
        self.sf_env.close()

    def _make_observation(self) -> Observation:
        if self._last_frame is None:
            raise RuntimeError("FlappySampleFactoryGpuEnvAdapter has no rendered observation; call reset() first")
        metadata = dict(self._last_info)
        metadata.update(
            {
                "env_step": self.env_step,
                "sim_time_ms": self.env_step * self.frame_ms,
                "observation_mode": "sample_factory_gpu",
            }
        )
        return Observation(
            data=self._last_frame,
            env_step=self.env_step,
            sim_time_ms=self.env_step * self.frame_ms,
            metadata=metadata,
        )


def _parse_obs_resize(value: Any) -> tuple[int, int] | None:
    if value in (None, ""):
        return None
    if isinstance(value, str):
        parts = [item.strip() for item in value.split(",") if item.strip()]
    else:
        parts = list(value)
    if len(parts) != 2:
        raise ValueError("env.obs_resize must be [height, width]")
    return int(parts[0]), int(parts[1])


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach") and callable(value.detach):
        value = value.detach().cpu().numpy()
    elif hasattr(value, "cpu") and callable(value.cpu):
        value = value.cpu().numpy()
    return np.asarray(value)


def _single_agent_obs(obs: Any) -> np.ndarray:
    value = obs.get("obs") if isinstance(obs, dict) else obs
    arr = _to_numpy(value)
    if arr.ndim == 4:
        if int(arr.shape[0]) != 1:
            raise RuntimeError(f"Expected Flappy GPU batch size 1, got observation shape {arr.shape!r}")
        arr = arr[0]
    if arr.ndim != 3:
        raise RuntimeError(f"Expected Flappy GPU CHW observation, got shape {arr.shape!r}")
    return np.asarray(arr, dtype=np.uint8)


def _single_agent_info(infos: Any) -> dict[str, Any]:
    if isinstance(infos, (list, tuple)):
        if not infos:
            return {}
        info = infos[0]
        return dict(info or {}) if isinstance(info, dict) else {}
    if isinstance(infos, dict):
        return dict(infos)
    return {}


def _single_agent_scalar(value: Any) -> Any:
    arr = _to_numpy(value)
    if arr.shape == ():
        return arr.item()
    if arr.size < 1:
        raise RuntimeError("Expected at least one scalar value for Flappy GPU env")
    return arr.reshape(-1)[0].item()


def _clean_sf_info(info: dict[str, Any]) -> dict[str, Any]:
    cleaned = dict(info)
    for key in _FLAPPY_INTERNAL_LATENCY_INFO_KEYS:
        cleaned.pop(key, None)
    return cleaned


def _import_flappy_gymnasium():
    try:
        import flappy_bird_gymnasium  # noqa: F401
        import gymnasium

        return gymnasium
    except ImportError:
        repo_root = Path(__file__).resolve().parents[2]
        flappy_root = repo_root / "flappy-bird-gymnasium"
        if flappy_root.exists() and str(flappy_root) not in sys.path:
            sys.path.insert(0, str(flappy_root))
        import flappy_bird_gymnasium  # noqa: F401
        import gymnasium

        return gymnasium


class FlappySampleFactoryGpuEnvAdapter(EnvAdapter):
    """Latency-bench adapter for the Sample Factory Flappy GPU-rendered env."""

    def __init__(
        self,
        *,
        config: dict[str, Any],
        noop_action: Action | None = None,
    ):
        env_cfg = config["env"]
        if float(env_cfg["fps_env"]) <= 0:
            raise ValueError("fps_env must be positive")

        from training.flappy_sf.integration import make_flappy_sf_env

        self.fps_env = float(env_cfg["fps_env"])
        self.noop_action = noop_action or Action(value=0, name="noop", is_noop=True)
        self.env_step = 0
        self._last_info: dict[str, Any] = {}
        self._last_frame: Any = None
        self.sf_env = make_flappy_sf_env(
            "latency_flappy",
            cfg=self._sf_cfg(config),
            env_config=SimpleNamespace(worker_index=0, vector_index=0, env_id=0),
        )

    @property
    def frame_ms(self) -> float:
        return 1000.0 / self.fps_env

    def reset(self, seed: int | None = None) -> Observation:
        self.env_step = 0
        obs, infos = self.sf_env.reset(seed=seed)
        info = self._single_info(infos)
        frame = self._single_observation(obs)
        self._last_info = info
        self._last_frame = frame
        return self._make_observation(frame, info)

    def step(self, action: Action) -> StepResult:
        obs, reward, terminated, truncated, infos = self.sf_env.step(int(action.value))
        self.env_step += 1
        info = self._single_info(infos)
        frame = self._single_observation(obs)
        self._last_info = info
        self._last_frame = frame
        step_info = dict(info)
        step_info.update(
            {
                "env_step": self.env_step,
                "sim_time_ms": self.env_step * self.frame_ms,
                "applied_action": int(action.value),
                "applied_action_name": action.name,
                "observation_mode": "sample_factory_gpu",
            }
        )
        return StepResult(
            observation=self._make_observation(frame, info),
            reward=float(self._single_scalar(reward)),
            done=bool(self._single_scalar(terminated)),
            truncated=bool(self._single_scalar(truncated)),
            info=step_info,
        )

    def render_frame(self) -> Any:
        return self._last_frame

    def get_state_info(self) -> dict[str, Any]:
        info = dict(self._last_info)
        info.update(
            {
                "env_step": self.env_step,
                "sim_time_ms": self.env_step * self.frame_ms,
                "score": info.get("score"),
                "observation_mode": "sample_factory_gpu",
            }
        )
        return info

    def close(self) -> None:
        self.sf_env.close()

    def _make_observation(self, frame: Any, info: dict[str, Any]) -> Observation:
        metadata = dict(info)
        metadata.update(
            {
                "env_step": self.env_step,
                "sim_time_ms": self.env_step * self.frame_ms,
                "observation_mode": "sample_factory_gpu",
            }
        )
        return Observation(
            data=frame,
            env_step=self.env_step,
            sim_time_ms=self.env_step * self.frame_ms,
            metadata=metadata,
        )

    def _sf_cfg(self, config: dict[str, Any]) -> SimpleNamespace:
        env_cfg = config["env"]
        experiment_cfg = config.get("experiment", {})
        return SimpleNamespace(
            gym_id=env_cfg.get("gym_id", "FlappyBird-v0"),
            use_lidar=bool(env_cfg.get("use_lidar", False)),
            normalize_obs=bool(env_cfg.get("normalize_obs", True)),
            audio_on=bool(env_cfg.get("audio_on", False)),
            pipe_gap=int(env_cfg.get("pipe_gap", 100)),
            bird_color=env_cfg.get("bird_color", "yellow"),
            pipe_color=env_cfg.get("pipe_color", "green"),
            background=env_cfg.get("background", "day"),
            debug=bool(env_cfg.get("debug", False)),
            debug_timelimit_diagnostics=bool(env_cfg.get("debug_timelimit_diagnostics", False)),
            screen_size=env_cfg.get("screen_size", ""),
            score_limit=env_cfg.get("score_limit", -1),
            obs_resize=env_cfg.get("obs_resize"),
            obs_mode="image_only",
            frame_stack=int(env_cfg.get("frame_stack", 1) or 1),
            max_episode_steps=int(env_cfg.get("max_episode_steps", 0) or 0),
            queue_latency_values="0",
            latency_mode="episode_fixed",
            latency_sampling_mode="uniform_all",
            episodes_per_stage=int(10**9),
            randomize_within_stage=False,
            within_episode_change_interval=1,
            within_episode_change_prob=1.0,
            add_latency_info=False,
            seed=int(experiment_cfg.get("seed", 0) or 0),
            episode_metrics_path="",
            use_gpu_render=True,
            gpu_render_device=env_cfg.get("gpu_render_device", "auto"),
            gpu_render_batch_size=1,
            gpu_render_profile=bool(env_cfg.get("gpu_render_profile", False)),
            gpu_render_profile_interval=int(env_cfg.get("gpu_render_profile_interval", 1000) or 1000),
        )

    @staticmethod
    def _single_observation(obs: Any) -> np.ndarray:
        if isinstance(obs, dict):
            obs = obs["obs"]
        if hasattr(obs, "detach") and callable(obs.detach):
            obs = obs.detach().cpu().numpy()
        elif hasattr(obs, "cpu") and callable(obs.cpu):
            obs = obs.cpu().numpy()
        arr = np.asarray(obs)
        if arr.ndim == 4:
            if arr.shape[0] != 1:
                raise ValueError(f"Expected one Flappy GPU env observation, got batch shape {arr.shape!r}")
            arr = arr[0]
        if arr.ndim != 3:
            raise ValueError(f"Expected Flappy GPU CHW observation, got shape {arr.shape!r}")
        return np.asarray(arr, dtype=np.uint8)

    @staticmethod
    def _single_info(infos: Any) -> dict[str, Any]:
        if isinstance(infos, (list, tuple)):
            if len(infos) != 1:
                raise ValueError(f"Expected one Flappy GPU env info dict, got {len(infos)}")
            return dict(infos[0] or {})
        return dict(infos or {})

    @staticmethod
    def _single_scalar(value: Any) -> float:
        if hasattr(value, "detach") and callable(value.detach):
            value = value.detach().cpu().numpy()
        elif hasattr(value, "cpu") and callable(value.cpu):
            value = value.cpu().numpy()
        arr = np.asarray(value)
        if arr.shape == ():
            return float(arr.item())
        if arr.size != 1:
            raise ValueError(f"Expected scalar Flappy GPU env value, got shape {arr.shape!r}")
        return float(arr.reshape(-1)[0])
