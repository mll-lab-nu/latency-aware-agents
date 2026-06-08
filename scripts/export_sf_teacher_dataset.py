from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
import inspect
import json
from pathlib import Path
import sys
from typing import Any, Mapping, NamedTuple, Sequence

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.sample_factory_common import import_integration_module, load_training_config
from training.sample_factory_common.checkpoint_io import load_sample_factory_checkpoint_from_paths
from training.sample_factory_common.common import maybe_set_cli_override
from training.sample_factory_common.queue_eval import prepare_queue_eval_cfg


class EpisodeSpec(NamedTuple):
    local_idx: int
    episode_idx: int
    split: str
    seed: int


class JitterLatencySpec(NamedTuple):
    distribution: str
    mean: float | None
    std: float | None
    min_latency: int
    max_latency: int


def _register_components(module, env_name: str) -> None:
    component_fn = getattr(module, f"register_{env_name}_components", None)
    if callable(component_fn):
        component_fn()
        return

    env_fn = getattr(module, f"register_{env_name}_envs", None)
    if callable(env_fn):
        env_fn()
        return

    raise ValueError(f"No environment registration entrypoint found for env '{env_name}'")


def _infer_env_name_from_experiment_dir(experiment_dir: Path) -> str:
    name = experiment_dir.name.strip().lower()
    for env_name in ("deadly_corridor", "demon_attack", "flappy"):
        if name == env_name or name.startswith(f"{env_name}_"):
            return env_name
    raise ValueError(
        f"Could not infer env name from experiment directory '{experiment_dir.name}'. "
        "Pass --config with a matching env.name if the checkpoint naming scheme is different."
    )


def _parse_module_cfg(module, cli_args: list[str]):
    parse_cfg = getattr(module, "parse_cfg")
    try:
        return parse_cfg(cli_args, evaluation=True)
    except TypeError:
        return parse_cfg(cli_args)


def _parse_image_size(value: str) -> tuple[int, int] | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    parts = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if len(parts) != 2:
        raise ValueError(f"Expected image size in 'W,H' format, got: {value!r}")
    width, height = parts
    if width <= 0 or height <= 0:
        raise ValueError(f"Image size must be positive, got: {value!r}")
    return (width, height)


def _minimal_export_cfg(module, *, experiment_dir: Path, device_override: str | None):
    env_id = str(getattr(module, "SF_ENV_NAME"))
    checkpoint_cfg: dict[str, Any] = {}
    checkpoint_config_path = experiment_dir / "config.json"
    if checkpoint_config_path.exists():
        try:
            checkpoint_cfg = json.loads(checkpoint_config_path.read_text(encoding="utf-8"))
            saved_env_id = checkpoint_cfg.get("env")
            if saved_env_id:
                env_id = str(saved_env_id)
        except Exception:
            checkpoint_cfg = {}

    cli_args = [
        "--mode",
        "eval",
        "--algo",
        "APPO",
        "--env",
        env_id,
        "--experiment",
        str(experiment_dir.name),
        "--train_dir",
        str(experiment_dir.parent),
    ]

    # Seed parse_cfg with the saved checkpoint env/render settings so the env is
    # reconstructed with the correct observation shape before loading weights.
    saved_cli_fields = (
        "device",
        "gym_id",
        "env_frameskip",
        "eval_env_frameskip",
        "frame_stack",
        "max_episode_steps",
        "res_w",
        "res_h",
        "wide_aspect_ratio",
        "fps",
        "screen_size",
        "policy_screen_size",
        "use_gpu_render",
        "gpu_render_device",
        "gpu_render_batch_size",
        "gpu_render_profile",
        "gpu_render_profile_interval",
        "action_repeat",
        "full_action_space",
        "terminal_on_life_loss",
        "clip_reward",
        "repeat_action_probability",
        "noop_max",
        "use_lidar",
        "normalize_obs",
        "audio_on",
        "pipe_gap",
        "bird_color",
        "pipe_color",
        "background",
        "score_limit",
        "debug",
    )
    for field in saved_cli_fields:
        value = checkpoint_cfg.get(field)
        if value is None:
            continue
        cli_args.extend([f"--{field}", str(value)])

    if device_override:
        cli_args.extend(["--device", str(device_override)])
    return _parse_module_cfg(module, cli_args)


def _latency_value_csv(min_latency: int, max_latency: int) -> str:
    if int(max_latency) < int(min_latency):
        raise ValueError(
            f"max_latency must be greater than or equal to min_latency, got {max_latency} < {min_latency}"
        )
    return ",".join(str(value) for value in range(int(min_latency), int(max_latency) + 1))


def _apply_jitter_latency_overrides(cfg, jitter_spec: JitterLatencySpec) -> None:
    maybe_set_cli_override(
        cfg,
        "queue_latency_values",
        _latency_value_csv(jitter_spec.min_latency, jitter_spec.max_latency),
    )
    maybe_set_cli_override(cfg, "latency_mode", "within_episode_random")
    maybe_set_cli_override(cfg, "latency_sampling_mode", "uniform_all")
    maybe_set_cli_override(cfg, "latency_distribution", str(jitter_spec.distribution))
    if jitter_spec.distribution == "normal":
        if jitter_spec.mean is None or jitter_spec.std is None:
            raise ValueError("Normal jitter latency requires mean and std")
        maybe_set_cli_override(cfg, "latency_normal_mean", float(jitter_spec.mean))
        maybe_set_cli_override(cfg, "latency_normal_std", float(jitter_spec.std))
        maybe_set_cli_override(cfg, "latency_normal_min", int(jitter_spec.min_latency))
        maybe_set_cli_override(cfg, "latency_normal_max", int(jitter_spec.max_latency))
    elif jitter_spec.distribution == "uniform":
        maybe_set_cli_override(cfg, "latency_uniform_min", int(jitter_spec.min_latency))
        maybe_set_cli_override(cfg, "latency_uniform_max", int(jitter_spec.max_latency))
    else:
        raise ValueError(f"Unsupported jitter latency distribution: {jitter_spec.distribution}")
    maybe_set_cli_override(cfg, "within_episode_change_interval", 1)
    maybe_set_cli_override(cfg, "within_episode_change_prob", 1.0)
    maybe_set_cli_override(cfg, "add_latency_info", True)


def _prepare_export_cfg(
    module,
    env_name: str,
    base_cfg,
    latency: int,
    jitter_spec: JitterLatencySpec | None,
):
    from sample_factory.cfg.arguments import load_from_checkpoint

    if env_name == "deadly_corridor" and hasattr(module, "_prepare_eval_cfg"):
        cfg = module._prepare_eval_cfg(base_cfg, latency)
        cfg = load_from_checkpoint(cfg)
        if jitter_spec is not None:
            _apply_jitter_latency_overrides(cfg, jitter_spec)
            return cfg
        cfg.queue_latency_values = str(int(latency))
        cfg.latency_mode = "episode_fixed"
        cfg.latency_sampling_mode = "uniform_all"
        cfg.eval_env_frameskip = cfg.env_frameskip
        return cfg

    cfg = prepare_queue_eval_cfg(base_cfg, latency)
    cfg = load_from_checkpoint(cfg)
    if jitter_spec is not None:
        _apply_jitter_latency_overrides(cfg, jitter_spec)
        if env_name == "demon_attack" and bool(getattr(base_cfg, "eval_raw_reward", False)):
            maybe_set_cli_override(cfg, "clip_reward", False)
        if env_name == "demon_attack":
            maybe_set_cli_override(cfg, "terminal_on_life_loss", False)
        return cfg
    maybe_set_cli_override(cfg, "queue_latency_values", str(int(latency)))
    maybe_set_cli_override(cfg, "latency_mode", "episode_fixed")
    maybe_set_cli_override(cfg, "latency_sampling_mode", "uniform_all")
    if env_name == "demon_attack" and bool(getattr(base_cfg, "eval_raw_reward", False)):
        maybe_set_cli_override(cfg, "clip_reward", False)
    if env_name == "demon_attack":
        maybe_set_cli_override(cfg, "terminal_on_life_loss", False)
    return cfg


def _build_env_and_policy(module, cfg, env_name: str, num_envs: int):
    import torch
    from sample_factory.algo.sampling.batched_sampling import preprocess_actions
    from sample_factory.algo.utils.action_distributions import argmax_actions
    from sample_factory.algo.utils.env_info import extract_env_info
    from sample_factory.algo.utils.make_env import make_env_func_batched
    from sample_factory.algo.utils.rl_utils import prepare_and_normalize_obs
    from sample_factory.algo.utils.tensor_utils import unsqueeze_tensor
    from sample_factory.model.actor_critic import create_actor_critic
    from sample_factory.model.model_utils import get_rnn_size
    from sample_factory.utils.attr_dict import AttrDict
    from sample_factory.algo.learning.learner import Learner

    env_count = int(num_envs)
    if env_count < 1:
        raise ValueError(f"num_envs must be positive, got {num_envs}")

    def build_env(local_cfg, env_id: int):
        local_env = make_env_func_batched(
            local_cfg,
            env_config=AttrDict(worker_index=0, vector_index=int(env_id), env_id=int(env_id)),
            render_mode="rgb_array" if env_name in {"flappy", "demon_attack"} else None,
        )
        return local_env

    def build_env_and_model(local_cfg):
        local_env = build_env(local_cfg, 0)
        local_env_info = extract_env_info(local_env, local_cfg)
        local_actor_critic = create_actor_critic(local_cfg, local_env.observation_space, local_env.action_space)
        local_actor_critic.eval()
        return local_env, local_env_info, local_actor_critic

    def checkpoint_dict_for_cfg(local_cfg, device):
        if hasattr(module, "_load_policy_state_dict"):
            policy_id = local_cfg.policy_index
            name_prefix = dict(latest="checkpoint", best="best")[local_cfg.load_checkpoint_kind]
            checkpoints = Learner.get_checkpoints(Learner.checkpoint_dir(local_cfg, policy_id), f"{name_prefix}_*")
            return load_sample_factory_checkpoint_from_paths(checkpoints, device)
        policy_id = local_cfg.policy_index
        name_prefix = dict(latest="checkpoint", best="best")[local_cfg.load_checkpoint_kind]
        checkpoints = Learner.get_checkpoints(Learner.checkpoint_dir(local_cfg, policy_id), f"{name_prefix}_*")
        return load_sample_factory_checkpoint_from_paths(checkpoints, device)

    def obs_shape_from_space(obs_space):
        if hasattr(obs_space, "spaces") and "obs" in getattr(obs_space, "spaces", {}):
            obs_space = obs_space.spaces["obs"]
        return tuple(int(dim) for dim in getattr(obs_space, "shape", ()))

    def expected_obs_shape_from_checkpoint(checkpoint_dict):
        model_state = checkpoint_dict.get("model", {})
        keys_to_try = (
            "obs_normalizer.running_mean_std.running_mean_std.obs.running_mean",
            "obs_normalizer.running_mean_std.running_mean_std.running_mean",
        )
        for key in keys_to_try:
            tensor = model_state.get(key)
            if tensor is not None and hasattr(tensor, "shape"):
                shape = tuple(int(dim) for dim in tensor.shape)
                if len(shape) == 3:
                    return shape
        return None

    device = torch.device("cpu" if str(cfg.device) == "cpu" else "cuda")
    checkpoint_dict = checkpoint_dict_for_cfg(cfg, device)

    env, env_info, actor_critic = build_env_and_model(cfg)
    actor_critic.model_to_device(device)

    expected_obs_shape = expected_obs_shape_from_checkpoint(checkpoint_dict)
    actual_obs_shape = obs_shape_from_space(env.observation_space)
    if expected_obs_shape and actual_obs_shape and expected_obs_shape != actual_obs_shape:
        env.close()
        if len(expected_obs_shape) == 3:
            expected_h = int(expected_obs_shape[1])
            expected_w = int(expected_obs_shape[2])
            if env_name == "flappy":
                maybe_set_cli_override(cfg, "screen_size", "288,512")
                maybe_set_cli_override(cfg, "policy_screen_size", f"{expected_w},{expected_h}")
            if hasattr(cfg, "res_h"):
                maybe_set_cli_override(cfg, "res_h", expected_h)
            if hasattr(cfg, "res_w"):
                maybe_set_cli_override(cfg, "res_w", expected_w)
            if hasattr(cfg, "screen_size") and env_name != "flappy":
                maybe_set_cli_override(cfg, "screen_size", f"{expected_w},{expected_h}")
        env, env_info, actor_critic = build_env_and_model(cfg)
        actor_critic.model_to_device(device)

    if hasattr(module, "_load_policy_state_dict"):
        actor_critic.load_state_dict(checkpoint_dict["model"])
    else:
        from training.sample_factory_common.queue_eval import load_sf_policy_state_dict

        try:
            actor_critic.load_state_dict(checkpoint_dict["model"])
        except Exception:
            load_sf_policy_state_dict(cfg, actor_critic, device)

    envs = [env]
    for env_id in range(1, env_count):
        envs.append(build_env(cfg, env_id))

    return {
        "env": env,
        "envs": envs,
        "env_info": env_info,
        "actor_critic": actor_critic,
        "device": device,
        "preprocess_actions": preprocess_actions,
        "argmax_actions": argmax_actions,
        "prepare_and_normalize_obs": prepare_and_normalize_obs,
        "unsqueeze_tensor": unsqueeze_tensor,
        "rnn_size": int(get_rnn_size(cfg)),
    }


def _to_python_scalar(value: Any):
    import torch

    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.detach().cpu().item()
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        if value.size == 1:
            return value.reshape(-1)[0].item()
        return value.tolist()
    if isinstance(value, (list, tuple)) and len(value) == 1:
        return _to_python_scalar(value[0])
    return value


def _extract_action_id(action_value: Any) -> int:
    scalar = _to_python_scalar(action_value)
    if isinstance(scalar, np.ndarray):
        scalar = np.asarray(scalar).reshape(-1)[0]
    if isinstance(scalar, (list, tuple)):
        if not scalar:
            raise ValueError("Encountered empty action container while exporting teacher data")
        scalar = _extract_action_id(scalar[0])
        return int(scalar)
    return int(scalar)


def _extract_action_components(action_value: Any) -> list[int]:
    scalar = _to_python_scalar(action_value)
    if isinstance(scalar, np.ndarray):
        if scalar.ndim == 0:
            return [int(scalar.item())]
        return [int(x) for x in scalar.reshape(-1).tolist()]
    if isinstance(scalar, (list, tuple)):
        components: list[int] = []
        for item in scalar:
            components.extend(_extract_action_components(item))
        return components
    return [int(scalar)]


def _extract_single_info(info: Any) -> dict[str, Any]:
    if isinstance(info, list):
        return dict(info[0] or {}) if info else {}
    if isinstance(info, tuple):
        return dict(info[0] or {}) if info else {}
    if not isinstance(info, dict):
        return {}

    extracted: dict[str, Any] = {}
    for key, value in info.items():
        if isinstance(value, np.ndarray):
            if value.ndim == 0:
                extracted[str(key)] = value.item()
            elif value.shape[0] > 0:
                item = value[0]
                extracted[str(key)] = item.item() if isinstance(item, np.generic) else item
            else:
                extracted[str(key)] = value
        elif isinstance(value, list):
            extracted[str(key)] = value[0] if value else None
        else:
            extracted[str(key)] = value
    return extracted


def _extract_single_observation(obs: Any):
    if isinstance(obs, Mapping):
        return {
            str(key): _extract_single_observation(value)
            for key, value in obs.items()
        }

    array = np.asarray(obs)
    if array.ndim >= 4:
        return array[0]
    if array.ndim >= 1 and array.shape[0] == 1:
        return array[0]
    return array


def _latest_frame_from_observation(single_obs: Any, *, env_name: str, frame_stack: int) -> np.ndarray:
    if isinstance(single_obs, Mapping):
        if "obs" in single_obs:
            return _latest_frame_from_observation(single_obs["obs"], env_name=env_name, frame_stack=frame_stack)
        raise ValueError(f"Unsupported dict observation keys for export: {sorted(single_obs)}")

    array = np.asarray(single_obs, dtype=np.uint8)
    if array.ndim == 2:
        array = np.repeat(array[..., None], 3, axis=2)
        return array

    if array.ndim != 3:
        raise ValueError(f"Unsupported observation shape for export: {array.shape!r}")

    if array.shape[0] <= 16 and array.shape[1] > 16 and array.shape[2] > 16:
        total_channels = int(array.shape[0])
        if frame_stack > 1 and total_channels % frame_stack == 0:
            base_channels = total_channels // frame_stack
        elif env_name == "demon_attack":
            base_channels = 1
        else:
            base_channels = min(3, total_channels)
        frame = array[-base_channels:, :, :]
        if frame.shape[0] == 1:
            frame = np.repeat(frame, 3, axis=0)
        elif frame.shape[0] > 3:
            frame = frame[:3]
        return np.transpose(frame, (1, 2, 0))

    total_channels = int(array.shape[2])
    if frame_stack > 1 and total_channels % frame_stack == 0:
        base_channels = total_channels // frame_stack
    else:
        base_channels = min(3, total_channels)
    frame = array[:, :, -base_channels:]
    if frame.shape[2] == 1:
        frame = np.repeat(frame, 3, axis=2)
    elif frame.shape[2] > 3:
        frame = frame[:, :, :3]
    return frame


def _save_observation_image(
    obs: Any,
    *,
    env: Any,
    env_name: str,
    frame_stack: int,
    image_path: Path,
    export_image_size: tuple[int, int] | None,
) -> str:
    image_path.parent.mkdir(parents=True, exist_ok=True)
    frame = None
    if env_name in {"flappy", "demon_attack"}:
        render_fn = getattr(env, "render", None)
        if callable(render_fn):
            rendered = render_fn()
            if rendered is not None:
                frame = np.asarray(rendered, dtype=np.uint8)
    if frame is None:
        single_obs = _extract_single_observation(obs)
        frame = _latest_frame_from_observation(single_obs, env_name=env_name, frame_stack=frame_stack)
    image = Image.fromarray(np.asarray(frame, dtype=np.uint8))
    if export_image_size is not None and env_name not in {"flappy", "demon_attack"} and image.size != export_image_size:
        image = image.resize(export_image_size, Image.BILINEAR)
    image.save(image_path)
    return str(image_path)


def _frame_from_observation(obs: Any, env_name: str, frame_stack: int) -> np.ndarray:
    single_obs = _extract_single_observation(obs)
    return _latest_frame_from_observation(single_obs, env_name=env_name, frame_stack=frame_stack)


def _iter_env_render_candidates(env: Any) -> list[Any]:
    candidates: list[Any] = []
    visited: set[int] = set()
    queue: list[Any] = [env]
    while queue:
        current = queue.pop(0)
        if current is None or id(current) in visited:
            continue
        visited.add(id(current))
        candidates.append(current)

        child = getattr(current, "env", None)
        if child is not None:
            queue.append(child)
        unwrapped = getattr(current, "unwrapped", None)
        if unwrapped is not None and unwrapped is not current:
            queue.append(unwrapped)
        envs = getattr(current, "envs", None)
        if isinstance(envs, list):
            queue.extend(envs)
        sub_envs = getattr(current, "sub_envs", None)
        if isinstance(sub_envs, list):
            queue.extend(sub_envs)
    return candidates


def _render_accepts_agent_idx(render_fn: Any) -> bool:
    try:
        signature = inspect.signature(render_fn)
    except (TypeError, ValueError):
        return False
    for parameter in signature.parameters.values():
        if parameter.kind == inspect.Parameter.VAR_KEYWORD:
            return True
        if parameter.name == "agent_idx":
            return True
    return False


def _render_frame_from_env(env: Any, env_name: str, agent_idx: int) -> np.ndarray:
    if env_name == "flappy":
        for candidate in _iter_env_render_candidates(env):
            render_fn = getattr(candidate, "render", None)
            if callable(render_fn) and _render_accepts_agent_idx(render_fn):
                rendered = render_fn(agent_idx=int(agent_idx))
                if rendered is None:
                    raise RuntimeError(
                        f"Environment {env_name} returned None from render(agent_idx={agent_idx})"
                    )
                return np.asarray(rendered, dtype=np.uint8)
        if int(agent_idx) != 0:
            raise RuntimeError(
                f"Environment {env_name} does not expose render(agent_idx=...) for agent_idx={agent_idx}"
            )

    render_fn = getattr(env, "render", None)
    if not callable(render_fn):
        raise RuntimeError(f"Environment {env_name} does not expose render() for image export")
    rendered = render_fn()
    if rendered is None:
        raise RuntimeError(f"Environment {env_name} returned None from render()")
    return np.asarray(rendered, dtype=np.uint8)


def _frame_for_image_source(
    obs: Any,
    env: Any,
    env_name: str,
    frame_stack: int,
    image_source: str,
    agent_idx: int = 0,
) -> np.ndarray:
    source = str(image_source)
    if source == "observation":
        return _frame_from_observation(obs, env_name=env_name, frame_stack=frame_stack)
    if source != "render":
        raise ValueError(f"Unsupported image source: {image_source}")
    if env_name not in {"flappy", "demon_attack"}:
        return _frame_from_observation(obs, env_name=env_name, frame_stack=frame_stack)
    return _render_frame_from_env(env, env_name, int(agent_idx))


def _save_frame_image(
    frame: np.ndarray,
    image_path: Path,
    export_image_size: tuple[int, int] | None,
    resize_image: bool,
) -> str:
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.fromarray(np.asarray(frame, dtype=np.uint8))
    if resize_image and export_image_size is not None and image.size != export_image_size:
        image = image.resize(export_image_size, Image.BILINEAR)
    image.save(image_path)
    return str(image_path)


def _drain_completed_image_writes(pending_futures: set[Future[str]]) -> set[Future[str]]:
    remaining: set[Future[str]] = set()
    for future in pending_futures:
        if future.done():
            future.result()
        else:
            remaining.add(future)
    return remaining


def _submit_image_write(
    executor: ThreadPoolExecutor | None,
    pending_futures: set[Future[str]],
    frame: np.ndarray,
    image_path: Path,
    export_image_size: tuple[int, int] | None,
    resize_image: bool,
    max_pending_writes: int,
) -> tuple[str, set[Future[str]]]:
    frame_to_write = np.array(frame, copy=True)
    if executor is None:
        return _save_frame_image(frame_to_write, image_path, export_image_size, resize_image), pending_futures

    limit = max(1, int(max_pending_writes))
    pending = _drain_completed_image_writes(pending_futures)
    while len(pending) >= limit:
        completed, pending = wait(pending, return_when=FIRST_COMPLETED)
        for future in completed:
            future.result()
    future = executor.submit(_save_frame_image, frame_to_write, image_path, export_image_size, resize_image)
    pending.add(future)
    return str(image_path), pending


def _wait_for_image_writes(pending_futures: set[Future[str]]) -> None:
    for future in pending_futures:
        future.result()


def _flappy_action_labels() -> dict[int, str]:
    return {0: "NOOP", 1: "FLAP"}


def _demon_action_labels(cfg: Any) -> dict[int, str]:
    if bool(getattr(cfg, "full_action_space", False)):
        return {idx: f"ACTION_{idx}" for idx in range(18)}
    return {
        0: "NOOP",
        1: "FIRE",
        2: "RIGHT",
        3: "LEFT",
        4: "RIGHTFIRE",
        5: "LEFTFIRE",
    }


def _extract_vizdoom_buttons(env) -> list[str]:
    visited: set[int] = set()
    queue: list[Any] = [env]
    while queue:
        current = queue.pop(0)
        if current is None or id(current) in visited:
            continue
        visited.add(id(current))

        game = getattr(current, "game", None)
        if game is not None and hasattr(game, "get_available_buttons"):
            try:
                buttons = game.get_available_buttons()
                names = []
                for button in buttons:
                    name = getattr(button, "name", None)
                    if name is None:
                        name = str(button).split(".")[-1]
                    names.append(str(name).upper())
                if names:
                    return names
            except Exception:
                pass

        child = getattr(current, "env", None)
        if child is not None:
            queue.append(child)
        unwrapped = getattr(current, "unwrapped", None)
        if unwrapped is not None and unwrapped is not current:
            queue.append(unwrapped)
        envs = getattr(current, "envs", None)
        if isinstance(envs, list):
            queue.extend(envs)
    return []


def _unwrap_env_attr(env, attr_name: str):
    visited: set[int] = set()
    queue: list[Any] = [env]
    while queue:
        current = queue.pop(0)
        if current is None or id(current) in visited:
            continue
        visited.add(id(current))
        if hasattr(current, attr_name):
            return getattr(current, attr_name)
        child = getattr(current, "env", None)
        if child is not None:
            queue.append(child)
        unwrapped = getattr(current, "unwrapped", None)
        if unwrapped is not None and unwrapped is not current:
            queue.append(unwrapped)
        envs = getattr(current, "envs", None)
        if isinstance(envs, list):
            queue.extend(envs)
    return None


def _deadly_action_labels(env) -> dict[int, str]:
    import gymnasium as gym

    action_space = getattr(env, "action_space", None)
    wrapper_labels = _unwrap_env_attr(env, "action_labels")
    if isinstance(wrapper_labels, dict) and wrapper_labels:
        return {int(key): str(value) for key, value in wrapper_labels.items()}
    if isinstance(action_space, gym.spaces.Tuple):
        expected_component_labels = [
            ["NOOP", "TURN_LEFT", "TURN_RIGHT"],
            ["NOOP", "MOVE_FORWARD", "MOVE_BACKWARD"],
            ["NOOP", "MOVE_LEFT", "MOVE_RIGHT"],
            ["NOOP", "ATTACK"],
        ]
        if (
            len(action_space.spaces) == 4
            and all(isinstance(space, gym.spaces.Discrete) for space in action_space.spaces)
            and [int(space.n) for space in action_space.spaces] == [3, 3, 3, 2]
        ):
            labels: dict[int, str] = {}
            action_id = 0
            for turn in range(3):
                for move_fb in range(3):
                    for move_lr in range(3):
                        for attack in range(2):
                            components = [turn, move_fb, move_lr, attack]
                            parts = [
                                expected_component_labels[idx][value]
                                for idx, value in enumerate(components)
                                if value != 0
                            ]
                            labels[action_id] = " + ".join(parts) if parts else "NOOP"
                            action_id += 1
            return labels

    n_actions = int(getattr(action_space, "n", 0) or 0)
    buttons = _extract_vizdoom_buttons(env)
    if n_actions <= 0:
        return {}
    if buttons:
        if n_actions == len(buttons) + 1:
            names = ["NOOP", *buttons]
            return {idx: names[idx] for idx in range(n_actions)}
        if n_actions == len(buttons):
            return {idx: buttons[idx] for idx in range(n_actions)}
    return {idx: f"ACTION_{idx}" for idx in range(n_actions)}


def _action_labels_for_env(env_name: str, cfg: Any, env) -> dict[int, str]:
    if env_name == "flappy":
        return _flappy_action_labels()
    if env_name == "demon_attack":
        return _demon_action_labels(cfg)
    if env_name == "deadly_corridor":
        return _deadly_action_labels(env)
    raise ValueError(f"Unsupported env for action labels: {env_name}")


def _deadly_action_id_from_policy_action(action_value: Any) -> int:
    components = _extract_action_components(action_value)
    if len(components) != 4:
        return _extract_action_id(action_value)
    turn, move_fb, move_lr, attack = [int(x) for x in components]
    return (((turn * 3) + move_fb) * 3 + move_lr) * 2 + attack


def _policy_action_id(env_name: str, action_value: Any) -> int:
    if env_name == "deadly_corridor" and len(_extract_action_components(action_value)) == 4:
        return _deadly_action_id_from_policy_action(action_value)
    return _extract_action_id(action_value)


def _prompt_for_episode(
    *,
    dataset_mode: str,
    prompt: str,
    mixed_prompt_template: str,
    latency: int,
    latency_ms: float,
) -> str:
    if dataset_mode in {"mixed_latency", "jitter_latency"}:
        return mixed_prompt_template.format(
            prompt=prompt,
            latency=int(latency),
            latency_ms=f"{float(latency_ms):.2f}",
        )
    return prompt


def _decision_period_ms_from_cfg(env_name: str, cfg: Any) -> float:
    if env_name == "flappy":
        return 1000.0 / 30.0
    if env_name == "demon_attack":
        action_repeat = int(getattr(cfg, "action_repeat", 4) or 4)
        return action_repeat * (1000.0 / 60.0)
    if env_name == "deadly_corridor":
        env_frameskip = int(getattr(cfg, "env_frameskip", 4) or 4)
        return env_frameskip * (1000.0 / 35.0)
    raise ValueError(f"Unsupported env for latency_ms conversion: {env_name}")


def _select_val_indices(total: int, val_count: int, split_seed: int) -> set[int]:
    if val_count <= 0:
        return set()
    rng = np.random.RandomState(split_seed)
    indices = np.arange(total)
    rng.shuffle(indices)
    return set(int(item) for item in indices[:val_count])


def _build_episode_specs(
    episode_count: int,
    val_indices: set[int],
    global_episode_start: int,
    base_seed: int,
    latency: int,
) -> list[EpisodeSpec]:
    specs: list[EpisodeSpec] = []
    for local_idx in range(int(episode_count)):
        split = "val" if int(local_idx) in val_indices else "train"
        specs.append(
            EpisodeSpec(
                local_idx=int(local_idx),
                episode_idx=int(global_episode_start) + int(local_idx),
                split=split,
                seed=int(base_seed) + int(latency) * 100_000 + int(local_idx),
            )
        )
    return specs


def _take_next_episode_specs(
    specs: Sequence[EpisodeSpec],
    start_index: int,
    slot_count: int,
) -> tuple[list[EpisodeSpec], int]:
    start = int(start_index)
    if start < 0:
        raise ValueError(f"start_index must be non-negative, got {start_index}")
    slots = int(slot_count)
    if slots < 0:
        raise ValueError(f"slot_count must be non-negative, got {slot_count}")
    end = min(len(specs), start + slots)
    return list(specs[start:end]), end


def _batch_observations(observations: Sequence[Any]) -> Any:
    if not observations:
        raise ValueError("Cannot batch an empty observation sequence")
    first = observations[0]
    if isinstance(first, Mapping):
        keys = tuple(first.keys())
        for obs in observations:
            if not isinstance(obs, Mapping):
                raise TypeError("All observations must have the same structure")
            if tuple(obs.keys()) != keys:
                raise ValueError("All mapping observations must have identical keys")
        return {
            key: _batch_observations([obs[key] for obs in observations])
            for key in keys
        }

    try:
        import torch
    except ImportError:
        torch = None
    if torch is not None and isinstance(first, torch.Tensor):
        if first.ndim >= 1 and int(first.shape[0]) == 1:
            return torch.cat(list(observations), dim=0)
        return torch.stack(list(observations), dim=0)

    arrays = [np.asarray(obs) for obs in observations]
    if arrays[0].ndim >= 1 and arrays[0].shape[0] == 1:
        return np.concatenate(arrays, axis=0)
    return np.stack(arrays, axis=0)


def _select_batch_item(value: Any, index: int) -> Any:
    if isinstance(value, Mapping):
        return {
            key: _select_batch_item(item, index)
            for key, item in value.items()
        }
    try:
        import torch
    except ImportError:
        torch = None
    if torch is not None and isinstance(value, torch.Tensor):
        return value[int(index) : int(index) + 1]
    if isinstance(value, np.ndarray):
        return value[int(index) : int(index) + 1]
    if isinstance(value, list):
        if value:
            first_item = value[0]
            is_structured_item = isinstance(first_item, (Mapping, list, tuple, np.ndarray))
            is_tensor_item = torch is not None and isinstance(first_item, torch.Tensor)
            if is_structured_item or is_tensor_item:
                return [_select_batch_item(item, index) for item in value]
        return [value[int(index)]]
    if isinstance(value, tuple):
        return tuple(_select_batch_item(item, index) for item in value)
    raise TypeError(f"Unsupported batched value type: {type(value).__name__}")


def _find_experiment_dir(checkpoint_root: Path, env_name: str, latency: int, seed: int) -> Path:
    expected_name = f"{env_name}_sf_same_latency_l{latency}_seed{seed}"
    direct = checkpoint_root / expected_name
    if direct.exists():
        return direct
    if checkpoint_root.name == expected_name:
        return checkpoint_root
    matches = sorted(checkpoint_root.glob(f"{expected_name}*"))
    if not matches:
        raise FileNotFoundError(
            f"Could not find experiment directory for latency {latency} under {checkpoint_root}"
        )
    return matches[0]


def _format_latency_number_for_path(value: float) -> str:
    numeric = float(value)
    if numeric.is_integer():
        return str(int(numeric))
    return f"{numeric:g}".replace("-", "m").replace(".", "p")


def _jitter_latency_label(jitter_spec: JitterLatencySpec) -> str:
    if jitter_spec.distribution == "uniform":
        return (
            "uniform"
            f"_min_{_format_latency_number_for_path(float(jitter_spec.min_latency))}"
            f"_max_{_format_latency_number_for_path(float(jitter_spec.max_latency))}"
        )
    if jitter_spec.distribution != "normal":
        raise ValueError(f"Unsupported jitter latency distribution: {jitter_spec.distribution}")
    if jitter_spec.mean is None or jitter_spec.std is None:
        raise ValueError("Normal jitter latency requires mean and std")
    return (
        "normal"
        f"_mean_{_format_latency_number_for_path(float(jitter_spec.mean))}"
        f"_std_{_format_latency_number_for_path(float(jitter_spec.std))}"
        f"_min_{_format_latency_number_for_path(float(jitter_spec.min_latency))}"
        f"_max_{_format_latency_number_for_path(float(jitter_spec.max_latency))}"
    )


def _find_jitter_experiment_dir(
    *,
    checkpoint_root: Path,
    env_name: str,
    jitter_spec: JitterLatencySpec,
    seed: int,
) -> Path:
    label = _jitter_latency_label(jitter_spec)
    expected_name = f"{env_name}_sf_latency_{label}_seed{int(seed)}"
    direct = checkpoint_root / expected_name
    if direct.exists():
        return direct
    if checkpoint_root.name == expected_name:
        return checkpoint_root
    matches = sorted(checkpoint_root.glob(f"{expected_name}*"))
    if not matches:
        raise FileNotFoundError(
            f"Could not find jitter experiment directory {expected_name} under {checkpoint_root}"
        )
    return matches[0]


def _find_best_checkpoint_name(experiment_dir: Path) -> str:
    checkpoints = sorted(experiment_dir.glob("checkpoint_p*/best_*.pth"))
    if not checkpoints:
        raise FileNotFoundError(f"No best_* checkpoint found under {experiment_dir}")
    return checkpoints[-1].name


def _empty_records() -> dict[str, list[Any]]:
    return {
        "image": [],
        "split": [],
        "prompt": [],
        "action_id": [],
        "action_text": [],
        "latency": [],
        "latency_ms": [],
        "reward": [],
        "done": [],
        "episode_idx": [],
        "t": [],
        "seed": [],
        "deterministic": [],
    }


def _write_parquet_splits(output_dir: Path, train_rows: list[dict[str, Any]], val_rows: list[dict[str, Any]]) -> None:
    try:
        from datasets import Dataset, DatasetDict, Features, Image as HFImage, Value
    except ImportError as exc:
        raise RuntimeError(
            "Exporting parquet datasets requires the 'datasets' package and a parquet backend such as 'pyarrow'."
        ) from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    features = Features(
        {
            "image": HFImage(),
            "split": Value("string"),
            "prompt": Value("string"),
            "action_id": Value("int64"),
            "action_text": Value("string"),
            "latency": Value("int64"),
            "latency_ms": Value("float64"),
            "reward": Value("float64"),
            "done": Value("bool"),
            "episode_idx": Value("int64"),
            "t": Value("int64"),
            "seed": Value("int64"),
            "deterministic": Value("bool"),
        }
    )

    def build_dataset(rows: list[dict[str, Any]]):
        if rows:
            return Dataset.from_list(rows, features=features)
        return Dataset.from_dict(_empty_records(), features=features)

    train_ds = build_dataset(train_rows)
    val_ds = build_dataset(val_rows)
    train_ds.to_parquet(str(output_dir / "train.parquet"))
    val_ds.to_parquet(str(output_dir / "val.parquet"))
    return DatasetDict({"train": train_ds, "val": val_ds})


def _push_dataset_to_hub(
    dataset_dict,
    *,
    repo_id: str,
    private: bool,
    token: str | None,
    max_shard_size: str,
) -> None:
    push_kwargs: dict[str, Any] = {
        "repo_id": repo_id,
        "private": bool(private),
        "max_shard_size": str(max_shard_size),
    }
    if token:
        push_kwargs["token"] = token
    dataset_dict.push_to_hub(**push_kwargs)


def _reward_from_step(reward_value: Any, info: dict[str, Any], *, prefer_raw_reward: bool) -> float:
    if prefer_raw_reward and "raw_reward" in info:
        return float(info["raw_reward"])
    return float(_to_python_scalar(reward_value))


def _latency_from_step_info(info: dict[str, Any], fallback_latency: int) -> int:
    for key in ("queue_latency", "frame_skip", "latency_frame_skip"):
        value = info.get(key)
        if value is not None:
            return int(_to_python_scalar(value))
    return int(fallback_latency)


def _build_export_row(
    image_path: str,
    split: str,
    prompt_text: str,
    action_id: int,
    action_text: str,
    latency: int,
    latency_ms: float,
    reward_value: float,
    done: bool,
    episode_idx: int,
    t: int,
    seed: int,
    deterministic: bool,
) -> dict[str, Any]:
    return {
        "image": image_path,
        "split": split,
        "prompt": prompt_text,
        "action_id": int(action_id),
        "action_text": action_text,
        "latency": int(latency),
        "latency_ms": float(latency_ms),
        "reward": float(reward_value),
        "done": bool(done),
        "episode_idx": int(episode_idx),
        "t": int(t),
        "seed": int(seed),
        "deterministic": bool(deterministic),
    }


def _batched_reward_item(reward: Any, index: int) -> Any:
    try:
        import torch
    except ImportError:
        torch = None
    if torch is not None and isinstance(reward, torch.Tensor):
        return reward[int(index)]
    if isinstance(reward, np.ndarray):
        return reward[int(index)]
    if isinstance(reward, (list, tuple)):
        return reward[int(index)]
    return reward


def _make_dones(terminated: Any, truncated: Any) -> Any:
    try:
        import torch
    except ImportError:
        torch = None
    if torch is not None and isinstance(terminated, torch.Tensor):
        return torch.logical_or(terminated, truncated)
    return np.logical_or(np.asarray(terminated), np.asarray(truncated))


def _collect_latency_dataset_batched(
    *,
    env_name: str,
    module,
    base_cfg,
    latency: int,
    episode_count: int,
    val_indices: set[int],
    max_steps_per_episode: int,
    prompt: str,
    mixed_prompt_template: str,
    dataset_mode: str,
    output_dir: Path,
    global_episode_start: int,
    deterministic: bool,
    export_image_size: tuple[int, int] | None,
    num_envs: int,
    image_writer_workers: int,
    row_buffer_size: int,
    image_source: str,
    jitter_spec: JitterLatencySpec | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    import torch

    cfg = _prepare_export_cfg(module, env_name, base_cfg, latency, jitter_spec)
    if env_name == "flappy" and bool(getattr(cfg, "use_gpu_render", False)):
        return _collect_flappy_gpu_latency_dataset_batched(
            module=module,
            cfg=cfg,
            latency=latency,
            episode_count=episode_count,
            val_indices=val_indices,
            max_steps_per_episode=max_steps_per_episode,
            prompt=prompt,
            mixed_prompt_template=mixed_prompt_template,
            dataset_mode=dataset_mode,
            output_dir=output_dir,
            global_episode_start=global_episode_start,
            deterministic=deterministic,
            export_image_size=export_image_size,
            num_envs=num_envs,
            image_writer_workers=image_writer_workers,
            row_buffer_size=row_buffer_size,
            image_source=image_source,
        )
    runtime = _build_env_and_policy(module, cfg, env_name, int(num_envs))
    envs = list(runtime["envs"])
    env_info = runtime["env_info"]
    actor_critic = runtime["actor_critic"]
    device = runtime["device"]
    preprocess_actions = runtime["preprocess_actions"]
    argmax_actions = runtime["argmax_actions"]
    prepare_and_normalize_obs = runtime["prepare_and_normalize_obs"]
    unsqueeze_tensor = runtime["unsqueeze_tensor"]
    rnn_size = runtime["rnn_size"]
    action_labels = _action_labels_for_env(env_name, cfg, envs[0])
    prefer_raw_reward = bool(getattr(cfg, "eval_raw_reward", False))
    frame_stack = int(getattr(cfg, "frame_stack", 1) or 1)
    decision_period_ms = _decision_period_ms_from_cfg(env_name, cfg)
    base_seed = int(getattr(cfg, "seed", 0) or 0)
    specs = _build_episode_specs(
        episode_count=int(episode_count),
        val_indices=val_indices,
        global_episode_start=int(global_episode_start),
        base_seed=base_seed,
        latency=int(latency),
    )

    train_rows: list[dict[str, Any]] = []
    val_rows: list[dict[str, Any]] = []
    active_specs: list[EpisodeSpec | None] = [None for _ in envs]
    active_obs: list[Any | None] = [None for _ in envs]
    active_t: list[int] = [0 for _ in envs]
    rnn_states = torch.zeros([len(envs), rnn_size], dtype=torch.float32, device=device)
    pending_futures: set[Future[str]] = set()
    executor = (
        ThreadPoolExecutor(max_workers=int(image_writer_workers))
        if int(image_writer_workers) > 0
        else None
    )

    def start_episode(slot: int, spec: EpisodeSpec) -> None:
        obs, _ = envs[int(slot)].reset(seed=int(spec.seed))
        active_specs[int(slot)] = spec
        active_obs[int(slot)] = obs
        active_t[int(slot)] = 0
        rnn_states[int(slot)] = torch.zeros_like(rnn_states[int(slot)])

    try:
        next_spec_index = 0
        initial_specs, next_spec_index = _take_next_episode_specs(
            specs=specs,
            start_index=next_spec_index,
            slot_count=len(envs),
        )
        for slot, spec in enumerate(initial_specs):
            start_episode(slot, spec)

        completed_episodes = 0
        while completed_episodes < int(episode_count):
            active_slots = [idx for idx, spec in enumerate(active_specs) if spec is not None]
            if not active_slots:
                break

            obs_batch = _batch_observations([active_obs[idx] for idx in active_slots])
            rnn_batch = rnn_states[active_slots]
            normalized_obs = prepare_and_normalize_obs(actor_critic, obs_batch)
            with torch.no_grad():
                policy_outputs = actor_critic(normalized_obs, rnn_batch)
                actions = policy_outputs["actions"]
                if deterministic:
                    actions = argmax_actions(actor_critic.action_distribution())
                if actions.ndim == 1:
                    actions = unsqueeze_tensor(actions, dim=-1)
                policy_actions = actions
                processed_actions = preprocess_actions(env_info, actions)
                new_rnn_states = policy_outputs["new_rnn_states"]

            for batch_idx, slot in enumerate(active_slots):
                spec = active_specs[slot]
                obs = active_obs[slot]
                if spec is None or obs is None:
                    raise RuntimeError(f"Active slot {slot} has no episode state")

                split_rows = val_rows if spec.split == "val" else train_rows
                t = int(active_t[slot])
                image_path = output_dir / "images" / spec.split / f"episode_{spec.episode_idx:06d}" / f"t_{t:06d}.png"
                frame = _frame_for_image_source(
                    obs=obs,
                    env=envs[slot],
                    env_name=env_name,
                    frame_stack=frame_stack,
                    image_source=image_source,
                )
                saved_image, pending_futures = _submit_image_write(
                    executor,
                    pending_futures,
                    frame,
                    image_path,
                    export_image_size,
                    env_name not in {"flappy", "demon_attack"},
                    int(row_buffer_size),
                )

                slot_policy_action = _select_batch_item(policy_actions, batch_idx)
                action_id = _policy_action_id(env_name, slot_policy_action)
                action_text = action_labels.get(action_id, str(action_id))
                next_obs, reward, terminated, truncated, info = envs[slot].step(
                    _select_batch_item(processed_actions, batch_idx)
                )
                dones = _make_dones(terminated, truncated)
                done = bool(_to_python_scalar(dones[0]))
                info_single = _extract_single_info(info)
                actual_latency = _latency_from_step_info(info_single, fallback_latency=int(latency))
                actual_latency_ms = float(actual_latency) * decision_period_ms
                prompt_text = _prompt_for_episode(
                    dataset_mode=dataset_mode,
                    prompt=prompt,
                    mixed_prompt_template=mixed_prompt_template,
                    latency=actual_latency,
                    latency_ms=actual_latency_ms,
                )
                reward_value = _reward_from_step(
                    _batched_reward_item(reward, 0),
                    info_single,
                    prefer_raw_reward=prefer_raw_reward,
                )
                split_rows.append(
                    _build_export_row(
                        image_path=saved_image,
                        split=spec.split,
                        prompt_text=prompt_text,
                        action_id=action_id,
                        action_text=action_text,
                        latency=int(actual_latency),
                        latency_ms=float(actual_latency_ms),
                        reward_value=float(reward_value),
                        done=bool(done),
                        episode_idx=int(spec.episode_idx),
                        t=t,
                        seed=int(spec.seed),
                        deterministic=bool(deterministic),
                    )
                )

                rnn_states[slot] = new_rnn_states[batch_idx]
                next_t = t + 1
                episode_finished = bool(done) or next_t >= int(max_steps_per_episode)
                if episode_finished:
                    completed_episodes += 1
                    replacement_specs, next_spec_index = _take_next_episode_specs(
                        specs=specs,
                        start_index=next_spec_index,
                        slot_count=1,
                    )
                    if replacement_specs:
                        start_episode(slot, replacement_specs[0])
                    else:
                        active_specs[slot] = None
                        active_obs[slot] = None
                        active_t[slot] = 0
                        rnn_states[slot] = torch.zeros_like(rnn_states[slot])
                else:
                    active_obs[slot] = next_obs
                    active_t[slot] = next_t
    finally:
        try:
            _wait_for_image_writes(pending_futures)
        finally:
            if executor is not None:
                executor.shutdown(wait=True)
            for env in envs:
                env.close()

    return train_rows, val_rows, {"latency": int(latency), "action_labels": action_labels}


def _collect_flappy_gpu_latency_dataset_batched(
    *,
    module,
    cfg,
    latency: int,
    episode_count: int,
    val_indices: set[int],
    max_steps_per_episode: int,
    prompt: str,
    mixed_prompt_template: str,
    dataset_mode: str,
    output_dir: Path,
    global_episode_start: int,
    deterministic: bool,
    export_image_size: tuple[int, int] | None,
    num_envs: int,
    image_writer_workers: int,
    row_buffer_size: int,
    image_source: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    import torch

    maybe_set_cli_override(cfg, "gpu_render_batch_size", int(num_envs))
    runtime = _build_env_and_policy(module, cfg, "flappy", 1)
    env = runtime["envs"][0]
    env_info = runtime["env_info"]
    actor_critic = runtime["actor_critic"]
    device = runtime["device"]
    preprocess_actions = runtime["preprocess_actions"]
    argmax_actions = runtime["argmax_actions"]
    prepare_and_normalize_obs = runtime["prepare_and_normalize_obs"]
    unsqueeze_tensor = runtime["unsqueeze_tensor"]
    rnn_size = runtime["rnn_size"]
    action_labels = _flappy_action_labels()
    frame_stack = int(getattr(cfg, "frame_stack", 1) or 1)
    decision_period_ms = _decision_period_ms_from_cfg("flappy", cfg)
    base_seed = int(getattr(cfg, "seed", 0) or 0)
    specs = _build_episode_specs(
        episode_count=int(episode_count),
        val_indices=val_indices,
        global_episode_start=int(global_episode_start),
        base_seed=base_seed,
        latency=int(latency),
    )
    slot_count = int(getattr(env, "num_agents", int(num_envs)))
    train_rows: list[dict[str, Any]] = []
    val_rows: list[dict[str, Any]] = []
    active_specs: list[EpisodeSpec | None] = [None for _slot in range(slot_count)]
    active_obs: list[Any | None] = [None for _slot in range(slot_count)]
    active_t: list[int] = [0 for _slot in range(slot_count)]
    rnn_states = torch.zeros([slot_count, rnn_size], dtype=torch.float32, device=device)
    pending_futures: set[Future[str]] = set()
    executor = (
        ThreadPoolExecutor(max_workers=int(image_writer_workers))
        if int(image_writer_workers) > 0
        else None
    )

    def assign_specs(start_index: int, slots: Sequence[int]) -> int:
        nonlocal active_obs
        next_specs, next_index = _take_next_episode_specs(
            specs=specs,
            start_index=start_index,
            slot_count=len(slots),
        )
        for slot in slots:
            active_specs[int(slot)] = None
            active_obs[int(slot)] = None
            active_t[int(slot)] = 0
            rnn_states[int(slot)] = torch.zeros_like(rnn_states[int(slot)])
        reset_plan = {
            int(slot): int(spec.seed)
            for slot, spec in zip(slots, next_specs)
        }
        if reset_plan:
            active_mask = [spec is not None for spec in active_specs]
            for slot in reset_plan:
                active_mask[int(slot)] = True
            obs_batch, _infos = env.reset_agents(reset_plan, active_mask=active_mask)
            for slot, spec in zip(slots, next_specs):
                active_specs[int(slot)] = spec
                active_obs[int(slot)] = _select_batch_item(obs_batch, int(slot))
                active_t[int(slot)] = 0
                rnn_states[int(slot)] = torch.zeros_like(rnn_states[int(slot)])
        return next_index

    try:
        if hasattr(env, "set_auto_reset_done"):
            env.set_auto_reset_done(False)
        env.reset(seed=base_seed)
        next_spec_index = assign_specs(0, list(range(slot_count)))
        completed_episodes = 0
        while completed_episodes < int(episode_count):
            active_slots = [idx for idx, spec in enumerate(active_specs) if spec is not None]
            if not active_slots:
                break
            if hasattr(env, "set_active_agents"):
                env.set_active_agents([idx in set(active_slots) for idx in range(slot_count)])

            obs_batch = _batch_observations([active_obs[idx] for idx in active_slots])
            rnn_batch = rnn_states[active_slots]
            normalized_obs = prepare_and_normalize_obs(actor_critic, obs_batch)
            with torch.no_grad():
                policy_outputs = actor_critic(normalized_obs, rnn_batch)
                actions = policy_outputs["actions"]
                if deterministic:
                    actions = argmax_actions(actor_critic.action_distribution())
                if actions.ndim == 1:
                    actions = unsqueeze_tensor(actions, dim=-1)
                policy_actions = actions
                processed_actions = preprocess_actions(env_info, actions)
                new_rnn_states = policy_outputs["new_rnn_states"]

            env_actions = np.zeros(slot_count, dtype=np.int64)
            for batch_idx, slot in enumerate(active_slots):
                env_actions[int(slot)] = _extract_action_id(_select_batch_item(processed_actions, batch_idx))

            saved_images: dict[int, str] = {}
            for slot in active_slots:
                spec = active_specs[slot]
                obs = active_obs[slot]
                if spec is None or obs is None:
                    raise RuntimeError(f"Active Flappy slot {slot} has no episode state")
                t = int(active_t[slot])
                image_path = output_dir / "images" / spec.split / f"episode_{spec.episode_idx:06d}" / f"t_{t:06d}.png"
                frame = _frame_for_image_source(
                    obs=obs,
                    env=env,
                    env_name="flappy",
                    frame_stack=frame_stack,
                    image_source=image_source,
                    agent_idx=int(slot),
                )
                saved_image, pending_futures = _submit_image_write(
                    executor,
                    pending_futures,
                    frame,
                    image_path,
                    export_image_size,
                    False,
                    int(row_buffer_size),
                )
                saved_images[int(slot)] = saved_image

            next_obs, reward, terminated, truncated, info = env.step(env_actions)
            dones = _make_dones(terminated, truncated)
            replacement_slots: list[int] = []
            for batch_idx, slot in enumerate(active_slots):
                spec = active_specs[slot]
                obs = active_obs[slot]
                if spec is None or obs is None:
                    raise RuntimeError(f"Active Flappy slot {slot} has no episode state")
                split_rows = val_rows if spec.split == "val" else train_rows
                t = int(active_t[slot])
                saved_image = saved_images[int(slot)]
                slot_policy_action = _select_batch_item(policy_actions, batch_idx)
                action_id = _policy_action_id("flappy", slot_policy_action)
                info_single = dict(info[int(slot)] or {}) if isinstance(info, list) else _extract_single_info(_select_batch_item(info, int(slot)))
                actual_latency = _latency_from_step_info(info_single, fallback_latency=int(latency))
                actual_latency_ms = float(actual_latency) * decision_period_ms
                reward_value = _reward_from_step(
                    _batched_reward_item(reward, int(slot)),
                    info_single,
                    prefer_raw_reward=False,
                )
                split_rows.append(
                    _build_export_row(
                        image_path=saved_image,
                        split=spec.split,
                        prompt_text=_prompt_for_episode(
                            dataset_mode=dataset_mode,
                            prompt=prompt,
                            mixed_prompt_template=mixed_prompt_template,
                            latency=actual_latency,
                            latency_ms=actual_latency_ms,
                        ),
                        action_id=action_id,
                        action_text=action_labels.get(action_id, str(action_id)),
                        latency=int(actual_latency),
                        latency_ms=float(actual_latency_ms),
                        reward_value=float(reward_value),
                        done=bool(_to_python_scalar(_select_batch_item(dones, int(slot)))),
                        episode_idx=int(spec.episode_idx),
                        t=t,
                        seed=int(spec.seed),
                        deterministic=bool(deterministic),
                    )
                )
                rnn_states[slot] = new_rnn_states[batch_idx]
                next_t = t + 1
                episode_finished = bool(_to_python_scalar(_select_batch_item(dones, int(slot)))) or next_t >= int(max_steps_per_episode)
                if episode_finished:
                    completed_episodes += 1
                    replacement_slots.append(int(slot))
                else:
                    active_obs[slot] = _select_batch_item(next_obs, int(slot))
                    active_t[slot] = next_t

            if replacement_slots:
                next_spec_index = assign_specs(next_spec_index, replacement_slots)
    finally:
        try:
            _wait_for_image_writes(pending_futures)
        finally:
            if executor is not None:
                executor.shutdown(wait=True)
            env.close()

    return train_rows, val_rows, {"latency": int(latency), "action_labels": action_labels}


def _collect_latency_dataset(
    *,
    env_name: str,
    module,
    base_cfg,
    latency: int,
    episode_count: int,
    val_indices: set[int],
    max_steps_per_episode: int,
    prompt: str,
    mixed_prompt_template: str,
    dataset_mode: str,
    output_dir: Path,
    global_episode_start: int,
    deterministic: bool,
    export_image_size: tuple[int, int] | None,
    jitter_spec: JitterLatencySpec | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    import torch

    cfg = _prepare_export_cfg(module, env_name, base_cfg, latency, jitter_spec)
    runtime = _build_env_and_policy(module, cfg, env_name, 1)
    env = runtime["env"]
    env_info = runtime["env_info"]
    actor_critic = runtime["actor_critic"]
    device = runtime["device"]
    preprocess_actions = runtime["preprocess_actions"]
    argmax_actions = runtime["argmax_actions"]
    prepare_and_normalize_obs = runtime["prepare_and_normalize_obs"]
    unsqueeze_tensor = runtime["unsqueeze_tensor"]
    rnn_size = runtime["rnn_size"]
    action_labels = _action_labels_for_env(env_name, cfg, env)
    prefer_raw_reward = bool(getattr(cfg, "eval_raw_reward", False))
    frame_stack = int(getattr(cfg, "frame_stack", 1) or 1)
    decision_period_ms = _decision_period_ms_from_cfg(env_name, cfg)

    train_rows: list[dict[str, Any]] = []
    val_rows: list[dict[str, Any]] = []
    for episode_local_idx in range(episode_count):
        episode_idx = global_episode_start + episode_local_idx
        split = "val" if episode_local_idx in val_indices else "train"
        split_rows = val_rows if split == "val" else train_rows
        episode_seed = int(getattr(cfg, "seed", 0) or 0) + int(latency) * 100_000 + episode_local_idx

        obs, _ = env.reset(seed=episode_seed)
        rnn_states = torch.zeros([env.num_agents, rnn_size], dtype=torch.float32, device=device)

        for t in range(max_steps_per_episode):
            image_path = output_dir / "images" / split / f"episode_{episode_idx:06d}" / f"t_{t:06d}.png"
            saved_image = _save_observation_image(
                obs,
                env=env,
                env_name=env_name,
                frame_stack=frame_stack,
                image_path=image_path,
                export_image_size=export_image_size,
            )

            normalized_obs = prepare_and_normalize_obs(actor_critic, obs)
            policy_outputs = actor_critic(normalized_obs, rnn_states)
            actions = policy_outputs["actions"]
            if deterministic:
                actions = argmax_actions(actor_critic.action_distribution())
            if actions.ndim == 1:
                actions = unsqueeze_tensor(actions, dim=-1)
            policy_actions = actions
            actions = preprocess_actions(env_info, actions)
            rnn_states = policy_outputs["new_rnn_states"]

            action_id = _policy_action_id(env_name, policy_actions[0])
            action_text = action_labels.get(action_id, str(action_id))
            next_obs, reward, terminated, truncated, info = env.step(actions)
            dones = _make_dones(terminated, truncated)
            done = bool(_to_python_scalar(dones[0]))
            info_single = _extract_single_info(info)
            actual_latency = _latency_from_step_info(info_single, fallback_latency=int(latency))
            actual_latency_ms = float(actual_latency) * decision_period_ms
            prompt_text = _prompt_for_episode(
                dataset_mode=dataset_mode,
                prompt=prompt,
                mixed_prompt_template=mixed_prompt_template,
                latency=actual_latency,
                latency_ms=actual_latency_ms,
            )
            reward_value = _reward_from_step(
                reward[0] if isinstance(reward, (list, tuple, np.ndarray)) else reward,
                info_single,
                prefer_raw_reward=prefer_raw_reward,
            )

            split_rows.append(
                _build_export_row(
                    image_path=saved_image,
                    split=split,
                    prompt_text=prompt_text,
                    action_id=action_id,
                    action_text=action_text,
                    latency=int(actual_latency),
                    latency_ms=float(actual_latency_ms),
                    reward_value=float(reward_value),
                    done=bool(done),
                    episode_idx=int(episode_idx),
                    t=int(t),
                    seed=int(episode_seed),
                    deterministic=bool(deterministic),
                )
            )

            obs = next_obs
            if done:
                break

    env.close()
    return train_rows, val_rows, {"latency": int(latency), "action_labels": action_labels}


def _build_plan(args) -> list[tuple[int, int, int]]:
    if args.dataset_mode in {"zero_latency", "fixed_latency", "jitter_latency"}:
        val_count = int(round(args.max_episodes * args.val_fraction))
        val_count = max(0, min(int(args.max_episodes), val_count))
        latency = int(args.fixed_latency) if args.dataset_mode == "fixed_latency" else 0
        return [(latency, int(args.max_episodes), val_count)]

    latencies = list(range(int(args.latency_start), int(args.latency_end) + 1))
    return [(latency, int(args.episodes_per_latency), int(args.val_episodes_per_latency)) for latency in latencies]


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Sample Factory teacher trajectories as parquet datasets for SFT.")
    parser.add_argument("--config", default="", help="Optional Sample Factory training YAML used only to disambiguate env.name if needed.")
    parser.add_argument("--env-name", default="", choices=["", "deadly_corridor", "demon_attack", "flappy"], help="Optional environment name. Use this instead of --config when checkpoint root contains multiple environments.")
    parser.add_argument("--checkpoint-root", required=True, help="Directory containing latency-specific experiment folders.")
    parser.add_argument("--output-dir", required=True, help="Directory where parquet files and images will be written.")
    parser.add_argument("--export-image-size", type=str, default="", help="Optional saved image size in 'W,H' format, e.g. '512,288'. This does not affect teacher inference.")
    parser.add_argument("--dataset-mode", required=True, choices=["zero_latency", "fixed_latency", "mixed_latency", "jitter_latency"])
    parser.add_argument("--prompt", required=True, help="Base prompt used for the zero-latency dataset.")
    parser.add_argument(
        "--mixed-prompt-template",
        default="{prompt} Your actions take {latency_ms} milliseconds to take effect. Choose the best next action while accounting for the delay.",
        help="Template for mixed-latency prompts. Supported fields: {prompt}, {latency}, {latency_ms}.",
    )
    parser.add_argument("--max-steps-per-episode", type=int, required=True)
    parser.add_argument("--device", type=str, default="", help="Optional override for Sample Factory device, e.g. cpu or gpu.")
    parser.add_argument("--deterministic", action="store_true", default=True)
    parser.add_argument("--collector-mode", type=str, default="batched", choices=["serial", "batched"], help="Rollout collector implementation.")
    parser.add_argument("--num-envs", type=int, default=1, help="Number of independent environments to roll out concurrently per latency.")
    parser.add_argument("--image-writer-workers", type=int, default=0, help="Background PNG writer threads. Use 0 for synchronous writes.")
    parser.add_argument("--row-buffer-size", type=int, default=8192, help="Maximum pending background image writes.")
    parser.add_argument("--image-source", type=str, default="render", choices=["render", "observation"], help="Source for exported images. 'observation' avoids render() and is fastest.")
    parser.add_argument("--checkpoint-seed", type=int, default=None, help="Seed encoded in checkpoint folder names. Defaults to experiment.seed from the config.")
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--push-to-hub", action="store_true", help="Upload the resulting dataset to Hugging Face after writing local parquet files.")
    parser.add_argument("--hf-repo-id", type=str, default="", help="Dataset repo id, e.g. username/dataset_name.")
    parser.add_argument("--hf-private", action="store_true", help="Create or update the HF dataset repo as private.")
    parser.add_argument("--hf-token", type=str, default="", help="Optional HF token. If omitted, the script will use HF_TOKEN or HUGGINGFACE_HUB_TOKEN from the environment.")
    parser.add_argument("--hf-max-shard-size", type=str, default="500MB", help="Shard size forwarded to datasets.push_to_hub.")

    parser.add_argument("--max-episodes", type=int, default=0, help="Zero/fixed/jitter-latency modes only.")
    parser.add_argument("--val-fraction", type=float, default=0.1, help="Zero/fixed/jitter-latency modes only.")
    parser.add_argument("--fixed-latency", type=int, default=0, help="Fixed-latency mode only.")

    parser.add_argument("--jitter-latency-distribution", type=str, default="normal", choices=["normal", "uniform"], help="Jitter-latency mode only.")
    parser.add_argument("--jitter-latency-mean", type=float, default=None, help="Normal jitter-latency mode only.")
    parser.add_argument("--jitter-latency-std", type=float, default=None, help="Normal jitter-latency mode only.")
    parser.add_argument("--jitter-latency-min", type=int, default=0, help="Jitter-latency mode only.")
    parser.add_argument("--jitter-latency-max", type=int, default=None, help="Jitter-latency mode only.")

    parser.add_argument("--latency-start", type=int, default=0, help="Mixed-latency mode only.")
    parser.add_argument("--latency-end", type=int, default=0, help="Mixed-latency mode only.")
    parser.add_argument("--episodes-per-latency", type=int, default=0, help="Mixed-latency mode only.")
    parser.add_argument("--val-episodes-per-latency", type=int, default=0, help="Mixed-latency mode only.")

    args = parser.parse_args()

    checkpoint_root = Path(args.checkpoint_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    export_image_size = _parse_image_size(args.export_image_size)
    env_name_hint = str(args.env_name).strip().lower() or None
    if args.config:
        hint_config = load_training_config(args.config)
        config_env_name = str(hint_config["env"]["name"]).strip().lower()
        if env_name_hint is None:
            env_name_hint = config_env_name
        elif env_name_hint != config_env_name:
            raise ValueError(
                f"--env-name={env_name_hint} conflicts with env.name={config_env_name} in {args.config}"
            )
        default_seed = int(hint_config["experiment"].get("seed", 0))
    else:
        hint_config = None
        default_seed = 0
    checkpoint_seed = int(args.checkpoint_seed if args.checkpoint_seed is not None else default_seed)

    if args.dataset_mode in {"zero_latency", "fixed_latency", "jitter_latency"} and args.max_episodes <= 0:
        raise ValueError("--max-episodes must be positive for fixed-latency or jitter-latency export")
    if args.dataset_mode == "fixed_latency" and args.fixed_latency < 0:
        raise ValueError("--fixed-latency must be non-negative")
    if args.dataset_mode == "jitter_latency":
        jitter_distribution = str(args.jitter_latency_distribution)
        if jitter_distribution == "normal" and args.jitter_latency_mean is None:
            raise ValueError("--jitter-latency-mean is required for normal jitter-latency export")
        if jitter_distribution == "normal" and args.jitter_latency_std is None:
            raise ValueError("--jitter-latency-std is required for normal jitter-latency export")
        if args.jitter_latency_max is None:
            raise ValueError("--jitter-latency-max is required for jitter-latency export")
        if jitter_distribution == "normal" and float(args.jitter_latency_std) < 0.0:
            raise ValueError("--jitter-latency-std must be non-negative")
        if int(args.jitter_latency_min) < 0:
            raise ValueError("--jitter-latency-min must be non-negative")
        if int(args.jitter_latency_max) < int(args.jitter_latency_min):
            raise ValueError("--jitter-latency-max must be >= --jitter-latency-min")
    if args.dataset_mode == "mixed_latency":
        if args.episodes_per_latency <= 0:
            raise ValueError("--episodes-per-latency must be positive for mixed-latency export")
        if args.latency_end < args.latency_start:
            raise ValueError("--latency-end must be >= --latency-start")
    if args.num_envs <= 0:
        raise ValueError("--num-envs must be positive")
    if args.image_writer_workers < 0:
        raise ValueError("--image-writer-workers must be non-negative")
    if args.row_buffer_size <= 0:
        raise ValueError("--row-buffer-size must be positive")

    jitter_spec = None
    if args.dataset_mode == "jitter_latency":
        jitter_distribution = str(args.jitter_latency_distribution)
        jitter_spec = JitterLatencySpec(
            distribution=jitter_distribution,
            mean=float(args.jitter_latency_mean) if jitter_distribution == "normal" else None,
            std=float(args.jitter_latency_std) if jitter_distribution == "normal" else None,
            min_latency=int(args.jitter_latency_min),
            max_latency=int(args.jitter_latency_max),
        )

    train_rows: list[dict[str, Any]] = []
    val_rows: list[dict[str, Any]] = []
    metadata: dict[str, Any] = {
        "env_name": None,
        "dataset_mode": args.dataset_mode,
        "prompt": args.prompt,
        "mixed_prompt_template": args.mixed_prompt_template,
        "max_steps_per_episode": int(args.max_steps_per_episode),
        "deterministic": bool(args.deterministic),
        "checkpoint_root": str(checkpoint_root),
        "export_image_size": list(export_image_size) if export_image_size is not None else None,
        "collector_mode": str(args.collector_mode),
        "num_envs": int(args.num_envs),
        "image_writer_workers": int(args.image_writer_workers),
        "row_buffer_size": int(args.row_buffer_size),
        "image_source": str(args.image_source),
        "decision_period_ms": None,
        "jitter_latency": None if jitter_spec is None else {
            "distribution": str(jitter_spec.distribution),
            "mean": None if jitter_spec.mean is None else float(jitter_spec.mean),
            "std": None if jitter_spec.std is None else float(jitter_spec.std),
            "min": int(jitter_spec.min_latency),
            "max": int(jitter_spec.max_latency),
        },
        "latencies": [],
        "action_labels": {},
    }

    episode_offset = 0
    plan = _build_plan(args)
    for latency, episode_count, val_count in plan:
        val_indices = _select_val_indices(episode_count, val_count, split_seed=args.split_seed + latency)
        env_name = env_name_hint
        if env_name is None:
            matches: list[tuple[str, Path]] = []
            for candidate in ("deadly_corridor", "demon_attack", "flappy"):
                try:
                    if jitter_spec is None:
                        experiment_candidate = _find_experiment_dir(checkpoint_root, candidate, latency, checkpoint_seed)
                    else:
                        experiment_candidate = _find_jitter_experiment_dir(
                            checkpoint_root=checkpoint_root,
                            env_name=candidate,
                            jitter_spec=jitter_spec,
                            seed=checkpoint_seed,
                        )
                    matches.append((candidate, experiment_candidate))
                except FileNotFoundError:
                    continue
            if len(matches) != 1:
                raise ValueError(
                    f"Could not uniquely infer env for latency {latency} under {checkpoint_root}. "
                    "Pass --config with the correct env.name."
                )
            env_name, experiment_dir = matches[0]
        else:
            if jitter_spec is None:
                experiment_dir = _find_experiment_dir(checkpoint_root, env_name, latency, checkpoint_seed)
            else:
                experiment_dir = _find_jitter_experiment_dir(
                    checkpoint_root=checkpoint_root,
                    env_name=env_name,
                    jitter_spec=jitter_spec,
                    seed=checkpoint_seed,
                )

        module = import_integration_module({"env": {"name": env_name}})
        _register_components(module, env_name)
        base_cfg = _minimal_export_cfg(
            module,
            experiment_dir=experiment_dir,
            device_override=args.device or None,
        )
        if metadata["env_name"] is None:
            metadata["env_name"] = env_name
        if metadata["decision_period_ms"] is None:
            prepared_cfg = _prepare_export_cfg(module, env_name, base_cfg, latency, jitter_spec)
            metadata["decision_period_ms"] = float(_decision_period_ms_from_cfg(env_name, prepared_cfg))
        if args.collector_mode == "batched":
            latency_train_rows, latency_val_rows, latency_meta = _collect_latency_dataset_batched(
                env_name=env_name,
                module=module,
                base_cfg=base_cfg,
                latency=latency,
                episode_count=episode_count,
                val_indices=val_indices,
                max_steps_per_episode=int(args.max_steps_per_episode),
                prompt=args.prompt,
                mixed_prompt_template=args.mixed_prompt_template,
                dataset_mode=args.dataset_mode,
                output_dir=output_dir,
                global_episode_start=episode_offset,
                deterministic=bool(args.deterministic),
                export_image_size=export_image_size,
                num_envs=int(args.num_envs),
                image_writer_workers=int(args.image_writer_workers),
                row_buffer_size=int(args.row_buffer_size),
                image_source=str(args.image_source),
                jitter_spec=jitter_spec,
            )
        else:
            latency_train_rows, latency_val_rows, latency_meta = _collect_latency_dataset(
                env_name=env_name,
                module=module,
                base_cfg=base_cfg,
                latency=latency,
                episode_count=episode_count,
                val_indices=val_indices,
                max_steps_per_episode=int(args.max_steps_per_episode),
                prompt=args.prompt,
                mixed_prompt_template=args.mixed_prompt_template,
                dataset_mode=args.dataset_mode,
                output_dir=output_dir,
                global_episode_start=episode_offset,
                deterministic=bool(args.deterministic),
                export_image_size=export_image_size,
                jitter_spec=jitter_spec,
            )
        train_rows.extend(latency_train_rows)
        val_rows.extend(latency_val_rows)
        latency_meta_key = str(latency)
        latency_metadata: dict[str, Any] = {
            "episode_count": int(episode_count),
            "val_episode_count": int(val_count),
            "experiment_dir": str(experiment_dir),
            "best_checkpoint": _find_best_checkpoint_name(experiment_dir),
        }
        if jitter_spec is None:
            latency_metadata["latency"] = int(latency)
            latency_metadata["latency_ms"] = float(latency) * float(metadata["decision_period_ms"])
        else:
            latency_meta_key = _jitter_latency_label(jitter_spec)
            latency_metadata["latency_distribution"] = str(jitter_spec.distribution)
            if jitter_spec.mean is not None:
                latency_metadata["latency_mean"] = float(jitter_spec.mean)
            if jitter_spec.std is not None:
                latency_metadata["latency_std"] = float(jitter_spec.std)
            latency_metadata["latency_min"] = int(jitter_spec.min_latency)
            latency_metadata["latency_max"] = int(jitter_spec.max_latency)
        metadata["latencies"].append(latency_metadata)
        metadata["action_labels"][latency_meta_key] = latency_meta["action_labels"]
        episode_offset += episode_count

    dataset_dict = _write_parquet_splits(output_dir, train_rows, val_rows)
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")

    if args.push_to_hub:
        repo_id = str(args.hf_repo_id).strip()
        if not repo_id:
            raise ValueError("--hf-repo-id is required when --push-to-hub is enabled")
        hf_token = str(args.hf_token).strip() or None
        if hf_token is None:
            import os

            hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        _push_dataset_to_hub(
            dataset_dict,
            repo_id=repo_id,
            private=bool(args.hf_private),
            token=hf_token,
            max_shard_size=args.hf_max_shard_size,
        )

    print(
        f"[export] wrote train={len(train_rows)} rows val={len(val_rows)} rows "
        f"to {output_dir}"
    )
    if args.push_to_hub:
        print(f"[export] pushed dataset to Hugging Face repo {args.hf_repo_id}")


if __name__ == "__main__":
    main()
