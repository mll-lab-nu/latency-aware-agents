"""Export Sample Factory teacher rollouts as parquet datasets.

This exporter is for offline teacher data collection. It loads a Sample
Factory checkpoint, reconstructs the training environment, rolls out the
teacher policy, saves policy observations as images, and writes
``train.parquet``, ``val.parquet``, and ``metadata.json``.

Default latency behavior:
    The rollout uses the latency configuration stored in the checkpoint
    ``config.json``. To intentionally collect data under another latency
    distribution, pass ``--latency-config`` with a YAML file containing a
    training-compatible ``latency:`` section.

Minimal local smoke test:
    python scripts/rollout_data/export_sf_teacher_dataset.py \
      --checkpoint-root results/checkpoints/flappy_sf_frame_stack_4_seed0 \
      --output-dir /tmp/flappy_teacher_export_smoke \
      --max-episodes 1 \
      --max-steps-per-episode 8 \
      --val-fraction 0.0 \
      --num-envs 1 \
      --device cpu

Direct export and upload:
    export HF_TOKEN="your-token"
    python scripts/rollout_data/export_sf_teacher_dataset.py \
      --checkpoint-root results/checkpoints/flappy_sf_frame_stack_4_seed0 \
      --output-dir /tmp/flappy_teacher_export_smoke \
      --max-episodes 1 \
      --max-steps-per-episode 8 \
      --val-fraction 0.0 \
      --num-envs 1 \
      --device cpu \
      --push-to-hub \
      --hf-repo-id your-hf-name/flappy_teacher_export_smoke \
      --hf-token "$HF_TOKEN" \
      --hf-private

Upload an existing export:
    python scripts/rollout_data/push_sf_teacher_dataset.py \
      --dataset /tmp/flappy_teacher_export_smoke your-hf-name/flappy_teacher_export_smoke \
      --upload-mode direct \
      --hf-token "$HF_TOKEN" \
      --hf-private

Dataset row prompts are rendered from scripts/rollout_data/prompts.py.
"""

from __future__ import annotations

import torch
import argparse
import contextlib
import logging
import yaml
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
import json
import os
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image
import datasets
from datasets import Dataset, DatasetDict, Features, Image as HFImage, Sequence as HFSequence, Value
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.rollout_data.prompts import BASE_PROMPTS, TIMING_SUFFIX, render_teacher_prompt
from latency_bench.policy.sample_factory_common import (
    _load_sf_policy_state_dict,
    _normalize_actions_for_space,
    _resolve_device,
    _split_batch_actions,
    _to_python_action,
)
from training.common.config_adapter import import_integration_module, load_training_config
from training.common.latency_config import raw_frames_to_latency_ms, sample_factory_latency_params_from_flat_config
from training.common.utils import maybe_set_cli_override


SF_ENV_TO_BENCH_ENV = {
    "latency_flappy": "flappy",
    "latency_demon_attack": "demon_attack",
}


@dataclass(frozen=True)
class CheckpointRoot:
    experiment_dir: Path
    checkpoint_path: Path | None


def _infer_env_name_from_sf_env(sf_env_name: str) -> str:
    normalized = str(sf_env_name).strip()
    if normalized in SF_ENV_TO_BENCH_ENV:
        return SF_ENV_TO_BENCH_ENV[normalized]
    if normalized.startswith("doom_"):
        return "deadly_corridor"
    for env_name in ("flappy", "demon_attack", "deadly_corridor"):
        if normalized == env_name or normalized.startswith(f"{env_name}_"):
            return env_name
    raise ValueError(f"Could not infer benchmark env name from Sample Factory env {sf_env_name!r}")


def _resolve_checkpoint_root(checkpoint_root: str | Path) -> CheckpointRoot:
    path = Path(checkpoint_root).expanduser().resolve()
    if path.is_file():
        if path.parent.name.startswith("checkpoint_p"):
            return CheckpointRoot(experiment_dir=path.parent.parent, checkpoint_path=path)
        return CheckpointRoot(experiment_dir=path.parent, checkpoint_path=path)
    return CheckpointRoot(experiment_dir=path, checkpoint_path=None)


def _read_checkpoint_config(experiment_dir: Path) -> dict[str, Any]:
    config_path = experiment_dir / "config.json"
    with config_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _get_present(flat_cfg: Mapping[str, Any], keys: Sequence[str]) -> dict[str, Any]:
    return {key: flat_cfg[key] for key in keys if key in flat_cfg and flat_cfg[key] is not None}


def _latency_from_flat_checkpoint_config(flat_cfg: Mapping[str, Any]) -> dict[str, Any]:
    latency_cfg: dict[str, Any] = {"type": flat_cfg.get("latency_type", "zero")}
    latency_cfg.update(
        _get_present(
            flat_cfg,
            (
                "fixed_latency_ms",
                "mean_latency_ms",
                "std_latency_ms",
                "min_latency_ms",
                "max_latency_ms",
                "max_pending_actions",
            ),
        )
    )
    if flat_cfg.get("latency_seed") is not None:
        latency_cfg["seed"] = flat_cfg["latency_seed"]
    if "add_latency_info" in flat_cfg:
        latency_cfg["add_latency_info"] = bool(flat_cfg["add_latency_info"])
    return latency_cfg


def _env_from_flat_checkpoint_config(env_name: str, flat_cfg: Mapping[str, Any]) -> dict[str, Any]:
    common = {
        "name": env_name,
        **_get_present(flat_cfg, ("env_fps", "obs_fps", "frame_stack", "max_episode_steps")),
    }
    if env_name == "flappy":
        env_cfg = {
            **common,
            **_get_present(
                flat_cfg,
                (
                    "gym_id",
                    "use_lidar",
                    "normalize_obs",
                    "audio_on",
                    "obs_resize",
                    "use_gpu_render",
                    "gpu_render_device",
                    "gpu_render_batch_size",
                    "gpu_render_profile",
                    "gpu_render_profile_interval",
                    "pipe_gap",
                    "bird_color",
                    "pipe_color",
                    "background",
                    "score_limit",
                    "debug",
                    "debug_timelimit_diagnostics",
                ),
            ),
        }
        screen_size = flat_cfg.get("screen_size")
        if screen_size not in (None, ""):
            env_cfg["screen_size"] = screen_size
        return env_cfg
    if env_name == "demon_attack":
        return {
            **common,
            **_get_present(
                flat_cfg,
                (
                    "env_id",
                    "screen_size",
                    "noop_max",
                    "terminal_on_life_loss",
                    "clip_reward",
                    "repeat_action_probability",
                    "full_action_space",
                ),
            ),
        }
    return {
        **common,
        "env_id": flat_cfg.get("env", flat_cfg.get("env_id", "doom_deadly_corridor")),
        **_get_present(
            flat_cfg,
            (
                "res_w",
                "res_h",
                "wide_aspect_ratio",
                "vizdoom_init_max_parallel",
                "vizdoom_init_lock_timeout_sec",
                "vizdoom_init_lock_max_wait_sec",
                "vizdoom_init_retry_jitter_sec",
            ),
        ),
    }


def _scheduler_from_flat_checkpoint_config(env_name: str, flat_cfg: Mapping[str, Any]) -> dict[str, Any]:
    default_hold_policy = "one_frame_then_noop" if env_name == "flappy" else "hold"
    return {
        "hold_policy": flat_cfg.get("hold_policy", default_hold_policy),
        "ordering_policy": flat_cfg.get("ordering_policy", "latest_ready"),
    }


def _training_from_flat_checkpoint_config(flat_cfg: Mapping[str, Any]) -> dict[str, Any]:
    return _get_present(
        flat_cfg,
        (
            "train_for_env_steps",
            "num_workers",
            "num_envs_per_worker",
            "num_policies",
            "batch_size",
            "rollout",
            "recurrence",
            "num_epochs",
            "num_batches_per_epoch",
            "worker_num_splits",
            "num_batches_to_accumulate",
            "learning_rate",
            "gamma",
            "gae_lambda",
            "ppo_clip_ratio",
            "ppo_clip_value",
            "exploration_loss",
            "exploration_loss_coeff",
            "value_loss_coeff",
            "max_grad_norm",
            "rnn_type",
            "rnn_size",
            "async_rl",
            "batched_sampling",
            "use_rnn",
            "normalize_returns",
            "normalize_input",
            "save_every_sec",
            "keep_checkpoints",
            "save_milestones_sec",
            "save_best_every_sec",
            "save_best_after",
            "stats_avg",
            "experiment_summaries_interval",
        ),
    )


def _structured_config_from_checkpoint_config(
    flat_cfg: Mapping[str, Any],
    *,
    experiment_dir: Path,
    checkpoint_path: Path | None,
) -> dict[str, Any]:
    env_name = _infer_env_name_from_sf_env(str(flat_cfg.get("env", "")))
    device = _resolve_device(flat_cfg.get("device", "cpu"))
    return {
        "experiment": {
            "name": experiment_dir.name,
            "seed": flat_cfg.get("seed", 0),
        },
        "backend": {
            "type": "sample_factory",
            "algo": flat_cfg.get("algo", "APPO"),
            "device": device,
            "train_dir": str(experiment_dir.parent),
            "restart_behavior": flat_cfg.get("restart_behavior", "resume"),
            "run_mode": "eval",
            "extra_args": ["--load_checkpoint_kind", flat_cfg.get("load_checkpoint_kind", "best")],
        },
        "executor": {"mode": "simulated"},
        "env": _env_from_flat_checkpoint_config(env_name, flat_cfg),
        "latency": _latency_from_flat_checkpoint_config(flat_cfg),
        "scheduler": _scheduler_from_flat_checkpoint_config(env_name, flat_cfg),
        "policy": {
            "type": f"{env_name}_sf",
            "checkpoint_path": None if checkpoint_path is None else str(checkpoint_path),
            "deterministic": True,
            "device": device,
        },
        "training": _training_from_flat_checkpoint_config(flat_cfg),
        "evaluation": {
            "eval_episodes": flat_cfg.get("eval_episodes", 1),
            "eval_parallel_envs": flat_cfg.get("eval_parallel_envs", 1),
            "eval_max_steps": flat_cfg.get("eval_max_steps", 0),
            "eval_deterministic": flat_cfg.get("eval_deterministic", True),
            "eval_raw_reward": flat_cfg.get("eval_raw_reward", False),
        },
        "logging": {
            "output_dir": flat_cfg.get("output_dir", ""),
        },
    }


def _load_latency_override(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    with resolved.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if "latency" in payload:
        return deepcopy(payload["latency"])
    return deepcopy(payload)


def _rollout_config_from_user_config(
    config_path: str | Path,
    *,
    checkpoint: CheckpointRoot,
) -> dict[str, Any]:
    config = load_training_config(config_path)
    updated = deepcopy(config)
    updated.setdefault("backend", {})
    updated.setdefault("experiment", {})
    updated.setdefault("policy", {})
    updated["backend"]["train_dir"] = str(checkpoint.experiment_dir.parent)
    updated["backend"]["run_mode"] = "eval"
    updated["experiment"]["name"] = checkpoint.experiment_dir.name
    updated["policy"]["checkpoint_path"] = None if checkpoint.checkpoint_path is None else str(checkpoint.checkpoint_path)
    return updated


def _force_rollout_latency_info(config: dict[str, Any]) -> None:
    config.setdefault("latency", {})["add_latency_info"] = True


def load_rollout_training_config(
    checkpoint_root: str | Path,
    *,
    config_path: str | Path | None = None,
    latency_config_path: str | Path | None = None,
    device_override: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    checkpoint = _resolve_checkpoint_root(checkpoint_root)
    if config_path is None:
        flat_cfg = _read_checkpoint_config(checkpoint.experiment_dir)
        config = _structured_config_from_checkpoint_config(
            flat_cfg,
            experiment_dir=checkpoint.experiment_dir,
            checkpoint_path=checkpoint.checkpoint_path,
        )
        config_source = str((checkpoint.experiment_dir / "config.json").resolve())
        latency_source = "checkpoint"
    else:
        config = _rollout_config_from_user_config(config_path, checkpoint=checkpoint)
        config_source = str(Path(config_path).expanduser().resolve())
        latency_source = config_source

    if latency_config_path is not None:
        latency_override_path = Path(latency_config_path).expanduser().resolve()
        config["latency"] = _load_latency_override(latency_override_path)
        latency_source = str(latency_override_path)

    if device_override:
        normalized_device = _resolve_device(device_override)
        config.setdefault("backend", {})["device"] = normalized_device
        config.setdefault("policy", {})["device"] = normalized_device
        if normalized_device == "cpu" and config.get("env", {}).get("name") == "flappy":
            config.setdefault("env", {})["gpu_render_device"] = "cpu"
    _force_rollout_latency_info(config)
    return config, {
        "checkpoint_experiment_dir": str(checkpoint.experiment_dir),
        "checkpoint_path": None if checkpoint.checkpoint_path is None else str(checkpoint.checkpoint_path),
        "config_source": config_source,
        "latency_source": latency_source,
        "latency_config": deepcopy(config.get("latency", {})),
    }


def _register_components(module: Any, env_name: str) -> None:
    if env_name == "deadly_corridor":
        module.register_deadly_corridor_components()
    elif env_name == "demon_attack":
        module.register_demon_attack_envs()
    else:
        module.register_flappy_envs()


def _apply_latency_to_flat_cfg(cfg: Any, latency_cfg: Mapping[str, Any]) -> None:
    field_map = {
        "type": "latency_type",
        "fixed_latency_ms": "fixed_latency_ms",
        "mean_latency_ms": "mean_latency_ms",
        "std_latency_ms": "std_latency_ms",
        "min_latency_ms": "min_latency_ms",
        "max_latency_ms": "max_latency_ms",
        "seed": "latency_seed",
        "add_latency_info": "add_latency_info",
        "max_pending_actions": "max_pending_actions",
    }
    for source_key, target_key in field_map.items():
        if source_key in latency_cfg:
            maybe_set_cli_override(cfg, target_key, latency_cfg[source_key])
    maybe_set_cli_override(cfg, "add_latency_info", True)


def build_sample_factory_rollout_cfg(config: dict[str, Any]):
    from sample_factory.cfg.arguments import load_from_checkpoint

    module = import_integration_module(config)
    env_name = str(config["env"]["name"])
    _register_components(module, env_name)
    cfg = module.parse_cfg(module.build_cli_args_from_config(config))
    cfg = load_from_checkpoint(cfg)
    _apply_latency_to_flat_cfg(cfg, config.get("latency", {}))
    return module, cfg


def _build_env_and_policy(module: Any, cfg: Any, config: Mapping[str, Any], num_envs: int):
    from sample_factory.algo.sampling.batched_sampling import preprocess_actions
    from sample_factory.algo.utils.action_distributions import argmax_actions
    from sample_factory.algo.utils.env_info import extract_env_info
    from sample_factory.algo.utils.make_env import make_env_func_batched
    from sample_factory.algo.utils.rl_utils import prepare_and_normalize_obs
    from sample_factory.algo.utils.tensor_utils import unsqueeze_tensor
    from sample_factory.model.actor_critic import create_actor_critic
    from sample_factory.model.model_utils import get_rnn_size
    from sample_factory.utils.attr_dict import AttrDict

    device = torch.device(_resolve_device(getattr(cfg, "device", config.get("backend", {}).get("device", "cpu"))))
    envs = [
        make_env_func_batched(
            cfg,
            env_config=AttrDict(worker_index=0, vector_index=int(env_id), env_id=int(env_id)),
            render_mode=None,
        )
        for env_id in range(int(num_envs))
    ]
    env_info = extract_env_info(envs[0], cfg)
    actor_critic = create_actor_critic(cfg, envs[0].observation_space, envs[0].action_space)
    actor_critic.eval()
    actor_critic.model_to_device(device)
    checkpoint_path = config.get("policy", {}).get("checkpoint_path") or None
    checkpoint = _load_sf_policy_state_dict(cfg, actor_critic, device, checkpoint_path)
    return {
        "envs": envs,
        "env_info": env_info,
        "actor_critic": actor_critic,
        "device": device,
        "preprocess_actions": preprocess_actions,
        "argmax_actions": argmax_actions,
        "prepare_and_normalize_obs": prepare_and_normalize_obs,
        "unsqueeze_tensor": unsqueeze_tensor,
        "rnn_size": int(get_rnn_size(cfg)),
        "checkpoint": checkpoint,
    }


def _to_numpy(value: Any) -> np.ndarray:
    if torch is not None and isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _image_frame_array(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 3 and frame.shape[-1] == 1:
        frame = frame[..., 0]
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    return frame


def _chw_frame_to_image(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 3 and int(frame.shape[0]) in (1, 3):
        frame = np.transpose(frame, (1, 2, 0))
    return _image_frame_array(frame)


def _hwc_frame_to_image(frame: np.ndarray) -> np.ndarray:
    return _image_frame_array(frame)


def frames_from_policy_observation(observation: Any, *, frame_stack: int) -> list[np.ndarray]:
    if isinstance(observation, Mapping):
        observation = observation.get("obs")
    frame = _to_numpy(observation)
    if frame.ndim == 4 and int(frame.shape[0]) == 1:
        frame = frame[0]
    if frame.ndim == 3:
        channels = int(frame.shape[0])
        if int(frame_stack) > 1 and channels % int(frame_stack) == 0:
            channels_per_frame = channels // int(frame_stack)
            if channels_per_frame in (1, 3):
                return [
                    _chw_frame_to_image(frame[index * channels_per_frame : (index + 1) * channels_per_frame])
                    for index in range(int(frame_stack))
                ]
        if channels in (1, 3):
            return [_chw_frame_to_image(frame[-channels:])]
        channels = int(frame.shape[-1])
        if int(frame_stack) > 1 and channels % int(frame_stack) == 0:
            channels_per_frame = channels // int(frame_stack)
            if channels_per_frame in (1, 3):
                return [
                    _hwc_frame_to_image(frame[..., index * channels_per_frame : (index + 1) * channels_per_frame])
                    for index in range(int(frame_stack))
                ]
        if channels in (1, 3):
            return [_hwc_frame_to_image(frame)]
    return [_image_frame_array(frame)]


def frame_from_policy_observation(observation: Any, *, frame_stack: int) -> np.ndarray:
    return frames_from_policy_observation(observation, frame_stack=frame_stack)[-1]


def _stack_image_paths(image_path: Path, frame_count: int) -> list[Path]:
    stack_dir = image_path.parent / image_path.stem
    return [stack_dir / f"stack_{index:02d}.png" for index in range(int(frame_count))]


def _save_image_frame(frame: np.ndarray, image_path: Path, export_image_size: tuple[int, int] | None) -> None:
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.fromarray(frame)
    if export_image_size is not None:
        image = image.resize(export_image_size, Image.Resampling.BILINEAR)
    image.save(image_path)


def _save_frame_stack_images(
    frames: Sequence[np.ndarray],
    *,
    image_path: Path,
    image_stack_paths: Sequence[Path],
    export_image_size: tuple[int, int] | None,
) -> None:
    for frame, stack_path in zip(frames, image_stack_paths):
        _save_image_frame(frame, stack_path, export_image_size)
    _save_image_frame(frames[-1], image_path, export_image_size)


def _submit_image_write(
    executor: ThreadPoolExecutor | None,
    pending_futures: set[Future[None]],
    observation: Any,
    *,
    image_path: Path,
    frame_stack: int,
    export_image_size: tuple[int, int] | None,
    row_buffer_size: int,
) -> tuple[str, list[str], set[Future[None]]]:
    frames = frames_from_policy_observation(observation, frame_stack=frame_stack)
    image_stack_paths = _stack_image_paths(image_path, len(frames))
    if executor is None:
        _save_frame_stack_images(
            frames,
            image_path=image_path,
            image_stack_paths=image_stack_paths,
            export_image_size=export_image_size,
        )
        return str(image_path), [str(path) for path in image_stack_paths], pending_futures
    future = executor.submit(
        _save_frame_stack_images,
        frames,
        image_path=image_path,
        image_stack_paths=image_stack_paths,
        export_image_size=export_image_size,
    )
    pending_futures.add(future)
    if len(pending_futures) >= int(row_buffer_size):
        done, pending = wait(pending_futures, return_when=FIRST_COMPLETED)
        for item in done:
            item.result()
        pending_futures = set(pending)
    return str(image_path), [str(path) for path in image_stack_paths], pending_futures


def _wait_for_image_writes(pending_futures: set[Future[None]]) -> None:
    for future in pending_futures:
        future.result()


def _batch_observations(observations: Sequence[Any]) -> Any:
    first = observations[0]
    if isinstance(first, Mapping):
        if len(observations) == 1:
            arrays = {key: _to_numpy(value) for key, value in first.items()}
            if any(array.ndim >= 4 for array in arrays.values()):
                return arrays
        arrays = {}
        for key in first:
            values = [_to_numpy(obs[key]) for obs in observations]
            arrays[key] = (
                np.concatenate(values, axis=0)
                if values[0].ndim >= 4 and int(values[0].shape[0]) == 1
                else np.stack(values, axis=0)
            )
        return arrays
    return {"obs": np.stack([_to_numpy(obs) for obs in observations], axis=0)}


def _single_env_info(info: Any) -> Mapping[str, Any]:
    if isinstance(info, Mapping):
        return info
    return info[0]


def _latency_from_info(info: Mapping[str, Any], *, fallback_raw_frames: int, env_fps: float) -> tuple[int, float]:
    raw_frames = info.get("latency_raw_frames", fallback_raw_frames)
    latency_ms = info.get("latency_ms")
    raw_frames_int = int(_to_python_action(raw_frames))
    if latency_ms is None:
        latency_ms = raw_frames_to_latency_ms(raw_frames_int, env_fps=env_fps)
    return raw_frames_int, float(_to_python_action(latency_ms))


def _action_labels_for_env(env_name: str, env: Any) -> dict[int, str]:
    if env_name == "flappy":
        return {0: "noop", 1: "flap"}
    meanings = getattr(getattr(env, "unwrapped", env), "get_action_meanings", None)
    if callable(meanings):
        return {index: str(label) for index, label in enumerate(meanings())}
    if env_name == "deadly_corridor":
        return {
            0: "MOVE_FORWARD",
            1: "MOVE_BACKWARD",
            2: "MOVE_LEFT",
            3: "MOVE_RIGHT",
            4: "TURN_LEFT",
            5: "TURN_RIGHT",
            6: "ATTACK",
        }
    return {0: "NOOP", 1: "FIRE", 2: "RIGHT", 3: "LEFT", 4: "RIGHTFIRE", 5: "LEFTFIRE"}


def _action_id(raw_action: Any) -> int:
    action = _to_python_action(raw_action)
    if isinstance(action, (list, tuple)):
        return int(action[0])
    return int(action)


def _empty_records() -> dict[str, list[Any]]:
    return {
        "image": [],
        "image_stack": [],
        "split": [],
        "prompt": [],
        "action_id": [],
        "action_text": [],
        "latency_raw_frames": [],
        "latency_ms": [],
        "reward": [],
        "done": [],
        "episode_idx": [],
        "t": [],
        "seed": [],
        "deterministic": [],
    }


def _write_dataset_parquet(dataset: Dataset, path: Path) -> None:
    if len(dataset) == 0:
        import pyarrow.parquet as pq

        pq.write_table(dataset.data.table, path)
        return
    dataset.to_parquet(str(path))


def _sort_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: (int(row["episode_idx"]), int(row["t"])))


def _write_parquet_splits(output_dir: Path, train_rows: list[dict[str, Any]], val_rows: list[dict[str, Any]]):
    datasets.disable_progress_bars()
    output_dir.mkdir(parents=True, exist_ok=True)
    features = Features(
        {
            "image": HFImage(),
            "image_stack": HFSequence(HFImage()),
            "split": Value("string"),
            "prompt": Value("string"),
            "action_id": Value("int64"),
            "action_text": Value("string"),
            "latency_raw_frames": Value("int64"),
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

    train_ds = build_dataset(_sort_rows(train_rows))
    val_ds = build_dataset(_sort_rows(val_rows))
    _write_dataset_parquet(train_ds, output_dir / "train.parquet")
    _write_dataset_parquet(val_ds, output_dir / "val.parquet")
    return DatasetDict({"train": train_ds, "val": val_ds})


def _episode_row_shard_path(output_dir: Path, split: str, episode_idx: int) -> Path:
    return output_dir / "episode_rows" / str(split) / f"episode_{int(episode_idx):06d}.jsonl"


def _write_episode_row_shard(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    split = str(rows[0]["split"])
    episode_idx = int(rows[0]["episode_idx"])
    shard_path = _episode_row_shard_path(output_dir, split, episode_idx)
    shard_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = shard_path.with_name(f"{shard_path.name}.tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        for row in _sort_rows(rows):
            json.dump(row, handle, sort_keys=True)
            handle.write("\n")
    tmp_path.replace(shard_path)


def _load_episode_row_shards(output_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], set[int]]:
    train_rows: list[dict[str, Any]] = []
    val_rows: list[dict[str, Any]] = []
    completed_episode_indices: set[int] = set()
    for split in ("train", "val"):
        for shard_path in sorted((output_dir / "episode_rows" / split).glob("episode_*.jsonl")):
            with shard_path.open("r", encoding="utf-8") as handle:
                rows = [json.loads(line) for line in handle if line.strip()]
            if rows:
                completed_episode_indices.add(int(rows[0]["episode_idx"]))
                if split == "train":
                    train_rows.extend(rows)
                else:
                    val_rows.extend(rows)
    return _sort_rows(train_rows), _sort_rows(val_rows), completed_episode_indices


@contextlib.contextmanager
def _sample_factory_init_log(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "sample_factory_init.log"
    with log_path.open("a", encoding="utf-8") as handle:
        sys.stdout.flush()
        sys.stderr.flush()
        stdout_fd = os.dup(1)
        stderr_fd = os.dup(2)
        os.dup2(handle.fileno(), 1)
        os.dup2(handle.fileno(), 2)
        sf_logger = logging.getLogger("rl")
        previous_propagate = sf_logger.propagate
        redirected_handlers = []
        temporary_handler = None
        for handler in sf_logger.handlers:
            if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
                redirected_handlers.append((handler, handler.stream))
                handler.setStream(handle)
        if not redirected_handlers:
            temporary_handler = logging.StreamHandler(handle)
            sf_logger.addHandler(temporary_handler)
        sf_logger.propagate = False
        try:
            with contextlib.redirect_stdout(handle), contextlib.redirect_stderr(handle):
                yield
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            handle.flush()
            os.dup2(stdout_fd, 1)
            os.dup2(stderr_fd, 2)
            os.close(stdout_fd)
            os.close(stderr_fd)
            sf_logger.propagate = previous_propagate
            if temporary_handler is not None:
                sf_logger.removeHandler(temporary_handler)
            for handler, stream in redirected_handlers:
                handler.setStream(stream)


def collect_teacher_dataset(
    config: dict[str, Any],
    *,
    output_dir: Path,
    max_episodes: int,
    max_steps_per_episode: int,
    val_fraction: float,
    split_seed: int,
    num_envs: int,
    deterministic: bool,
    export_image_size: tuple[int, int] | None,
    image_writer_workers: int,
    row_buffer_size: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    with _sample_factory_init_log(output_dir):
        module, cfg = build_sample_factory_rollout_cfg(config)
        runtime = _build_env_and_policy(module, cfg, config, num_envs)
    envs = list(runtime["envs"])
    actor_critic = runtime["actor_critic"]
    device = runtime["device"]
    prepare_and_normalize_obs = runtime["prepare_and_normalize_obs"]
    argmax_actions = runtime["argmax_actions"]
    preprocess_actions = runtime["preprocess_actions"]
    unsqueeze_tensor = runtime["unsqueeze_tensor"]
    rnn_size = runtime["rnn_size"]
    env_name = str(config["env"]["name"])
    action_labels = _action_labels_for_env(env_name, envs[0])
    frame_stack = int(getattr(cfg, "frame_stack", 1) or 1)
    env_fps = float(getattr(cfg, "env_fps"))
    obs_fps = float(getattr(cfg, "obs_fps"))
    obs_stride_raw_frames = int(env_fps / obs_fps)
    latency_params = sample_factory_latency_params_from_flat_config(cfg, env_fps=env_fps)
    fallback_latency = int(latency_params.fixed_latency_raw_frames)
    base_seed = int(getattr(cfg, "seed", 0) or 0)
    val_count = int(round(int(max_episodes) * float(val_fraction)))
    val_indices = set(np.random.default_rng(int(split_seed)).choice(int(max_episodes), size=val_count, replace=False).tolist())
    episode_specs = [
        {
            "episode_idx": idx,
            "split": "val" if idx in val_indices else "train",
            "seed": base_seed + idx,
        }
        for idx in range(int(max_episodes))
    ]
    train_rows, val_rows, completed_episode_indices = _load_episode_row_shards(output_dir)
    episode_specs = [spec for spec in episode_specs if int(spec["episode_idx"]) not in completed_episode_indices]
    active_specs: list[dict[str, Any] | None] = [None for _ in envs]
    active_obs: list[Any | None] = [None for _ in envs]
    active_episode_rows: list[list[dict[str, Any]]] = [[] for _ in envs]
    active_t = [0 for _ in envs]
    rnn_states = torch.zeros([len(envs), rnn_size], dtype=torch.float32, device=device)
    pending_futures: set[Future[None]] = set()
    executor = ThreadPoolExecutor(max_workers=int(image_writer_workers)) if int(image_writer_workers) > 0 else None

    def start_episode(slot: int, spec: dict[str, Any]) -> None:
        obs, _ = envs[slot].reset(seed=int(spec["seed"]))
        active_specs[slot] = spec
        active_obs[slot] = obs
        active_episode_rows[slot] = []
        active_t[slot] = 0
        rnn_states[slot].zero_()

    try:
        next_spec_idx = 0
        for slot in range(len(envs)):
            if next_spec_idx >= len(episode_specs):
                break
            start_episode(slot, episode_specs[next_spec_idx])
            next_spec_idx += 1

        completed = len(completed_episode_indices)
        with tqdm(total=int(max_episodes), initial=completed, unit="episode") as progress:
            while completed < int(max_episodes):
                active_slots = [slot for slot, spec in enumerate(active_specs) if spec is not None]
                if not active_slots:
                    break
                obs_batch = _batch_observations([active_obs[slot] for slot in active_slots])
                normalized_obs = prepare_and_normalize_obs(actor_critic, obs_batch)
                with torch.no_grad():
                    policy_outputs = actor_critic(normalized_obs, rnn_states[active_slots])
                    actions = policy_outputs["actions"]
                    if deterministic:
                        actions = argmax_actions(actor_critic.action_distribution())
                    actions = _normalize_actions_for_space(
                        actions,
                        batch_size=len(active_slots),
                        action_space=envs[0].action_space,
                        device=device,
                        unsqueeze_tensor=unsqueeze_tensor,
                    )
                    processed_actions = preprocess_actions(runtime["env_info"], actions)
                    raw_actions = _split_batch_actions(processed_actions, len(active_slots), action_space=envs[0].action_space)
                    new_rnn_states = policy_outputs["new_rnn_states"]

                for batch_idx, slot in enumerate(active_slots):
                    spec = active_specs[slot]
                    obs = active_obs[slot]
                    split = str(spec["split"])
                    t = int(active_t[slot])
                    image_path = output_dir / "images" / split / f"episode_{int(spec['episode_idx']):06d}" / f"t_{t:06d}.png"
                    saved_image, image_stack, pending_futures = _submit_image_write(
                        executor,
                        pending_futures,
                        obs,
                        image_path=image_path,
                        frame_stack=frame_stack,
                        export_image_size=export_image_size,
                        row_buffer_size=int(row_buffer_size),
                    )
                    raw_action = raw_actions[batch_idx]
                    action_id = _action_id(raw_action)
                    next_obs, reward, terminated, truncated, info = envs[slot].step([raw_action])
                    info_dict = dict(_single_env_info(info))
                    latency_raw_frames, latency_ms = _latency_from_info(
                        info_dict,
                        fallback_raw_frames=fallback_latency,
                        env_fps=env_fps,
                    )
                    done = bool(_to_python_action(terminated)) or bool(_to_python_action(truncated))
                    row = {
                        "image": saved_image,
                        "image_stack": image_stack,
                        "split": split,
                        "prompt": render_teacher_prompt(
                            env_name,
                            latency_raw_frames=latency_raw_frames,
                            latency_ms=latency_ms,
                            env_fps=env_fps,
                            obs_fps=obs_fps,
                        ),
                        "action_id": int(action_id),
                        "action_text": action_labels.get(int(action_id), str(action_id)),
                        "latency_raw_frames": int(latency_raw_frames),
                        "latency_ms": float(latency_ms),
                        "reward": float(_to_python_action(reward)),
                        "done": done,
                        "episode_idx": int(spec["episode_idx"]),
                        "t": t,
                        "seed": int(spec["seed"]),
                        "deterministic": bool(deterministic),
                    }
                    active_episode_rows[slot].append(row)
                    if split == "val":
                        val_rows.append(row)
                    else:
                        train_rows.append(row)

                    rnn_states[slot] = new_rnn_states[batch_idx]
                    next_t = t + 1
                    if bool(row["done"]) or next_t >= int(max_steps_per_episode):
                        _wait_for_image_writes(pending_futures)
                        pending_futures = set()
                        _write_episode_row_shard(output_dir, active_episode_rows[slot])
                        progress.update(1)
                        completed += 1
                        if next_spec_idx < len(episode_specs):
                            start_episode(slot, episode_specs[next_spec_idx])
                            next_spec_idx += 1
                        else:
                            active_specs[slot] = None
                            active_obs[slot] = None
                            active_episode_rows[slot] = []
                            active_t[slot] = 0
                            rnn_states[slot].zero_()
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

    return _sort_rows(train_rows), _sort_rows(val_rows), {
        "action_labels": action_labels,
        "checkpoint_train_step": int(runtime["checkpoint"].get("train_step", runtime["checkpoint"].get("env_steps", 0))),
        "frame_stack": frame_stack,
        "env_fps": env_fps,
        "obs_fps": obs_fps,
        "obs_stride_raw_frames": obs_stride_raw_frames,
    }


def _parse_image_size(value: str) -> tuple[int, int] | None:
    if not value:
        return None
    width, height = [int(item.strip()) for item in str(value).split(",")]
    return width, height


def _resolve_token(explicit_token: str) -> str | None:
    token = str(explicit_token).strip()
    if token:
        return token
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")


def _push_dataset_to_hub(dataset_dict: Any, *, repo_id: str, private: bool, token: str | None, max_shard_size: str) -> None:
    push_kwargs: dict[str, Any] = {
        "repo_id": repo_id,
        "private": bool(private),
        "max_shard_size": str(max_shard_size),
    }
    if token:
        push_kwargs["token"] = token
    dataset_dict.push_to_hub(**push_kwargs)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export Sample Factory teacher rollouts as parquet datasets.")
    parser.add_argument("--checkpoint-root", required=True, help="Sample Factory experiment directory or checkpoint file.")
    parser.add_argument("--output-dir", required=True, help="Directory where parquet files and observation images are written.")
    parser.add_argument("--config", default="", help="Optional current training YAML. Defaults to checkpoint config.json.")
    parser.add_argument("--latency-config", default="", help="Optional YAML containing a training-compatible latency: section.")
    parser.add_argument("--env-name", default="", choices=["", "deadly_corridor", "demon_attack", "flappy"])
    parser.add_argument("--max-episodes", type=int, required=True)
    parser.add_argument("--max-steps-per-episode", type=int, required=True)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--export-image-size", type=str, default="")
    parser.add_argument("--image-writer-workers", type=int, default=0)
    parser.add_argument("--row-buffer-size", type=int, default=8192)
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--hf-repo-id", type=str, default="")
    parser.add_argument("--hf-private", action="store_true")
    parser.add_argument("--hf-token", type=str, default="")
    parser.add_argument("--hf-max-shard-size", type=str, default="500MB")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    output_dir = Path(args.output_dir).expanduser().resolve()
    config, config_metadata = load_rollout_training_config(
        args.checkpoint_root,
        config_path=args.config or None,
        latency_config_path=args.latency_config or None,
        device_override=args.device or None,
    )
    if args.env_name:
        config.setdefault("env", {})["name"] = str(args.env_name)
    train_rows, val_rows, rollout_metadata = collect_teacher_dataset(
        config,
        output_dir=output_dir,
        max_episodes=int(args.max_episodes),
        max_steps_per_episode=int(args.max_steps_per_episode),
        val_fraction=float(args.val_fraction),
        split_seed=int(args.split_seed),
        num_envs=int(args.num_envs),
        deterministic=bool(args.deterministic),
        export_image_size=_parse_image_size(str(args.export_image_size)),
        image_writer_workers=int(args.image_writer_workers),
        row_buffer_size=int(args.row_buffer_size),
    )
    dataset_dict = _write_parquet_splits(output_dir, train_rows, val_rows)
    metadata = {
        **config_metadata,
        **rollout_metadata,
        "env_name": config["env"]["name"],
        "prompt_source": "scripts/rollout_data/prompts.py",
        "base_prompt": BASE_PROMPTS[str(config["env"]["name"])],
        "prompt_timing_suffix": TIMING_SUFFIX,
        "max_episodes": int(args.max_episodes),
        "max_steps_per_episode": int(args.max_steps_per_episode),
        "val_fraction": float(args.val_fraction),
        "split_seed": int(args.split_seed),
        "num_envs": int(args.num_envs),
        "deterministic": bool(args.deterministic),
        "image_source": "policy_observation",
        "image_stack_source": "policy_observation_frame_stack",
        "image_stack_order": "oldest_to_newest",
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
    }
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")

    if args.push_to_hub:
        _push_dataset_to_hub(
            dataset_dict,
            repo_id=str(args.hf_repo_id),
            private=bool(args.hf_private),
            token=_resolve_token(str(args.hf_token)),
            max_shard_size=str(args.hf_max_shard_size),
        )
    print(f"[export] wrote train={len(train_rows)} rows val={len(val_rows)} rows to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
