from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.sample_factory_common import import_integration_module, load_training_config
from training.sample_factory_common.common import maybe_set_cli_override
from training.sample_factory_common.queue_eval import prepare_queue_eval_cfg


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


def _minimal_export_cfg(module, *, experiment_dir: Path, device_override: str | None):
    cli_args = [
        "--mode",
        "eval",
        "--algo",
        "APPO",
        "--env",
        str(getattr(module, "SF_ENV_NAME")),
        "--experiment",
        str(experiment_dir.name),
        "--train_dir",
        str(experiment_dir.parent),
    ]
    if device_override:
        cli_args.extend(["--device", str(device_override)])
    return _parse_module_cfg(module, cli_args)


def _prepare_export_cfg(module, env_name: str, base_cfg, latency: int):
    from sample_factory.cfg.arguments import load_from_checkpoint

    if env_name == "deadly_corridor" and hasattr(module, "_prepare_eval_cfg"):
        cfg = module._prepare_eval_cfg(base_cfg, latency)
        cfg = load_from_checkpoint(cfg)
        cfg.queue_latency_values = str(int(latency))
        cfg.latency_mode = "episode_fixed"
        cfg.latency_sampling_mode = "uniform_all"
        cfg.eval_env_frameskip = cfg.env_frameskip
        return cfg

    cfg = prepare_queue_eval_cfg(base_cfg, latency)
    cfg = load_from_checkpoint(cfg)
    maybe_set_cli_override(cfg, "queue_latency_values", str(int(latency)))
    maybe_set_cli_override(cfg, "latency_mode", "episode_fixed")
    maybe_set_cli_override(cfg, "latency_sampling_mode", "uniform_all")
    if env_name == "demon_attack" and bool(getattr(base_cfg, "eval_raw_reward", False)):
        maybe_set_cli_override(cfg, "clip_reward", False)
    return cfg


def _build_env_and_policy(module, cfg):
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

    env = make_env_func_batched(
        cfg,
        env_config=AttrDict(worker_index=0, vector_index=0, env_id=0),
        render_mode=None,
    )
    env_info = extract_env_info(env, cfg)
    actor_critic = create_actor_critic(cfg, env.observation_space, env.action_space)
    actor_critic.eval()
    device = torch.device("cpu" if str(cfg.device) == "cpu" else "cuda")
    actor_critic.model_to_device(device)

    if hasattr(module, "_load_policy_state_dict"):
        module._load_policy_state_dict(cfg, actor_critic, device)
    else:
        from training.sample_factory_common.queue_eval import load_sf_policy_state_dict

        load_sf_policy_state_dict(cfg, actor_critic, device)

    return {
        "env": env,
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
    env_name: str,
    frame_stack: int,
    image_path: Path,
) -> str:
    image_path.parent.mkdir(parents=True, exist_ok=True)
    single_obs = _extract_single_observation(obs)
    frame = _latest_frame_from_observation(single_obs, env_name=env_name, frame_stack=frame_stack)
    Image.fromarray(np.asarray(frame, dtype=np.uint8)).save(image_path)
    return str(image_path)


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


def _deadly_action_labels(env) -> dict[int, str]:
    action_space = getattr(env, "action_space", None)
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


def _prompt_for_episode(
    *,
    dataset_mode: str,
    prompt: str,
    mixed_prompt_template: str,
    latency: int,
    latency_ms: float,
) -> str:
    if dataset_mode == "mixed_latency":
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
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    import torch
    from sample_factory.algo.utils.rl_utils import make_dones

    cfg = _prepare_export_cfg(module, env_name, base_cfg, latency)
    runtime = _build_env_and_policy(module, cfg)
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
    latency_ms = float(latency) * decision_period_ms

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
                env_name=env_name,
                frame_stack=frame_stack,
                image_path=image_path,
            )

            normalized_obs = prepare_and_normalize_obs(actor_critic, obs)
            policy_outputs = actor_critic(normalized_obs, rnn_states)
            actions = policy_outputs["actions"]
            if deterministic:
                actions = argmax_actions(actor_critic.action_distribution())
            if actions.ndim == 1:
                actions = unsqueeze_tensor(actions, dim=-1)
            actions = preprocess_actions(env_info, actions)
            rnn_states = policy_outputs["new_rnn_states"]

            action_id = int(_to_python_scalar(actions[0]))
            action_text = action_labels.get(action_id, str(action_id))
            prompt_text = _prompt_for_episode(
                dataset_mode=dataset_mode,
                prompt=prompt,
                mixed_prompt_template=mixed_prompt_template,
                latency=latency,
                latency_ms=latency_ms,
            )

            next_obs, reward, terminated, truncated, info = env.step(actions)
            dones = make_dones(terminated, truncated)
            done = bool(_to_python_scalar(dones[0]))
            info_single = _extract_single_info(info)
            reward_value = _reward_from_step(reward[0] if isinstance(reward, (list, tuple, np.ndarray)) else reward, info_single, prefer_raw_reward=prefer_raw_reward)

            split_rows.append(
                {
                    "image": saved_image,
                    "split": split,
                    "prompt": prompt_text,
                    "action_id": action_id,
                    "action_text": action_text,
                    "latency": int(latency),
                    "latency_ms": float(latency_ms),
                    "reward": float(reward_value),
                    "done": bool(done),
                    "episode_idx": int(episode_idx),
                    "t": int(t),
                    "seed": int(episode_seed),
                    "deterministic": bool(deterministic),
                }
            )

            obs = next_obs
            if done:
                break

    env.close()
    return train_rows, val_rows, {"latency": int(latency), "action_labels": action_labels}


def _build_plan(args) -> list[tuple[int, int, int]]:
    if args.dataset_mode == "zero_latency":
        val_count = int(round(args.max_episodes * args.val_fraction))
        val_count = max(0, min(int(args.max_episodes), val_count))
        return [(0, int(args.max_episodes), val_count)]

    latencies = list(range(int(args.latency_start), int(args.latency_end) + 1))
    return [(latency, int(args.episodes_per_latency), int(args.val_episodes_per_latency)) for latency in latencies]


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Sample Factory teacher trajectories as parquet datasets for SFT.")
    parser.add_argument("--config", default="", help="Optional Sample Factory training YAML used only to disambiguate env.name if needed.")
    parser.add_argument("--env-name", default="", choices=["", "deadly_corridor", "demon_attack", "flappy"], help="Optional environment name. Use this instead of --config when checkpoint root contains multiple environments.")
    parser.add_argument("--checkpoint-root", required=True, help="Directory containing latency-specific experiment folders.")
    parser.add_argument("--output-dir", required=True, help="Directory where parquet files and images will be written.")
    parser.add_argument("--dataset-mode", required=True, choices=["zero_latency", "mixed_latency"])
    parser.add_argument("--prompt", required=True, help="Base prompt used for the zero-latency dataset.")
    parser.add_argument(
        "--mixed-prompt-template",
        default="{prompt} Your actions take {latency_ms} milliseconds to take effect. Choose the best next action while accounting for the delay.",
        help="Template for mixed-latency prompts. Supported fields: {prompt}, {latency}, {latency_ms}.",
    )
    parser.add_argument("--max-steps-per-episode", type=int, required=True)
    parser.add_argument("--device", type=str, default="", help="Optional override for Sample Factory device, e.g. cpu or gpu.")
    parser.add_argument("--deterministic", action="store_true", default=True)
    parser.add_argument("--checkpoint-seed", type=int, default=None, help="Seed encoded in checkpoint folder names. Defaults to experiment.seed from the config.")
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--push-to-hub", action="store_true", help="Upload the resulting dataset to Hugging Face after writing local parquet files.")
    parser.add_argument("--hf-repo-id", type=str, default="", help="Dataset repo id, e.g. username/dataset_name.")
    parser.add_argument("--hf-private", action="store_true", help="Create or update the HF dataset repo as private.")
    parser.add_argument("--hf-token", type=str, default="", help="Optional HF token. If omitted, the script will use HF_TOKEN or HUGGINGFACE_HUB_TOKEN from the environment.")
    parser.add_argument("--hf-max-shard-size", type=str, default="500MB", help="Shard size forwarded to datasets.push_to_hub.")

    parser.add_argument("--max-episodes", type=int, default=0, help="Zero-latency mode only.")
    parser.add_argument("--val-fraction", type=float, default=0.1, help="Zero-latency mode only.")

    parser.add_argument("--latency-start", type=int, default=0, help="Mixed-latency mode only.")
    parser.add_argument("--latency-end", type=int, default=0, help="Mixed-latency mode only.")
    parser.add_argument("--episodes-per-latency", type=int, default=0, help="Mixed-latency mode only.")
    parser.add_argument("--val-episodes-per-latency", type=int, default=0, help="Mixed-latency mode only.")

    args = parser.parse_args()

    checkpoint_root = Path(args.checkpoint_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
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

    if args.dataset_mode == "zero_latency" and args.max_episodes <= 0:
        raise ValueError("--max-episodes must be positive for zero-latency export")
    if args.dataset_mode == "mixed_latency":
        if args.episodes_per_latency <= 0:
            raise ValueError("--episodes-per-latency must be positive for mixed-latency export")
        if args.latency_end < args.latency_start:
            raise ValueError("--latency-end must be >= --latency-start")

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
        "decision_period_ms": None,
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
                    matches.append((candidate, _find_experiment_dir(checkpoint_root, candidate, latency, checkpoint_seed)))
                except FileNotFoundError:
                    continue
            if len(matches) != 1:
                raise ValueError(
                    f"Could not uniquely infer env for latency {latency} under {checkpoint_root}. "
                    "Pass --config with the correct env.name."
                )
            env_name, experiment_dir = matches[0]
        else:
            experiment_dir = _find_experiment_dir(checkpoint_root, env_name, latency, checkpoint_seed)

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
            prepared_cfg = _prepare_export_cfg(module, env_name, base_cfg, latency)
            metadata["decision_period_ms"] = float(_decision_period_ms_from_cfg(env_name, prepared_cfg))
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
        )
        train_rows.extend(latency_train_rows)
        val_rows.extend(latency_val_rows)
        metadata["latencies"].append(
            {
                "latency": int(latency),
                "latency_ms": float(latency) * float(metadata["decision_period_ms"]),
                "episode_count": int(episode_count),
                "val_episode_count": int(val_count),
                "experiment_dir": str(experiment_dir),
                "best_checkpoint": _find_best_checkpoint_name(experiment_dir),
            }
        )
        metadata["action_labels"][str(latency)] = latency_meta["action_labels"]
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
