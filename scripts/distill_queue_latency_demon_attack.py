from __future__ import annotations

import argparse
import importlib.util
import json
import os
import pickle
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from stable_baselines3 import PPO


# =========================
# Helpers
# =========================


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def maybe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_json(obj: Any, path: str) -> None:
    maybe_mkdir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def import_module_from_path(module_path: str):
    spec = importlib.util.spec_from_file_location("teacher_ppo_module", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import module from path: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def serialize_jsonable(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): serialize_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [serialize_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    return x


# =========================
# Image layout helpers
# =========================


def infer_image_layout(obs_shape: Sequence[int]) -> Tuple[str, int, int, int]:
    """
    Returns: (layout, channels, height, width)
    """
    if len(obs_shape) != 3:
        raise ValueError(f"Expected 3D image observation, got shape={tuple(obs_shape)}")

    d0, d1, d2 = [int(x) for x in obs_shape]
    if d0 <= 16 and d1 >= 16 and d2 >= 16:
        return "CHW", d0, d1, d2
    return "HWC", d2, d0, d1


def maybe_image_to_chw_tensor(x: torch.Tensor, obs_shape: Sequence[int]) -> torch.Tensor:
    layout, c, h, w = infer_image_layout(obs_shape)

    if x.ndim == 3:
        x = x.unsqueeze(0)

    if x.ndim != 4:
        raise ValueError(f"Expected image tensor with 3 or 4 dims, got shape={tuple(x.shape)}")

    if x.shape[1] == c and x.shape[2] == h and x.shape[3] == w:
        return x

    if x.shape[-1] == c and x.shape[1] == h and x.shape[2] == w:
        return x.permute(0, 3, 1, 2).contiguous()

    if layout == "HWC":
        # Most common failure mode for these Atari stacked observations.
        return x.permute(0, 3, 1, 2).contiguous()

    return x


# =========================
# Student models
# =========================


class ImageBCPolicy(nn.Module):
    """
    Layout-safe CNN policy for Atari-style image observations.

    Important fix:
    - stacked DemonAttack observations come out as HWC=(84, 84, 4)
    - conv2d expects NCHW=(N, 4, 84, 84)
    - this module infers layout from obs_shape and permutes in forward()
    """

    def __init__(self, obs_shape: Sequence[int], num_actions: int):
        super().__init__()
        self.obs_shape = tuple(int(x) for x in obs_shape)
        self.layout, c, h, w = infer_image_layout(self.obs_shape)

        self.net = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, c, h, w, dtype=torch.float32)
            feat_dim = int(self.net(dummy).shape[1])

        self.head = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.ReLU(),
            nn.Linear(512, int(num_actions)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        if x.max().item() > 1.0:
            x = x / 255.0
        x = maybe_image_to_chw_tensor(x, self.obs_shape)
        return self.head(self.net(x))

    @torch.no_grad()
    def predict(self, obs: np.ndarray, device: torch.device) -> int:
        x = torch.as_tensor(obs, dtype=torch.float32, device=device)
        logits = self.forward(x)
        return int(torch.argmax(logits, dim=1).item())


class ImageLatencyConcatBCPolicy(nn.Module):
    """
    Optional model for obs_mode=latency_channel when the latency channel is already in the image.
    Uses the same layout-safe image path as ImageBCPolicy.
    """

    def __init__(self, obs_shape: Sequence[int], num_actions: int):
        super().__init__()
        self.base = ImageBCPolicy(obs_shape=obs_shape, num_actions=num_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x)

    @torch.no_grad()
    def predict(self, obs: np.ndarray, device: torch.device) -> int:
        return self.base.predict(obs=obs, device=device)


# =========================
# Dataset
# =========================


class DistillDataset(Dataset):
    def __init__(self, observations: np.ndarray, actions: np.ndarray):
        if len(observations) != len(actions):
            raise ValueError("observations and actions must have the same length")
        self.observations = observations.astype(np.uint8, copy=False)
        self.actions = actions.astype(np.int64, copy=False)

    def __len__(self) -> int:
        return int(len(self.actions))

    def __getitem__(self, idx: int):
        obs = torch.from_numpy(self.observations[idx])
        action = torch.tensor(int(self.actions[idx]), dtype=torch.long)
        return obs, action


# =========================
# Rollout collection and shifting
# =========================


@dataclass
class EpisodeRollout:
    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    infos: List[Dict[str, Any]]



def _build_single_env(
    ppo_module,
    seed: int,
    env_id: str,
    obs_mode: str,
    frame_skip_values: Sequence[int],
    screen_size: int,
    noop_max: int,
    clip_reward: bool,
    terminal_on_life_loss: bool,
    repeat_action_probability: float,
    full_action_space: bool,
    stack_frames: int,
):
    if not hasattr(ppo_module, "make_env"):
        raise AttributeError("The PPO module must expose make_env(...) for env construction.")

    env_fn = ppo_module.make_env(
        seed=seed,
        env_id=env_id,
        obs_mode=obs_mode,
        frame_skip_values=list(frame_skip_values),
        episodes_per_stage=10**9,
        randomize_within_stage=False,
        latency_mode="episode_fixed",
        latency_sampling_mode="uniform_all",
        within_episode_change_interval=1,
        within_episode_change_prob=1.0,
        screen_size=screen_size,
        noop_max=noop_max,
        clip_reward=clip_reward,
        terminal_on_life_loss=terminal_on_life_loss,
        repeat_action_probability=repeat_action_probability,
        full_action_space=full_action_space,
        stack_frames=stack_frames,
        history_len=8,
        memory_use_latency=False,
        memory_latency_repr="separate",
    )
    return env_fn()



def collect_teacher_rollouts(
    ppo_module,
    teacher_model_path: str,
    teacher_rollout_path: str,
    teacher_episodes: int,
    teacher_max_steps: int,
    teacher_obs_mode: str,
    env_id: str,
    screen_size: int,
    noop_max: int,
    clip_reward: bool,
    terminal_on_life_loss: bool,
    repeat_action_probability: float,
    full_action_space: bool,
    stack_frames: int,
    seed: int,
) -> Dict[str, Any]:
    maybe_mkdir(os.path.dirname(teacher_rollout_path) or ".")

    if os.path.exists(teacher_rollout_path):
        with open(teacher_rollout_path, "rb") as f:
            return pickle.load(f)

    if teacher_obs_mode not in {"image_only", "latency_channel"}:
        raise ValueError(f"This corrected distill script supports image obs only. Got {teacher_obs_mode=}")

    env = _build_single_env(
        ppo_module=ppo_module,
        seed=seed,
        env_id=env_id,
        obs_mode=teacher_obs_mode,
        frame_skip_values=[0],
        screen_size=screen_size,
        noop_max=noop_max,
        clip_reward=clip_reward,
        terminal_on_life_loss=terminal_on_life_loss,
        repeat_action_probability=repeat_action_probability,
        full_action_space=full_action_space,
        stack_frames=stack_frames,
    )

    model = PPO.load(teacher_model_path, device="cpu")

    episodes: List[EpisodeRollout] = []
    obs_shape: Optional[Tuple[int, ...]] = None
    num_actions = int(env.action_space.n)

    for ep in range(int(teacher_episodes)):
        obs, info = env.reset(seed=seed + ep)
        obs = np.asarray(obs)
        if obs_shape is None:
            obs_shape = tuple(int(x) for x in obs.shape)
            print(f"[teacher] first obs shape: {obs_shape}")

        obs_list: List[np.ndarray] = []
        act_list: List[int] = []
        rew_list: List[float] = []
        done_list: List[bool] = []
        info_list: List[Dict[str, Any]] = []

        for _ in range(int(teacher_max_steps)):
            action, _ = model.predict(obs, deterministic=True)
            action_int = int(action)

            obs_list.append(np.asarray(obs, dtype=np.uint8).copy())
            act_list.append(action_int)

            next_obs, reward, terminated, truncated, info = env.step(action_int)
            done = bool(terminated or truncated)

            rew_list.append(float(reward))
            done_list.append(done)
            info_list.append(serialize_jsonable(info if isinstance(info, dict) else {}))

            obs = np.asarray(next_obs)
            if done:
                break

        episodes.append(
            EpisodeRollout(
                observations=np.stack(obs_list, axis=0).astype(np.uint8),
                actions=np.asarray(act_list, dtype=np.int64),
                rewards=np.asarray(rew_list, dtype=np.float32),
                dones=np.asarray(done_list, dtype=np.bool_),
                infos=info_list,
            )
        )
        print(f"[teacher] episode={ep} steps={len(act_list)} return={float(np.sum(rew_list)):.3f}")

    env.close()

    payload = {
        "obs_shape": tuple(obs_shape or ()),
        "num_actions": num_actions,
        "episodes": episodes,
    }
    with open(teacher_rollout_path, "wb") as f:
        pickle.dump(payload, f)
    return payload



def build_shifted_dataset_for_latency(
    teacher_rollouts: Dict[str, Any],
    latency: int,
    shifted_dataset_root: str,
) -> Dict[str, Any]:
    maybe_mkdir(shifted_dataset_root)
    out_path = os.path.join(shifted_dataset_root, f"latency_{int(latency)}.pkl")

    if os.path.exists(out_path):
        with open(out_path, "rb") as f:
            return pickle.load(f)

    all_obs: List[np.ndarray] = []
    all_actions: List[int] = []
    episode_counts: List[int] = []

    for ep in teacher_rollouts["episodes"]:
        observations = ep.observations
        actions = ep.actions
        if latency < 0:
            raise ValueError("latency must be non-negative")
        if len(actions) <= latency:
            episode_counts.append(0)
            continue

        usable = len(actions) - latency
        all_obs.append(observations[:usable])
        all_actions.append(actions[latency: latency + usable])
        episode_counts.append(int(usable))

    if len(all_obs) == 0:
        raise RuntimeError(f"No usable samples found for latency={latency}")

    observations = np.concatenate(all_obs, axis=0).astype(np.uint8)
    actions = np.concatenate(all_actions, axis=0).astype(np.int64)

    payload = {
        "latency": int(latency),
        "obs_shape": tuple(int(x) for x in teacher_rollouts["obs_shape"]),
        "num_actions": int(teacher_rollouts["num_actions"]),
        "num_samples": int(len(actions)),
        "per_episode_counts": episode_counts,
        "observations": observations,
        "actions": actions,
    }
    with open(out_path, "wb") as f:
        pickle.dump(payload, f)
    print(f"[shift] latency={latency} samples={len(actions)} -> {out_path}")
    return payload


# =========================
# Training
# =========================



def make_student_model(obs_shape: Sequence[int], num_actions: int, student_obs_mode: str) -> nn.Module:
    if student_obs_mode in {"image_only", "latency_channel"}:
        if student_obs_mode == "latency_channel":
            return ImageLatencyConcatBCPolicy(obs_shape=obs_shape, num_actions=num_actions)
        return ImageBCPolicy(obs_shape=obs_shape, num_actions=num_actions)
    raise ValueError(f"This corrected distill script supports image obs only. Got {student_obs_mode=}")



def train_student_for_latency(
    dataset_payload: Dict[str, Any],
    student_obs_mode: str,
    student_save_dir: str,
    batch_size: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    grad_clip: float,
    seed: int,
    device: torch.device,
    wandb_run=None,
) -> Dict[str, Any]:
    maybe_mkdir(student_save_dir)

    latency = int(dataset_payload["latency"])
    obs_shape = tuple(int(x) for x in dataset_payload["obs_shape"])
    num_actions = int(dataset_payload["num_actions"])

    dataset = DistillDataset(
        observations=dataset_payload["observations"],
        actions=dataset_payload["actions"],
    )

    n_total = len(dataset)
    n_val = max(1, int(round(0.1 * n_total))) if n_total >= 10 else max(1, n_total // 5)
    n_train = max(1, n_total - n_val)
    if n_train + n_val > n_total:
        n_val = n_total - n_train
    train_ds, val_ds = random_split(
        dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(seed + latency),
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = make_student_model(obs_shape=obs_shape, num_actions=num_actions, student_obs_mode=student_obs_mode)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    best_state: Optional[Dict[str, torch.Tensor]] = None
    history: List[Dict[str, float]] = []

    print(f"[train] latency={latency} obs_shape={obs_shape} num_actions={num_actions} n_train={n_train} n_val={n_val}")

    for epoch in range(1, int(epochs) + 1):
        model.train()
        total_train_loss = 0.0
        total_train_correct = 0
        total_train_count = 0

        for obs_batch, act_batch in train_loader:
            obs_batch = obs_batch.to(device=device, dtype=torch.float32)
            act_batch = act_batch.to(device=device, dtype=torch.long)

            optimizer.zero_grad(set_to_none=True)
            logits = model(obs_batch)
            loss = criterion(logits, act_batch)
            loss.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            total_train_loss += float(loss.item()) * int(act_batch.shape[0])
            preds = torch.argmax(logits, dim=1)
            total_train_correct += int((preds == act_batch).sum().item())
            total_train_count += int(act_batch.shape[0])

        model.eval()
        total_val_loss = 0.0
        total_val_correct = 0
        total_val_count = 0
        with torch.no_grad():
            for obs_batch, act_batch in val_loader:
                obs_batch = obs_batch.to(device=device, dtype=torch.float32)
                act_batch = act_batch.to(device=device, dtype=torch.long)
                logits = model(obs_batch)
                loss = criterion(logits, act_batch)
                total_val_loss += float(loss.item()) * int(act_batch.shape[0])
                preds = torch.argmax(logits, dim=1)
                total_val_correct += int((preds == act_batch).sum().item())
                total_val_count += int(act_batch.shape[0])

        train_loss = total_train_loss / max(1, total_train_count)
        val_loss = total_val_loss / max(1, total_val_count)
        train_acc = total_train_correct / max(1, total_train_count)
        val_acc = total_val_correct / max(1, total_val_count)

        row = {
            "epoch": float(epoch),
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "train_acc": float(train_acc),
            "val_acc": float(val_acc),
        }
        history.append(row)
        print(
            f"[train] latency={latency} epoch={epoch}/{epochs} "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"train_acc={train_acc:.4f} val_acc={val_acc:.4f}"
        )
        if wandb_run is not None:
            wandb_run.log({f"latency_{latency}/" + k: v for k, v in row.items()})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    save_path = os.path.join(student_save_dir, f"student_latency_{latency}.pt")
    torch.save(
        {
            "latency": latency,
            "obs_shape": obs_shape,
            "num_actions": num_actions,
            "student_obs_mode": student_obs_mode,
            "state_dict": model.state_dict(),
            "history": history,
        },
        save_path,
    )

    return {
        "latency": latency,
        "student_path": save_path,
        "obs_shape": obs_shape,
        "num_actions": num_actions,
        "best_val_loss": float(best_val_loss),
        "history": history,
    }


# =========================
# Evaluation
# =========================



def load_student_checkpoint(student_path: str, device: torch.device) -> Tuple[nn.Module, Dict[str, Any]]:
    ckpt = torch.load(student_path, map_location=device)
    model = make_student_model(
        obs_shape=ckpt["obs_shape"],
        num_actions=int(ckpt["num_actions"]),
        student_obs_mode=str(ckpt["student_obs_mode"]),
    )
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()
    return model, ckpt


@torch.no_grad()
def evaluate_student(
    ppo_module,
    student_path: str,
    eval_latency: int,
    eval_episodes: int,
    env_id: str,
    student_obs_mode: str,
    screen_size: int,
    noop_max: int,
    clip_reward: bool,
    terminal_on_life_loss: bool,
    repeat_action_probability: float,
    full_action_space: bool,
    stack_frames: int,
    eval_max_steps: int,
    seed: int,
    device: torch.device,
) -> Dict[str, Any]:
    model, ckpt = load_student_checkpoint(student_path=student_path, device=device)

    env = _build_single_env(
        ppo_module=ppo_module,
        seed=seed,
        env_id=env_id,
        obs_mode=student_obs_mode,
        frame_skip_values=[int(eval_latency)],
        screen_size=screen_size,
        noop_max=noop_max,
        clip_reward=clip_reward,
        terminal_on_life_loss=terminal_on_life_loss,
        repeat_action_probability=repeat_action_probability,
        full_action_space=full_action_space,
        stack_frames=stack_frames,
    )

    returns: List[float] = []
    lengths: List[int] = []

    for ep in range(int(eval_episodes)):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        ep_ret = 0.0
        ep_len = 0
        while (not done) and (int(eval_max_steps) <= 0 or ep_len < int(eval_max_steps)):
            action = model.predict(np.asarray(obs), device=device)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = bool(terminated or truncated)
            ep_ret += float(reward)
            ep_len += 1
        returns.append(ep_ret)
        lengths.append(ep_len)

    env.close()
    return {
        "trained_latency": int(ckpt["latency"]),
        "eval_latency": int(eval_latency),
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "mean_length": float(np.mean(lengths)),
        "num_episodes": int(eval_episodes),
    }


# =========================
# Main
# =========================



def parse_args():
    ap = argparse.ArgumentParser()

    ap.add_argument("--ppo-module-path", type=str, required=True)
    ap.add_argument("--teacher-model-path", type=str, required=True)
    ap.add_argument("--teacher-rollout-path", type=str, required=True)
    ap.add_argument("--shifted-dataset-root", type=str, required=True)
    ap.add_argument("--student-save-dir", type=str, required=True)
    ap.add_argument("--eval-json-path", type=str, required=True)

    ap.add_argument("--mode", type=str, default="all", choices=["all", "teacher", "shift", "train", "eval"])
    ap.add_argument("--teacher-episodes", type=int, default=100)
    ap.add_argument("--teacher-max-steps", type=int, default=4000)
    ap.add_argument("--teacher-obs-mode", type=str, default="image_only")
    ap.add_argument("--student-obs-mode", type=str, default="image_only")
    ap.add_argument("--train-latencies", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--eval-latencies", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5])
    ap.add_argument("--latency-normalization-max", type=int, default=15)

    ap.add_argument("--env-id", type=str, default="ALE/DemonAttack-v5")
    ap.add_argument("--screen-size", type=int, default=84)
    ap.add_argument("--noop-max", type=int, default=30)
    ap.add_argument("--stack-frames", type=int, default=4)
    ap.add_argument("--clip-reward", action="store_true")
    ap.add_argument("--terminal-on-life-loss", action="store_true")
    ap.add_argument("--repeat-action-probability", type=float, default=0.0)
    ap.add_argument("--full-action-space", action="store_true")

    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--eval-episodes", type=int, default=5)
    ap.add_argument("--eval-max-steps", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="cpu")

    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--wandb-run-name", type=str, default="")
    ap.add_argument("--wandb-project", type=str, default="demon_attack_queue_distill")
    return ap.parse_args()



def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)

    ppo_module = import_module_from_path(args.ppo_module_path)

    wandb_run = None
    if args.wandb:
        try:
            import wandb

            wandb_run = wandb.init(
                project=args.wandb_project,
                name=(args.wandb_run_name or None),
                config=vars(args),
            )
        except Exception as e:
            print(f"[wandb] failed to init wandb ({e}); continuing without it")
            wandb_run = None

    teacher_rollouts: Optional[Dict[str, Any]] = None

    if args.mode in {"all", "teacher", "shift", "train", "eval"}:
        teacher_rollouts = collect_teacher_rollouts(
            ppo_module=ppo_module,
            teacher_model_path=args.teacher_model_path,
            teacher_rollout_path=args.teacher_rollout_path,
            teacher_episodes=args.teacher_episodes,
            teacher_max_steps=args.teacher_max_steps,
            teacher_obs_mode=args.teacher_obs_mode,
            env_id=args.env_id,
            screen_size=args.screen_size,
            noop_max=args.noop_max,
            clip_reward=args.clip_reward,
            terminal_on_life_loss=args.terminal_on_life_loss,
            repeat_action_probability=args.repeat_action_probability,
            full_action_space=args.full_action_space,
            stack_frames=args.stack_frames,
            seed=args.seed,
        )

    train_results: List[Dict[str, Any]] = []
    eval_results: List[Dict[str, Any]] = []

    if args.mode in {"all", "shift", "train", "eval"}:
        assert teacher_rollouts is not None
        shifted_payloads = {
            int(latency): build_shifted_dataset_for_latency(
                teacher_rollouts=teacher_rollouts,
                latency=int(latency),
                shifted_dataset_root=args.shifted_dataset_root,
            )
            for latency in args.train_latencies
        }
    else:
        shifted_payloads = {}

    if args.mode in {"all", "train", "eval"}:
        for latency in args.train_latencies:
            latency = int(latency)
            result = train_student_for_latency(
                dataset_payload=shifted_payloads[latency],
                student_obs_mode=args.student_obs_mode,
                student_save_dir=args.student_save_dir,
                batch_size=args.batch_size,
                epochs=args.epochs,
                lr=args.lr,
                weight_decay=args.weight_decay,
                grad_clip=args.grad_clip,
                seed=args.seed,
                device=device,
                wandb_run=wandb_run,
            )
            train_results.append(result)

    if args.mode in {"all", "eval"}:
        if not train_results:
            for latency in args.train_latencies:
                student_path = os.path.join(args.student_save_dir, f"student_latency_{int(latency)}.pt")
                if not os.path.exists(student_path):
                    raise FileNotFoundError(f"Missing trained student checkpoint: {student_path}")
                train_results.append({"latency": int(latency), "student_path": student_path})

        for train_row in train_results:
            student_path = train_row["student_path"]
            for eval_latency in args.eval_latencies:
                row = evaluate_student(
                    ppo_module=ppo_module,
                    student_path=student_path,
                    eval_latency=int(eval_latency),
                    eval_episodes=args.eval_episodes,
                    env_id=args.env_id,
                    student_obs_mode=args.student_obs_mode,
                    screen_size=args.screen_size,
                    noop_max=args.noop_max,
                    clip_reward=args.clip_reward,
                    terminal_on_life_loss=args.terminal_on_life_loss,
                    repeat_action_probability=args.repeat_action_probability,
                    full_action_space=args.full_action_space,
                    stack_frames=args.stack_frames,
                    eval_max_steps=args.eval_max_steps,
                    seed=args.seed,
                    device=device,
                )
                eval_results.append(row)
                print(
                    f"[eval] trained_latency={row['trained_latency']} eval_latency={row['eval_latency']} "
                    f"mean_return={row['mean_return']:.3f} std_return={row['std_return']:.3f} "
                    f"mean_length={row['mean_length']:.1f}"
                )
                if wandb_run is not None:
                    wandb_run.log({
                        "eval/trained_latency": row["trained_latency"],
                        "eval/eval_latency": row["eval_latency"],
                        "eval/mean_return": row["mean_return"],
                        "eval/std_return": row["std_return"],
                        "eval/mean_length": row["mean_length"],
                    })

    final_payload = {
        "teacher_rollout_path": args.teacher_rollout_path,
        "shifted_dataset_root": args.shifted_dataset_root,
        "student_save_dir": args.student_save_dir,
        "train_results": train_results,
        "eval_results": eval_results,
    }
    save_json(final_payload, args.eval_json_path)
    print(f"[done] wrote eval results to {args.eval_json_path}")

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()