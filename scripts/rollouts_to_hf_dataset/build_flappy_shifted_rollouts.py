#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import pickle
from pathlib import Path
from typing import Any, Dict, List, Sequence


def _load_pickle(path: str) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


def _is_rollout_episode_dict(x: Any) -> bool:
    return (
        isinstance(x, dict)
        and ("base_observations" in x or "observations" in x)
        and "actions" in x
        and "rewards" in x
        and "dones" in x
    )


def _extract_rollout_episodes(obj: Any) -> List[Dict[str, Any]]:
    if isinstance(obj, (list, tuple)) and len(obj) > 0 and all(_is_rollout_episode_dict(ep) for ep in obj):
        return list(obj)
    if _is_rollout_episode_dict(obj):
        return [obj]
    raise ValueError(
        "Unsupported pickle structure. Expected a rollout pickle containing episode dicts "
        "with base_observations/observations, actions, rewards, dones."
    )


def _normalize_episode(ep: Dict[str, Any]) -> Dict[str, Any]:
    obs = ep.get("base_observations", ep.get("observations"))
    actions = ep.get("actions")
    rewards = ep.get("rewards")
    dones = ep.get("dones")

    if obs is None or actions is None or rewards is None or dones is None:
        raise ValueError("Episode missing one of base_observations/observations, actions, rewards, dones")

    n = len(actions)
    if not (len(obs) == len(rewards) == len(dones) == n):
        raise ValueError(
            f"Inconsistent lengths: obs={len(obs)}, actions={len(actions)}, rewards={len(rewards)}, dones={len(dones)}"
        )

    return {
        "base_observations": list(obs),
        "actions": list(actions),
        "rewards": list(rewards),
        "dones": list(dones),
    }


def _shift_episode(
    ep: Dict[str, Any],
    latency: int,
    reward_mode: str = "truncate",
) -> Dict[str, Any] | None:
    obs = ep["base_observations"]
    actions = ep["actions"]
    rewards = ep["rewards"]

    T = len(actions)
    if latency < 0:
        raise ValueError(f"Latency must be non-negative, got {latency}")
    if latency == 0:
        return {
            "base_observations": list(obs),
            "actions": list(actions),
            "rewards": list(rewards),
            "dones": list(ep["dones"]),
        }
    if T <= latency:
        return None

    new_len = T - latency
    shifted_obs = list(obs[:new_len])
    shifted_actions = list(actions[latency:])

    if reward_mode == "truncate":
        shifted_rewards = list(rewards[:new_len])
    elif reward_mode == "zero":
        shifted_rewards = [0.0 for _ in range(new_len)]
    else:
        raise ValueError(f"Unsupported reward_mode: {reward_mode}")

    shifted_dones = [False] * new_len
    shifted_dones[-1] = True

    return {
        "base_observations": shifted_obs,
        "actions": shifted_actions,
        "rewards": shifted_rewards,
        "dones": shifted_dones,
    }


def _save_pickle(path: str, payload: Any) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)


def _latency_dir(root: str, latency: int) -> Path:
    return Path(root) / f"latency_{int(latency)}"


def _latency_output_path(root: str, latency: int) -> Path:
    return _latency_dir(root, latency) / f"shifted_rollout_latency_{int(latency)}.pkl"


def build_shifted_rollouts(
    teacher_rollout_path: str,
    output_root: str,
    latencies: Sequence[int],
    reward_mode: str,
) -> Dict[str, Any]:
    obj = _load_pickle(teacher_rollout_path)
    episodes = [_normalize_episode(ep) for ep in _extract_rollout_episodes(obj)]

    summary: Dict[str, Any] = {
        "source_teacher_rollout_path": teacher_rollout_path,
        "num_input_episodes": len(episodes),
        "reward_mode": reward_mode,
        "latencies": {},
    }

    for latency in latencies:
        shifted_eps: List[Dict[str, Any]] = []
        skipped = 0
        total_input_steps = 0
        total_output_steps = 0
        input_lengths: List[int] = []
        output_lengths: List[int] = []

        for ep in episodes:
            ep_len = len(ep["actions"])
            total_input_steps += ep_len
            input_lengths.append(ep_len)

            shifted = _shift_episode(ep, latency=latency, reward_mode=reward_mode)
            if shifted is None:
                skipped += 1
                continue

            shifted_eps.append(shifted)
            out_len = len(shifted["actions"])
            total_output_steps += out_len
            output_lengths.append(out_len)

        out_path = _latency_output_path(output_root, latency)
        _save_pickle(str(out_path), shifted_eps)

        meta = {
            "latency": int(latency),
            "reward_mode": reward_mode,
            "num_output_episodes": len(shifted_eps),
            "num_skipped_short_episodes": int(skipped),
            "total_input_steps": int(total_input_steps),
            "total_output_steps": int(total_output_steps),
            "mean_input_length": float(sum(input_lengths) / len(input_lengths)) if input_lengths else 0.0,
            "mean_output_length": float(sum(output_lengths) / len(output_lengths)) if output_lengths else 0.0,
            "output_pickle": str(out_path),
        }

        meta_path = _latency_dir(output_root, latency) / "metadata.json"
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        summary["latencies"][str(latency)] = meta
        print(json.dumps(meta, indent=2))

    summary_path = Path(output_root) / "build_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[info] wrote summary to: {summary_path}")
    return summary


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Create per-latency shifted episodic rollout pickles in the same format as the teacher rollout."
    )
    ap.add_argument("--teacher-rollout-path", type=str, required=True)
    ap.add_argument("--output-root", type=str, required=True)
    ap.add_argument("--latencies", type=int, nargs="+", required=True)
    ap.add_argument("--reward-mode", type=str, default="truncate", choices=["truncate", "zero"])
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_root, exist_ok=True)
    build_shifted_rollouts(
        teacher_rollout_path=args.teacher_rollout_path,
        output_root=args.output_root,
        latencies=args.latencies,
        reward_mode=args.reward_mode,
    )


if __name__ == "__main__":
    main()
