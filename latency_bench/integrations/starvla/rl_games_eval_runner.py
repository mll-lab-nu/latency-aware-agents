"""Drive starVLA mid/post-train eval through latency_bench instead of eval_core.

``LatencyBenchRlGamesEvalRunner`` is a drop-in for
``starVLA.training.rl_games.RlGamesEvalRunner``: same constructor and the same
``run / merge_results / save / is_enabled / interval_steps`` surface, returning
the same ``EvalResult`` shape. It only overrides ``run()`` — instead of
eval_core's own rollout loop it builds a latency_bench config per (task,
latency), wraps the trainer's *live* model with
:func:`latency_bench.policy.starvla.build_live_starvla_policy` (no checkpoint
reload), and runs :func:`latency_bench.eval.driver.run_from_config`. The
returned per-episode metrics are folded back into eval_core's per-latency dict
so wandb keys, distributed merge, and best-checkpoint scoring are unchanged.

Why this matches eval_core's no-reload behaviour: eval_core never loads a model
either — it calls ``model.predict_action`` on the live object. latency_bench
only reloaded because its ``starvla`` policy factory builds a
``PolicyServerWrapper`` from a checkpoint; the live wrapper removes that.

Note: the resulting numbers are the latency_bench ("corrected") numbers, not a
clone of eval_core — preprocessing differs by design (PIL BICUBIC to match
training vs eval_core's cv2.INTER_AREA / BILINEAR).
"""
from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path
from latency_bench.envs.factory import environment_integration
import shutil
import sys
import time
from typing import Any
from collections.abc import Mapping

import numpy as np
from omegaconf import OmegaConf
from torch import distributed as dist

from latency_bench.core.actions import ActionResolver
from latency_bench.core.clock import EnvClock
from latency_bench.core.config import load_config
from latency_bench.envs.tasks.contracts import canonical_gymnasium_contract
from latency_bench.eval.config import _apply_eval_latency
from latency_bench.eval.driver import run_from_config
from latency_bench.envs.tasks.contracts import gymnasium_task_contract
from latency_bench.policy._starvla_input import apply_starvla_model_input_config
from latency_bench.policy.starvla import build_live_starvla_policy
from latency_bench.policy.starvla_prompts import load_latency_prompt_map
from latency_bench.utils.paths import REPO_ROOT

# eval_core lives in the vendored starVLA directory (editable-installed).
from starVLA.training.rl_games import RlGamesEvalRunner
from starVLA.training.rl_games.eval_core import EvalResult


BASE_CONFIGS = {
    "flappy": REPO_ROOT / "configs" / "base" / "flappy.yaml",
    "demon_attack": REPO_ROOT / "configs" / "base" / "demon_attack.yaml",
    "deadly_corridor": REPO_ROOT / "configs" / "base" / "deadly_corridor.yaml",
    "asterix": REPO_ROOT / "configs" / "base" / "asterix.yaml",
    "atlantis": REPO_ROOT / "configs" / "base" / "atlantis.yaml",
    "air_raid": REPO_ROOT / "configs" / "base" / "air_raid.yaml",
}


def _plain_mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if OmegaConf.is_config(value):
        value = OmegaConf.to_container(value, resolve=True)
    return dict(value)


def _obs_stride_raw_frames(env_cfg: Mapping[str, Any]) -> int:
    return EnvClock(
        env_fps=float(env_cfg["env_fps"]),
        obs_fps=float(env_cfg["obs_fps"]),
    ).obs_stride_raw_frames


def _cfg_get(container: Any, key: str, default_value: Any = None) -> Any:
    if container is None:
        return default_value
    if isinstance(container, Mapping):
        return container.get(key, default_value)
    try:
        return container[key]
    except (AttributeError, KeyError, TypeError):
        return getattr(container, key, default_value)


def _starvla_latency_label_to_raw_frames(
    config: dict[str, Any],
    latency_steps: int,
    latency_prompt_map_path: str | None,
) -> int:
    if latency_prompt_map_path is not None:
        prompt_map = load_latency_prompt_map(latency_prompt_map_path)
        key = str(latency_steps)
        if key not in prompt_map:
            available = ", ".join(sorted(prompt_map.keys(), key=lambda x: int(x))[:20])
            raise KeyError(
                f"Latency prompt key {key!r} not found in {latency_prompt_map_path}. "
                f"Available keys include: {available}"
            )
        entry = prompt_map[key]
        if "latency_raw_frames" in entry:
            return entry["latency_raw_frames"]
        return entry["latency"]

    return latency_steps * round(config["env"]["env_fps"] / config["env"]["obs_fps"])


def _gymnasium_cross_task_action_dim(config: dict[str, Any], action_layout: str | None) -> int:
    task_name = str(config["env"].get("name", ""))
    layout = str(action_layout or "").strip().lower()
    if task_name == "asterix" and layout in {
        "factorized_6",
        "factorized6",
        "asterix_factorized_6",
        "asterix_factorized6",
    }:
        return 6
    return len(config["env"].get("action_order", []))


class _EvalProgressReporter:
    def __init__(
        self,
        *,
        output_dir: str | Path,
        stage: str,
        step: int,
        total_episode_runs: int,
        shard_rank: int,
        shard_count: int,
    ) -> None:
        self.stage = stage
        self.step = int(step)
        self.total_episode_runs = int(total_episode_runs)
        self.shard_rank = int(shard_rank)
        self.shard_count = int(shard_count)
        self.local_completed = 0
        self.last_printed = 0
        self.distributed = (
            self.shard_count > 1
            and dist.is_available()
            and dist.is_initialized()
        )
        self.enabled = self.shard_count == 1 or self.distributed
        self.progress_dir = (
            Path(output_dir) / "latency_bench_eval" / stage / f"step_{int(step)}" / "_progress"
        )
        if self.distributed and self.shard_rank == 0:
            if self.progress_dir.exists():
                shutil.rmtree(self.progress_dir)
            self.progress_dir.mkdir(parents=True, exist_ok=True)
            for rank in range(self.shard_count):
                self._write_rank_progress(rank=rank, completed=0)
        if self.distributed:
            dist.barrier()

    def on_episode_complete(self, _episode_metrics: Any) -> None:
        self.local_completed += 1
        if not self.enabled:
            return
        if not self.distributed:
            self._print(self.local_completed)
            return
        self._write_rank_progress(rank=self.shard_rank, completed=self.local_completed)
        if self.shard_rank == 0:
            self._print_global_progress()

    def finish(self) -> None:
        if not self.distributed or self.shard_rank != 0:
            return
        while self.last_printed < self.total_episode_runs:
            completed = self._global_completed()
            if completed > self.last_printed:
                self._print(completed)
            if completed >= self.total_episode_runs:
                return
            time.sleep(1.0)

    def _write_rank_progress(self, *, rank: int, completed: int) -> None:
        self.progress_dir.mkdir(parents=True, exist_ok=True)
        path = self.progress_dir / f"rank_{int(rank)}.json"
        tmp_path = self.progress_dir / f"rank_{int(rank)}.json.tmp"
        tmp_path.write_text(json.dumps({"completed": int(completed)}), encoding="utf-8")
        tmp_path.replace(path)

    def _global_completed(self) -> int:
        completed = 0
        for rank in range(self.shard_count):
            path = self.progress_dir / f"rank_{rank}.json"
            completed += int(json.loads(path.read_text(encoding="utf-8"))["completed"])
        return completed

    def _print_global_progress(self) -> None:
        completed = self._global_completed()
        if completed > self.last_printed:
            self._print(completed)

    def _print(self, completed: int) -> None:
        self.last_printed = int(completed)
        print(
            f"[bench] {self.stage} step={self.step} "
            f"episodes {int(completed)}/{self.total_episode_runs} done",
            file=sys.stderr,
            flush=True,
        )


def build_latency_bench_config_for_rl_games(
    *,
    env_name: str,
    model_cfg: Mapping[str, Any],
    seed: int,
    image_size: int,
    num_episodes: int,
    max_env_steps: int,
    latency_steps: int,
    parallel_envs: int,
    unnorm_key: str | None,
    latency_prompt_map_path: str | None,
    image_transform_config: dict[str, Any] | None,
    actions: list[Any] | None,
    output_dir: str | Path,
    gymnasium_make_kwargs: Mapping[str, Any] | None = None,
    prompt_mode: str | None = None,
    action_layout: str | None = None,
    multibinary_threshold: float | None = None,
) -> dict[str, Any]:
    """Translate one starVLA rl_games (task, latency) eval into a lb config.

    Starts from ``configs/base/<env>.yaml`` (which carries env_fps, scheduler,
    and rendering defaults) and overrides only what the eval needs: a single
    fixed latency, starVLA policy fields, the 224 obs resize, episode/step
    budgets, and disabled artifacts.
    """
    if env_name == "gymnasium":
        task_contract = canonical_gymnasium_contract(model_cfg["datasets"]["vla_data"]["gymnasium_task_contract"])
        env_config = {**task_contract, "name": task_contract["task_name"]}
        env_config.pop("task_name")
        env_config["make_kwargs"] = {
            **env_config["make_kwargs"],
            **dict(gymnasium_make_kwargs or {}),
        }
        if "action_labels" in task_contract:
            action_labels = list(task_contract["action_labels"])
            env_config.update(
                {
                    "action_order": action_labels,
                    "action_map": dict(zip(action_labels, task_contract["action_values"])),
                    "noop_action": action_labels[task_contract["noop_action_id"]],
                }
            )
        config = {
            "experiment": {},
            "backend": {"run_mode": "eval"},
            "executor": {"mode": "simulated"},
            "env": env_config,
            "latency": {"method": "zero", "add_latency_info": False},
            "scheduler": {
                "hold_policy": "hold",
                "ordering_policy": "issue_order_fifo",
            },
            "policy": {"type": "random"},
            "evaluation": {},
            "logging": {
                "output_dir": str(output_dir),
                "video": {"enabled": False, "num_bins": 1},
                "save_step_records": False,
                "save_action_records": False,
                "save_latency_records": False,
            },
        }
    else:
        config = deepcopy(load_config(BASE_CONFIGS[env_name]))

    config["experiment"] = {
        **config.get("experiment", {}),
        "name": f"starvla_{env_name}_train_eval",
        "seed": int(seed),
    }
    config["executor"] = {"mode": "simulated"}
    config.setdefault("backend", {})["run_mode"] = "eval"

    config["env"] = {**config["env"], "obs_resize": [int(image_size), int(image_size)]}
    transform_cfg = dict(image_transform_config or {})
    image_transform = str(transform_cfg.get("image_transform", "raw_rgb") or "raw_rgb").strip().lower()
    if image_transform not in {"", "none", "raw", "raw_rgb"}:
        if image_transform == "flappy_ghost_trail":
            if env_name != "flappy":
                raise ValueError("image_transform=flappy_ghost_trail is only supported for env=flappy")
            history_frames = int(transform_cfg.get("history_frames", 5))
        elif image_transform == "demon_attack_ghost_trail":
            if env_name != "demon_attack":
                raise ValueError("image_transform=demon_attack_ghost_trail is only supported for env=demon_attack")
            history_frames = int(transform_cfg.get("history_frames", 30))
        else:
            raise ValueError(f"Unsupported rl_games.env_eval.image_transform={image_transform!r}")
        frame_stack_min = history_frames if image_transform == "demon_attack_ghost_trail" else history_frames + 1
        config["env"]["frame_stack"] = max(int(config["env"].get("frame_stack", 1)), frame_stack_min)
    apply_starvla_model_input_config(config, model_cfg=model_cfg, image_transform=image_transform)

    # Single fixed latency for this rollout. Prompt maps carry exact source
    # raw-frame latency; legacy runs without one use observation-step labels.
    latency_raw_frames = _starvla_latency_label_to_raw_frames(
        config,
        int(latency_steps),
        latency_prompt_map_path,
    )
    latency_ms = (
        latency_raw_frames
        * 1000.0
        / float(config["env"]["env_fps"])
    )
    _apply_eval_latency(config, latency_raw_frames, latency_ms)

    policy: dict[str, Any] = {
        "type": "starvla",
        "source": "local", "root": str(output_dir), "checkpoint": None,
    }
    if environment_integration(config) == "gymnasium":
        data_cfg = model_cfg["datasets"]["vla_data"]
        if env_name == "gymnasium":
            task_contract = deepcopy(data_cfg["gymnasium_task_contract"])
            policy["gymnasium_active_action_dim"] = len(
                task_contract["action_labels"] if "action_labels" in task_contract
                else task_contract["action_space"]["labels"]
            )
        else:
            task_contract = gymnasium_task_contract(config["env"])
            policy["gymnasium_cross_task_eval"] = True
            policy["gymnasium_active_action_dim"] = _gymnasium_cross_task_action_dim(config, action_layout)
        policy["gymnasium_training_task_contract"] = task_contract
    if unnorm_key is not None:
        policy["unnorm_key"] = unnorm_key
    if latency_prompt_map_path:
        policy["latency_prompt_map_path"] = str(latency_prompt_map_path)
        policy["latency_prompt_key"] = int(latency_steps)
    if prompt_mode:
        policy["prompt_mode"] = str(prompt_mode)
    # Deadly Corridor multibinary head: pass layout + threshold so the live
    # wrapper threshold-decodes (matching eval_core) and the factory selects the
    # MultiBinary gymnasium env. Only meaningful for env_name=deadly_corridor.
    if action_layout:
        policy["action_layout"] = str(action_layout)
    if multibinary_threshold is not None:
        policy["multibinary_threshold"] = float(multibinary_threshold)
    if image_transform not in {"", "none", "raw", "raw_rgb"}:
        policy["image_transform_config"] = {
            "image_transform": image_transform,
            "history_frames": history_frames,
            "gamma": float(transform_cfg.get("gamma", 1.3)),
            "min_alpha": int(transform_cfg.get("min_alpha", 35)),
            "ground_fraction": float(transform_cfg.get("ground_fraction", 0.22)),
            "scroll_px_per_step": float(transform_cfg.get("scroll_px_per_step", 4.0)),
        }
    if actions:
        policy["actions"] = list(actions)
    config["policy"] = policy

    raw_frame_eval_max_steps = int(max_env_steps) * _obs_stride_raw_frames(config["env"])
    config["evaluation"] = {
        **config.get("evaluation", {}),
        "episodes": int(num_episodes),
        "max_env_steps": int(raw_frame_eval_max_steps),
        "eval_max_decision_steps": int(max_env_steps),
    }
    # We drive one fixed latency per call, so suites / sweeps must not re-expand.
    config["evaluation"]["suites"] = []
    config["executor"]["parallel_envs"] = parallel_envs
    config["executor"]["inference_devices"] = ["cuda"]

    logging_cfg = dict(config["logging"])
    logging_cfg["output_dir"] = str(output_dir)
    logging_cfg["save_step_records"] = False
    logging_cfg["save_action_records"] = False
    logging_cfg["save_latency_records"] = False
    logging_cfg["video"] = {**logging_cfg.get("video", {}), "enabled": False}
    config["logging"] = logging_cfg
    return config


class LatencyBenchRlGamesEvalRunner(RlGamesEvalRunner):
    """eval_core-compatible runner that evaluates via latency_bench in-process."""
    def _image_size(self) -> int:
        env_eval = self.cfg.rl_games.env_eval
        return int(getattr(env_eval, "image_size", 224))

    def _parallel_envs(self, stage: str, task: str | None = None) -> int:
        # Stage overrides take precedence over the global setting.
        stage_cfg = self._task_stage_cfg(task, stage)
        if stage_cfg is not None and getattr(stage_cfg, "eval_parallel_envs", None) is not None:
            return int(stage_cfg.eval_parallel_envs)
        return int(getattr(self.cfg.rl_games.env_eval, "eval_parallel_envs", 1))

    def _image_transform_config(self, stage: str, task: str | None = None) -> dict[str, Any] | None:
        env_eval = self.cfg.rl_games.env_eval
        task_cfg = self._cross_task_cfg(task) if task is not None else None
        stage_cfg = self._task_stage_cfg(task, stage)
        image_transform = str(
            _cfg_get(
                stage_cfg,
                "image_transform",
                _cfg_get(task_cfg, "image_transform", _cfg_get(env_eval, "image_transform", "raw_rgb")),
            )
            or "raw_rgb"
        ).strip().lower()
        if image_transform in {"", "none", "raw", "raw_rgb"}:
            return None

        ghost_cfg = {
            **_plain_mapping(_cfg_get(env_eval, "ghost_trail", None)),
            **_plain_mapping(_cfg_get(task_cfg, "ghost_trail", None)),
            **_plain_mapping(_cfg_get(stage_cfg, "ghost_trail", None)),
        }
        return {
            "image_transform": image_transform,
            "history_frames": int(ghost_cfg.get("history_frames", 5)),
            "gamma": float(ghost_cfg.get("gamma", 1.3)),
            "min_alpha": int(ghost_cfg.get("min_alpha", 35)),
            "ground_fraction": float(ghost_cfg.get("ground_fraction", 0.22)),
            "scroll_px_per_step": float(ghost_cfg.get("scroll_px_per_step", 4.0)),
        }

    def _deadly_action_layout(self, task: str | None = None) -> str | None:
        """Resolve the Deadly Corridor action layout (only meaningful for deadly).

        Source order: rl_games.env_eval.deadly.action_layout, else the deadly
        entry in cross_task.train_tasks. Returns None for non-deadly tasks so the
        flag is never passed for flappy/demon.
        """
        if task is not None and str(task) != "deadly_corridor":
            return None
        deadly_cfg = getattr(self.cfg.rl_games.env_eval, "deadly", None)
        layout = _cfg_get(deadly_cfg, "action_layout", None)
        if not layout:
            cross_cfg = getattr(self.cfg.rl_games, "cross_task", None)
            train_tasks = getattr(cross_cfg, "train_tasks", None) if cross_cfg is not None else None
            for entry in (train_tasks or []):
                if str(_cfg_get(entry, "name", "")) == "deadly_corridor":
                    layout = _cfg_get(entry, "action_layout", None)
                    if layout:
                        break
        layout = str(layout or "").strip().lower()
        return layout or None

    def _task_action_layout(self, task: str | None = None) -> str | None:
        if task == "deadly_corridor":
            return self._deadly_action_layout(task)
        if task != "asterix":
            return None
        asterix_cfg = getattr(self.cfg.rl_games.env_eval, "asterix", None)
        layout = _cfg_get(asterix_cfg, "action_layout", None)
        if not layout:
            cross_cfg = getattr(self.cfg.rl_games, "cross_task", None)
            train_tasks = getattr(cross_cfg, "train_tasks", None) if cross_cfg is not None else None
            for entry in (train_tasks or []):
                if str(_cfg_get(entry, "name", "")) == "asterix":
                    layout = _cfg_get(entry, "action_layout", None)
                    if layout:
                        break
        layout = str(layout or "").strip().lower()
        return layout or None

    def _deadly_multibinary_threshold(self) -> float | None:
        deadly_cfg = getattr(self.cfg.rl_games.env_eval, "deadly", None)
        threshold = _cfg_get(deadly_cfg, "multibinary_threshold", None)
        return None if threshold is None else float(threshold)

    def _prompt_mode(self, stage: str, task: str | None = None) -> str | None:
        env_eval = self.cfg.rl_games.env_eval
        task_cfg = self._cross_task_cfg(task) if task is not None else None
        stage_cfg = self._task_stage_cfg(task, stage)
        mode = _cfg_get(
            stage_cfg,
            "prompt_mode",
            _cfg_get(task_cfg, "prompt_mode", _cfg_get(env_eval, "prompt_mode", None)),
        )
        mode = str(mode or "").strip().lower()
        return None if mode in {"", "default", "none", "raw"} else mode

    def _get_latency_values(self, stage: str, task: str | None = None) -> list[int]:
        stage_cfg = self._task_stage_cfg(task, stage)
        if bool(_cfg_get(stage_cfg, "latencies_from_prompt_map", False)):
            prompt_map_path = self._prompt_map_path(task)
            if not prompt_map_path:
                raise ValueError(
                    f"rl_games.env_eval.{stage}.latencies_from_prompt_map=true "
                    "but no latency prompt map path was configured"
                )
            prompt_map = load_latency_prompt_map(prompt_map_path)
            if not prompt_map:
                raise ValueError(f"Latency prompt map is empty: {prompt_map_path}")
            return sorted(int(key) for key in prompt_map.keys())
        return super()._get_latency_values(stage=stage, task=task)

    def _model_cfg_from_training_cfg(self) -> dict[str, Any]:
        return OmegaConf.to_container(self.cfg, resolve=True)

    def _base_actions(self, env_name: str) -> list[Any] | None:
        if env_name == "gymnasium":
            return None
        base = load_config(BASE_CONFIGS[env_name])
        policy_cfg = base.get("policy", {})
        return list(policy_cfg["actions"]) if "actions" in policy_cfg else None

    def _gymnasium_make_kwargs(self) -> dict[str, Any] | None:
        gymnasium_cfg = _cfg_get(self.cfg.rl_games.env_eval, "gymnasium", None)
        make_kwargs = _cfg_get(gymnasium_cfg, "make_kwargs", None)
        return _plain_mapping(make_kwargs) or None

    def _fold_episode_metrics(
        self,
        metrics_list: list[Any],
        *,
        latency: int,
        eval_seed: int,
        obs_stride_raw_frames: int,
    ) -> dict[str, Any]:
        """Fold latency_bench EpisodeMetrics into eval_core's per-latency dict."""
        ordered = sorted(metrics_list, key=lambda m: int(m.episode_id))
        rewards = [float(m.episode_return) for m in ordered]
        lengths = [
            int(math.ceil(int(m.survival_steps) / obs_stride_raw_frames))
            for m in ordered
        ]
        indices = [int(m.episode_id) for m in ordered]
        return {
            "latency": int(latency),
            "num_episodes": len(rewards),
            "mean_reward": float(np.mean(rewards)),
            "mean_length": float(np.mean(lengths)),
            "std_reward": float(np.std(rewards)),
            "std_length": float(np.std(lengths)),
            "episode_rewards": rewards,
            "episode_lengths": lengths,
            "decoded_action_hist": {},
            "fixed_episode_seeds": True,
            "eval_seed": int(eval_seed),
            "episode_seeds": [None] * len(rewards),
            "episode_indices": indices,
        }

    def run(
        self,
        model,
        step: int,
        stage: str = "mid_train",
        shard_rank: int = 0,
        shard_count: int = 1,
        save: bool = True,
        episode_seed_overrides: dict[str, dict[int, int | None]] | None = None,
    ) -> EvalResult:
        del episode_seed_overrides  # seeds are derived from latency_bench episode ids
        tasks = self._get_tasks()
        image_size = self._image_size()
        model_cfg = self._model_cfg_from_training_cfg()
        normalization_statistics = None
        env_eval = self.cfg.rl_games.env_eval
        eval_seed = int(getattr(env_eval, "seed", getattr(self.cfg, "seed", 42)))

        per_latency: dict[str, dict] = {}
        rollout_plan = [
            (task_name, latency)
            for task_name in tasks
            if getattr(self._task_stage_cfg(task_name, stage), "enabled", True)
            for latency in self._get_latency_values(stage=stage, task=task_name)
        ]
        progress = _EvalProgressReporter(
            output_dir=self.output_dir,
            stage=stage,
            step=int(step),
            total_episode_runs=sum(
                self._num_episodes(stage=stage, task=task_name)
                for task_name, _latency in rollout_plan
            ),
            shard_rank=int(shard_rank),
            shard_count=shard_count,
        )
        for task_name, latency in rollout_plan:
            num_episodes = self._num_episodes(stage=stage, task=task_name)
            episode_indices = [
                episode
                for episode in range(num_episodes)
                if episode % shard_count == shard_rank
            ]
            if not episode_indices:
                continue

            output_dir = Path(self.output_dir) / "latency_bench_eval" / stage / f"step_{int(step)}" / task_name / f"latency_{int(latency)}"
            lb_config = build_latency_bench_config_for_rl_games(
                env_name=task_name,
                model_cfg=model_cfg,
                seed=eval_seed,
                image_size=image_size,
                num_episodes=num_episodes,
                max_env_steps=self._max_steps_per_episode(stage=stage, task=task_name),
                latency_steps=int(latency),
                parallel_envs=self._parallel_envs(stage=stage, task=task_name),
                unnorm_key=getattr(self.cfg.rl_games.env_eval, "unnorm_key", None),
                latency_prompt_map_path=self._prompt_map_path(task_name),
                image_transform_config=self._image_transform_config(stage=stage, task=task_name),
                gymnasium_make_kwargs=self._gymnasium_make_kwargs(),
                prompt_mode=self._prompt_mode(stage=stage, task=task_name),
                actions=self._base_actions(task_name),
                output_dir=output_dir,
                action_layout=self._task_action_layout(task_name),
                multibinary_threshold=self._deadly_multibinary_threshold(),
            )
            if environment_integration(lb_config) == "gymnasium" and normalization_statistics is None:
                normalization_statistics = json.loads(
                    (Path(self.output_dir) / "dataset_statistics.json").read_text(encoding="utf-8")
                )
            policy = build_live_starvla_policy(
                framework=model,
                model_cfg=model_cfg,
                config=lb_config,
                normalization_statistics=normalization_statistics,
                action_resolver=ActionResolver(lb_config["env"]),
            )
            metrics_list = run_from_config(
                lb_config,
                write_summary=False,
                episode_ids=episode_indices,
                policy=policy,
                on_episode_complete=progress.on_episode_complete,
            )
            per_latency[f"{task_name}/latency_{latency}"] = self._fold_episode_metrics(
                metrics_list,
                latency=int(latency),
                eval_seed=eval_seed,
                obs_stride_raw_frames=_obs_stride_raw_frames(lb_config["env"]),
            )
        progress.finish()

        aggregate = self._aggregate_result(
            per_latency=per_latency,
            stage=stage,
            step=step,
            cfg=self.cfg,
            task_count=len(tasks),
        )
        aggregate["distributed_eval"] = int(shard_count) > 1
        result = EvalResult(per_latency=per_latency, aggregate=aggregate)
        if save:
            self.save(result=result, step=step, stage=stage)
        return result
