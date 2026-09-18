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
from pathlib import Path
import shutil
import sys
import time
from typing import Any
from collections.abc import Mapping

import numpy as np
from torch import distributed as dist

from latency_bench.core.actions import ActionResolver
from latency_bench.core.config import load_config
from latency_bench.eval.config import _apply_eval_latency
from latency_bench.eval.driver import run_from_config
from latency_bench.policy.starvla import (
    apply_starvla_model_input_config,
    build_live_starvla_policy,
)
from latency_bench.policy.starvla_prompts import load_latency_prompt_map

# eval_core lives in the starVLA submodule (editable-installed).
from starVLA.training.rl_games import RlGamesEvalRunner
from starVLA.training.rl_games.eval_core import EvalResult


REPO_ROOT = Path(__file__).resolve().parents[2]
BASE_CONFIGS = {
    "flappy": REPO_ROOT / "configs" / "base" / "flappy.yaml",
    "demon_attack": REPO_ROOT / "configs" / "base" / "demon_attack.yaml",
    "deadly_corridor": REPO_ROOT / "configs" / "base" / "deadly_corridor.yaml",
    "asterix": REPO_ROOT / "configs" / "base" / "asterix.yaml",
    "atlantis": REPO_ROOT / "configs" / "base" / "atlantis.yaml",
}


def _plain_mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    try:
        from omegaconf import OmegaConf

        if OmegaConf.is_config(value):
            converted = OmegaConf.to_container(value, resolve=True)
            return dict(converted or {}) if isinstance(converted, Mapping) else {}
    except Exception:
        pass
    return dict(value) if isinstance(value, Mapping) else {}


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
        entry = load_latency_prompt_map(latency_prompt_map_path)[str(latency_steps)]
        if "latency_raw_frames" in entry:
            return entry["latency_raw_frames"]
        return entry["latency"]

    return latency_steps * round(config["env"]["env_fps"] / config["env"]["obs_fps"])


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
    eval_max_steps: int,
    latency_steps: int,
    eval_parallel_envs: int,
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
        task_contract = deepcopy(model_cfg["datasets"]["vla_data"]["gymnasium_task_contract"])
        env_config = {"name": "gymnasium", **task_contract}
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
            history_frames = max(0, int(transform_cfg.get("history_frames", 5)))
        elif image_transform == "demon_attack_ghost_trail":
            if env_name != "demon_attack":
                raise ValueError("image_transform=demon_attack_ghost_trail is only supported for env=demon_attack")
            history_frames = max(0, int(transform_cfg.get("history_frames", 30)))
        else:
            raise ValueError(f"Unsupported rl_games.env_eval.image_transform={image_transform!r}")
        frame_stack_min = history_frames if image_transform == "demon_attack_ghost_trail" else history_frames + 1
        config["env"]["frame_stack"] = max(int(config["env"].get("frame_stack", 1) or 1), frame_stack_min)
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
        "checkpoint_path": str(output_dir),  # metadata only; live model is injected
        "device": "cuda",
    }
    if str(config["env"].get("name", "")) == "gymnasium":
        data_cfg = model_cfg["datasets"]["vla_data"]
        policy["task_manifest_path"] = str(
            Path(data_cfg["data_root_dir"]) / data_cfg["data_mix"] / "manifest.json"
        )
        task_contract = deepcopy(data_cfg["gymnasium_task_contract"])
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

    config["evaluation"] = {
        **config.get("evaluation", {}),
        "eval_episodes": int(num_episodes),
        "eval_parallel_envs": max(1, int(eval_parallel_envs)),
        "eval_max_steps": int(eval_max_steps),
        "eval_deterministic": True,
    }
    # We drive one fixed latency per call, so suites / sweeps must not re-expand.
    config["evaluation"].pop("eval_suites", None)
    config["evaluation"].pop("eval_latency_values", None)

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
        return int(getattr(env_eval, "image_size", 224) or 224)

    def _eval_parallel_envs(self, stage: str, task: str | None = None) -> int:
        # Number of envs evaluated in parallel (batched policy forward). Per-stage
        # override wins, else the global env_eval value, else 1 (sequential).
        # Settable from the launch script as rl_games.env_eval.eval_parallel_envs=N
        # (or per-stage rl_games.env_eval.{mid_train,post_train}.eval_parallel_envs).
        stage_cfg = self._task_stage_cfg(task, stage)
        if stage_cfg is not None and getattr(stage_cfg, "eval_parallel_envs", None) is not None:
            return max(1, int(stage_cfg.eval_parallel_envs))
        return max(1, int(getattr(self.cfg.rl_games.env_eval, "eval_parallel_envs", 1) or 1))

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

    def _model_cfg_from_training_cfg(self) -> dict[str, Any]:
        from omegaconf import OmegaConf

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
    ) -> dict[str, Any]:
        """Fold latency_bench EpisodeMetrics into eval_core's per-latency dict."""
        ordered = sorted(metrics_list, key=lambda m: int(m.episode_id))
        rewards = [float(m.episode_return) for m in ordered]
        lengths = [int(m.survival_steps) for m in ordered]
        indices = [int(m.episode_id) for m in ordered]
        return {
            "latency": int(latency),
            "num_episodes": len(rewards),
            "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
            "mean_length": float(np.mean(lengths)) if lengths else 0.0,
            "std_reward": float(np.std(rewards)) if rewards else 0.0,
            "std_length": float(np.std(lengths)) if lengths else 0.0,
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
        env_eval = self.cfg.rl_games.env_eval
        eval_seed = int(getattr(env_eval, "seed", getattr(self.cfg, "seed", 42)) or 42)

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
            shard_count=max(1, int(shard_count)),
        )
        for task_name, latency in rollout_plan:
            num_episodes = self._num_episodes(stage=stage, task=task_name)
            episode_indices = [
                episode
                for episode in range(num_episodes)
                if episode % max(1, int(shard_count)) == int(shard_rank)
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
                eval_max_steps=self._max_steps_per_episode(stage=stage, task=task_name),
                latency_steps=int(latency),
                eval_parallel_envs=self._eval_parallel_envs(stage=stage, task=task_name),
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
            policy = build_live_starvla_policy(
                framework=model,
                model_cfg=model_cfg,
                config=lb_config,
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
                metrics_list, latency=int(latency), eval_seed=eval_seed
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
