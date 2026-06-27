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
from pathlib import Path
from typing import Any

import numpy as np

from latency_bench.core.actions import ActionResolver
from latency_bench.core.config import load_config
from latency_bench.eval.config import _apply_eval_latency, _latency_steps_to_ms
from latency_bench.eval.driver import run_from_config
from latency_bench.policy.starvla import build_live_starvla_policy

# eval_core lives in the starVLA submodule (editable-installed).
from starVLA.training.rl_games import RlGamesEvalRunner
from starVLA.training.rl_games.eval_core import EvalResult


REPO_ROOT = Path(__file__).resolve().parents[2]
BASE_CONFIGS = {
    "flappy": REPO_ROOT / "configs" / "base" / "flappy.yaml",
    "demon_attack": REPO_ROOT / "configs" / "base" / "demon_attack.yaml",
    "deadly_corridor": REPO_ROOT / "configs" / "base" / "deadly_corridor.yaml",
}


def build_latency_bench_config_for_rl_games(
    *,
    env_name: str,
    seed: int,
    image_size: int,
    num_episodes: int,
    eval_max_steps: int,
    latency_steps: int,
    eval_parallel_envs: int,
    unnorm_key: str | None,
    latency_prompt_map_path: str | None,
    actions: list[Any] | None,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Translate one starVLA rl_games (task, latency) eval into a lb config.

    Starts from ``configs/base/<env>.yaml`` (which carries env_fps, scheduler,
    and rendering defaults) and overrides only what the eval needs: a single
    fixed latency, starVLA policy fields, the 224 obs resize, episode/step
    budgets, and disabled artifacts.
    """
    if env_name not in BASE_CONFIGS:
        raise ValueError(f"Unsupported rl_games eval task for latency_bench: {env_name!r}")
    config = deepcopy(load_config(BASE_CONFIGS[env_name]))

    config["experiment"] = {
        **config.get("experiment", {}),
        "name": f"starvla_{env_name}_train_eval",
        "seed": int(seed),
    }
    config["executor"] = {"mode": "simulated"}
    config.setdefault("backend", {})["run_mode"] = "eval"

    config["env"] = {**config["env"], "obs_resize": [int(image_size), int(image_size)]}

    # Single fixed latency for this rollout (frames -> ms via env_fps); reuse the
    # same zero/fixed mapping the sweep driver uses.
    latency_ms = _latency_steps_to_ms(config, int(latency_steps))
    _apply_eval_latency(config, int(latency_steps), latency_ms)

    policy: dict[str, Any] = {
        "type": "starvla",
        "checkpoint_path": str(output_dir),  # metadata only; live model is injected
        "device": "cuda",
    }
    if unnorm_key is not None:
        policy["unnorm_key"] = unnorm_key
    if latency_prompt_map_path:
        policy["latency_prompt_map_path"] = str(latency_prompt_map_path)
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

    def _model_cfg_from_training_cfg(self) -> dict[str, Any]:
        from omegaconf import OmegaConf

        return OmegaConf.to_container(self.cfg, resolve=True)

    def _base_actions(self, env_name: str) -> list[Any] | None:
        base = load_config(BASE_CONFIGS[env_name])
        policy_cfg = base.get("policy", {})
        return list(policy_cfg["actions"]) if "actions" in policy_cfg else None

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
                seed=eval_seed,
                image_size=image_size,
                num_episodes=num_episodes,
                eval_max_steps=self._max_steps_per_episode(stage=stage, task=task_name),
                latency_steps=int(latency),
                eval_parallel_envs=self._eval_parallel_envs(stage=stage, task=task_name),
                unnorm_key=getattr(self.cfg.rl_games.env_eval, "unnorm_key", None),
                latency_prompt_map_path=self._prompt_map_path(task_name),
                actions=self._base_actions(task_name),
                output_dir=output_dir,
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
            )
            per_latency[f"{task_name}/latency_{latency}"] = self._fold_episode_metrics(
                metrics_list, latency=int(latency), eval_seed=eval_seed
            )

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
