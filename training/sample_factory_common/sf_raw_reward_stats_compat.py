from __future__ import annotations

import inspect
from importlib import metadata
from typing import Any

import numpy as np

from training.atari_episode_stats import (
    TRUE_EPISODE_END_INFO_KEY,
    _apply_raw_full_game_return_to_report,
    _batched_info_for_agent,
    _flatten_numeric_episode_extra_stats,
    _raw_full_game_return_from_info,
    is_true_atari_episode_end,
)


SUPPORTED_SAMPLE_FACTORY_VERSION = "2.1.1"
SUPPORTED_RAW_REWARD_ENV_NAMES = frozenset({"latency_demon_attack"})

_PATCHED_ATTR = "_latency_bench_raw_reward_console_stats_patched"
_ENABLED_ENV_NAMES_ATTR = "_latency_bench_raw_reward_console_stats_env_names"
_ORIGINAL_RECORD_ENV_STEP_ATTR = "_latency_bench_original_record_env_step"
_ORIGINAL_PROCESS_ENV_STEP_ATTR = "_latency_bench_original_process_env_step"

_EXPECTED_RECORD_ENV_STEP_PARAMS = ("self", "reward", "terminated", "truncated", "info", "rollout_step")
_EXPECTED_PROCESS_ENV_STEP_PARAMS = ("self", "rewards", "dones_orig", "infos")


def ensure_sample_factory_raw_reward_console_stats_installed(*, env_name: str) -> bool:
    """Install Sample Factory raw-return episode reporting for supported envs.

    Sample Factory computes console/W&B ``reward/reward`` and best-policy rewards
    from its native episodic report. Demon Attack still trains on clipped rewards,
    so this compat layer patches the episode report boundary only when the SF cfg
    is for the Demon Attack env.
    """

    env_name = str(env_name)
    if env_name not in SUPPORTED_RAW_REWARD_ENV_NAMES:
        return False

    try:
        import sample_factory
        from sample_factory.algo.sampling.batched_sampling import BatchedVectorEnvRunner
        from sample_factory.algo.sampling.non_batched_sampling import ActorState
        from sample_factory.algo.utils.misc import EPISODIC
    except ImportError:
        return False

    _validate_sample_factory_version(sample_factory)
    _patch_non_batched_record_env_step(ActorState, EPISODIC, env_name)
    _patch_batched_process_env_step(BatchedVectorEnvRunner, EPISODIC, env_name)
    return True


def _validate_sample_factory_version(sample_factory_module: Any) -> None:
    try:
        installed_version = metadata.version("sample-factory")
    except metadata.PackageNotFoundError:
        installed_version = str(getattr(sample_factory_module, "__version__", "") or "")

    if installed_version != SUPPORTED_SAMPLE_FACTORY_VERSION:
        raise RuntimeError(
            "Unsupported Sample Factory version for raw reward stats compat: "
            f"expected {SUPPORTED_SAMPLE_FACTORY_VERSION}, got {installed_version or 'unknown'}"
        )


def _validate_signature(callable_obj: Any, *, expected_params: tuple[str, ...], label: str) -> None:
    signature = inspect.signature(callable_obj)
    actual_params = tuple(signature.parameters)
    if actual_params != expected_params:
        raise RuntimeError(
            f"Sample Factory {label} signature changed: expected {expected_params}, got {actual_params}"
        )


def _patch_non_batched_record_env_step(ActorState: Any, episodic_key: str, env_name: str) -> None:
    current = ActorState.record_env_step
    _validate_signature(
        current,
        expected_params=_EXPECTED_RECORD_ENV_STEP_PARAMS,
        label="ActorState.record_env_step",
    )

    if getattr(current, _PATCHED_ATTR, False):
        getattr(current, _ENABLED_ENV_NAMES_ATTR).add(env_name)
        return

    original_record_env_step = current
    enabled_env_names = {env_name}

    def record_env_step_raw_return(self, reward, terminated: bool, truncated: bool, info, rollout_step):
        if not _is_enabled_for_sample_factory_env(self, enabled_env_names):
            return original_record_env_step(self, reward, terminated, truncated, info, rollout_step)

        done = bool(terminated or truncated)
        true_episode_end = is_true_atari_episode_end(terminated, truncated, info)
        if not done or true_episode_end:
            report = original_record_env_step(self, reward, terminated, truncated, info, rollout_step)
            if done and true_episode_end:
                _apply_raw_full_game_return_to_report(report, info, episodic_key)
            return report

        episode_reward = self.last_episode_reward
        duration_before_step = self.last_episode_duration
        original_record_env_step(self, reward, terminated, truncated, info, rollout_step)

        duration_increment = self.env_info.frameskip if self.cfg.summaries_use_frameskip else 1
        self.last_episode_reward = episode_reward
        self.last_episode_duration = duration_before_step + duration_increment
        return None

    setattr(record_env_step_raw_return, _PATCHED_ATTR, True)
    setattr(record_env_step_raw_return, _ENABLED_ENV_NAMES_ATTR, enabled_env_names)
    setattr(record_env_step_raw_return, _ORIGINAL_RECORD_ENV_STEP_ATTR, original_record_env_step)
    ActorState.record_env_step = record_env_step_raw_return


def _patch_batched_process_env_step(BatchedVectorEnvRunner: Any, episodic_key: str, env_name: str) -> None:
    current = BatchedVectorEnvRunner._process_env_step
    _validate_signature(
        current,
        expected_params=_EXPECTED_PROCESS_ENV_STEP_PARAMS,
        label="BatchedVectorEnvRunner._process_env_step",
    )

    if getattr(current, _PATCHED_ATTR, False):
        getattr(current, _ENABLED_ENV_NAMES_ATTR).add(env_name)
        return

    original_process_env_step = current
    enabled_env_names = {env_name}

    def process_env_step_raw_return(self, rewards, dones_orig, infos):
        if not _is_enabled_for_sample_factory_env(self, enabled_env_names):
            return original_process_env_step(self, rewards, dones_orig, infos)

        done_indices = _done_indices(dones_orig)
        if not done_indices:
            return original_process_env_step(self, rewards, dones_orig, infos)

        report_dones = _clone_done_mask(dones_orig)
        for agent_idx in done_indices:
            info = _batched_info_for_agent(infos, int(agent_idx))
            if not _is_true_batched_episode_end(info):
                report_dones[agent_idx] = False

        reports = original_process_env_step(self, rewards, report_dones, infos)
        if reports:
            finished_agents = _done_indices(report_dones)
            raw_returns = _raw_full_game_returns_for_finished_agents(infos, finished_agents)
            extra_stats = _episode_extra_stats_for_finished_agents(infos, finished_agents)
            for report in reports:
                episodic_stats = report.get(episodic_key)
                if isinstance(episodic_stats, dict):
                    if raw_returns is not None:
                        episodic_stats["reward"] = raw_returns
                    if extra_stats:
                        episodic_stats.update(extra_stats)
        return reports

    setattr(process_env_step_raw_return, _PATCHED_ATTR, True)
    setattr(process_env_step_raw_return, _ENABLED_ENV_NAMES_ATTR, enabled_env_names)
    setattr(process_env_step_raw_return, _ORIGINAL_PROCESS_ENV_STEP_ATTR, original_process_env_step)
    BatchedVectorEnvRunner._process_env_step = process_env_step_raw_return


def _is_enabled_for_sample_factory_env(obj: Any, enabled_env_names: set[str]) -> bool:
    cfg = getattr(obj, "cfg", None)
    env_name = _cfg_value(cfg, "env")
    if env_name is None:
        env_name = _cfg_value(cfg, "env_name")
    if env_name is None:
        env_name = _cfg_value(cfg, "full_env_name")
    return env_name in enabled_env_names


def _cfg_value(cfg: Any, key: str) -> str | None:
    if cfg is None:
        return None
    if isinstance(cfg, dict):
        value = cfg.get(key)
    else:
        value = getattr(cfg, key, None)
    if value is None:
        return None
    return str(value)


def _done_indices(done_mask: Any) -> list[int]:
    done_array = _as_numpy_array(done_mask).astype(bool, copy=False).reshape(-1)
    return [int(idx) for idx, done in enumerate(done_array.tolist()) if done]


def _as_numpy_array(value: Any) -> np.ndarray:
    item = value
    if hasattr(item, "detach") and callable(item.detach):
        item = item.detach()
    if hasattr(item, "cpu") and callable(item.cpu):
        item = item.cpu()
    if hasattr(item, "numpy") and callable(item.numpy):
        item = item.numpy()
    return np.asarray(item)


def _clone_done_mask(done_mask: Any) -> Any:
    if hasattr(done_mask, "clone") and callable(done_mask.clone):
        return done_mask.clone()
    if hasattr(done_mask, "copy") and callable(done_mask.copy):
        return done_mask.copy()
    return np.asarray(done_mask).copy()


def _is_true_batched_episode_end(info: dict[str, Any]) -> bool:
    if TRUE_EPISODE_END_INFO_KEY in info:
        return bool(info[TRUE_EPISODE_END_INFO_KEY])
    truncated = bool(info.get("TimeLimit.truncated", False))
    return is_true_atari_episode_end(terminated=True, truncated=truncated, info=info)


def _raw_full_game_returns_for_finished_agents(infos: Any, finished_agents: list[int]) -> np.ndarray | None:
    raw_returns: list[float] = []
    for agent_idx in finished_agents:
        raw_return = _raw_full_game_return_from_info(_batched_info_for_agent(infos, int(agent_idx)))
        if raw_return is None:
            return None
        raw_returns.append(float(raw_return))
    return np.asarray(raw_returns, dtype=np.float64)


def _episode_extra_stats_for_finished_agents(infos: Any, finished_agents: list[int]) -> dict[str, np.ndarray]:
    collected: dict[str, list[float]] = {}
    for agent_idx in finished_agents:
        info = _batched_info_for_agent(infos, int(agent_idx))
        extra_stats = _flatten_numeric_episode_extra_stats(info.get("episode_extra_stats") or {})
        for key, value in extra_stats.items():
            collected.setdefault(key, []).append(value)
    return {key: np.asarray(values, dtype=np.float64) for key, values in collected.items()}
