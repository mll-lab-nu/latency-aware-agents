from __future__ import annotations

import inspect
from importlib import metadata
from typing import Any

import numpy as np

from training.atari_episode_stats import _batched_info_for_agent, _flatten_numeric_episode_extra_stats


SUPPORTED_SAMPLE_FACTORY_VERSION = "2.1.1"

_PATCHED_ATTR = "_latency_bench_episode_extra_stats_patched"
_ENABLED_ENV_NAMES_ATTR = "_latency_bench_episode_extra_stats_env_names"
_ORIGINAL_PROCESS_ENV_STEP_ATTR = "_latency_bench_original_episode_extra_stats_process_env_step"
_KNOWN_ORIGINAL_PROCESS_ENV_STEP_ATTRS = (
    _ORIGINAL_PROCESS_ENV_STEP_ATTR,
    "_latency_bench_original_process_env_step",
)

_EXPECTED_PROCESS_ENV_STEP_PARAMS = ("self", "rewards", "dones_orig", "infos")


def ensure_sample_factory_episode_extra_stats_installed(*, env_name: str) -> bool:
    """Forward numeric ``episode_extra_stats`` from batched list infos into SF episodic reports."""

    env_name = str(env_name)
    try:
        import sample_factory
        from sample_factory.algo.sampling.batched_sampling import BatchedVectorEnvRunner
        from sample_factory.algo.utils.misc import EPISODIC, POLICY_ID_KEY
    except ImportError:
        return False

    _validate_sample_factory_version(sample_factory)
    _patch_batched_process_env_step(BatchedVectorEnvRunner, EPISODIC, POLICY_ID_KEY, env_name)
    return True


def _validate_sample_factory_version(sample_factory_module: Any) -> None:
    try:
        installed_version = metadata.version("sample-factory")
    except metadata.PackageNotFoundError:
        installed_version = str(getattr(sample_factory_module, "__version__", "") or "")

    if installed_version != SUPPORTED_SAMPLE_FACTORY_VERSION:
        raise RuntimeError(
            "Unsupported Sample Factory version for episode extra stats compat: "
            f"expected {SUPPORTED_SAMPLE_FACTORY_VERSION}, got {installed_version or 'unknown'}"
        )


def _validate_signature(callable_obj: Any, *, expected_params: tuple[str, ...], label: str) -> None:
    signature = inspect.signature(callable_obj)
    actual_params = tuple(signature.parameters)
    if actual_params != expected_params:
        raise RuntimeError(
            f"Sample Factory {label} signature changed: expected {expected_params}, got {actual_params}"
        )


def _patch_batched_process_env_step(
    BatchedVectorEnvRunner: Any,
    episodic_key: str,
    policy_id_key: str,
    env_name: str,
) -> None:
    current = BatchedVectorEnvRunner._process_env_step
    _validate_signature(
        current,
        expected_params=_EXPECTED_PROCESS_ENV_STEP_PARAMS,
        label="BatchedVectorEnvRunner._process_env_step",
    )

    if getattr(current, _PATCHED_ATTR, False):
        getattr(current, _ENABLED_ENV_NAMES_ATTR).add(env_name)
        return

    existing_patch = _find_existing_patch(current)
    if existing_patch is not None:
        getattr(existing_patch, _ENABLED_ENV_NAMES_ATTR).add(env_name)
        return

    original_process_env_step = current
    enabled_env_names = {env_name}

    def process_env_step_episode_extra_stats(self, rewards, dones_orig, infos):
        if not _is_enabled_for_sample_factory_env(self, enabled_env_names):
            return original_process_env_step(self, rewards, dones_orig, infos)
        if not isinstance(infos, (list, tuple)):
            return original_process_env_step(self, rewards, dones_orig, infos)

        finished_agents = _done_indices(dones_orig)
        if not finished_agents:
            return original_process_env_step(self, rewards, dones_orig, infos)

        extra_stats = _episode_extra_stats_for_finished_agents(infos, finished_agents)
        if not extra_stats:
            return original_process_env_step(self, rewards, dones_orig, infos)

        native_stats_snapshot = _native_episode_stats_snapshot(self, rewards, finished_agents)
        reports = original_process_env_step(self, rewards, dones_orig, infos)
        if reports:
            _augment_episode_reports(reports, episodic_key, extra_stats)
            return reports

        return [
            {
                policy_id_key: getattr(self, "policy_id", 0),
                episodic_key: _merge_missing_episode_extra_stats(native_stats_snapshot, extra_stats),
            }
        ]

    setattr(process_env_step_episode_extra_stats, _PATCHED_ATTR, True)
    setattr(process_env_step_episode_extra_stats, _ENABLED_ENV_NAMES_ATTR, enabled_env_names)
    setattr(process_env_step_episode_extra_stats, _ORIGINAL_PROCESS_ENV_STEP_ATTR, original_process_env_step)
    BatchedVectorEnvRunner._process_env_step = process_env_step_episode_extra_stats


def _find_existing_patch(callable_obj: Any) -> Any | None:
    seen: set[int] = set()
    current = callable_obj
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if getattr(current, _PATCHED_ATTR, False):
            return current
        next_callable = None
        for attr in _KNOWN_ORIGINAL_PROCESS_ENV_STEP_ATTRS:
            candidate = getattr(current, attr, None)
            if candidate is not None:
                next_callable = candidate
                break
        current = next_callable
    return None


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


def _select_finished(value: Any, finished_agents: list[int]) -> np.ndarray:
    array = _as_numpy_array(value)
    if array.ndim == 0:
        return np.asarray([array.item() for _ in finished_agents])
    return np.asarray([array[int(agent_idx)] for agent_idx in finished_agents])


def _native_episode_stats_snapshot(self: Any, rewards: Any, finished_agents: list[int]) -> dict[str, np.ndarray]:
    curr_reward = _as_numpy_array(getattr(self, "curr_episode_reward", 0.0))
    reward_values = _as_numpy_array(rewards)
    episode_reward = curr_reward + reward_values

    curr_len = _as_numpy_array(getattr(self, "curr_episode_len", 0))
    frame_increment = int(getattr(getattr(self, "env_info", None), "frameskip", 1) or 1)
    if not bool(getattr(getattr(self, "cfg", None), "summaries_use_frameskip", False)):
        frame_increment = 1
    episode_len = curr_len + frame_increment

    stats = {
        "reward": _select_finished(episode_reward, finished_agents),
        "len": _select_finished(episode_len, finished_agents),
    }
    if hasattr(self, "min_raw_rewards"):
        stats["min_raw_reward"] = _select_finished(getattr(self, "min_raw_rewards"), finished_agents)
    if hasattr(self, "max_raw_rewards"):
        stats["max_raw_reward"] = _select_finished(getattr(self, "max_raw_rewards"), finished_agents)
    return stats


def _episode_extra_stats_for_finished_agents(infos: Any, finished_agents: list[int]) -> dict[str, np.ndarray]:
    collected: dict[str, list[float]] = {}
    for agent_idx in finished_agents:
        info = _batched_info_for_agent(infos, int(agent_idx))
        extra_stats = _flatten_numeric_episode_extra_stats(info.get("episode_extra_stats") or {})
        for key, value in extra_stats.items():
            collected.setdefault(key, []).append(value)
    return {key: np.asarray(values, dtype=np.float64) for key, values in collected.items()}


def _augment_episode_reports(reports: Any, episodic_key: str, extra_stats: dict[str, np.ndarray]) -> None:
    for report in reports:
        if not isinstance(report, dict):
            continue
        episodic_stats = report.get(episodic_key)
        if not isinstance(episodic_stats, dict):
            continue
        _merge_missing_episode_extra_stats(episodic_stats, extra_stats)


def _merge_missing_episode_extra_stats(
    stats: dict[str, Any],
    extra_stats: dict[str, np.ndarray],
) -> dict[str, Any]:
    for key, value in extra_stats.items():
        if key not in stats:
            stats[key] = value
    return stats
