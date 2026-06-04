from __future__ import annotations

import inspect
from importlib import metadata
from typing import Any

import torch


SUPPORTED_SAMPLE_FACTORY_VERSION = "2.1.1"

_PATCHED_ATTR = "_latency_bench_gpu_sampling_patched"
_ENABLED_ENV_NAMES_ATTR = "_latency_bench_gpu_sampling_env_names"
_ORIGINAL_ADVANCE_ROLLOUTS_ATTR = "_latency_bench_original_advance_rollouts"

_EXPECTED_ADVANCE_ROLLOUTS_PARAMS = ("self", "policy_id", "timing")


def ensure_sample_factory_gpu_sampling_compat_installed(*, env_name: str) -> bool:
    env_name = str(env_name)
    try:
        import sample_factory
        from sample_factory.algo.sampling.batched_sampling import BatchedVectorEnvRunner
    except ImportError:
        return False

    _validate_sample_factory_version(sample_factory)
    _patch_batched_advance_rollouts(BatchedVectorEnvRunner, env_name)
    return True


def _validate_sample_factory_version(sample_factory_module: Any) -> None:
    try:
        installed_version = metadata.version("sample-factory")
    except metadata.PackageNotFoundError:
        installed_version = str(sample_factory_module.__version__ or "")

    if installed_version != SUPPORTED_SAMPLE_FACTORY_VERSION:
        raise RuntimeError(
            "Unsupported Sample Factory version for GPU sampling compat: "
            f"expected {SUPPORTED_SAMPLE_FACTORY_VERSION}, got {installed_version or 'unknown'}"
        )


def _validate_signature(callable_obj: Any, *, expected_params: tuple[str, ...], label: str) -> None:
    signature = inspect.signature(callable_obj)
    actual_params = tuple(signature.parameters)
    if actual_params != expected_params:
        raise RuntimeError(
            f"Sample Factory {label} signature changed: expected {expected_params}, got {actual_params}"
        )


def _patch_batched_advance_rollouts(BatchedVectorEnvRunner: Any, env_name: str) -> None:
    current = BatchedVectorEnvRunner.advance_rollouts
    _validate_signature(
        current,
        expected_params=_EXPECTED_ADVANCE_ROLLOUTS_PARAMS,
        label="BatchedVectorEnvRunner.advance_rollouts",
    )

    if getattr(current, _PATCHED_ATTR, False):
        getattr(current, _ENABLED_ENV_NAMES_ATTR).add(env_name)
        return

    original_advance_rollouts = current
    enabled_env_names = {env_name}

    def advance_rollouts_gpu_sampling(self, policy_id, timing):
        if not _is_enabled_for_sample_factory_env(self, enabled_env_names):
            return original_advance_rollouts(self, policy_id, timing)

        from sample_factory.algo.sampling.batched_sampling import preprocess_actions

        with timing.add_time("process_policy_outputs"):
            self.curr_step[:] = self.policy_output_tensors
            actions = preprocess_actions(self.env_info, self.policy_output_tensors["actions"])

        complete_rollouts, episodic_stats = [], []

        with timing.add_time("env_step"):
            fast_step = _sample_factory_gpu_step(self.vec_env)
            if fast_step is None:
                self.last_obs, rewards, terminated, truncated, infos = self.vec_env.step(actions)
            else:
                self.last_obs, rewards, terminated, truncated, infos = fast_step(actions)
            dones = terminated | truncated

        with timing.add_time("post_env_step"):
            self.policy_id_buffer[:] = self.policy_id
            cpu_tensors = _sample_factory_step_cpu_tensors(self.vec_env)
            rewards_cpu = cpu_tensors["rewards"]
            dones_for_stats = cpu_tensors["dones"]
            processed_rewards = self._process_rewards(rewards, rewards_cpu)
            self.curr_step[:] = dict(
                rewards=processed_rewards,
                dones=dones,
                time_outs=truncated,
                policy_id=self.policy_id_buffer,
            )

            not_done = (1.0 - self.curr_step["dones"].float()).unsqueeze(-1)
            self.last_rnn_state = self.policy_output_tensors["new_rnn_states"] * not_done

            with timing.add_time("process_env_step"):
                stats = self._process_env_step(rewards_cpu, dones_for_stats, infos)
            episodic_stats.extend(stats)

        self.rollout_step += 1

        with timing.add_time("finalize_trajectories"):
            if self.rollout_step == self.cfg.rollout:
                complete_rollouts = self._finalize_trajectories()
                self.rollout_step = 0
                self.curr_traj_slice = self.curr_traj = None

                if self.training_info[self.policy_id] is not None:
                    from sample_factory.envs.env_utils import set_reward_shaping, set_training_info

                    reward_shaping = self.training_info[self.policy_id].get("reward_shaping", None)
                    set_reward_shaping(self.vec_env, reward_shaping, slice(0, self.vec_env.num_agents))
                    set_training_info(self.env_training_info_interface, self.training_info[self.policy_id])

        self.env_step_ready = True
        return complete_rollouts, episodic_stats

    setattr(advance_rollouts_gpu_sampling, _PATCHED_ATTR, True)
    setattr(advance_rollouts_gpu_sampling, _ENABLED_ENV_NAMES_ATTR, enabled_env_names)
    setattr(advance_rollouts_gpu_sampling, _ORIGINAL_ADVANCE_ROLLOUTS_ATTR, original_advance_rollouts)
    BatchedVectorEnvRunner.advance_rollouts = advance_rollouts_gpu_sampling


def _sample_factory_step_cpu_tensors(vec_env: Any) -> dict[str, torch.Tensor]:
    current = vec_env
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        current_vars = vars(current)
        if "sample_factory_step_cpu_tensors" in current_vars:
            tensors = current_vars["sample_factory_step_cpu_tensors"]
            if tensors:
                return tensors
        current = getattr(current, "env", None)
    return {}


def _sample_factory_gpu_step(vec_env: Any) -> Any:
    current = vec_env
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        current_vars = vars(current)
        if "sample_factory_gpu_step" in current_vars or hasattr(type(current), "sample_factory_gpu_step"):
            return current.sample_factory_gpu_step
        current = getattr(current, "env", None)
    return None


def _is_enabled_for_sample_factory_env(obj: Any, enabled_env_names: set[str]) -> bool:
    env_name = str(obj.cfg.env)
    return env_name in enabled_env_names and _has_sample_factory_step_cpu_tensors_attr(obj.vec_env)


def _has_sample_factory_step_cpu_tensors_attr(vec_env: Any) -> bool:
    current = vec_env
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if "sample_factory_step_cpu_tensors" in vars(current):
            return True
        current = getattr(current, "env", None)
    return False
