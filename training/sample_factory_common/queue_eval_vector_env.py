from __future__ import annotations

import multiprocessing as mp
import os
import traceback
from typing import Any, Mapping, Sequence

import gymnasium as gym
import numpy as np
import torch
from gymnasium.spaces import Dict as GymDict
from sample_factory.utils.attr_dict import AttrDict


_THREAD_ENV_DEFAULTS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def _set_subproc_thread_env_defaults() -> None:
    for key in _THREAD_ENV_DEFAULTS:
        os.environ.setdefault(key, "1")


def _subproc_start_method() -> str:
    default_method = "fork" if "fork" in mp.get_all_start_methods() else mp.get_start_method()
    method = str(os.environ.get("DEMON_EVAL_MP_START_METHOD", default_method)).strip()
    if method not in mp.get_all_start_methods():
        raise ValueError(f"Unsupported DEMON_EVAL_MP_START_METHOD={method!r}")
    return method


def demon_attack_queue_eval_mp_start_method() -> str:
    return _subproc_start_method()


def _slot_env_config(base_env_config: Any, slot_idx: int) -> AttrDict:
    worker_index = int(getattr(base_env_config, "worker_index", 0) or 0)
    base_vector_index = int(getattr(base_env_config, "vector_index", 0) or 0)
    base_env_id = int(getattr(base_env_config, "env_id", 0) or 0)
    return AttrDict(
        worker_index=worker_index,
        vector_index=base_vector_index + int(slot_idx),
        env_id=base_env_id + int(slot_idx),
    )


def _reset_payload(reset_value: Any) -> dict[str, int | None]:
    if isinstance(reset_value, (list, tuple)) and len(reset_value) == 2:
        episode_id, seed = reset_value
    else:
        episode_id, seed = None, reset_value
    return {
        "episode_id": None if episode_id is None else int(episode_id),
        "seed": None if seed is None else int(seed),
    }


def _annotate_worker_info(info: dict[str, Any], *, slot_idx: int, episode_id: int | None) -> dict[str, Any]:
    annotated = dict(info or {})
    annotated["_queue_eval_worker_pid"] = int(os.getpid())
    annotated["_queue_eval_slot_idx"] = int(slot_idx)
    if episode_id is not None:
        annotated["_queue_eval_episode_id"] = int(episode_id)
    return annotated


def _demon_attack_queue_eval_worker(remote, cfg: Any, env_config: Any, slot_idx: int) -> None:
    _set_subproc_thread_env_defaults()
    env = None
    current_episode_id: int | None = None
    try:
        from training.demon_attack_sf.integration import SF_ENV_NAME, make_demon_attack_sf_env

        env = make_demon_attack_sf_env(SF_ENV_NAME, cfg=cfg, env_config=env_config, render_mode=None)
        remote.send(
            (
                "ready",
                {
                    "observation_space": env.observation_space,
                    "action_space": env.action_space,
                    "worker_pid": int(os.getpid()),
                },
            )
        )
        while True:
            try:
                command, payload = remote.recv()
            except EOFError:
                break

            if command == "close":
                if env is not None:
                    env.close()
                    env = None
                remote.send(("closed", None))
                break

            if command == "reset":
                payload = dict(payload or {})
                current_episode_id = payload.get("episode_id")
                reset_kwargs = {}
                if payload.get("seed") is not None:
                    reset_kwargs["seed"] = int(payload["seed"])
                obs, info = env.reset(**reset_kwargs)
                remote.send(
                    (
                        "result",
                        (
                            obs,
                            _annotate_worker_info(
                                dict(info or {}),
                                slot_idx=int(slot_idx),
                                episode_id=current_episode_id,
                            ),
                        ),
                    )
                )
                continue

            if command == "step":
                obs, reward, terminated, truncated, info = env.step(payload)
                remote.send(
                    (
                        "result",
                        (
                            obs,
                            float(reward),
                            bool(terminated),
                            bool(truncated),
                            _annotate_worker_info(
                                dict(info or {}),
                                slot_idx=int(slot_idx),
                                episode_id=current_episode_id,
                            ),
                        ),
                    )
                )
                continue

            raise ValueError(f"Unsupported worker command: {command!r}")
    except BaseException:
        try:
            remote.send(("error", traceback.format_exc()))
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        if env is not None:
            env.close()
        remote.close()


class DemonAttackQueueEvalVectorEnv(gym.Env):
    """Manual-reset vector wrapper for DemonAttack queue eval.

    The underlying environments are still created by the DemonAttack Sample Factory
    integration; this class only controls batching, auto-reset, and inactive slots.
    """

    is_multiagent = True

    def __init__(
        self,
        *,
        cfg: Any,
        num_slots: int,
        env_config: Any | None = None,
        auto_reset_done: bool = True,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.base_env_config = env_config or AttrDict(worker_index=0, vector_index=0, env_id=0)
        self.num_agents = int(num_slots)
        if self.num_agents <= 0:
            raise ValueError("num_slots must be positive")

        self.envs = [self._make_env(slot_idx) for slot_idx in range(self.num_agents)]
        self.single_observation_space = self.envs[0].observation_space
        self.observation_space = GymDict({"obs": self.single_observation_space})
        self.action_space = self.envs[0].action_space
        self._auto_reset_done = bool(auto_reset_done)
        self._active_agents = np.ones(self.num_agents, dtype=np.bool_)
        self._last_obs: list[Any | None] = [None for _ in range(self.num_agents)]
        self._last_infos: list[dict[str, Any]] = [{} for _ in range(self.num_agents)]
        self._has_reset = False

    def _make_env(self, slot_idx: int) -> gym.Env:
        from training.demon_attack_sf.integration import SF_ENV_NAME, make_demon_attack_sf_env

        env_config = _slot_env_config(self.base_env_config, int(slot_idx))
        return make_demon_attack_sf_env(SF_ENV_NAME, cfg=self.cfg, env_config=env_config, render_mode=None)

    def set_auto_reset_done(self, enabled: bool) -> None:
        self._auto_reset_done = bool(enabled)

    def set_active_agents(self, active_mask: Sequence[bool]) -> None:
        mask = np.asarray(list(active_mask), dtype=np.bool_).reshape(-1)
        if int(mask.shape[0]) != self.num_agents:
            raise ValueError(f"Expected active mask with {self.num_agents} entries, got {int(mask.shape[0])}")
        self._active_agents = mask

    def reset(self, **kwargs):
        seed = kwargs.pop("seed", None)
        observations: list[Any] = []
        infos: list[dict[str, Any]] = []
        for slot_idx, env in enumerate(self.envs):
            reset_kwargs = dict(kwargs)
            if seed is not None:
                reset_kwargs["seed"] = int(seed) + int(slot_idx)
            obs, info = env.reset(**reset_kwargs)
            observations.append(obs)
            infos.append(dict(info or {}))
            self._last_obs[slot_idx] = obs
            self._last_infos[slot_idx] = dict(info or {})
        self._has_reset = True
        return self._obs(observations), infos

    def reset_agents(
        self,
        reset_plan: Mapping[int, int | None],
        *,
        active_mask: Sequence[bool] | None = None,
    ):
        if active_mask is not None:
            self.set_active_agents(active_mask)
        if not self._has_reset:
            return self.reset()

        observations = [self._require_last_obs(slot_idx) for slot_idx in range(self.num_agents)]
        infos = [dict(info) for info in self._last_infos]
        for raw_slot_idx, seed in reset_plan.items():
            slot_idx = int(raw_slot_idx)
            if slot_idx < 0 or slot_idx >= self.num_agents:
                raise IndexError(f"slot index out of range: {slot_idx}")
            payload = _reset_payload(seed)
            reset_kwargs = {}
            if payload["seed"] is not None:
                reset_kwargs["seed"] = int(payload["seed"])
            obs, info = self.envs[slot_idx].reset(**reset_kwargs)
            observations[slot_idx] = obs
            infos[slot_idx] = dict(info or {})
            self._last_obs[slot_idx] = obs
            self._last_infos[slot_idx] = dict(info or {})
        return self._obs(observations), infos

    def step(self, action):
        if not self._has_reset:
            self.reset()

        actions = self._normalize_actions(action)
        observations = [self._require_last_obs(slot_idx) for slot_idx in range(self.num_agents)]
        rewards = np.zeros(self.num_agents, dtype=np.float32)
        terminated = np.zeros(self.num_agents, dtype=np.bool_)
        truncated = np.zeros(self.num_agents, dtype=np.bool_)
        infos: list[dict[str, Any]] = []

        for slot_idx, (env, scalar_action) in enumerate(zip(self.envs, actions)):
            if not bool(self._active_agents[slot_idx]):
                info = dict(self._last_infos[slot_idx])
                info["inactive"] = True
                infos.append(info)
                continue

            obs, reward, term, trunc, info = env.step(scalar_action)
            info = dict(info or {})
            done = bool(term or trunc)
            rewards[slot_idx] = float(reward)
            terminated[slot_idx] = bool(term)
            truncated[slot_idx] = bool(trunc)
            if done and self._auto_reset_done:
                reset_obs, reset_info = env.reset()
                info["reset_info"] = dict(reset_info or {})
                obs = reset_obs
                self._last_infos[slot_idx] = dict(reset_info or {})
            else:
                self._last_infos[slot_idx] = dict(info)
            observations[slot_idx] = obs
            self._last_obs[slot_idx] = obs
            infos.append(info)

        return (
            self._obs(observations),
            torch.as_tensor(rewards, dtype=torch.float32),
            torch.as_tensor(terminated, dtype=torch.bool),
            torch.as_tensor(truncated, dtype=torch.bool),
            infos,
        )

    def _obs(self, observations: Sequence[Any]) -> dict[str, torch.Tensor]:
        return {"obs": torch.as_tensor(np.stack(observations, axis=0))}

    def _require_last_obs(self, slot_idx: int) -> Any:
        obs = self._last_obs[slot_idx]
        if obs is None:
            raise RuntimeError("environment must be reset before step/reset_agents")
        return obs

    def _normalize_actions(self, action: Any) -> list[Any]:
        if isinstance(action, torch.Tensor):
            action = action.detach().cpu().numpy()
        if isinstance(action, np.ndarray):
            arr = action
            if arr.ndim == 0 and self.num_agents == 1:
                return [int(arr.item())]
            if arr.ndim == 2 and arr.shape[1] == 1:
                arr = arr[:, 0]
            if arr.ndim != 1 or int(arr.shape[0]) != self.num_agents:
                raise ValueError(f"Expected action array with shape ({self.num_agents},), got {arr.shape!r}")
            return [int(item) for item in arr.tolist()]
        if isinstance(action, (list, tuple)):
            if len(action) != self.num_agents:
                raise ValueError(f"Expected {self.num_agents} actions, got {len(action)}")
            return [self._scalar_action(item) for item in action]
        if self.num_agents == 1:
            return [self._scalar_action(action)]
        raise ValueError(f"Expected batched action for {self.num_agents} agents, got {type(action).__name__}")

    @staticmethod
    def _scalar_action(action: Any) -> Any:
        if isinstance(action, torch.Tensor):
            action = action.detach().cpu().numpy()
        if isinstance(action, np.ndarray):
            if action.size != 1:
                raise ValueError(f"Expected scalar action, got array with shape {action.shape!r}")
            return int(action.reshape(-1)[0])
        if isinstance(action, (list, tuple)) and len(action) == 1:
            return DemonAttackQueueEvalVectorEnv._scalar_action(action[0])
        return int(action)

    def close(self):
        for env in self.envs:
            env.close()
        super().close()


class DemonAttackQueueEvalSubprocVectorEnv(gym.Env):
    """Subprocess vector wrapper for DemonAttack queue eval.

    Workers own the standard DemonAttack Sample Factory env. The main process
    only sends actions/resets and stacks results for batched policy inference.
    """

    is_multiagent = True

    def __init__(
        self,
        *,
        cfg: Any,
        num_slots: int,
        env_config: Any | None = None,
        auto_reset_done: bool = True,
    ) -> None:
        super().__init__()
        _set_subproc_thread_env_defaults()
        self.cfg = cfg
        self.base_env_config = env_config or AttrDict(worker_index=0, vector_index=0, env_id=0)
        self.num_agents = int(num_slots)
        if self.num_agents <= 0:
            raise ValueError("num_slots must be positive")

        self._auto_reset_done = bool(auto_reset_done)
        self._active_agents = np.ones(self.num_agents, dtype=np.bool_)
        self._last_obs: list[Any | None] = [None for _ in range(self.num_agents)]
        self._last_infos: list[dict[str, Any]] = [{} for _ in range(self.num_agents)]
        self._has_reset = False
        self._closed = False
        self._ctx = mp.get_context(_subproc_start_method())
        self._remotes = []
        self._processes = []
        self._worker_pids: list[int] = []

        try:
            for slot_idx in range(self.num_agents):
                parent_remote, worker_remote = self._ctx.Pipe()
                process = self._ctx.Process(
                    target=_demon_attack_queue_eval_worker,
                    args=(
                        worker_remote,
                        self.cfg,
                        _slot_env_config(self.base_env_config, slot_idx),
                        int(slot_idx),
                    ),
                    daemon=True,
                )
                process.start()
                worker_remote.close()
                self._remotes.append(parent_remote)
                self._processes.append(process)

            observation_space = None
            action_space = None
            for slot_idx in range(self.num_agents):
                payload = self._recv(slot_idx, expected_tag="ready")
                if observation_space is None:
                    observation_space = payload["observation_space"]
                    action_space = payload["action_space"]
                self._worker_pids.append(int(payload["worker_pid"]))

            self.single_observation_space = observation_space
            self.observation_space = GymDict({"obs": self.single_observation_space})
            self.action_space = action_space
        except BaseException:
            self.close()
            raise

    def worker_pids(self) -> list[int]:
        return [int(pid) for pid in self._worker_pids]

    def set_auto_reset_done(self, enabled: bool) -> None:
        self._auto_reset_done = bool(enabled)

    def set_active_agents(self, active_mask: Sequence[bool]) -> None:
        mask = np.asarray(list(active_mask), dtype=np.bool_).reshape(-1)
        if int(mask.shape[0]) != self.num_agents:
            raise ValueError(f"Expected active mask with {self.num_agents} entries, got {int(mask.shape[0])}")
        self._active_agents = mask

    def reset(self, **kwargs):
        seed = kwargs.pop("seed", None)
        for slot_idx in range(self.num_agents):
            payload = {
                "episode_id": int(slot_idx),
                "seed": None if seed is None else int(seed) + int(slot_idx),
            }
            self._send(slot_idx, "reset", payload)

        observations: list[Any] = []
        infos: list[dict[str, Any]] = []
        for slot_idx in range(self.num_agents):
            obs, info = self._recv(slot_idx, expected_tag="result")
            observations.append(obs)
            infos.append(dict(info or {}))
            self._last_obs[slot_idx] = obs
            self._last_infos[slot_idx] = dict(info or {})
        self._has_reset = True
        return self._obs(observations), infos

    def reset_agents(
        self,
        reset_plan: Mapping[int, int | None],
        *,
        active_mask: Sequence[bool] | None = None,
    ):
        if active_mask is not None:
            self.set_active_agents(active_mask)
        if not self._has_reset:
            return self.reset()

        observations = [self._require_last_obs(slot_idx) for slot_idx in range(self.num_agents)]
        infos = [dict(info) for info in self._last_infos]
        reset_slots: list[int] = []
        for raw_slot_idx, reset_value in reset_plan.items():
            slot_idx = int(raw_slot_idx)
            if slot_idx < 0 or slot_idx >= self.num_agents:
                raise IndexError(f"slot index out of range: {slot_idx}")
            self._send(slot_idx, "reset", _reset_payload(reset_value))
            reset_slots.append(slot_idx)

        for slot_idx in reset_slots:
            obs, info = self._recv(slot_idx, expected_tag="result")
            observations[slot_idx] = obs
            infos[slot_idx] = dict(info or {})
            self._last_obs[slot_idx] = obs
            self._last_infos[slot_idx] = dict(info or {})
        return self._obs(observations), infos

    def step(self, action):
        if not self._has_reset:
            self.reset()

        actions = self._normalize_actions(action)
        observations = [self._require_last_obs(slot_idx) for slot_idx in range(self.num_agents)]
        rewards = np.zeros(self.num_agents, dtype=np.float32)
        terminated = np.zeros(self.num_agents, dtype=np.bool_)
        truncated = np.zeros(self.num_agents, dtype=np.bool_)
        infos: list[dict[str, Any]] = [dict(self._last_infos[slot_idx]) for slot_idx in range(self.num_agents)]

        active_slots: list[int] = []
        for slot_idx, scalar_action in enumerate(actions):
            if not bool(self._active_agents[slot_idx]):
                infos[slot_idx]["inactive"] = True
                continue
            self._send(slot_idx, "step", scalar_action)
            active_slots.append(slot_idx)

        for slot_idx in active_slots:
            obs, reward, term, trunc, info = self._recv(slot_idx, expected_tag="result")
            info = dict(info or {})
            done = bool(term or trunc)
            rewards[slot_idx] = float(reward)
            terminated[slot_idx] = bool(term)
            truncated[slot_idx] = bool(trunc)
            if done and self._auto_reset_done:
                self._send(slot_idx, "reset", {"episode_id": None, "seed": None})
                reset_obs, reset_info = self._recv(slot_idx, expected_tag="result")
                info["reset_info"] = dict(reset_info or {})
                obs = reset_obs
                self._last_infos[slot_idx] = dict(reset_info or {})
            else:
                self._last_infos[slot_idx] = dict(info)
            observations[slot_idx] = obs
            self._last_obs[slot_idx] = obs
            infos[slot_idx] = info

        return (
            self._obs(observations),
            torch.as_tensor(rewards, dtype=torch.float32),
            torch.as_tensor(terminated, dtype=torch.bool),
            torch.as_tensor(truncated, dtype=torch.bool),
            infos,
        )

    def _send(self, slot_idx: int, command: str, payload: Any) -> None:
        if self._closed:
            raise RuntimeError("DemonAttackQueueEvalSubprocVectorEnv is closed")
        self._remotes[int(slot_idx)].send((command, payload))

    def _recv(self, slot_idx: int, *, expected_tag: str):
        try:
            tag, payload = self._remotes[int(slot_idx)].recv()
        except EOFError as exc:
            raise RuntimeError(f"DemonAttack eval worker {int(slot_idx)} exited unexpectedly") from exc
        if tag == "error":
            raise RuntimeError(f"DemonAttack eval worker {int(slot_idx)} failed:\n{payload}")
        if tag != expected_tag:
            raise RuntimeError(f"Expected worker {int(slot_idx)} response {expected_tag!r}, got {tag!r}")
        return payload

    def _obs(self, observations: Sequence[Any]) -> dict[str, torch.Tensor]:
        return {"obs": torch.as_tensor(np.stack(observations, axis=0))}

    def _require_last_obs(self, slot_idx: int) -> Any:
        obs = self._last_obs[slot_idx]
        if obs is None:
            raise RuntimeError("environment must be reset before step/reset_agents")
        return obs

    def _normalize_actions(self, action: Any) -> list[Any]:
        return DemonAttackQueueEvalVectorEnv._normalize_actions(self, action)

    @staticmethod
    def _scalar_action(action: Any) -> Any:
        return DemonAttackQueueEvalVectorEnv._scalar_action(action)

    def close(self):
        if self._closed:
            return
        self._closed = True
        for remote in list(self._remotes):
            try:
                remote.send(("close", None))
            except (BrokenPipeError, EOFError, OSError):
                pass
        for remote in list(self._remotes):
            try:
                if remote.poll(1.0):
                    remote.recv()
            except (BrokenPipeError, EOFError, OSError):
                pass
            try:
                remote.close()
            except OSError:
                pass
        for process in list(self._processes):
            process.join(timeout=2.0)
            if process.is_alive():
                process.terminate()
                process.join(timeout=2.0)
        super().close()
