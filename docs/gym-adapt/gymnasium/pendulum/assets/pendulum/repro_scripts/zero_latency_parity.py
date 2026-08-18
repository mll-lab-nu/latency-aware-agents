"""Compare the Pendulum task wrapper with the generic adapter at zero latency."""

import json
from pathlib import Path

import gymnasium as gym
import numpy as np

from latency_bench.core.types import Action
from latency_bench.envs.factory import build_env_adapter
import latency_bench.envs.gymnasium_pendulum  # registers the task-owned wrapper


MAKE_KWARGS = {
    "base_env_id": "Pendulum-v1",
    "render_mode": "rgb_array",
    "torque_values": [-2.0, -1.0, 0.0, 1.0, 2.0],
    "base_make_kwargs": {"g": 10.0},
}
ACTIONS = [2, 4, 1, 3, 0, 2, 4, 0]


def main() -> None:
    config = {
        "env": {
            "name": "gymnasium",
            "task_name": "pendulum_discrete",
            "env_id": "LatencyBench/PendulumDiscrete-v0",
            "registration_imports": ["latency_bench.envs.gymnasium_pendulum"],
            "make_kwargs": MAKE_KWARGS,
            "env_fps": 20.0,
            "obs_fps": 20.0,
            "frame_stack": 1,
        },
        "policy": {"type": "random"},
    }
    direct = gym.make("LatencyBench/PendulumDiscrete-v0", **MAKE_KWARGS)
    adapter = build_env_adapter(config, Action(value=2, name="zero_torque", is_noop=True))
    direct_obs, _ = direct.reset(seed=123)
    adapter_obs = adapter.reset(seed=123).data
    reset_obs_equal = bool(np.array_equal(direct_obs, adapter_obs))
    steps = []
    for step, action_id in enumerate(ACTIONS, start=1):
        direct_result = direct.step(action_id)
        adapted_result = adapter.step(Action(value=action_id, name=f"action_{action_id}"))
        direct_obs, direct_reward, direct_terminated, direct_truncated, _ = direct_result
        steps.append(
            {
                "step": step,
                "issued_action_id": action_id,
                "obs_equal": bool(np.array_equal(direct_obs, adapted_result.observation.data)),
                "obs_max_abs_diff": float(np.max(np.abs(direct_obs - adapted_result.observation.data))),
                "reward_equal": bool(direct_reward == adapted_result.reward),
                "terminated_equal": bool(direct_terminated == adapted_result.done),
                "truncated_equal": bool(direct_truncated == adapted_result.truncated),
                "adapter_applied_action": adapted_result.info["applied_action"],
                "adapter_env_step": adapted_result.info["env_step"],
            }
        )
        if direct_terminated or direct_truncated:
            break
    payload = {
        "seed": 123,
        "actions": ACTIONS,
        "reset_obs_equal": reset_obs_equal,
        "all_steps_equal": bool(
            all(
                all(
                    row[key]
                    for key in (
                        "obs_equal",
                        "reward_equal",
                        "terminated_equal",
                        "truncated_equal",
                    )
                )
                for row in steps
            )
        ),
        "steps": steps,
    }
    output_path = Path("artifacts/pendulum/zero_latency_parity.json")
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    direct.close()
    adapter.close()


if __name__ == "__main__":
    main()
