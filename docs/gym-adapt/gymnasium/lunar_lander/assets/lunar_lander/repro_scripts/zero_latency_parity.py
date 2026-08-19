"""Compare direct LunarLander-v2 with the generic adapter at zero latency."""

import json
from pathlib import Path

import gymnasium as gym
import numpy as np

from latency_bench.core.types import Action
from latency_bench.envs.factory import build_env_adapter


MAKE_KWARGS = {
    "render_mode": "rgb_array",
    "continuous": False,
    "gravity": -10.0,
    "enable_wind": False,
    "wind_power": 15.0,
    "turbulence_power": 1.5,
}
ACTIONS = [0, 2, 1, 3, 0, 2, 3, 1]


def main() -> None:
    config = {
        "env": {
            "name": "gymnasium",
            "task_name": "lunar_lander_discrete",
            "env_id": "LunarLander-v2",
            "registration_imports": [],
            "make_kwargs": MAKE_KWARGS,
            "env_fps": 50.0,
            "obs_fps": 50.0,
            "frame_stack": 1,
        },
        "policy": {"type": "random"},
    }
    direct = gym.make("LunarLander-v2", **MAKE_KWARGS)
    adapter = build_env_adapter(config, Action(value=0, name="do_nothing", is_noop=True))
    direct_obs, _ = direct.reset(seed=123)
    adapter_obs = adapter.reset(seed=123).data
    reset_obs_equal = bool(np.array_equal(direct_obs, adapter_obs))
    steps = []
    for step, action_id in enumerate(ACTIONS, start=1):
        direct_obs, direct_reward, direct_terminated, direct_truncated, _ = direct.step(action_id)
        adapted = adapter.step(Action(value=action_id, name=f"action_{action_id}"))
        steps.append(
            {
                "step": step,
                "issued_action_id": action_id,
                "obs_equal": bool(np.array_equal(direct_obs, adapted.observation.data)),
                "obs_max_abs_diff": float(np.max(np.abs(direct_obs - adapted.observation.data))),
                "reward_equal": bool(direct_reward == adapted.reward),
                "terminated_equal": bool(direct_terminated == adapted.done),
                "truncated_equal": bool(direct_truncated == adapted.truncated),
                "adapter_applied_action": adapted.info["applied_action"],
                "adapter_env_step": adapted.info["env_step"],
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
    output_path = Path("artifacts/lunar_lander/zero_latency_parity.json")
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    direct.close()
    adapter.close()


if __name__ == "__main__":
    main()
