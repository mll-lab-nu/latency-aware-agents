"""Compare native Hopper torques with the adapted catalog at zero latency."""

import json
from pathlib import Path

import gymnasium as gym
import numpy as np

from latency_bench.core.config import load_config
from latency_bench.core.types import Action
from latency_bench.envs.factory import build_env_adapter


ACTIONS = [13, 26, 0, 23, 4, 22, 9, 17]


def main() -> None:
    config = load_config("configs/examples/gymnasium/hopper/small_model_train.yaml")
    make_kwargs = config["env"]["make_kwargs"]
    direct = gym.make(
        make_kwargs["base_env_id"],
        render_mode=make_kwargs["render_mode"],
        **make_kwargs["base_make_kwargs"],
    )
    adapter = build_env_adapter(config, Action(value=13, name="t_z_l_z_f_z", is_noop=True))
    direct_obs, _ = direct.reset(seed=123)
    adapter_obs = adapter.reset(seed=123).data
    reset_obs_equal = bool(np.array_equal(direct_obs, adapter_obs))
    steps = []
    for step, action_id in enumerate(ACTIONS, start=1):
        native_action = np.asarray(make_kwargs["action_values"][action_id], dtype=direct.action_space.dtype)
        direct_obs, direct_reward, direct_terminated, direct_truncated, _ = direct.step(native_action)
        adapted = adapter.step(Action(value=action_id, name=f"action_{action_id}"))
        steps.append(
            {
                "step": step,
                "issued_action_id": action_id,
                "native_action": native_action.tolist(),
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
                all(row[key] for key in ("obs_equal", "reward_equal", "terminated_equal", "truncated_equal"))
                for row in steps
            )
        ),
        "steps": steps,
    }
    output_path = Path("artifacts/hopper/zero_latency_parity.json")
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    direct.close()
    adapter.close()


if __name__ == "__main__":
    main()
