"""Compare direct wrapped Asterix with the generic adapter at zero latency."""

import json
from pathlib import Path

import gymnasium as gym
import numpy as np

from latency_bench.core.config import load_config
from latency_bench.core.types import Action
from latency_bench.envs.factory import build_env_adapter
import latency_bench.envs.tasks.asterix  # registers task wrapper


ACTIONS = [0, 1, 2, 3, 4, 5, 6, 7, 8]


def main() -> None:
    config = load_config("configs/examples/gymnasium/asterix/small_model_train.yaml")
    direct = gym.make(config["env"]["env_id"], **config["env"]["make_kwargs"])
    adapter = build_env_adapter(config, Action(value=0, name="noop", is_noop=True))
    direct_obs, _ = direct.reset(seed=123)
    adapter_obs = adapter.reset(seed=123).data
    reset_obs_equal = bool(np.array_equal(direct_obs, adapter_obs))
    steps = []
    for step, action_id in enumerate(ACTIONS, start=1):
        direct_obs, direct_reward, direct_terminated, direct_truncated, _ = direct.step(action_id)
        adapted = adapter.step(Action(value=action_id, name=f"action_{action_id}"))
        steps.append({
            "step": step,
            "issued_action_id": action_id,
            "obs_equal": bool(np.array_equal(direct_obs, adapted.observation.data)),
            "reward_equal": bool(direct_reward == adapted.reward),
            "terminated_equal": bool(direct_terminated == adapted.done),
            "truncated_equal": bool(direct_truncated == adapted.truncated),
            "adapter_applied_action": adapted.info["applied_action"],
            "adapter_env_step": adapted.info["env_step"],
        })
        if direct_terminated or direct_truncated:
            break
    payload = {
        "seed": 123,
        "actions": ACTIONS,
        "policy_observation_shape": list(adapter_obs.shape),
        "reset_obs_equal": reset_obs_equal,
        "all_steps_equal": bool(all(all(row[key] for key in ("obs_equal","reward_equal","terminated_equal","truncated_equal")) for row in steps)),
        "steps": steps,
    }
    Path("artifacts/asterix/zero_latency_parity.json").write_text(json.dumps(payload,indent=2)+"\n")
    print(json.dumps(payload,indent=2))
    direct.close(); adapter.close()


if __name__ == "__main__": main()
