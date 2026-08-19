"""Record the real AirRaid raw-action trace for one-frame latency."""

import json
from pathlib import Path

import gymnasium as gym

from latency_bench.core.config import load_config
import latency_bench.envs.gymnasium_air_raid  # registers the task wrapper
from training.gymnasium_sf.integration import GymnasiumActionCatalogWrapper, GymnasiumActionLatencyWrapper


class ActionTraceWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self.actions: list[int] = []

    def step(self, action):
        self.actions.append(action)
        return self.env.step(action)


def main() -> None:
    config = load_config("configs/examples/gymnasium/air_raid/small_model_train.yaml")
    traced = ActionTraceWrapper(gym.make(config["env"]["env_id"], **config["env"]["make_kwargs"]))
    env = GymnasiumActionCatalogWrapper(traced, list(range(6)))
    env = GymnasiumActionLatencyWrapper(
        env,
        fixed_latency_raw_frames=1,
        env_fps=50,
        obs_fps=12.5,
        hold_policy="hold",
        ordering_policy="issue_order_fifo",
        noop_action=0,
        add_latency_info=True,
    )
    _, reset_info = env.reset(seed=7)
    trace = []
    for issued_action_id in [4, 5, 1]:
        start = len(traced.actions)
        _, reward, terminated, truncated, info = env.step(issued_action_id)
        trace.append(
            {
                "issued_action_id": issued_action_id,
                "raw_applied_actions": traced.actions[start:],
                "ready_raw_frame": info["ready_raw_frame"],
                "raw_frames_advanced": info["raw_frames_advanced"],
                "latency_raw_frames": info["latency_raw_frames"],
                "reward": reward,
                "terminated": terminated,
                "truncated": truncated,
            }
        )
    payload = {
        "fixed_latency_ms": 20.0,
        "env_fps": 50,
        "obs_fps": 12.5,
        "expected_latency_raw_frames": 1,
        "expected_obs_stride_raw_frames": 4,
        "reset_applied_action": reset_info["applied_action"],
        "trace": trace,
        "claim": "The first issued action is preceded by one raw NOOP frame, then held for the remaining three raw frames in the decision interval.",
    }
    output_path = Path("artifacts/air_raid/l1_scheduler_trace.json")
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    env.close()


if __name__ == "__main__":
    main()
