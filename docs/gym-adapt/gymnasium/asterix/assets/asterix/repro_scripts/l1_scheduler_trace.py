"""Record Asterix raw actions for one-frame latency after random noop reset."""

import json
from pathlib import Path

import gymnasium as gym

from latency_bench.core.config import load_config
import latency_bench.envs.tasks.asterix  # registers task wrapper
from training.gymnasium_sf.integration import GymnasiumActionCatalogWrapper, GymnasiumActionLatencyWrapper


class ActionTraceWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env) -> None:
        super().__init__(env); self.actions: list[int] = []
    def step(self, action):
        self.actions.append(action); return self.env.step(action)


def main() -> None:
    config=load_config("configs/examples/gymnasium/asterix/small_model_train.yaml")
    traced=ActionTraceWrapper(gym.make(config["env"]["env_id"],**config["env"]["make_kwargs"]))
    env=GymnasiumActionCatalogWrapper(traced,list(range(9)))
    env=GymnasiumActionLatencyWrapper(env,fixed_latency_raw_frames=1,env_fps=60,obs_fps=15,hold_policy="hold",ordering_policy="issue_order_fifo",noop_action=0,add_latency_info=True)
    _,reset_info=env.reset(seed=7); trace=[]
    for issued_action_id in [5,8,1]:
        start=len(traced.actions)
        _,reward,terminated,truncated,info=env.step(issued_action_id)
        trace.append({"issued_action_id":issued_action_id,"raw_applied_actions":traced.actions[start:],"ready_raw_frame":info["ready_raw_frame"],"raw_frames_advanced":info["raw_frames_advanced"],"latency_raw_frames":info["latency_raw_frames"],"reward":reward,"terminated":terminated,"truncated":truncated})
    payload={"fixed_latency_ms":1000/60,"env_fps":60,"obs_fps":15,"noop_max":30,"expected_latency_raw_frames":1,"expected_obs_stride_raw_frames":4,"reset_applied_action":reset_info["applied_action"],"trace":trace,"claim":"Random noops occur only during reset; the first policy action is still preceded by one raw NOOP latency frame."}
    Path("artifacts/asterix/l1_scheduler_trace.json").write_text(json.dumps(payload,indent=2)+"\n")
    print(json.dumps(payload,indent=2)); env.close()


if __name__ == "__main__": main()
