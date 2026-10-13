"""Construct the environment and policy for the standalone CPU examples."""

from latency_bench.envs.adapters.gymnasium import GymnasiumEnvAdapter
from latency_bench.envs.factory import register_env_adapter
from latency_bench.policy.base import ConstantPolicyRunner


@register_env_adapter("custom_mountain_car", env_id="MountainCar-v0")
def make_env(config, noop_action):
    env = config["env"]
    return GymnasiumEnvAdapter(
        env_id=env["env_id"], registration_imports=[], make_kwargs=env.get("make_kwargs", {}),
        noop_action=noop_action, env_fps=env["env_fps"], frame_stack=1,
        export_env_raw_rgb_frames=False,
    )


def make_policy(config, action_resolver):
    return ConstantPolicyRunner(action_resolver.resolve(config["policy"]["action"]))
