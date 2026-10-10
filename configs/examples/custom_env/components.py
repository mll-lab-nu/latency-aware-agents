"""User-owned factories for the standalone CPU profiling example."""

from latency_bench.envs.adapters.gymnasium import GymnasiumEnvAdapter
from latency_bench.policy.base import ConstantPolicyRunner


def make_env(config, noop_action):
    env = config["env"]
    return GymnasiumEnvAdapter(
        env_id=env["env_id"], registration_imports=[], make_kwargs={},
        noop_action=noop_action, env_fps=env["env_fps"], frame_stack=1,
        export_env_raw_rgb_frames=False,
    )


def make_policy(config, action_resolver):
    return ConstantPolicyRunner(action_resolver.resolve(config["policy"]["action"]))
