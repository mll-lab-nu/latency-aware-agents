"""Native InterceptGrab task construction for the simulator interpreter."""

from __future__ import annotations

import sys

import gymnasium as gym

from latency_bench.utils.paths import REPO_ROOT


# Keep the pinned source available to PPO, VLA wrappers, and per-slot resets.
# Simulator packages stay lazily imported so model-worker spawn remains isolated.
sys.path.insert(0, str(REPO_ROOT / "third_party" / "MIKASA-Robo"))

ENV_ID = "InterceptGrabFast-VLA-v0"
INSTRUCTION = "Intercept the rolling ball and grasp it to stop it."
MIKASA_IMAGE_VIEWS_INFO_KEY = "mikasa_image_views"
MIKASA_STATE_INFO_KEY = "mikasa_proprio"
MIKASA_STATE_LABELS = ("tcp_x", "tcp_y", "tcp_z", "tcp_roll", "tcp_pitch", "tcp_yaw", "gripper_width")
MIKASA_ACTION_LABELS = ("ee_delta_x", "ee_delta_y", "ee_delta_z", "ee_rotation_vector_x",
                        "ee_rotation_vector_y", "ee_rotation_vector_z", "gripper")


def make_raw_env(
    obs_mode: str,
    num_envs: int = 1,
    simulator_device: str = "gpu",
):
    import mikasa_robo_suite.vla.memory_envs  # noqa: F401

    return gym.make(
        ENV_ID,
        num_envs=num_envs,
        obs_mode=obs_mode,
        control_mode="pd_ee_delta_pose",
        render_mode="all",
        sim_backend=simulator_device,
        render_backend=simulator_device,
        reward_mode="normalized_dense",
    )


def make_ppo_env(num_envs: int = 1, simulator_device: str = "gpu"):
    from baselines.ppo.ppo_memtasks import FlattenRGBDObservationWrapper
    from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
    from mikasa_robo_suite.vla.dataset_collectors.get_mikasa_robo_datasets import (
        env_info,
    )

    env = make_raw_env(
        "state",
        num_envs=num_envs,
        simulator_device=simulator_device,
    )
    wrappers, _ = env_info(ENV_ID)
    for wrapper, kwargs in wrappers:
        env = wrapper(env, **kwargs)
    env = FlattenRGBDObservationWrapper(env, rgb=False, depth=False, state=True)
    return ManiSkillVectorEnv(
        env,
        num_envs,
        ignore_terminations=True,
        record_metrics=True,
    )


def make_vla_env(num_envs: int = 1, simulator_device: str = "gpu"):
    from mikasa_robo_suite.vla.utils.apply_wrappers import apply_mikasa_vla_wrappers

    return apply_mikasa_vla_wrappers(
        make_raw_env(
            "rgb",
            num_envs=num_envs,
            simulator_device=simulator_device,
        ),
        include_overlays=False,
    )
