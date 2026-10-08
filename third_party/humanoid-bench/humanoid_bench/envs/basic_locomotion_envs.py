import os

import numpy as np
import mujoco
import gymnasium as gym
from gymnasium.spaces import Box
from dm_control.utils import rewards

from humanoid_bench.tasks import Task


# Height of head above which stand reward is 1.
_STAND_HEIGHT = 1.65
_CRAWL_HEIGHT = 0.8

# Horizontal speeds above which move reward is 1.
_WALK_SPEED = 1
_RUN_SPEED = 5


class Walk(Task):
    qpos0_robot = {
        "h1": "0 0 0.98 1 0 0 0 0 0 -0.4 0.8 -0.4 0 0 -0.4 0.8 -0.4 0 0 0 0 0 0 0 0 0",
        "h1hand": "0 0 0.98 1 0 0 0 0 0 -0.4 0.8 -0.4 0 0 -0.4 0.8 -0.4 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0",
        "h1touch": "0 0 0.98 1 0 0 0 0 0 -0.4 0.8 -0.4 0 0 -0.4 0.8 -0.4 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0",
        "g1": "0 0 0.75 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 -1.57 0 0 0 0 0 0 0 0 0 0 0 1.57 0 0 0 0 0 0 0"
    }
    _move_speed = _WALK_SPEED
    htarget_low = np.array([-1.0, -1.0, 0.8])
    htarget_high = np.array([1000.0, 1.0, 2.0])
    success_bar = 700

    def __init__(self, robot=None, env=None, **kwargs):
        super().__init__(robot, env, **kwargs)
        if robot.__class__.__name__ == "G1":
            global _STAND_HEIGHT, _CRAWL_HEIGHT
            _STAND_HEIGHT = 1.28
            _CRAWL_HEIGHT = 0.6

    @property
    def observation_space(self):
        return Box(
            low=-np.inf, high=np.inf, shape=(self.robot.dof * 2 - 1,), dtype=np.float64
        )

    def get_reward(self):
        standing = rewards.tolerance(
            self.robot.head_height(),
            bounds=(_STAND_HEIGHT, float("inf")),
            margin=_STAND_HEIGHT / 4,
        )
        upright = rewards.tolerance(
            self.robot.torso_upright(),
            bounds=(0.9, float("inf")),
            sigmoid="linear",
            margin=1.9,
            value_at_margin=0,
        )
        stand_reward = standing * upright
        small_control = rewards.tolerance(
            self.robot.actuator_forces(),
            margin=10,
            value_at_margin=0,
            sigmoid="quadratic",
        ).mean()
        small_control = (4 + small_control) / 5
        if self._move_speed == 0:
            horizontal_velocity = self.robot.center_of_mass_velocity()[[0, 1]]
            dont_move = rewards.tolerance(horizontal_velocity, margin=2).mean()
            return small_control * stand_reward * dont_move, {
                "small_control": small_control,
                "stand_reward": stand_reward,
                "dont_move": dont_move,
                "standing": standing,
                "upright": upright,
            }
        else:
            com_velocity = self.robot.center_of_mass_velocity()[0]
            move = rewards.tolerance(
                com_velocity,
                bounds=(self._move_speed, float("inf")),
                margin=self._move_speed,
                value_at_margin=0,
                sigmoid="linear",
            )
            move = (5 * move + 1) / 6
            reward = small_control * stand_reward * move
            return reward, {
                "stand_reward": stand_reward,
                "small_control": small_control,
                "move": move,
                "standing": standing,
                "upright": upright,
            }

    def get_terminated(self):
        return self._env.data.qpos[2] < 0.2, {}


class Stand(Walk):
    _move_speed = 0
    success_bar = 800


class Run(Walk):
    _move_speed = _RUN_SPEED


class Crawl(Walk):
    def get_reward(self):
        small_control = rewards.tolerance(
            self.robot.actuator_forces(),
            margin=10,
            value_at_margin=0,
            sigmoid="quadratic",
        ).mean()
        small_control = (4 + small_control) / 5

        com_velocity = self.robot.center_of_mass_velocity()[0]
        move = rewards.tolerance(
            com_velocity,
            bounds=(1, float("inf")),
            margin=1,
            value_at_margin=0,
            sigmoid="linear",
        )
        move = (5 * move + 1) / 6

        crawling_head = rewards.tolerance(
            self.robot.head_height(),
            bounds=(_CRAWL_HEIGHT - 0.2, _CRAWL_HEIGHT + 0.2),
            margin=1,
        )

        crawling = rewards.tolerance(
            self._env.named.data.site_xpos["imu", "z"],
            bounds=(_CRAWL_HEIGHT - 0.2, _CRAWL_HEIGHT + 0.2),
            margin=1,
        )

        reward_xquat = rewards.tolerance(
            np.linalg.norm(
                self._env.data.body("pelvis").xquat - np.array([0.75, 0, 0.65, 0])
            ),
            margin=1,
        )

        in_tunnel = rewards.tolerance(
            self._env.named.data.site_xpos["imu", "y"],
            bounds=(-1, 1),
            margin=0,
        )

        reward = (
            0.1 * small_control
            + 0.25 * min(crawling, crawling_head)
            + 0.4 * move
            + 0.25 * reward_xquat
        ) * in_tunnel
        return reward, {
            "crawling": crawling,
            "crawling_head": crawling_head,
            "small_control": small_control,
            "move": move,
            "in_tunnel": in_tunnel,
        }

    def get_terminated(self):
        return False, {}


class ClimbingUpwards(Walk):
    def get_reward(self):
        standing = rewards.tolerance(
            self.robot.head_height() - self.robot.left_foot_height(),
            bounds=(1.2, float("inf")),
            margin=0.45,
        ) * rewards.tolerance(
            self.robot.head_height() - self.robot.right_foot_height(),
            bounds=(1.2, float("inf")),
            margin=0.45,
        )
        upright = rewards.tolerance(
            self.robot.torso_upright(),
            bounds=(0.5, float("inf")),
            sigmoid="linear",
            margin=1.9,
            value_at_margin=0,
        )
        stand_reward = standing * upright
        small_control = rewards.tolerance(
            self.robot.actuator_forces(),
            margin=10,
            value_at_margin=0,
            sigmoid="quadratic",
        ).mean()
        small_control = (4 + small_control) / 5

        com_velocity = self.robot.center_of_mass_velocity()[0]
        move = rewards.tolerance(
            com_velocity,
            bounds=(_WALK_SPEED, float("inf")),
            margin=_WALK_SPEED,
            value_at_margin=0,
            sigmoid="linear",
        )
        move = (5 * move + 1) / 6
        return stand_reward * small_control * move, {  # small_control *
            "stand_reward": stand_reward,
            "small_control": small_control,
            "move": move,
            "standing": standing,
            "upright": upright,
        }

    def get_terminated(self):
        return self.robot.torso_upright() < 0.1, {}


class Stair(ClimbingUpwards):
    pass


class Slide(ClimbingUpwards):
    pass


class Hurdle(Walk):
    camera_name = "cam_hurdle"
    _hurdle_x_positions = np.arange(7.0, 71.0, 7.0)
    _hurdle_clearance = 3.0
    _hurdle_geom_names = tuple(
        f"hurdle_{hurdle_index}_collision" for hurdle_index in range(1, 11)
    )

    def __init__(
        self,
        robot=None,
        env=None,
        *,
        hurdle_count=10,
        hurdle_horizon=1000,
        **kwargs,
    ):
        super().__init__(robot, env, **kwargs)
        self._hurdle_count = hurdle_count
        self._hurdle_horizon = hurdle_horizon
        self._passed_hurdle_ids = set()
        self._hurdle_contact_ids = set()
        self._falls_or_nonfoot_ground_contact = False
        self._previous_root_x = None
        self._episode_steps = 0

        if env is not None:
            self._hurdle_geom_ids = tuple(
                env.model.geom(name).id for name in self._hurdle_geom_names
            )
            self._floor_geom_id = env.model.geom("floor").id
            foot_body_ids = tuple(
                env.model.site_bodyid[env.model.site(site_name).id]
                for site_name in ("left_foot", "right_foot")
            )
            self._foot_geom_ids = tuple(
                np.flatnonzero(np.isin(env.model.geom_bodyid, foot_body_ids))
            )

    def reset_model(self):
        self._passed_hurdle_ids = set()
        self._hurdle_contact_ids = set()
        self._falls_or_nonfoot_ground_contact = False
        self._previous_root_x = self._env.data.qpos[0]
        self._episode_steps = 0
        return super().reset_model()

    @property
    def observation_space(self):
        return Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.robot.dof * 2,),
            dtype=np.float64,
        )

    def get_obs(self):
        state = super().get_obs()
        self._record_contacts()
        prospective_passed_ids = (
            self._passed_hurdle_ids | self._newly_cleared_hurdle_ids()
        )
        terminal = (
            len(prospective_passed_ids) == self._hurdle_count
            or bool(self._hurdle_contact_ids)
            or self._falls_or_nonfoot_ground_contact
        )
        if terminal:
            next_hurdle_distance = 0.0
        else:
            next_hurdle_index = next(
                index
                for index, geom_id in enumerate(
                    self._hurdle_geom_ids[: self._hurdle_count]
                )
                if geom_id not in prospective_passed_ids
            )
            next_hurdle_distance = max(
                self._hurdle_x_positions[next_hurdle_index]
                - self._env.data.qpos[0],
                0.0,
            )
        return np.concatenate((state, [next_hurdle_distance]))

    def _newly_cleared_hurdle_ids(self):
        root_x = self._env.data.qpos[0]
        return {
            geom_id
            for hurdle_x, geom_id in zip(
                self._hurdle_x_positions[: self._hurdle_count],
                self._hurdle_geom_ids[: self._hurdle_count],
            )
            if self._previous_root_x < hurdle_x + self._hurdle_clearance <= root_x
            and geom_id not in self._passed_hurdle_ids
            and geom_id not in self._hurdle_contact_ids
        }

    def _current_contacts(self):
        current_hurdle_contact_ids = set()
        nonfoot_ground_contact = False
        for pair in self._env.data.contact.geom:
            pair_ids = set(pair)
            current_hurdle_contact_ids.update(
                pair_ids.intersection(self._hurdle_geom_ids[: self._hurdle_count])
            )
            if self._floor_geom_id in pair_ids and not pair_ids.intersection(
                self._foot_geom_ids
            ):
                nonfoot_ground_contact = True
        return current_hurdle_contact_ids, nonfoot_ground_contact

    def _record_contacts(self):
        current_hurdle_contact_ids, nonfoot_ground_contact = self._current_contacts()
        self._hurdle_contact_ids.update(current_hurdle_contact_ids)
        self._falls_or_nonfoot_ground_contact = (
            self._falls_or_nonfoot_ground_contact
            or self._env.data.qpos[2] < 0.2
            or nonfoot_ground_contact
        )

    def get_reward(self):
        self._episode_steps += 1
        self._record_contacts()

        root_x = self._env.data.qpos[0]
        previous_root_x = self._previous_root_x
        previous_hurdles_passed = len(self._passed_hurdle_ids)
        self._passed_hurdle_ids.update(self._newly_cleared_hurdle_ids())
        self._previous_root_x = root_x

        progress = root_x - previous_root_x
        hurdles_passed = len(self._passed_hurdle_ids)
        new_hurdles_passed = hurdles_passed - previous_hurdles_passed
        course_complete = hurdles_passed == self._hurdle_count

        root_y = self._env.data.qpos[1]
        w, x, y, z = self._env.data.qpos[3:7]
        yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        teacher_accepted = bool(
            course_complete
            and not self._hurdle_contact_ids
            and not self._falls_or_nonfoot_ground_contact
            and abs(root_y) <= 0.5
            and abs(yaw) <= np.deg2rad(5.0)
        )
        hurdle_contact = bool(self._hurdle_contact_ids)
        fall_or_nonfoot_ground_contact = self._falls_or_nonfoot_ground_contact
        physical_failure = hurdle_contact or fall_or_nonfoot_ground_contact
        quality_failure = (
            course_complete and not teacher_accepted and not physical_failure
        )
        timeout = self._episode_steps >= self._hurdle_horizon
        forward_quality = (
            np.clip(progress, 0.0, 0.05) / 0.05
            * np.exp(-2.0 * root_y**2 - yaw**2)
        )
        step_reward = 0.02 * forward_quality
        hurdle_pass_reward = 5.0 * new_hurdles_passed * (
            not physical_failure and (not course_complete or teacher_accepted)
        )
        course_complete_reward = 200.0 * teacher_accepted
        failure_penalty = -200.0 * physical_failure
        quality_failure_penalty = -200.0 * quality_failure
        timeout_penalty = -200.0 * (timeout and not course_complete)
        if physical_failure:
            reward = failure_penalty
        elif teacher_accepted:
            reward = step_reward + hurdle_pass_reward + course_complete_reward
        elif quality_failure:
            reward = quality_failure_penalty
        elif timeout:
            reward = timeout_penalty
        else:
            reward = step_reward + hurdle_pass_reward

        return reward, {
            "progress": progress,
            "forward_quality": forward_quality,
            "step_reward": step_reward,
            "new_hurdles_passed": new_hurdles_passed,
            "hurdle_pass_reward": hurdle_pass_reward,
            "course_complete_reward": course_complete_reward,
            "failure_penalty": failure_penalty,
            "quality_failure_penalty": quality_failure_penalty,
            "timeout_penalty": timeout_penalty,
            "root_y": root_y,
            "yaw": yaw,
            "hurdles_passed": hurdles_passed,
            "first_hurdle_clean_pass": self._hurdle_geom_ids[0]
                in self._passed_hurdle_ids,
            "hurdle_contacts": len(self._hurdle_contact_ids),
            "falls_or_nonfoot_ground_contact": self._falls_or_nonfoot_ground_contact,
            "course_complete": course_complete,
            "teacher_accepted": teacher_accepted,
            "timeout": timeout,
            "hurdle_horizon": self._hurdle_horizon,
        }

    def get_terminated(self):
        self._record_contacts()
        return (
            bool(
                self._hurdle_contact_ids
                or self._falls_or_nonfoot_ground_contact
                or len(self._passed_hurdle_ids) == self._hurdle_count
                or self._episode_steps >= self._hurdle_horizon
            ),
            {"timeout": self._episode_steps >= self._hurdle_horizon},
        )


class Sit(Task):
    qpos0_robot = {
        "h1": "0 0 0.98 1 0 0 0 0 0 -0.4 0.8 -0.4 0 0 -0.4 0.8 -0.4 0 0 0 0 0 0 0 0 0",
        "h1hand": "0 0 0.98 1 0 0 0 0 0 -0.4 0.8 -0.4 0 0 -0.4 0.8 -0.4 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0",
        "h1touch": "0 0 0.98 1 0 0 0 0 0 -0.4 0.8 -0.4 0 0 -0.4 0.8 -0.4 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0",
        "g1": "0 0 0.75 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 -1.57 0 0 0 0 0 0 0 0 0 0 0 1.57 0 0 0 0 0 0 0"
    }
    dof = 0
    vels = 0
    success_bar = 750

    @property
    def observation_space(self):
        return Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.robot.dof * 2 - 1 + self.dof + self.vels,),
            dtype=np.float64,
        )

    def get_reward(self):
        sitting = rewards.tolerance(
            self._env.data.qpos[2], bounds=(0.68, 0.72), margin=0.2
        )
        chair_location = self._env.named.data.xpos["chair"]
        on_chair = rewards.tolerance(
            self._env.data.qpos[0] - chair_location[0], bounds=(-0.19, 0.19), margin=0.2
        ) * rewards.tolerance(self._env.data.qpos[1] - chair_location[1], margin=0.1)
        sitting_posture = rewards.tolerance(
            self.robot.head_height() - self._env.named.data.site_xpos["imu", "z"],
            bounds=(0.35, 0.45),
            margin=0.3,
        )
        upright = rewards.tolerance(
            self.robot.torso_upright(),
            bounds=(0.95, float("inf")),
            sigmoid="linear",
            margin=0.9,
            value_at_margin=0,
        )
        sit_reward = (0.5 * sitting + 0.5 * on_chair) * upright * sitting_posture
        small_control = rewards.tolerance(
            self.robot.actuator_forces(),
            margin=10,
            value_at_margin=0,
            sigmoid="quadratic",
        ).mean()
        small_control = (4 + small_control) / 5

        horizontal_velocity = self.robot.center_of_mass_velocity()[[0, 1]]
        dont_move = rewards.tolerance(horizontal_velocity, margin=2).mean()
        return small_control * sit_reward * dont_move, {
            "small_control": small_control,
            "sit_reward": sit_reward,
            "dont_move": dont_move,
            "sitting": sitting,
            "upright": upright,
            "sitting_posture": sitting_posture,
        }

    def get_terminated(self):
        return self._env.data.qpos[2] < 0.5, {}

    @staticmethod
    def euler_to_quat(angles):
        cr, cp, cy = np.cos(angles[0] / 2), np.cos(angles[1] / 2), np.cos(angles[2] / 2)
        sr, sp, sy = np.sin(angles[0] / 2), np.sin(angles[1] / 2), np.sin(angles[2] / 2)
        return np.array(
            [
                cr * cp * cy + sr * sp * sy,
                sr * cp * cy - cr * sp * sy,
                cr * sp * cy + sr * cp * sy,
                cr * cp * sy - sr * sp * cy,
            ]
        )


class SitHard(Sit):
    qpos0_robot = {
        "h1": "0 0 0.98 1 0 0 0 0 0 -0.4 0.8 -0.4 0 0 -0.4 0.8 -0.4 0 0 0 0 0 0 0 0 0 -0.25 0 0 1 0 0 0",
        "h1hand": "0 0 0.98 1 0 0 0 0 0 -0.4 0.8 -0.4 0 0 -0.4 0.8 -0.4 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 -0.25 0 0 1 0 0 0",
        "h1touch": "0 0 0.98 1 0 0 0 0 0 -0.4 0.8 -0.4 0 0 -0.4 0.8 -0.4 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 -0.25 0 0 1 0 0 0",
        "g1": "0 0 0.75 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 -1.57 0 0 0 0 0 0 0 0 0 0 0 1.57 0 0 0 0 0 0 0 -0.25 0 0 1 0 0 0"
    }

    dof = 7
    vels = 6

    def reset_model(self):
        position = self._env.data.qpos.flat.copy()
        velocity = self._env.data.qvel.flat.copy()
        position[0] = np.random.uniform(0.2, 0.4)
        position[1] = np.random.uniform(-0.15, 0.15)
        rotation_angle = np.random.uniform(-1.8, 1.8)
        position[3:7] = self.euler_to_quat(np.array([0, 0, rotation_angle]))
        self._env.set_state(position, velocity)
        return super().reset_model()
