import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box

_STAND_HEIGHT = 1.65
_GAUSSIAN_SCALE = np.sqrt(-2.0 * np.log(0.1))

from humanoid_bench.tasks import Task


class BalanceBase(Task):
    frame_skip = 10
    success_bar = 800

    def __init__(self, robot=None, env=None, **kwargs):
        super().__init__(robot, env, **kwargs)
        if robot.__class__.__name__ == "G1":
            global _STAND_HEIGHT
            _STAND_HEIGHT = 1.28

    @property
    def observation_space(self):
        return Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.robot.dof * 2 - 1 + self.dof + self.vels,),
            dtype=np.float64,
        )

    def get_reward(self):
        standing_distance = max(
            0.0, (_STAND_HEIGHT + 0.37 - self.robot.head_height()) / (_STAND_HEIGHT / 4)
        )
        standing = np.exp(-0.5 * (standing_distance * _GAUSSIAN_SCALE) ** 2)
        upright_distance = max(0.0, (0.9 - self.robot.torso_upright()) / 1.9)
        upright = max(0.0, 1.0 - upright_distance)
        stand_reward = standing * upright
        force_distance = np.abs(self.robot.actuator_forces()) / 10.0
        small_control = np.maximum(0.0, 1.0 - force_distance**2).mean()
        small_control = (4 + small_control) / 5

        horizontal_velocity = self.robot.center_of_mass_velocity()[[0, 1]]
        velocity_distance = np.abs(horizontal_velocity) / 2.0
        dont_move = np.exp(
            -0.5 * (velocity_distance * _GAUSSIAN_SCALE) ** 2
        ).mean()
        return small_control * stand_reward * dont_move, {
            "small_control": small_control,
            "stand_reward": stand_reward,
            "dont_move": dont_move,
            "standing": standing,
            "upright": upright,
        }

    def get_terminated(self):
        if self._env.data.qpos[2] < 0.8:
            return True, {}
        sphere_collision_id = self._env.named.data.geom_xpos.axes.row.names.index(
            "pivot_sphere_collision"
        )
        board_collision_id = self._env.named.data.geom_xpos.axes.row.names.index(
            "stand_board_collision"
        )
        floor_collision_id = 0
        for pair in self._env.data.contact.geom:
            if sphere_collision_id in pair and all(
                [
                    allowed_collision not in pair
                    for allowed_collision in [floor_collision_id, board_collision_id]
                ]
            ):  # for no hand. if for hand, > 155
                return True, {}
            if floor_collision_id in pair and sphere_collision_id not in pair:
                return True, {}
        return False, {}


class BalanceSimple(BalanceBase):
    qpos0_robot = {
        "h1": """
            -0.1 0 1.38 1 0 0 0 0 0 -0.4 0.8 -0.4 0 0 -0.4 0.8 -0.4 0 0 0 0 0 0 0 0 0
            0 0 0.37 1 0 0 0
        """,
        "h1hand": """
            -0.1 0 1.38 1 0 0 0 0 0 -0.4 0.8 -0.4 0 0 -0.4 0.8 -0.4 0 0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            0 0 0.37 1 0 0 0
        """,
        "h1touch": """
            -0.1 0 1.38 1 0 0 0 0 0 -0.4 0.8 -0.4 0 0 -0.4 0.8 -0.4 0 0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            0 0 0.37 1 0 0 0
        """,
        "g1": """
            0 0 1.15
            1 0 0 0
            0 0 0 0 0 0
            0 0 0 0 0 0
            0
            0 0 0 0 -1.57
            0 0 0 0 0 0 0
            0 0 0 0 1.57
            0 0 0 0 0 0 0
            0 0 0.37 1 0 0 0
        """
    }
    dof = 7
    vels = 6


class BalanceHard(BalanceBase):
    qpos0_robot = {
        "h1": """
            -0.1 0 1.38 1 0 0 0 0 0 -0.4 0.8 -0.4 0 0 -0.4 0.8 -0.4 0 0 0 0 0 0 0 0 0
            0 0 0.37 1 0 0 0
            0 0 0.17 1 0 0 0
        """,
        "h1hand": """
            -0.1 0 1.38 1 0 0 0 0 0 -0.4 0.8 -0.4 0 0 -0.4 0.8 -0.4 0 0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            0 0 0.37 1 0 0 0
            0 0 0.17 1 0 0 0
        """,
        "h1touch": """
            -0.1 0 1.38 1 0 0 0 0 0 -0.4 0.8 -0.4 0 0 -0.4 0.8 -0.4 0 0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            0 0 0.37 1 0 0 0
            0 0 0.17 1 0 0 0
        """,
        "g1": """
            0 0 1.15
            1 0 0 0
            0 0 0 0 0 0
            0 0 0 0 0 0
            0
            0 0 0 0 -1.57
            0 0 0 0 0 0 0
            0 0 0 0 1.57
            0 0 0 0 0 0 0
            0 0 0.37 1 0 0 0
            0 0 0.17 1 0 0 0
        """
    }
    dof = 14
    vels = 12
