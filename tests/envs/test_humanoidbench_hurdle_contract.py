from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from humanoid_bench.envs.basic_locomotion_envs import Hurdle


class _Robot:
    dof = 7

    def head_height(self):
        return 2.0

    def torso_upright(self):
        return 1.0

    def actuator_forces(self):
        return np.zeros(1)

    def center_of_mass_velocity(self):
        return np.array([5.0, 0.0, 0.0])


class _Model:
    geom_ids = {
        "floor": 0,
        **{f"hurdle_{index}_collision": index for index in range(1, 11)},
    }
    geom_bodyid = np.array([0] + [99] * 10 + [10, 11])
    site_bodyid = np.array([10, 11])

    def geom(self, name):
        return SimpleNamespace(id=self.geom_ids[name])

    def site(self, name):
        return SimpleNamespace(id={"left_foot": 0, "right_foot": 1}[name])


class _Data:
    def __init__(self):
        self.qpos = np.array([0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0])
        self.qvel = np.zeros(6)
        self.contact = SimpleNamespace(geom=np.empty((0, 2), dtype=int))


class _Env:
    def __init__(self):
        self.render_mode = None
        self.model = _Model()
        self.data = _Data()
        self.named = SimpleNamespace(
            data=SimpleNamespace(
                geom_xpos=SimpleNamespace(
                    axes=SimpleNamespace(
                        row=SimpleNamespace(
                            names=[
                                "left_barrier_collision",
                                "right_barrier_collision",
                                "behind_barrier_collision",
                            ]
                        )
                    )
                )
            )
        )


def _task():
    env = _Env()
    task = Hurdle(_Robot(), env, hurdle_count=1)
    task.reset_model()
    return task, env


def test_hurdle_teacher_acceptance_requires_a_clean_crossing():
    successful_task, successful_env = _task()
    successful_env.data.qpos[0] = 10.0
    _, successful_info = successful_task.get_reward()

    failed_task, failed_env = _task()
    failed_env.data.contact.geom = np.array([[1, 11]])
    failed_env.data.qpos[0] = 10.0
    _, failed_info = failed_task.get_reward()

    assert successful_info["teacher_accepted"] is True
    assert successful_task.get_terminated()[0] is True
    assert failed_info["teacher_accepted"] is False
    assert failed_task.get_terminated()[0] is True
