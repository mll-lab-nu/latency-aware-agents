from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from latency_bench.envs.humanoidbench_sonic_contract import SONIC_ACTION_DIM
from latency_bench.envs.humanoidbench_sonic_teacher import (
    JUMP_STATE,
    RUN_STATE,
    SonicHurdleTeacher,
    evaluate_trigger_distances,
    match_reference_phase,
    select_best_result,
)


def _reference_qpos(length):
    qpos = np.zeros((length, 36), dtype=np.float32)
    qpos[:, 2] = 0.75
    qpos[:, 3] = 1.0
    return qpos


def test_teacher_switches_on_root_distance_and_loops_run_tokens():
    run = np.arange(3 * 64, dtype=np.float32).reshape(3, 64)
    jump = np.full((2, 64), 10.0, dtype=np.float32)
    run_reference = _reference_qpos(3)
    run_reference[:, 7:] = np.arange(3)[:, None]
    jump_reference = _reference_qpos(2)
    jump_reference[-1, 7:] = 1.0
    teacher = SonicHurdleTeacher(
        run,
        jump,
        run_reference,
        jump_reference,
        np.zeros((3, 2), dtype=bool),
        np.zeros((2, 2), dtype=bool),
        np.full(14, -1.0, dtype=np.float32),
        trigger_distance=1.0,
        hurdle_positions=(5.0,),
    )

    np.testing.assert_array_equal(teacher(0.0)[:64], run[0])
    np.testing.assert_array_equal(teacher(0.0)[:64], run[1])
    assert teacher.state == RUN_STATE

    np.testing.assert_array_equal(teacher(4.0)[:64], jump[0])
    assert teacher.state == JUMP_STATE

    np.testing.assert_array_equal(teacher(100.0)[:64], jump[1])
    assert teacher.state == RUN_STATE
    np.testing.assert_array_equal(
        teacher(100.0)[:64], run[teacher.resume_run_phase]
    )


def test_teacher_action_is_sonic_body_token_plus_neutral_hands():
    run = np.zeros((2, 64), dtype=np.float32)
    jump = np.ones((2, 64), dtype=np.float32)
    action = SonicHurdleTeacher(
        run,
        jump,
        _reference_qpos(2),
        _reference_qpos(2),
        np.zeros((2, 2), dtype=bool),
        np.zeros((2, 2), dtype=bool),
        np.linspace(-1.0, 1.0, 14, dtype=np.float32),
        trigger_distance=1.0,
        hurdle_positions=(1.0,),
    )(0.0)

    assert action.shape == (SONIC_ACTION_DIM,)
    assert action.dtype == np.float32
    np.testing.assert_array_equal(
        action[64:], np.linspace(-1.0, 1.0, 14, dtype=np.float32)
    )


def test_reference_phase_matching_uses_position_velocity_and_contact_cost():
    actual_q = np.zeros(29)
    actual_dq = np.zeros(29)
    actual_contact = np.asarray([True, False])
    reference_qpos = _reference_qpos(3)
    reference_qpos[:, 7:9] = np.asarray(
        [[0.2, 0.2], [0.0, 0.0], [0.1, 0.1]], dtype=np.float32
    )
    reference_qvel = np.zeros((3, 29), dtype=np.float32)
    reference_contacts = np.asarray(
        [[False, False], [True, False], [False, True]], dtype=bool
    )

    assert (
        match_reference_phase(
            actual_q,
            actual_dq,
            actual_contact,
            reference_qpos,
            reference_qvel,
            reference_contacts,
        )
        == 1
    )


def test_select_best_result_rejects_contacts_and_falls_before_progress():
    def result(distance, *, complete, contacts, passed, tracking):
        return {
            "trigger_distance": distance,
            "info": {
                "course_complete": complete,
                "hurdle_contacts": contacts,
                "falls_or_nonfoot_ground_contact": False,
                "hurdles_passed": passed,
                "reference_tracking_joint_rmse": tracking,
            },
        }

    results = [
        result(0.5, complete=True, contacts=1, passed=10, tracking=0.1),
        result(1.0, complete=False, contacts=0, passed=5, tracking=0.4),
        result(1.5, complete=False, contacts=0, passed=5, tracking=0.2),
    ]

    assert select_best_result(results) is results[2]


def test_trigger_sweep_reuses_seed_and_closes_each_environment():
    environments = []

    class Env:
        def __init__(self):
            self.spec = SimpleNamespace(max_episode_steps=1)
            self.unwrapped = self
            qpos = np.zeros(36)
            qpos[2] = 0.75
            qpos[3] = 1.0
            self.data = SimpleNamespace(
                qpos=qpos,
                qvel=np.zeros(35),
                site_xpos=np.ones((2, 3)),
            )
            self._body_qpos_addresses = np.arange(7, 36)
            self._body_qvel_addresses = np.arange(6, 35)
            self.model = SimpleNamespace(
                site=lambda name: SimpleNamespace(
                    id={"left_foot": 0, "right_foot": 1}[name]
                )
            )
            self.closed = False
            self.seed = None

        def reset(self, *, seed):
            self.seed = seed
            return None, {}

        def step(self, action):
            assert action.shape == (SONIC_ACTION_DIM,)
            return None, 0.0, False, False, {
                "course_complete": False,
                "hurdle_contacts": 0,
                "falls_or_nonfoot_ground_contact": False,
                "hurdles_passed": 0,
                "teacher_accepted": False,
            }

        def close(self):
            self.closed = True

    def factory():
        env = Env()
        environments.append(env)
        return env

    sweep = evaluate_trigger_distances(
        factory,
        np.zeros((2, 64), dtype=np.float32),
        np.ones((2, 64), dtype=np.float32),
        _reference_qpos(2),
        _reference_qpos(2),
        np.zeros((2, 2), dtype=bool),
        np.zeros((2, 2), dtype=bool),
        np.zeros(14, dtype=np.float32),
        (0.5, 1.0),
        seed=7,
        hurdle_positions=(2.0,),
    )

    assert len(sweep["results"]) == 2
    assert sweep["teacher_accepted"] is False
    assert [env.seed for env in environments] == [7, 7]
    assert all(env.closed for env in environments)
