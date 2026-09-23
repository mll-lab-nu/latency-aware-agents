import numpy as np
from gymnasium.spaces import Discrete, MultiBinary, Tuple

from latency_bench.core.decision_action_history import DecisionActionHistory


def test_flappy_history_keeps_only_completed_decisions_and_resets():
    history = DecisionActionHistory(Discrete(2), num_envs=1, decisions=8)
    assert history.observation().shape == (1, 24)
    history.append([0], [True], [1], [0])
    history.append([0], [False], [1], [1])
    np.testing.assert_array_equal(history.observation()[0, -6:], [1, 1, 0, 0, 0, 1])
    history.reset([0])
    np.testing.assert_array_equal(history.observation(), 0)


def test_demon_history_encodes_non_noop_action_without_ordering():
    history = DecisionActionHistory(Discrete(6), num_envs=2, decisions=8)
    history.append([0, 1], [True, False], [5, 5], [2, 0])
    assert history.observation().shape == (2, 88)
    np.testing.assert_array_equal(history.observation()[0, -11:], [1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0])
    np.testing.assert_array_equal(history.observation()[1, -11:], 0)


def test_deadly_history_encodes_each_tuple_component():
    history = DecisionActionHistory(Tuple((Discrete(3), Discrete(3), Discrete(3), Discrete(2))), num_envs=1, decisions=8)
    history.append([0], [True], [(1, 0, 2, 1)], [(0, 2, 1, 0)])
    assert history.observation().shape == (1, 120)
    np.testing.assert_array_equal(history.observation()[0, -15:], [1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0])


def test_deadly_vla_button_history_matches_teacher_tuple_history():
    teacher = DecisionActionHistory(
        Tuple((Discrete(3), Discrete(3), Discrete(3), Discrete(2))), num_envs=1, decisions=8
    )
    student = DecisionActionHistory(MultiBinary(7), num_envs=1, decisions=8)
    teacher.append([0], [True], [(1, 2, 1, 1)], [(2, 1, 2, 0)])
    student.append([0], [True], [[0, 1, 1, 0, 1, 0, 1]], [[1, 0, 0, 1, 0, 1, 0]])
    np.testing.assert_array_equal(student.observation(), teacher.observation())

    student.append([0], [False], [[1, 1, 0, 0, 0, 0, 0]], [[1, 1, 0, 0, 1, 1, 0]])
    np.testing.assert_array_equal(
        student.observation()[0, -15:], [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0]
    )
