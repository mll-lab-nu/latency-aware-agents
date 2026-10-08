import numpy as np

from starVLA.training.rl_games import action_cc_f1 as cc


def _demon(action_id):
    v = [0.0] * 6
    v[action_id] = 1.0
    return v


def _flap(x):
    v = [0.0, 0.0]
    v[x] = 1.0
    return v


def _fact(turn, move, strafe, attack):
    v = [0.0] * 11
    v[turn] = 1.0
    v[3 + move] = 1.0
    v[6 + strafe] = 1.0
    v[9 + attack] = 1.0
    return v


NOOP, FIRE, RIGHT, LEFT, RIGHTFIRE, LEFTFIRE = (
    _demon(0),
    _demon(1),
    _demon(2),
    _demon(3),
    _demon(4),
    _demon(5),
)


def _episode(frames, teacher_frames, model_frames, hit, miss):
    teacher = [hit if f in teacher_frames else miss for f in frames]
    model = [hit if f in model_frames else miss for f in frames]
    return {"frames": frames, "teacher": teacher, "model": model}


def test_demon_fire_worked_example_k0():
    # teacher fires {10,11,20}, model fires {11,12,50}: at K=0 only frame 11 overlaps.
    ep = _episode(list(range(60)), {10, 11, 20}, {11, 12, 50}, FIRE, NOOP)
    metrics, _ = cc.compute_from_episodes("demon_attack", [ep], k=0)
    assert abs(metrics["eval/demon_fire_f1"] - 1 / 3) < 1e-9


def test_demon_fire_worked_example_k1_forgives_one_frame_lag():
    # ±1 tolerance forgives the one-frame-late fires; only the true miss(20)/spurious(50) remain.
    ep = _episode(list(range(60)), {10, 11, 20}, {11, 12, 50}, FIRE, NOOP)
    metrics, _ = cc.compute_from_episodes("demon_attack", [ep], k=1)
    assert abs(metrics["eval/demon_fire_f1"] - 2 / 3) < 1e-9
    assert abs(metrics["eval/demon_fire_rate_teacher"] - 3 / 60) < 1e-9
    assert abs(metrics["eval/demon_fire_rate_model"] - 3 / 60) < 1e-9


def test_no_cross_episode_matching():
    # teacher fire at end of ep A, model fire at start of ep B: must not match across episodes.
    ep_a = _episode(list(range(10)), {9}, set(), FIRE, NOOP)
    ep_b = _episode(list(range(10)), set(), {0}, FIRE, NOOP)
    metrics, _ = cc.compute_from_episodes("demon_attack", [ep_a, ep_b], k=2)
    assert metrics["eval/demon_fire_recall"] == 0.0
    assert metrics["eval/demon_fire_precision"] == 0.0


def test_harmonic_mean_zeroes_on_collapsed_group():
    # perfect fire, but model never moves while teacher does -> move-F1 = 0 -> CC-F1 = 0.
    frames = list(range(20))
    teacher = [FIRE if f < 5 else (LEFT if f < 10 else NOOP) for f in frames]
    model = [FIRE if f < 5 else NOOP for f in frames]
    metrics, _ = cc.compute_from_episodes(
        "demon_attack", [{"frames": frames, "teacher": teacher, "model": model}], k=0
    )
    assert metrics["eval/demon_fire_f1"] > 0.9
    assert metrics["eval/demon_move_f1"] == 0.0
    assert metrics["eval/demon_cc_f1"] == 0.0


def test_demon_composite_actions_feed_fire_and_movement_metrics():
    frames = list(range(4))
    teacher = [RIGHTFIRE, LEFTFIRE, FIRE, NOOP]
    model = [RIGHTFIRE, LEFTFIRE, FIRE, NOOP]
    metrics, _ = cc.compute_from_episodes(
        "demon_attack", [{"frames": frames, "teacher": teacher, "model": model}], k=0
    )

    assert metrics["eval/demon_fire_f1"] == 1.0
    assert metrics["eval/demon_left_f1"] == 1.0
    assert metrics["eval/demon_right_f1"] == 1.0
    assert metrics["eval/demon_move_f1"] == 1.0
    assert metrics["eval/demon_cc_f1"] == 1.0


def test_flappy_single_component_frame_level():
    frames = list(range(100))
    teacher = [_flap(1) if f % 5 == 0 else _flap(0) for f in frames]
    model = [_flap(1) if f % 5 == 1 else _flap(0) for f in frames]  # always one frame late
    metrics, _ = cc.compute_from_episodes(
        "flappy", [{"frames": frames, "teacher": teacher, "model": model}], k=0
    )
    # 20 teacher flaps, 20 model flaps, zero overlap at K=0.
    assert metrics["eval/flap_f1"] == 0.0
    assert metrics["eval/flap_cc_f1"] == 0.0
    assert abs(metrics["eval/flap_rate_teacher"] - 0.2) < 1e-9
    assert abs(metrics["eval/flap_rate_model"] - 0.2) < 1e-9


def test_deadly_factorized_components():
    frames = list(range(10))
    teacher = [
        _fact(0, 0, 0, 1)
        if f < 3
        else (_fact(1, 0, 0, 0) if f < 5 else (_fact(0, 1, 0, 0) if f < 7 else _fact(0, 0, 0, 0)))
        for f in frames
    ]
    model = [
        _fact(0, 0, 0, 1) if f < 3 else (_fact(0, 1, 0, 0) if f < 7 else _fact(0, 0, 0, 0))
        for f in frames
    ]  # matches attack + move, misses turn
    metrics, _ = cc.compute_from_episodes(
        "deadly_corridor", [{"frames": frames, "teacher": teacher, "model": model}],
        deadly_layout=cc.DEADLY_FACTORIZED_11, k=1,
    )
    assert 0.0 <= metrics["eval/deadly_cc_f1"] <= 1.0
    assert metrics["eval/deadly_attack_f1"] > 0.9
    assert metrics["eval/deadly_turn_f1"] < 0.5


def test_deadly_multibinary_components_and_main_metric_names():
    frames = list(range(5))
    noop = [0, 0, 0, 0, 0, 0, 0]
    attack = [0, 0, 0, 0, 0, 0, 1]
    turn_left = [0, 0, 0, 0, 1, 0, 0]
    move_forward = [1, 0, 0, 0, 0, 0, 0]
    strafe_left = [0, 0, 1, 0, 0, 0, 0]
    teacher = [attack, turn_left, move_forward, strafe_left, noop]
    model = [attack, turn_left, move_forward, noop, noop]

    metrics, _ = cc.compute_from_episodes(
        "deadly_corridor", [{"frames": frames, "teacher": teacher, "model": model}],
        deadly_layout=cc.DEADLY_MULTIBINARY_7, k=0,
    )

    assert metrics["eval/deadly_attack_f1"] == 1.0
    assert metrics["eval/deadly_turn_f1"] == 1.0
    assert metrics["eval/deadly_move_f1"] == 1.0
    assert metrics["eval/deadly_strafe_left_recall"] == 0.0
    assert "eval/deadly_cc_f1" in metrics


def test_group_macro_ignores_components_absent_from_teacher_and_model():
    frames = list(range(3))
    teacher = [LEFT, LEFT, NOOP]
    model = [LEFT, LEFT, NOOP]

    metrics, _ = cc.compute_from_episodes(
        "demon_attack", [{"frames": frames, "teacher": teacher, "model": model}], k=0
    )

    assert metrics["eval/demon_left_f1"] == 1.0
    assert metrics["eval/demon_right_f1"] == 0.0
    assert metrics["eval/demon_move_f1"] == 1.0


def test_empty_input_is_safe():
    metrics, _ = cc.compute_from_episodes("demon_attack", [], k=1)
    assert all(v == 0.0 for v in metrics.values())


def test_distributed_counts_sum_equals_single_pass():
    # Two per-rank count vectors summed must equal one combined pass (validates all_reduce).
    ep1 = [_episode(list(range(10)), {0, 1, 2}, {0, 1, 2}, FIRE, NOOP)]
    ep2 = [_episode(list(range(10)), {0, 1, 2, 3}, set(), LEFT, NOOP)]
    spec = cc.get_spec("demon_attack")
    _, c1 = cc.compute_from_episodes("demon_attack", ep1, k=1)
    _, c2 = cc.compute_from_episodes("demon_attack", ep2, k=1)
    split = cc.reduce_metrics(spec, c1 + c2)
    combined, _ = cc.compute_from_episodes("demon_attack", ep1 + ep2, k=1)
    assert all(abs(split[key] - combined[key]) < 1e-9 for key in combined)
