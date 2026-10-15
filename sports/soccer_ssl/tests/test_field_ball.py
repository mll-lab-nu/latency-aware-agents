"""The field (rulebook §2.1, Division B), the frames, the free-kick placement positions (ssl-game-controller
placementPos.go) and the ball's physics (sim/ball.py: proposal §4.1-4.2, TIGERs' fitted ball model).
Run: python3 sports/soccer_ssl/tests/test_field_ball.py."""

from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from _util import F, run_all  # noqa: E402

from sim import ball as BALL  # noqa: E402
from sim import frames as FR  # noqa: E402


def _empty():
    spec, h = F.build_scene(n_per_team=0, seed=0, poses={})
    m = spec.compile()
    d = mujoco.MjData(m)
    b = BALL.BallState(m, d)
    return m, d, b


def _geom(m, d, name):
    g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, name)
    return d.geom_xpos[g].copy(), m.geom_size[g].copy()


def test_field_dimensions_division_b():
    """§2.1: 9 x 6 m playing area, walls 0.3 m beyond the lines, goals 1.0 m wide, 0.18 m deep, 0.16 m high."""
    m, d, _ = _empty()
    mujoco.mj_forward(m, d)
    p, s = _geom(m, d, "wall_pos_x")
    assert abs(p[0] - s[0] - 4.8) < 1e-9 and abs(2 * s[2] - 0.10) < 1e-9
    p, s = _geom(m, d, "wall_neg_y")
    assert abs(-p[1] - s[1] - 3.3) < 1e-9
    p, s = _geom(m, d, "goal_pos_foam_side_l")
    assert abs((p[1] - s[1]) - 0.5) < 1e-9 and abs(2 * s[2] - 0.16) < 1e-9  # inner width 1.0 m
    p, s = _geom(m, d, "goal_pos_foam_back")
    assert abs((p[0] - s[0]) - (4.505 + 0.18)) < 1e-9  # inner depth 0.18 m
    p, s = _geom(m, d, "carpet")
    assert abs(s[0] - 5.2) < 1e-9 and abs(s[1] - 3.7) < 1e-9  # 10.4 x 7.4 m
    assert F.defense_area(1) == (3.495, 4.505, -1.005, 1.005)
    assert F.penalty_mark(1) == (-1.5, 0.0) and F.penalty_mark(-1) == (1.5, 0.0)  # 6 m from the goal
    assert m.opt.timestep == 0.001 and m.opt.cone == mujoco.mjtCone.mjCONE_ELLIPTIC


def test_frames_round_trip_and_mirror():
    """interfaces §2: each team attacks +x in its own frame; the ends change at half time."""
    assert FR.attack_sign("blue", 1) == 1 and FR.attack_sign("yellow", 1) == -1 and FR.attack_sign("blue", 2) == -1
    for sgn in (1, -1):
        for pose in [(1.0, -2.0, 0.3), (-4.0, 2.5, -3.0)]:
            t = FR.to_team(sgn, *pose)
            back = FR.to_world(sgn, *t)
            assert np.allclose(back[:2], pose[:2]) and abs(FR.wrap(back[2] - pose[2])) < 1e-12
    assert FR.to_team(-1, 4.5, 0.0) == (-4.5, -0.0)


def test_placement_positions_follow_the_game_controller():
    """§5.3.3, §6: free kicks >= 0.2 m from the lines and >= 1 m from the defense areas (GC placementPos.go)."""
    assert F.goal_kick_pos(4.6, 0.7) == (3.5, 2.8)
    assert F.corner_kick_pos(-4.6, -0.7) == (-4.3, -2.8)
    assert F.throw_in_pos(4.45, 3.05) == (4.3, 2.8)
    assert F.valid_free_kick_pos(4.0, 0.2) == (2.5, 0.2)  # out of the defense area, the shorter way
    assert F.valid_free_kick_pos(4.0, 1.8) == (4.0, 2.0)
    assert F.in_goal(4.6, 0.3, 0.05) == 1 and F.in_goal(4.6, 0.6, 0.05) == 0 and F.in_goal(-4.6, 0.0, 0.2) == 0


def test_ball_rolling_deceleration():
    """§4.1: rolling deceleration 0.45 m/s^2 (TIGERs accRoll) within 3 %."""
    m, d, b = _empty()
    b.place(0.0, -2.0, BALL.RADIUS, vel=(2.0, 0.0, 0.0))
    d.qvel[b.va + 4] = 2.0 / BALL.RADIUS  # pure rolling about +y
    ts, vs = [], []
    for i in range(3000):
        mujoco.mj_step(m, d)
        if i % 10 == 0:
            ts.append(d.time)
            vs.append(b.vel[0])
    ts, vs = np.array(ts), np.array(vs)
    k = (vs > 0.5) & (vs < 1.8)
    a = -np.polyfit(ts[k], vs[k], 1)[0]
    assert abs(a - BALL.ROLL_DECEL) / BALL.ROLL_DECEL < 0.03, a


def test_ball_slides_then_rolls_and_does_not_hop():
    """§4.1: a 6.5 m/s flat ball without spin slides (about 3.4 m/s^2) and rolls at about 5/7 of its speed after
    about 0.7 s; it does not hop off the carpet by more than 3 mm."""
    m, d, b = _empty()
    for _ in range(200):
        mujoco.mj_step(m, d)
    z0 = float(b.pos[2])
    b.place(-4.0, 0.0, z0, vel=(6.5, 0.0, 0.0))
    zmax = 0.0
    for _ in range(1200):
        mujoco.mj_step(m, d)
        zmax = max(zmax, float(b.pos[2]) - z0)
    slip = b.vel[0] - b.spin[1] * BALL.RADIUS
    assert abs(slip) < 0.2 and 4.2 < b.vel[0] < 4.7, (b.vel[0], slip)
    assert zmax < 0.003, zmax


def test_ball_bounce_restitution():
    """§4.2: a chipped ball bounces with a vertical restitution of about 0.54 (TIGERs chipDampingZ 0.54)."""
    m, d, b = _empty()
    v = 4.0
    b.place(-3.0, 0.0, BALL.RADIUS + 0.001, vel=(v * math.cos(math.pi / 4), 0.0, v * math.sin(math.pi / 4)))
    peaks, last, rising = [], float(b.pos[2]), True
    for _ in range(4000):
        mujoco.mj_step(m, d)
        z = float(b.pos[2])
        if rising and z < last:
            peaks.append(last - BALL.RADIUS)
            rising = False
        if not rising and z > last + 1e-6:
            rising = True
        last = z
        if len(peaks) >= 3:
            break
    cor = [math.sqrt(peaks[i + 1] / peaks[i]) for i in range(2)]
    assert all(0.50 < c < 0.58 for c in cor), (peaks, cor)


def test_goal_foam_absorbs_and_walls_keep_the_ball_in():
    """§2.1.4: the goal's inner faces absorb the ball (rebound < 20 %); the 0.1 m walls stop a rolling ball."""
    m, d, b = _empty()
    b.place(4.4, 0.0, BALL.RADIUS, vel=(3.0, 0.0, 0.0))
    vmin = 0.0
    for _ in range(300):
        mujoco.mj_step(m, d)
        vmin = min(vmin, float(b.vel[0]))
    assert -vmin < 0.2 * 3.0 and abs(b.pos[0]) < 4.505 + 0.18, (vmin, b.pos)
    b.place(0.0, 3.1, BALL.RADIUS, vel=(0.0, 3.0, 0.0))
    for _ in range(500):
        mujoco.mj_step(m, d)
    assert b.pos[1] < 3.3 and b.vel[1] < 0.0


if __name__ == "__main__":
    sys.exit(1 if run_all(globals()) else 0)
