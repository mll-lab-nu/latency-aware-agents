"""bodies/trajopt.py: fixed-time jerk-limited trajectories (numpy only).
Run: python bodies/tests/test_trajopt.py (or pytest; the sim image has no pytest)"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from bodies import limits as AL  # noqa: E402
from bodies import trajopt as T  # noqa: E402

LIM = AL.load("franka_panda")
DT = 0.01
Q0 = np.array([0.0, -0.5, 0.0, -2.2, 0.0, 1.8, 0.8])


def _within(out, lim, margin, tol=1.02):
    assert out["viol"] <= margin * tol, out["viol"]
    assert np.all(out["q"] >= lim.range_lo - 1e-6) and np.all(out["q"] <= lim.range_hi + 1e-6)


def test_reachable_stroke_meets_contact_within_limits():
    q1 = Q0 + np.array([0.3, 0.2, -0.1, 0.3, 0.2, -0.2, 0.3])
    v1 = np.array([1.0, 0.6, -0.4, 0.9, 0.5, -0.8, 1.2])
    out = T.stroke(Q0, np.zeros(7), np.zeros(7), 60, DT, LIM, q1, v1)
    assert out["contact_err_q"] < 1e-3
    assert out["scale"] > 0.98
    assert out["n_stroke"] > 0                       # a wind-up at rest, then the stroke
    _within(out, LIM, 0.95)
    k = 60 - out["n_stroke"] - 1                     # at rest at the wind-up
    assert np.max(np.abs(out["v"][k])) < 0.05


def test_follow_through_ends_at_rest():
    q1 = Q0 + 0.2
    v1 = np.full(7, 0.8)
    out = T.stroke(Q0, np.zeros(7), np.zeros(7), 50, DT, LIM, q1, v1, n_follow=30)
    assert np.max(np.abs(out["v"][-1])) < 0.05 and np.max(np.abs(out["a"][-1])) < 0.5


def test_infeasible_keeps_limits_and_reports():
    q1 = Q0 + np.array([1.5, 0.8, -1.0, 0.9, 1.0, -0.9, 1.5])   # far too far for 0.2 s
    v1 = np.full(7, 1.5)
    out = T.stroke(Q0, np.zeros(7), np.zeros(7), 20, DT, LIM, q1, v1)
    _within(out, LIM, 0.95)
    assert out["contact_err_q"] > 0.05               # the residual says it could not


def test_moving_start_and_torque_rate_caps():
    caps = np.array([230.0, 280.0, 250.0, 390.0, 2000.0, 2000.0, 5500.0])
    lim = LIM.with_jerk(np.minimum(LIM.jerk, caps))
    v0 = np.array([0.5, -0.3, 0.2, 0.4, -0.5, 0.3, 0.6])
    q1 = Q0 + np.array([0.2, -0.1, 0.1, 0.2, -0.2, 0.1, 0.3])
    out = T.stroke(Q0, v0, np.zeros(7), 70, DT, lim, q1, np.full(7, 0.6))
    assert out["contact_err_q"] < 2e-3
    jerk = np.abs(out["j"])
    assert np.all(jerk <= lim.jerk * 0.95 * 1.02 + 1e-6)


def test_ramp_time():
    lim = AL.ArmLimits("x", "x", -np.ones(3), np.ones(3), np.full(3, 2.2), np.full(3, 10.0), np.full(3, 1000.0),
                       np.full(3, 87.0), None)
    t = T.ramp_time(np.array([0.0, 1.0, 2.0]), lim, margin=1.0)
    assert t[0] == 0.0
    assert abs(t[1] - (1.0 / 10.0 + 10.0 / 1000.0)) < 1e-9          # trapezoid: v >= A^2/J = 0.1
    assert t[2] > t[1]


def test_via_waypoint():
    q1 = Q0 + 0.25
    via = Q0 + np.array([0.3, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0])
    out = T.stroke(Q0, np.zeros(7), np.zeros(7), 70, DT, LIM, q1, np.full(7, 0.5), via=via, via_steps=20)
    e = float(np.max(np.abs(out["q"][70 - 20 - 1] - via)))
    assert e < 0.04, e                               # soft (tolerance 2e-3, traded against the contact and the wind-up)
    assert out["contact_err_q"] < 2e-3


if __name__ == "__main__":
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for k, f in tests:
        f()
        print("ok", k)
    print(f"{len(tests)} passed")
