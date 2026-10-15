"""bodies/limits.py: the setpoint generator (JerkLimiter) and the output stage (TorqueRateLimiter), numpy only.

The generator must follow a step without overshoot and without ringing at any jerk a step may use (the jerk limit or a
much lower jerk_cap, as limits.jerk_caps() gives on a Panda's shoulder in play: a few hundred rad/s^3), and never
exceed the velocity, acceleration and jerk limits, whatever the reference.
Run: python bodies/tests/test_limits.py (or pytest; the sim image has no pytest)"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from bodies import limits as AL  # noqa: E402
from bodies import servo as SV  # noqa: E402

DT = 1e-3
ARMS = ("franka_panda", "franka_fr3", "kuka_iiwa14")
CAPS = (None, 100.0, 231.0, 450.0, 1000.0)          # None: the jerk limit; else a jerk_cap on every joint


def swings(e, th):
    """half-cycles of an oscillation of amplitude > th: transitions between e > th and e < -th"""
    s = np.where(e > th, 1, np.where(e < -th, -1, 0))
    s = s[s != 0]
    return int(np.sum(np.diff(s) != 0)) if len(s) else 0


def step_response(body, cap, step, T=3.0, t0=0.1):
    """joint 1's setpoint after the reference steps from 0 to `step` at t0 (the other joints at rest)"""
    lim = AL.load(body)
    g = AL.JerkLimiter(lim, DT, q0=np.zeros(lim.n))
    jc = None if cap is None else np.full(lim.n, cap)
    x = []
    for i in range(int(T / DT)):
        r = np.zeros(lim.n)
        r[0] = step if i * DT >= t0 else 0.0
        x.append(g.step(r, np.zeros(lim.n), jc)[0][0])
    return np.array(x)


def test_step_without_overshoot():
    for body in ARMS:
        for cap in CAPS:
            for step in (0.001, 0.0506, 0.3, 1.5):
                x = step_response(body, cap, step)
                over = float(x.max() - step)
                assert over <= 2e-3, (body, cap, step, over)        # < 0.12 deg: none beyond the discrete steps
                assert np.all(np.abs(x[-1000:] - step) < 1e-4), (body, cap, step)   # settled, at rest


def test_low_jerk_does_not_ring():
    """the regression of 2026-10-11: a 2.9 deg step at jerk_cap 231 rad/s^3 (joint 1 of the table-tennis Panda in
    play) rang in a limit cycle of +-3.5 deg at 2.5 Hz under the former sqrt-law position loop"""
    for cap in (100.0, 231.0, 300.0, 450.0):
        x = step_response("franka_panda", cap, 0.0506, T=4.0)
        e = np.degrees(x[200:] - 0.0506)
        assert swings(e, 0.1) <= 1, (cap, swings(e, 0.1))          # at most the one crossing of the approach
        assert np.max(np.abs(e[-2000:])) < 0.01, cap


def test_hold_at_rest_does_not_ring():
    """a joint already at its reference stays there (no self-excited motion), with any jerk cap"""
    for cap in CAPS:
        lim = AL.load("franka_panda")
        q0 = np.array([0.1, -0.5, 0.2, -2.0, 0.3, 1.6, 0.7])
        g = AL.JerkLimiter(lim, DT, q0=q0)
        jc = None if cap is None else np.full(7, cap)
        for _ in range(2000):
            q, qd, qdd = g.step(q0, np.zeros(7), jc)
        assert np.max(np.abs(q - q0)) < 1e-12 and np.max(np.abs(qd)) < 1e-12, cap


def peaks(body, capmode, seed):
    lim = AL.load(body)
    g = AL.JerkLimiter(lim, DT, q0=np.zeros(lim.n))
    rng = np.random.default_rng(seed)
    pv = pa = pj = 0.0
    prev = np.zeros(lim.n)
    for _ in range(10):
        r0 = rng.uniform(lim.range_lo + 0.1, lim.range_hi - 0.1)
        rd = rng.uniform(-1.0, 1.0, lim.n) * lim.vel * rng.uniform(0.0, 1.3)    # moving, some beyond the limits
        for i in range(800):
            if capmode == "none":
                cap = None
            elif capmode == "fixed":
                cap = np.full(lim.n, 231.0)
            else:                                     # a pose-dependent cap that changes every step
                cap = rng.uniform(120.0, 600.0, lim.n)
            q, qd, qdd = g.step(r0 + rd * i * DT, rd, cap)
            pv = max(pv, float(np.max(np.abs(qd) / lim.vel)))
            pa = max(pa, float(np.max(np.abs(qdd) / lim.acc)))
            pj = max(pj, float(np.max(np.abs(qdd - prev) / DT / lim.jerk)))
            prev = qdd.copy()
    return pv, pa, pj


def test_setpoints_within_the_limits():
    for body in ARMS:
        for capmode in ("none", "fixed", "varying"):
            for seed in (1, 5):
                pv, pa, pj = peaks(body, capmode, seed)
                assert pv <= 1.0 and pa <= 1.0 + 1e-9 and pj <= 1.0 + 1e-9, (body, capmode, seed, pv, pa, pj)


def test_jerk_cap_respected_when_moving_within_the_limits():
    """a reference that is itself slow (well inside the velocity bound) is followed at the capped jerk"""
    lim = AL.load("franka_panda")
    g = AL.JerkLimiter(lim, DT, q0=np.zeros(7))
    cap = np.full(7, 231.0)
    prev, pj = np.zeros(7), 0.0
    for i in range(3000):
        t = i * DT
        r, rd = np.full(7, 0.2 * np.sin(2.0 * t)), np.full(7, 0.4 * np.cos(2.0 * t))
        q, qd, qdd = g.step(r, rd, cap)
        pj = max(pj, float(np.max(np.abs(qdd - prev) / DT / cap)))
        prev = qdd.copy()
    assert pj <= 1.0 + 1e-9, pj


def test_torque_rate_limiter_within_the_limits():
    lim = AL.load("franka_panda")
    trl = AL.TorqueRateLimiter(lim, DT)
    rng = np.random.default_rng(3)
    prev, pr, pt = trl.tau.copy(), 0.0, 0.0
    for _ in range(5000):
        t = trl(rng.normal(0.0, 200.0, 7))
        pr = max(pr, float(np.max(np.abs(t - prev) / DT / lim.dtau)))
        pt = max(pt, float(np.max(np.abs(t) / lim.tau)))
        prev = t.copy()
    assert pr <= 1.0 + 1e-9 and pt <= 1.0, (pr, pt)


def test_one_braking_law():
    assert SV.brake_speed is AL.brake_speed
    d = np.array([0.0, 1e-6, 1e-3, 0.1, 2.0])
    for A, J in ((15.0, 7500.0), (9.0, 231.0), (4.5, 4.5)):
        v = AL.brake_speed(d, A, J)
        # the stop from v (acceleration ramped to A at J, held, ramped back) covers at most d
        dist = np.where(v <= A * A / J, v * np.sqrt(v / J), v * v / (2 * A) + v * A / (2 * J))
        assert np.allclose(dist, d, rtol=1e-9, atol=1e-12), (A, J)


def test_follows_a_moving_reference():
    """the regression of 2026-10-12: a jerk-feasible stroke (jerk +-0.9 x the cap, as the table-tennis planner plans
    within limits.jerk_caps) must be followed to its end, not braked early. The position loop's coast prediction is
    relative to the reference's velocity (qd - rd); with qd alone it lagged 1.0 deg of a 2.6 deg stroke and reached
    82 % of the speed at cap 231 rad/s^3, and the table-tennis strokes delivered too little blade speed (B9 248/284)."""
    lim = AL.load("franka_panda")
    n = 0
    for cap in (231.0, 300.0, 450.0, 1000.0):
        for T in (0.02, 0.03, 0.045, 0.06):
            Jr = 0.9 * cap
            j = np.r_[np.full(int(T / DT), Jr), np.full(int(T / DT), -Jr)]
            a = np.cumsum(j) * DT; v = np.cumsum(a) * DT; x = np.cumsum(v) * DT
            if v.max() > 0.9 * lim.vel[0] or Jr * T > 0.9 * lim.acc[0]:
                continue                                  # within the velocity and acceleration limits only
            n += 1
            g = AL.JerkLimiter(lim, DT, q0=np.zeros(lim.n))
            jc = np.full(lim.n, cap)
            for i in range(len(x)):
                r = np.zeros(lim.n); rd = np.zeros(lim.n); r[0] = x[i]; rd[0] = v[i]
                q, qd, _ = g.step(r, rd, jc)
            lag = np.degrees(x[-1] - q[0])
            assert lag <= 0.25 + 0.1 * np.degrees(x[-1]), (cap, T, lag, np.degrees(x[-1]))
            assert qd[0] >= 0.9 * v[-1], (cap, T, qd[0], v[-1])
    assert n >= 8, n


def test_tracks_a_sine_reference():
    """sustained tracking (independent review of #94): joint 1 follows r = 0.3 sin(4 t) at jerk cap 231 rad/s^3
    (peak reference acceleration 4.8 rad/s^2, jerk 19.2 rad/s^3: well within the limits); RMS error after 1 s
    0.0185 rad with the stationary coast term, about 0.001 rad with the relative one"""
    lim = AL.load("franka_panda")
    g = AL.JerkLimiter(lim, DT, q0=np.zeros(lim.n))
    jc = np.full(lim.n, 231.0)
    err = []
    for i in range(3000):
        t = i * DT
        r = np.zeros(lim.n); rd = np.zeros(lim.n)
        r[0] = 0.3 * np.sin(4.0 * t); rd[0] = 1.2 * np.cos(4.0 * t)
        q, _, _ = g.step(r, rd, jc)
        if t >= 1.0:
            err.append(q[0] - r[0])
    rms = float(np.sqrt(np.mean(np.square(err))))
    assert rms < 0.003, rms


def test_servo_generators_track_moving_references():
    """servo.SetpointGenerator and PlanarBase take the coast distance relative to the reference's velocity too
    (2026-10-12): a joint with a slow jerk (60 rad/s^3) follows r = 0.3 sin(4 t) (RMS 0.036 rad with qd alone, about
    0.01 with qd - rd), and a planar base (3 m/s^2, 10 m/s^3; per axis and on the norm) follows a 0.5 m circle at
    1 rad/s (RMS 0.038 / 0.063 m with v alone, about 0.001 m with v - rd)"""
    lim = AL.load("franka_panda").with_jerk(np.full(7, 60.0))
    g = SV.SetpointGenerator(lim, DT, q0=np.zeros(lim.n))
    err = []
    for i in range(4000):
        t = i * DT
        r = np.zeros(lim.n); rd = np.zeros(lim.n)
        r[0] = 0.3 * np.sin(4.0 * t); rd[0] = 1.2 * np.cos(4.0 * t)
        q, _, _ = g.step(r, rd)
        if t >= 2.0:
            err.append(q[0] - r[0])
    rms = float(np.sqrt(np.mean(np.square(err))))
    assert rms < 0.02, rms
    for L in (dict(mode="axes", speed_m_s=[2.0, 2.0], accel_m_s2=[3.0, 3.0], jerk_m_s3=[10.0, 10.0]),
              dict(mode="norm", speed_m_s=2.0, accel_m_s2=3.0, jerk_m_s3=10.0)):
        b = SV.PlanarBase(SV.BaseLimits.of(dict(**L, yaw_rate_rad_s=2.0, yaw_accel_rad_s2=4.0, yaw_jerk_rad_s3=20.0)), DT)
        err = []
        for i in range(6000):
            t = i * DT
            r = np.array([0.5 * np.cos(t) - 0.5, 0.5 * np.sin(t), 0.2 * np.sin(t)])
            rd = np.array([-0.5 * np.sin(t), 0.5 * np.cos(t), 0.2 * np.cos(t)])
            q, _, _ = b.step(r, rd)
            if t >= 2.0:
                err.append(np.linalg.norm(q[:2] - r[:2]))
        rms = float(np.sqrt(np.mean(np.square(err))))
        assert rms < 0.005, (L["mode"], rms)


class _Model:
    """the one field of an MjModel that servo.Actuators reads"""
    def __init__(self, n):
        self.actuator_ctrlrange = np.tile([-1e3, 1e3], (n, 1))


class _Data:
    def __init__(self, n):
        self.qvel = np.zeros(n)
        self.ctrl = np.zeros(n)


def test_motor_model_thermal_and_power():
    """booster_t1 [motors.thermal] / [motors.bus] (t1-2): the peak torque from cold lasts about peak_s, then the I^2 t
    foldback brings the limit down to the rated torque; the rated torque holds for good; the summed positive power of a
    group never exceeds the bus cap; a torque against the motion is never power-capped; cool_down() restores the peak"""
    import tomllib
    decl = tomllib.loads((Path(__file__).resolve().parents[1] / "booster_t1" / "body.toml").read_text())
    keys = ["hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll"] * 2
    kw = SV.motor_model(decl, keys, groups=[np.arange(12)])
    peak = np.tile(np.asarray(decl["limits"]["leg"]["torque_nm"], float), 2)
    rated = decl["motors"]["thermal"]["rated_frac"] * peak
    dt = 0.002
    act = SV.Actuators(_Model(12), np.arange(12), np.arange(12), peak, dt, **kw)
    d = _Data(12)
    full = []                                      # the knee held at the peak torque from cold, joints still
    for i in range(int(3.0 * decl["motors"]["thermal"]["peak_s"] / dt)):
        full.append(act.write(d, peak * 2.0)[3])
    full = np.array(full)
    t_full = float(np.argmax(full < peak[3] - 1e-6) * dt)
    assert 1.0 < t_full < decl["motors"]["thermal"]["peak_s"] + 0.2, t_full
    assert abs(full[-1] - rated[3]) < 0.02 * peak[3], (full[-1], rated[3])
    act.cool_down()
    for i in range(int(30.0 / dt)):               # the rated torque is continuous
        t = act.write(d, rated * 0.999)
    assert np.allclose(t, rated * 0.999), t
    act.cool_down()
    rng = np.random.default_rng(3)
    cap = decl["motors"]["bus"]["power_w"]
    worst = 0.0
    for i in range(3000):
        d.qvel[:] = rng.uniform(-1, 1, 12) * np.tile(np.asarray(decl["limits"]["leg"]["speed_rad_s"], float), 2)
        t = act.write(d, rng.uniform(-1, 1, 12) * peak)
        worst = max(worst, float(np.maximum(t * d.qvel, 0).sum()) - cap)
        braking = t * d.qvel < 0
        assert np.all(np.abs(t[braking]) <= peak[braking] + 1e-9)
    assert worst <= 1e-6, worst


if __name__ == "__main__":
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for k, f in tests:
        f()
        print("ok", k)
    print(f"{len(tests)} passed")
