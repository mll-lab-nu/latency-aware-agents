#!/usr/bin/env python3
"""The dribbler's calibration experiments (README "Dribbler"): the benchmark of ZJUNlict's TestDribble (S6: hold
the ball, accelerate at 3 m/s^2 along -x / y until the ball is lost; turn ever faster until it is lost), the same
forwards, and ball reception (a ball rolling at 1-5 m/s head-on into the robot standing with its dribbler on:
caught? how far does it rebound?), plus KIKS's rebound test (S8: 2.3 m/s into a dribbler that does not turn).

    python bodies/ssl_robot/tools/dribble_lab.py [--rpm 3000] [--json out.json] [key=value ...]

key=value overrides body.toml [dribbler_model] (e.g. bar_friction=[1.2,0.005,0.0001] hinge_damping=0.01) and
'torque=0.02' the motor's torque limit, 'floor=-20000,-60' the carpet's solref, 'assist=0|1' the assist model.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from bodies.ssl_robot import body as SSL  # noqa: E402
from bodies.ssl_robot.tools.lab import Lab  # noqa: E402

LOST_M = 0.03          # the ball counts as lost once its centre is this far from where the dribbler held it


def make(cfg, ball=None, robots=None):
    decl = copy.deepcopy(SSL.load_decl())
    dm = decl.setdefault("dribbler_model", {})
    dm.update(cfg.get("model", {}))
    kw = {}
    if "ball" in cfg:                     # e.g. {"solref": [0.02, 1]}: a test of the ball's own contact
        kw["ball_kw"] = cfg["ball"]
    if ball is None:                      # touching the bar (its rest point)
        ball = (SSL.layout(decl["geometry"])["x_ball_rest"] - 0.0003, 0.0)
    lab = Lab(robots or [("b", "blue", 0, (0.0, 0.0, 0.0))], ball=ball, decl=decl, **kw)
    r = lab.rigs[0]
    if "torque" in cfg:
        r.dribbler_act.tau_max = np.array([float(cfg["torque"])])
        lab.m.actuator_ctrlrange[r.r_aid] = (-float(cfg["torque"]), float(cfg["torque"]))
    if "assist" in cfg:
        r.assist_on = bool(cfg["assist"])
    if "assist_p" in cfg:                 # e.g. {"side_speed_m_s": 2.3}: the grip's parameters
        r.assist.P = dict(r.assist.P, **cfg["assist_p"])
    return lab


def capture(cfg, rpm):
    """the ball just in front of the bar, dribbler on, 0.6 s to settle: (lab, held point) or (lab, None)"""
    lab = make(cfg)
    r = lab.rigs[0]
    r.dribble(rpm)
    lab.run(0.6)
    rel = lab.ball_rel()
    ok = r.state()["ball_in_beam"] and rel[0] < 0.1 and abs(rel[1]) < 0.01
    return lab, (rel if ok else None)


def translate(cfg, rpm, direction, accel=3.0, vmax=3.0, seconds=2.0):
    """accelerate along `direction` (robot frame unit vector) at `accel` up to vmax, until the ball is lost:
    (speed at loss or None if held, peak speed reached)"""
    lab, held = capture(cfg, rpm)
    if held is None:
        return dict(capture=False)
    r = lab.rigs[0]
    lim = {"base": {"accel_m_s2": accel, "speed_m_s": vmax}}
    r2 = SSL.rig(lab.m, lab.d, lab.handles[0], lim)
    r2.w_target = r.w_target
    r2.dribbler_act.tau_max = r.dribbler_act.tau_max
    r2.assist_on = getattr(r, "assist_on", r2.assist_on)
    r2.assist.P = r.assist.P
    lab.rigs[0] = r2
    d = np.asarray(direction, float)
    r2.set_base_target(d[0] * 20, d[1] * 20, 0.0)
    peak = 0.0
    for i in range(int(seconds / 0.01)):
        lab.step()
        v = r2.vel(lab.d)
        sp = float(np.hypot(v[0], v[1]))
        peak = max(peak, sp)
        rel = lab.ball_rel()
        if np.linalg.norm(rel[:2] - held[:2]) > LOST_M or rel[2] > 0.035:
            return dict(capture=True, lost_at_speed=round(sp, 2), t=round(i * 0.01, 2), peak=round(peak, 2))
    return dict(capture=True, lost_at_speed=None, peak=round(peak, 2))


def stop(cfg, rpm, v=1.5, decel=3.0):
    """carry forward at v, then brake at decel: held?"""
    lab, held = capture(cfg, rpm)
    if held is None:
        return dict(capture=False)
    r = lab.rigs[0]
    r.set_base_target(v * 1.0 + 0.4, 0.0, 0.0, v_end=(v, 0.0))
    lab.run(1.0)
    r.set_base_target(r.pose(lab.d)[0] + 0.05, 0.0, 0.0)
    for i in range(150):
        lab.step()
        rel = lab.ball_rel()
        if np.linalg.norm(rel[:2] - held[:2]) > LOST_M:
            return dict(capture=True, held=False, t=round(i * 0.01, 2))
    return dict(capture=True, held=True)


def rotate(cfg, rpm, yaw_accel=2.0, seconds=8.0):
    """turn on the spot ever faster (the yaw rate's reference ramps at yaw_accel) until the ball is lost"""
    lab, held = capture(cfg, rpm)
    if held is None:
        return dict(capture=False)
    r = lab.rigs[0]
    peak = 0.0
    for i in range(int(seconds / 0.01)):
        r._spin = min(20.0, yaw_accel * i * 0.01)
        lab.step()
        w = float(r.vel(lab.d)[2])
        peak = max(peak, w)
        rel = lab.ball_rel()
        if np.linalg.norm(rel[:2] - held[:2]) > LOST_M:
            return dict(capture=True, lost_at_rate=round(w, 2), t=round(i * 0.01, 2))
    return dict(capture=True, lost_at_rate=None, peak=round(peak, 2))


def receive(cfg, rpm, v_in, seconds=1.5):
    """a ball rolling head-on at v_in into the standing robot: caught (held at the end), the rebound (the largest
    distance of the ball's centre from the held point after the first contact), the final distance"""
    lab = make(cfg, ball=(1.0, 0.0))
    r = lab.rigs[0]
    r.dribble(rpm)
    lab.run(0.2)
    lab.set_ball((0.6, 0.0, SSL.BALL_R), vel=(-v_in, 0.0, 0.0), spin=(0.0, -v_in / SSL.BALL_R, 0.0))
    x_rest = lab.handles[0].extra["layout"]["x_ball_rest"]
    touched, reb, zmax = False, 0.0, 0.0
    for i in range(int(seconds / 0.01)):
        lab.step()
        rel = lab.ball_rel()
        zmax = max(zmax, rel[2] - SSL.BALL_R)
        if rel[0] < x_rest + 0.005:
            touched = True
        if touched:
            reb = max(reb, rel[0] - x_rest)
    rel = lab.ball_rel()
    held = r.state()["ball_in_beam"] and abs(rel[0] - x_rest) < 0.01 and abs(rel[1]) < 0.02
    return dict(v_in=v_in, caught=bool(held), rebound_m=round(reb, 3), final_m=round(float(rel[0] - x_rest), 3),
                hop_m=round(zmax, 4))


def receive_spread(cfg, rpm, v_in, n=10, seed=0, seconds=1.5):
    """n balls at v_in from slightly different lines (lateral offset +-15 mm, direction +-10 deg): the share caught
    and the rebounds"""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        off, ang = rng.uniform(-0.015, 0.015), math.radians(rng.uniform(-10, 10))
        lab = make(cfg, ball=(1.0, 0.0))
        r = lab.rigs[0]
        r.dribble(rpm)
        lab.run(0.2)
        x_rest = lab.handles[0].extra["layout"]["x_ball_rest"]
        start = np.array([x_rest, off]) + 0.5 * np.array([math.cos(ang), math.sin(ang)])
        v = -v_in * np.array([math.cos(ang), math.sin(ang)])
        lab.set_ball((start[0], start[1], SSL.BALL_R), vel=(v[0], v[1], 0.0),
                     spin=(-v[1] / SSL.BALL_R, v[0] / SSL.BALL_R, 0.0))
        touched, reb = False, 0.0
        for _ in range(int(seconds / 0.01)):
            lab.step()
            rel = lab.ball_rel()
            if rel[0] < x_rest + 0.005:
                touched = True
            if touched:
                reb = max(reb, float(np.hypot(rel[0] - x_rest, rel[1])))
        rel = lab.ball_rel()
        out.append((bool(r.state()["ball_in_beam"] and abs(rel[0] - x_rest) < 0.01 and abs(rel[1]) < 0.02), reb))
    return dict(v_in=v_in, caught=sum(c for c, _ in out) / n, rebound_median_m=round(float(np.median([b for _, b in out])), 3),
                rebound_max_m=round(max(b for _, b in out), 3))


def steal(cfg, rpm, direction=(0.0, 1.0), rate=1.0, seconds=4.0):
    """another robot taking the ball (the rules: a dribbler must let others take it, S7 3.2.4), as a force: a
    horizontal push on the held ball growing at `rate` N/s along `direction` (robot frame; lateral by default):
    the push at which the ball leaves (3 cm). A robot's drive pushes with up to 15.7 N (body.toml)."""
    lab, held = capture(cfg, rpm)
    if held is None:
        return dict(capture=False)
    r = lab.rigs[0]
    b = lab.ball
    dvec = np.array([direction[0], direction[1], 0.0])
    for i in range(int(seconds / 0.01)):
        F = rate * i * 0.01
        lab.d.xfrc_applied[b, 0:3] = F * dvec               # the robot faces +x: robot frame = world frame
        lab.step()
        rel = lab.ball_rel()
        if np.linalg.norm(rel[:2] - held[:2]) > LOST_M:
            return dict(capture=True, steal_force_n=round(F, 2))
    return dict(capture=True, steal_force_n=None)


def kiks(cfg, v_in=2.3, seconds=4.0):
    """KIKS (S8): 2.3 m/s head-on into a dribbler that does not turn; the robot fixed: where the ball stops"""
    lab = make(cfg, ball=(1.0, 0.0))
    lab.set_ball((0.6, 0.0, SSL.BALL_R), vel=(-v_in, 0.0, 0.0), spin=(0.0, -v_in / SSL.BALL_R, 0.0))
    lab.run(seconds)
    front = SSL.load_decl()["geometry"]["front"]
    rel = lab.ball_rel()
    return dict(stop_from_front_m=round(float(rel[0] - SSL.BALL_R - front), 3))


def suite(cfg, rpm):
    out = dict(cfg=cfg, rpm=rpm)
    out["back"] = translate(cfg, rpm, (-1, 0))
    out["side"] = translate(cfg, rpm, (0, 1))
    out["forward"] = translate(cfg, rpm, (1, 0))
    out["stop"] = stop(cfg, rpm)
    out["turn"] = rotate(cfg, rpm)
    out["receive"] = [receive(cfg, rpm, v) for v in (1.0, 2.0, 3.0, 4.0, 5.0)]
    out["kiks_0rpm"] = kiks(cfg)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--rpm", type=float, default=3000)
    ap.add_argument("--json")
    ap.add_argument("--only")
    ap.add_argument("kv", nargs="*")
    a = ap.parse_args(argv)
    cfg = {"model": {}}
    for kv in a.kv:
        k, v = kv.split("=", 1)
        val = json.loads(v) if v[:1] in "[{-0123456789" and "," not in v[:1] else v
        if k == "torque":
            cfg["torque"] = float(v)
        elif k == "floor":
            cfg["floor"] = [float(x) for x in v.split(",")]
        elif k == "assist":
            cfg["assist"] = int(v)
        else:
            cfg["model"][k] = json.loads(v)
    if a.only:
        fn = dict(back=lambda: translate(cfg, a.rpm, (-1, 0)), side=lambda: translate(cfg, a.rpm, (0, 1)),
                  forward=lambda: translate(cfg, a.rpm, (1, 0)), stop=lambda: stop(cfg, a.rpm),
                  turn=lambda: rotate(cfg, a.rpm), kiks=lambda: kiks(cfg),
                  receive=lambda: [receive(cfg, a.rpm, v) for v in (1.0, 2.0, 3.0, 4.0, 5.0)],
                  spread=lambda: [receive_spread(cfg, a.rpm, v) for v in (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0)],
                  steal=lambda: [steal(cfg, a.rpm), steal(cfg, a.rpm, (1.0, 0.0))])[a.only]
        res = dict(cfg=cfg, rpm=a.rpm, **{a.only: fn()})
    else:
        res = suite(cfg, a.rpm)
    print(json.dumps(res))
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    main()
