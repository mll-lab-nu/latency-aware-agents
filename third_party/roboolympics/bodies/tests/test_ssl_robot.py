"""bodies/ssl_robot: the SSL robot's admission tests (bodies/README.md "准入"; the planar robot's equivalents of
bodies/conformance.py, which is written for the arm robots' interface: see bodies/ssl_robot/README.md "Admission").

Run: python bodies/tests/test_ssl_robot.py   (or pytest; needs mujoco + numpy)
"""
import hashlib
import math
import sys
import tomllib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from bodies import load_body, sources  # noqa: E402
from bodies.ssl_robot import body as SSL  # noqa: E402
from bodies.ssl_robot.tools.lab import Lab  # noqa: E402

DECL = tomllib.loads((ROOT / "bodies" / "ssl_robot" / "body.toml").read_text())


# ------------------------------------------------------------------ 1 declared (conformance 1)
def test_limits_declared_with_sources():
    assert sources.source_problems(DECL) == []
    groups = [DECL["limits"]["kicker"], DECL["limits"]["dribbler"], DECL["dribbler_model"], DECL["dribbler_assist"]]
    for g, L in enumerate(groups):                          # groups sources.py does not know: the same rule here
        for k, v in L.items():
            if k in ("source", "enabled"):
                continue
            s = str(L["source"].get(k, ""))
            assert sources.SOURCE_RE.match(s), (g, k, s)
            for ref in __import__("re").findall(r"\bS\d+\b", s):
                assert ref in DECL["sources"], (g, k, ref)
    d = ROOT / "bodies" / "ssl_robot"
    assert (d / "LICENSE").exists() and (d / "tools" / "make_model.py").exists() and (d / "model.xml").exists()
    assert load_body("ssl_robot").revision == DECL["body"]["revision"]


# ------------------------------------------------------------------ 2 scene (conformance 2)
def test_ten_robots_one_scene_poses_and_patterns():
    robots = [(f"{t}_{k}", t, k, ((-1 if t == "blue" else 1) * (0.5 + 0.6 * k), 0.4 * k - 1.0, 0.3 * k))
              for t in ("blue", "yellow") for k in range(5)]
    lab = Lab(robots, ball=(0.0, 2.0))
    for (n, t, k, pose), r in zip(robots, lab.rigs):
        assert np.allclose(r.pose(lab.d), pose)           # qpos = the world pose, qpos0 = the mount pose
        xb = lab.d.xpos[r.body_id]
        assert np.allclose(xb[:2], pose[:2])
    codes = {(h.team, h.pattern) for h in lab.handles}
    assert len(codes) == 10                                # 16 distinct SSL patterns (S7 figure 6)
    assert SSL.PATTERN[0] == "PPGP" and SSL.PATTERN[8] == "GGGG" and SSL.PATTERN[9] == "PPPP"


def test_shape_fits_the_rules():
    import mujoco as mj
    lab = Lab([("b", "blue", 0, (0, 0, 0))], ball=None)
    m, d = lab.m, lab.d
    r = lab.rigs[0]
    pts = []
    for g in range(m.ngeom):
        if m.geom_bodyid[g] in (r.body_id, r.body_id + 1, r.body_id + 2):
            if m.geom_type[g] == mj.mjtGeom.mjGEOM_MESH:
                a, n = m.mesh_vertadr[m.geom_dataid[g]], m.mesh_vertnum[m.geom_dataid[g]]
                v = m.mesh_vert[a:a + n] @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]
                pts.append(v)
    P = np.vstack(pts)
    assert np.max(np.hypot(P[:, 0], P[:, 1])) <= 0.09 + 1e-9   # S7 3.2.2: inside 0.18 m diameter
    assert P[:, 2].max() <= 0.15 + 1e-9                        # and 0.15 m height
    Lg = lab.handles[0].extra["layout"]
    cov = SSL.coverage(Lg["x_ball_min"] - DECL["geometry"]["front"])
    assert cov <= 0.20 and abs(cov - 0.198) < 1e-3           # the shovel stops the ball at 19.8 % (S1), <= 20 % (S7)
    assert SSL.coverage(Lg["x_ball_rest"] - DECL["geometry"]["front"]) < cov


def test_side_band_rules_3_2_3_visual_only():
    """the team colour on the sides (rules 3.2.3, S12): >= 20 % of the side area, visible all round but the camera
    window, visual only (no contact, no mass), inside the 0.18 m cylinder, and not the vision pattern's colours"""
    import colorsys
    import mujoco as mj
    lab = Lab([("b", "blue", 0, (0, 0, 0)), ("y", "yellow", 1, (1.0, 0.0, 0.0))], ball=None)
    m, d = lab.m, lab.d
    for h, r in zip(lab.handles, lab.rigs):
        assert h.extra["side_band_share"] >= DECL["appearance"]["side_band_min_share"], h.extra["side_band_share"]
        ids = [mj.mj_name2id(m, mj.mjtObj.mjOBJ_GEOM, g) for g in h.geoms["side_band"]]
        assert len(ids) >= 30
        angles = []
        for g in ids:
            assert m.geom_contype[g] == 0 and m.geom_conaffinity[g] == 0
            c = d.geom_xpos[g] - d.xpos[r.body_id]
            R_ = d.geom_xmat[g].reshape(3, 3)
            for sx in (-1, 1):
                for sy in (-1, 1):
                    corner = c + R_ @ (m.geom_size[g] * np.array([sx, sy, 1]))
                    assert math.hypot(corner[0], corner[1]) <= 0.09 + 1e-9
            angles.append(math.degrees(math.atan2(c[1], c[0])))
        a = np.sort(np.mod(angles, 360))
        gaps = np.diff(np.r_[a, a[0] + 360])
        assert gaps.max() < 60          # panel centres: the widest gap is across the front window (+-15 deg + panels)
        assert DECL["appearance"]["side_band_front_gap_deg"] <= 15
        rgba = m.mat_rgba[m.geom_matid[ids[0]]]
        hh, ss, vv = colorsys.rgb_to_hsv(*rgba[:3])
        assert vv < 0.45 or ss < 0.5                            # not the pattern's saturated blue / yellow
    assert np.isclose(m.body_subtreemass[lab.rigs[0].body_id], 2.62)
    bright = colorsys.rgb_to_hsv(*DECL["appearance"]["side_band_rgba"]["yellow"][:3])[2]
    dark = colorsys.rgb_to_hsv(*DECL["appearance"]["side_band_rgba"]["blue"][:3])[2]
    assert bright > 0.8 and dark < 0.4                          # one bright, one dark


# ------------------------------------------------------------------ 3 + 4 base (conformance 3, 4)
def test_base_reaches_goals_within_limits():
    """the setpoint generator's knots (100 Hz) and the interpolated 1 kHz setpoints the PD follows stay within the
    declared speed / acceleration / jerk (robot frame, on the norm; yaw on its own)"""
    lab = Lab([("b", "blue", 0, (0, 0, 0))], ball=None)
    r = lab.rigs[0]
    L = DECL["limits"]["base"]
    worst = dict(v=0.0, a=0.0, w=0.0, wd=0.0, sv=0.0, sa=0.0, sw=0.0, j=0.0)
    prev_a = None
    for goal in ((1.5, 0.8, 2.5), (-1.0, 1.2, -1.0), (0.2, -0.3, 3.0), (0.0, 0.0, 0.0), (3.0, 0.0, 0.0)):
        r.set_base_target(*goal)
        for _ in range(300):
            r.step(lab.m, lab.d)
            b = r.pb
            worst["v"] = max(worst["v"], float(np.linalg.norm(b.v)) / L["speed_m_s"])
            worst["a"] = max(worst["a"], float(np.linalg.norm(b.a)) / L["accel_m_s2"])
            worst["w"] = max(worst["w"], abs(b.w) / L["yaw_rate_rad_s"])
            worst["wd"] = max(worst["wd"], abs(b.wd) / L["yaw_accel_rad_s2"])
            if prev_a is not None:
                worst["j"] = max(worst["j"], float(np.linalg.norm(b.a - prev_a)) / r.sp_dt / L["jerk_m_s3"])
            prev_a = b.a.copy()
            for _ in range(r.substeps):
                r.substep(lab.m, lab.d)
                j = r._substeps_seen - 1
                q, qd, qdd = (x[j] for x in r.sp_table)
                worst["sv"] = max(worst["sv"], float(np.hypot(qd[0], qd[1])) / L["speed_m_s"])
                ka = "sa" if abs(qd[2]) < 0.05 else "sa_turning"    # see below
                worst[ka] = max(worst.get(ka, 0.0), float(np.hypot(qdd[0], qdd[1])) / L["accel_m_s2"])
                worst["sw"] = max(worst["sw"], abs(qd[2]) / L["yaw_rate_rad_s"])
                lab.mj.mj_step(lab.m, lab.d)
        q = r.pose(lab.d)
        assert np.linalg.norm(q[:2] - goal[:2]) < 0.01, (goal, q)
        assert abs(SSL._wrap(q[2] - goal[2])) < 0.01
    for k in ("v", "a", "w", "wd", "j", "sv", "sw"):
        assert worst[k] <= 1 + 1e-3, worst
    assert worst["sa"] <= 1.03, worst
    # turning while translating, PlanarBase keeps the velocity in the robot's frame, so the world-frame velocity
    # turns with the robot: an extra v x w the generator does not limit (bodies/servo.py, README "Known gaps")
    assert worst["sa_turning"] < 2.0, worst


def test_group_path_equals_rig_path():
    """step_world (Group: vectorised) and the per-rig step / substep give the same motion"""
    import mujoco as mj
    out = []
    for fast in (True, False):
        lab = Lab([("b", "blue", 0, (0, 0, 0)), ("y", "yellow", 2, (1.0, 0.5, 1.0))], ball=(0.3, 0.0))
        a, b = lab.rigs
        a.set_base_target(0.8, 0.4, 1.0)
        b.set_base_target(-0.5, 0.0, -2.0, t=1.5)
        a.dribble(2000)
        for _ in range(150):
            if fast:
                SSL.step_world(lab.m, lab.d, lab.rigs)
            else:
                for r in lab.rigs:
                    r.step(lab.m, lab.d)
                for _ in range(10):
                    for r in lab.rigs:
                        r.substep(lab.m, lab.d)
                    mj.mj_step(lab.m, lab.d)
        out.append(lab.d.qpos.copy())
    assert np.allclose(out[0], out[1], atol=1e-9), np.abs(out[0] - out[1]).max()


def test_timed_arrival_and_moving_arrival():
    lab = Lab([("b", "blue", 0, (0, 0, 0))], ball=None)
    r = lab.rigs[0]
    rep = r.set_base_target(1.0, 0.0, 0.0, t=2.0)
    assert rep["status"] == "ok" and abs(rep["eta"] - 2.0) < 0.05
    lab.run(1.0)
    assert r.pose(lab.d)[0] < 0.75                             # slowed down to arrive at t
    lab.run(1.0)
    assert abs(r.pose(lab.d)[0] - 1.0) < 0.05
    rep = r.set_base_target(2.0, 0.0, 0.0, t=lab.d.time + 0.2)
    assert rep["status"] == "late"                             # too early for the limits: as fast as possible
    lab2 = Lab([("b", "blue", 0, (0, 0, 0))], ball=None)
    r2 = lab2.rigs[0]
    rep = r2.set_base_target(1.0, 0.0, 0.0, v_end=(1.0, 0.0))
    lab2.run(rep["eta"] + 0.05)
    q, v = r2.pose(lab2.d), r2.vel(lab2.d)
    assert abs(q[0] - 1.0) < 0.12 and abs(v[0] - 1.0) < 0.15, (q, v)


def test_sport_may_only_tighten():
    lab = Lab([("b", "blue", 0, (0, 0, 0))], ball=None)
    r = SSL.rig(lab.m, lab.d, lab.handles[0], {"base": {"speed_m_s": 1.5}})
    assert r.base_lim.speed == 1.5
    try:
        SSL.rig(lab.m, lab.d, lab.handles[0], {"base": {"accel_m_s2": 4.0}})
    except ValueError:
        pass
    else:
        raise AssertionError("a looser limit was accepted")


# ------------------------------------------------------------------ kicker
def _held_ball_lab(seed=0):
    lab = Lab([("b", "blue", 0, (0, 0, 0))], ball=(0.090, 0.0), seed=seed)
    lab.rigs[0].dribble(1000)
    lab.run(0.3)
    return lab


def test_flat_kick_speed_and_error():
    speeds = []
    for seed in range(12):
        lab = _held_ball_lab(seed)
        r = lab.rigs[0]
        assert r.state()["ball_in_beam"]
        assert r.kick(6.0, "flat") == "ok"
        lab.step()
        ev = [e for e in lab.events if e["event"] == "kicked"][-1]
        assert ev["ball"]
        speeds.append(ev["speed"])
        v = lab.ball_vel()
        assert v[0] > 5.0 and abs(v[1]) < 0.4
    s = np.array(speeds)
    assert abs(s.mean() - 6.0) < 0.15 and 0.05 < s.std() < 0.4, s     # sigma 0.18 m/s at 6 m/s (S6)


def test_chip_and_capacitor():
    lab = _held_ball_lab()
    r = lab.rigs[0]
    assert r.kick(9.0, "flat") == "clipped"                   # beyond 8.5 m/s (S1)
    lab.step()
    assert abs(r.state()["kicker_charge"] - 0.75) < 0.02      # a maximum kick uses 25 %
    assert r.kick(6.0, "flat", armed_for=0.5) == "ok"         # a slower kick needs less voltage: ready
    assert r.kick(0.0, "flat") == "disarmed"
    assert r.kick(8.5, "flat") == "kicker_not_ready"
    ev = [e for e in r.events if e["event"] == "kicker_not_ready"][-1] if r.events else \
        [e for e in lab.events if e["event"] == "kicker_not_ready"][-1]
    assert 0.4 < ev["wait_s"] < 0.55                           # refilled in 0.5 s (S1 3.2)
    lab.run(0.6)
    lab.set_ball((0.090, 0.0, SSL.BALL_R))
    lab.run(0.3)
    assert r.kick(5.5, "chip") == "ok"
    zmax, t0 = 0.0, lab.d.time
    land = None
    for _ in range(150):
        lab.step()
        b = lab.ball_pos()
        zmax = max(zmax, b[2])
        if land is None and zmax > 0.1 and b[2] < SSL.BALL_R + 0.003:
            land = b[0]
    # 5.5 m/s at 45 deg: apex 0.77 m, first landing about 3.1 m (proposal 4.2; no drag)
    assert 0.6 < zmax < 0.9 and land is not None and 2.6 < land < 3.6, (zmax, land)


def test_armed_kick_fires_on_the_beam_and_expires():
    lab = Lab([("b", "blue", 0, (0, 0, 0))], ball=(1.5, 0.0))
    r = lab.rigs[0]
    lab.set_ball((1.5, 0.0, SSL.BALL_R), vel=(-3.0, 0, 0), spin=(0, -3.0 / SSL.BALL_R, 0))
    assert r.kick(4.0, "flat", armed_for=1.5) == "ok"
    lab.run(1.0)
    ev = [e for e in lab.events if e["event"] == "kicked"]
    assert ev and ev[0]["ball"] and lab.ball_vel()[0] > 2.5
    assert r.kick(4.0, "flat", armed_for=0.2) == "ok"
    lab.run(0.4)
    assert [e for e in lab.events if e["event"] == "kick_expired"]


# ------------------------------------------------------------------ dribbler (README "Dribbler")
def _DL():
    sys.path.insert(0, str(ROOT / "bodies" / "ssl_robot" / "tools"))
    import dribble_lab
    return dribble_lab


def test_dribbler_with_ball_limits_match_the_published_ones():
    """ZJUNlict's TestDribble (S6) replayed: the ball is lost at about, never far beyond, 2.3 m/s backwards,
    2.6 m/s sideways, 9.3 rad/s turning (3000 rpm, the default grip)"""
    DL = _DL()
    cfg = {"model": {}}
    back = DL.translate(cfg, 3000, (-1, 0))["lost_at_speed"]
    side = DL.translate(cfg, 3000, (0, 1))["lost_at_speed"]
    turn = DL.rotate(cfg, 3000)["lost_at_rate"]
    assert back is not None and 1.9 <= back <= 2.4, back
    assert side is not None and 2.3 <= side <= 2.7, side
    assert turn is not None and 8.6 <= turn <= 9.5, turn
    assert DL.translate(cfg, 3000, (1, 0))["lost_at_speed"] is None       # forwards: carried to 3 m/s


def test_dribbler_contact_alone_loses_the_ball_sideways():
    """why the grip exists: the rigid contact alone loses the ball sideways below 1 m/s"""
    DL = _DL()
    side = DL.translate({"model": {}, "assist": 0}, 3000, (0, 1))["lost_at_speed"]
    assert side is not None and side < 1.0, side


def test_dribbler_receives_up_to_5_m_s_and_lets_others_take_the_ball():
    DL = _DL()
    cfg = {"model": {}}
    for v in (1.0, 3.0, 5.0):
        assert DL.receive(cfg, 3000, v)["caught"], v
    assert not DL.receive(cfg, 3000, 7.0)["caught"]
    f = DL.steal(cfg, 3000)["steal_force_n"]
    assert f is not None and f < 1.5, f                      # a robot pushes with up to 15.7 N


def test_grip_force_released_at_kick():
    lab = _held_ball_lab()
    r = lab.rigs[0]
    assert r.state()["dribbler_engaged"]
    assert np.any(lab.d.qfrc_applied != 0)
    r.kick(4.0, "flat")
    lab.step()
    assert not r.state()["dribbler_engaged"]
    assert np.allclose(lab.d.qfrc_applied, 0.0)


# ------------------------------------------------------------------ 6 replay (conformance 6)
def _run_hash():
    lab = Lab([("b", "blue", 0, (0, 0, 0)), ("y", "yellow", 1, (0.6, 0.0, math.pi))], ball=(0.3, 0.0))
    a, b = lab.rigs
    a.dribble(2000)
    a.set_base_target(0.4, 0.2, 0.5)
    b.set_base_target(0.2, -0.2, 2.0)
    h = hashlib.sha256()
    for i in range(200):
        if i == 100:
            a.kick(5.0, "flat", armed_for=1.0)
        lab.step()
        h.update(lab.d.qpos.tobytes()); h.update(lab.d.qvel.tobytes())
    return h.hexdigest()


def test_deterministic_replay():
    assert _run_hash() == _run_hash()


if __name__ == "__main__":
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("PASS", name, flush=True)
            except Exception as e:  # noqa: BLE001
                fails += 1
                print("FAIL", name, type(e).__name__, e, flush=True)
    sys.exit(1 if fails else 0)
