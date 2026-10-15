"""Table tennis' own tests: one per sport.toml [[exploits]] entry (core/testkit.py runs them in the sport's image; the
generic suite - determinism, no draw, whitelist, purity, malformed actions - comes from the core). Ported from the
legacy scripts/olympiad/tabletennis/test_modes.py where the rule is the same."""
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SPORT_DIR = HERE.parent
from core.episode import load_plugin  # noqa: E402
from core.sport import FINISHED, UNIT_OVER, MatchConfig, Refused  # noqa: E402

S = load_plugin(str(SPORT_DIR / "plugin.py"))
import action_low as AL  # noqa: E402  (the sport's sim/, on sys.path once the plugin is loaded)
import builtin_players as BP  # noqa: E402
import rcb_tt  # noqa: E402


def cfg(fmt="match", level="high", seed=101, first="A"):
    seats = ("A",) if fmt == "match" else ("A", "B")
    return MatchConfig(sport="tabletennis", format=fmt, seats=seats, seed=seed, first=first,
                       action_level=level, level=1 if fmt == "match" else None)


def refused(layer, m, body):
    try:
        layer.parse("A", body, m, {})
    except Refused:
        return True
    return False


def play_out(m, seats, k_max=20000):
    """No commands at all (the arm stays where the level's rest puts it); reviews end at once."""
    while m.phase().state != FINISHED and m.k < k_max:
        if m.phase().state == UNIT_OVER:
            m.resume()
        m.step()
    return m


def test_exploit_bad_commands_are_refused():
    q0 = [float(x) for x in rcb_tt.HOME] + [float(x) for x in rcb_tt.load_instance(str(SPORT_DIR / "sim" / "instance.npz"))["q0"]]
    lim = (AL.VEL_LIMIT * rcb_tt.CONTROL_DT).tolist()
    bad_high = [{"type": "act", "t": 0, "pos": [0, -1.2, .3], "normal": [0, 1, 0], "vel": [0, 1, 0]},
                {"type": "act", "t": 3, "pos": [0, -1.2, .3], "normal": [0, 1, 0], "vel": [0, 1, 0]},
                {"type": "act", "t": .5, "pos": [float("nan"), -1.2, .3], "normal": [0, 1, 0], "vel": [0, 1, 0]},
                {"type": "act", "t": .5, "pos": [0, 1.0, .3], "normal": [0, 1, 0], "vel": [0, 1, 0]},
                {"type": "act", "t": .5, "pos": [0, -1.2, .3], "normal": [0, 0, 0], "vel": [0, 1, 0]},
                {"type": "act", "t": .5, "pos": [0, -1.2, .3], "normal": [0, 1, 0], "vel": [0, 9, 0]},
                {"type": "act", "t": .5, "pos": [0, -1.2], "normal": [0, 1, 0], "vel": [0, 1, 0]},
                {"type": "act", "t": .5, "pos": [0, -1.2, .3], "normal": [0, 1, 0], "vel": [0, 1, 0], "base": [0, -1.2]},
                {"type": "act", "t": .5, "pos": [0, -1.2, .3], "normal": [0, 1, 0], "vel": [0, 1, 0], "base": [1.5, -2]},
                {"type": "move", "base": [0, -1.0]}, {"type": "move", "base": [0]}, {"type": "move"},
                {"type": "toss", "pos": [0.3, -1.45, 0.2], "height": 0.15}, {"type": "toss", "pos": [0.3, -1.45], "height": .3},
                {"type": "toss", "pos": [0.3, -1.45, 0.2]}, {"type": "toss", "pos": [0.3, -1.45, 0.2], "height": 1.2},
                {"type": "toss", "pos": [0.3, -1.45, 0.2], "vz": 2.0},
                {"type": "traj", "k0": 0, "q": [q0]}, {"type": "swing"}, {}, "act"]
    too_fast = [q0, [q0[0] + 2 * lim[0]] + q0[1:]]
    too_sharp = [q0, q0, q0[:3] + [q0[3] + 0.9 * lim[3]] + q0[4:]]     # within the speed limit, 196 rad/s^2 on joint 2
    bad_low = [{"type": "traj", "k0": 0.5, "q": [q0]}, {"type": "traj", "k0": -1, "q": [q0]},
               {"type": "traj", "k0": 0, "q": [q0[:6]]}, {"type": "traj", "k0": 0, "q": [q0] * 301},
               {"type": "traj", "k0": 0, "q": too_fast}, {"type": "traj", "k0": 0, "q": too_sharp},
               {"type": "traj", "k0": 0, "q": [[9.0] * 9]},
               {"type": "traj", "k0": 0, "q": [q0[2:]]}, {"type": "traj", "k0": 0, "q": [[0.0, -1.2] + q0[2:]]},
               {"type": "slew", "q": q0, "speed": 0}, {"type": "slew", "q": q0, "speed": 1.5},
               {"type": "slew", "q": [9.0] * 9}, {"type": "slew", "q": q0[2:]}, {"type": "act", "t": .5},
               {"type": "move"}, {"type": "toss", "pos": [0.3, -1.45, 0.2], "height": 0.1}, {}]
    for fmt in ("match", "duel"):
        for level, bodies in (("high", bad_high), ("low", bad_low)):
            m = S.build(cfg(fmt, level, seed=1))
            layer = S.action_layer(level)
            d0 = m.state_digest()
            leaked = [b for b in bodies if not refused(layer, m, b)]
            assert not leaked and m.state_digest() == d0, (fmt, level, leaked)
    m = S.build(cfg("match", "low"))
    assert not refused(S.action_layer("low"), m, {"type": "traj", "k0": 0, "q": [q0, q0]})
    assert not refused(S.action_layer("high"), m, {"type": "none"})
    assert not refused(S.action_layer("high"), m, {"type": "move", "base": [0.5, -2.0]})


def test_exploit_discontinuous_start_is_rejected():
    for fmt in ("match", "duel"):
        m = S.build(cfg(fmt, "low", seed=1))
        layer = S.action_layer("low")
        for _ in range(3):
            m.step()
        tgt = (m.R.s._target if fmt == "match" else m.g.target["A"]).copy()
        jump = tgt.copy()
        jump[0] += 3 * AL.VEL_LIMIT[0] * rcb_tt.CONTROL_DT
        cmd = layer.parse("A", {"type": "traj", "k0": m.k, "q": [jump.tolist()]}, m, {})
        info = m.apply("A", cmd)
        assert "rejected" in info and "discontinuous" in info["rejected"], info
        m.step()                                    # the simulator's own speed check would raise on a jump
        now = m.R.s._target if fmt == "match" else m.g.target["A"]
        assert np.abs(now - tgt).max() <= AL.VEL_LIMIT.max() * rcb_tt.CONTROL_DT + 1e-12


def test_exploit_real_limits_hold():
    """Whatever the targets, the robot obeys the real limits (principle B0): the servo's setpoints stay within the arm's
    velocity / acceleration / jerk limits and the rail's, the arm's torques within 87 / 12 N m changing by at most
    1 N m per ms; a traj that would accelerate a joint too sharply from the target in force is rejected; the scripted
    opponent's paddle obeys the rail's limits and its shots stay within the real arm's envelope."""
    import match as MT
    m = S.build(cfg("match", "low", seed=1))
    s = m.R.s
    rec = []
    orig = s.servo.ctrl

    def ctrl(*a, **k):
        out = orig(*a, **k)
        sv = s.servo
        rec.append(np.r_[sv.jl.q, sv.jl.qd, sv.jl.qdd, sv.jb.q, sv.jb.qd, sv.jb.qdd, out, sv.n_resync])
        return out
    s.servo.ctrl = ctrl
    step = rcb_tt.VEL_LIMIT * rcb_tt.CONTROL_DT
    q = s._target.copy()
    for k in range(120):                          # the speed limit, reversed every step (the simulator's own check
        sign = 1.0 if (k // 1) % 2 == 0 else -1.0  # passes it; the low level's validator would refuse it)
        q = np.clip(q + sign * step * 0.999, s.JOINT_LOW, s.JOINT_HIGH)
        s.step(q)
    for k in range(60):                           # then a hold
        s.step(q)
    R = np.array(rec)
    n_arm = 7
    dt = rcb_tt.PHYS_DT
    qd, qdd = R[:, 7:14], R[:, 14:21]
    bd, bdd = R[:, 23:25], R[:, 25:27]
    tau = R[:, 29:36]
    assert np.all(np.abs(qd) <= rcb_tt.ARM_VEL * (1 + 1e-9)), np.abs(qd).max(0)
    assert np.all(np.abs(qdd) <= rcb_tt.ARM_ACC * (1 + 1e-9)), np.abs(qdd).max(0)
    smooth = np.diff(R[:, 36]) == 0               # a re-sync (the arm pushed off its setpoint) restarts the setpoint
    assert np.all(np.abs(np.diff(qdd, axis=0))[smooth] / dt <= rcb_tt.ARM_JERK * (1 + 1e-6))
    assert np.all(np.abs(bd) <= rcb_tt.BASE_VMAX * (1 + 1e-9)) and np.all(np.abs(bdd) <= rcb_tt.BASE_AMAX * (1 + 1e-9))
    assert np.all(np.abs(np.diff(bdd, axis=0)) / dt <= rcb_tt.BASE_JMAX * (1 + 1e-6))
    assert np.all(np.abs(tau) <= rcb_tt.ARM_LIM.tau + 1e-9)
    assert np.all(np.abs(np.diff(tau, axis=0)) <= rcb_tt.TORQUE_RATE * dt + 1e-9)
    assert n_arm == 7
    # the low level: a first row that changes the velocity beyond the acceleration limit is rejected
    m = S.build(cfg("match", "low", seed=1))
    for _ in range(3):
        m.step()
    tgt = m.R.s._target.copy()
    jolt = tgt.copy()
    jolt[3] += 0.9 * step[3]                      # joint 2 at 90 % of its speed limit from rest: 196 rad/s^2
    info = m.apply("A", S.action_layer("low").parse("A", {"type": "traj", "k0": m.k, "q": [jolt.tolist()]}, m, {}))
    assert "rejected" in info and "acceleration" in info["rejected"], info
    # the opponent's paddle: a long chase from rest obeys the rail's limits (its setpoint at 1 kHz)
    R0 = S.build(cfg("match", "high", seed=1)).R
    R0.opp = {"tx": 0.9, "plan_k": None, "speed": 5.0}       # even a level asking for more than the rail can do
    jl, X = R0._opp_jl, []
    for _ in range(80):
        r = np.array([R0.opp["tx"]])
        jl.V = np.array([min(R0.opp["speed"], rcb_tt.BASE_VMAX) - rcb_tt.BASE_JMAX * dt ** 2])
        for _ in range(rcb_tt.SUBSTEPS):
            jl.step(r)
            X.append([jl.qd[0], jl.qdd[0]])
    X = np.array(X)
    assert np.all(np.abs(X[:, 0]) <= rcb_tt.BASE_VMAX + 1e-9) and np.all(np.abs(X[:, 1]) <= rcb_tt.BASE_AMAX + 1e-9)
    assert np.all(np.abs(np.diff(X[:, 1])) / dt <= rcb_tt.BASE_JMAX * (1 + 1e-6))
    # its shots: within the real arm's envelope (ball speed, blade speed by the contact law)
    v_in = np.array([0.2, 3.5, -0.5])
    assert MT.Match._within_limits(v_in, np.array([0.3, -4.5, 1.0]))
    assert not MT.Match._within_limits(v_in, np.array([0.3, -6.5, 1.0]))          # too fast off the paddle
    assert not MT.Match._within_limits(np.array([0.0, 0.5, 0.0]), np.array([0.0, -5.4, 0.5]))   # blade too fast
    assert MT.OPP_SPIN_MAX <= 34.0 and all(lv["spin"] <= 60 for lv in MT.LEVELS.values())
    # the platform's hit, both formats (match.plan_paddle, workers.prepare; revision 14/11): with the limits-aware
    # planner the contact velocity goes to the planner unscaled (it keeps the limits itself, as checked above); only
    # the hermite swing pre-scales it to 0.9 x the joint speed limits
    import rally
    assert rally.PLANNER == "limits" and rally.contact_prescale(2.0) == 1.0
    planner = rally.PLANNER
    try:
        rally.PLANNER = "hermite"
        assert abs(rally.contact_prescale(1.8) - 2.0) < 1e-12 and rally.contact_prescale(0.5) == 1.0
    finally:
        rally.PLANNER = planner
    # the joint-velocity rule (revision 15/12): the setpoints at 0.9 of the velocity limits (checked above against the
    # full limits; here the margin), and the reflex: an arm joint faster than 1.05 x its limit stops the robot at that
    # pose (targets ignored) for 1 s, then it resumes from rest
    assert rcb_tt.VMARGIN == 0.9 and rcb_tt.REFLEX == 1.05 and rcb_tt.REFLEX_RECOVER_S == 1.0
    assert np.all(np.abs(qd) <= rcb_tt.ARM_VEL * rcb_tt.VMARGIN * (1 + 1e-9)), np.abs(qd).max(0) / rcb_tt.ARM_VEL
    m = S.build(cfg("match", "low", seed=1))
    s = m.R.s
    for _ in range(3):
        s.step(s._target.copy())
    s.d.qvel[s._vadr[3]] = 1.2 * rcb_tt.ARM_VEL[1]               # joint 2 driven past its limit (e.g. an impact)
    q_hold = s._target.copy()
    s.step(q_hold)
    assert s.servo.reflex_at is not None and s.servo.n_reflex == 1
    pose = s.servo.reflex_at[1].copy()
    away = q_hold.copy()                                          # targets the stopped robot must ignore
    for _ in range(95):
        away[2] = min(away[2] + 0.01, s.JOINT_HIGH[2] - 0.1)
        s.step(away)
    assert s.servo.reflex_at is not None and np.max(np.abs(s.d.qpos[s._qadr] - pose)) < 0.05
    for _ in range(10):
        s.step(away)
    assert s.servo.reflex_at is None                              # recovered after 1 s: it follows the targets again
    # the table-avoidance via: on the contact velocity's line, VIA_STEPS_LIMITS steps before the contact, raised by dz
    p_c, v_p = np.array([0.1, -1.2, 0.08]), np.array([0.2, 2.0, 1.0])
    via = rally.via_pose(p_c, v_p, np.array([0.0, 1.0, 0.0]), 0.05)
    assert np.allclose(via, p_c - v_p * rally.VIA_STEPS_LIMITS * rcb_tt.CONTROL_DT + np.array([0.0, 0.0, 0.05]))


def _keys(o, acc):
    if isinstance(o, dict):
        for k, v in o.items():
            acc.add(str(k))
            _keys(v, acc)
    elif isinstance(o, list):
        for v in o:
            _keys(v, acc)
    return acc


def test_exploit_no_hidden_state():
    from core.local import seat_spec
    forbidden = {"seed", "ball_pos", "ball_vel", "ball_spin", "opp_x", "opp", "serve_ids", "serve_log", "judge",
                 "lv", "events", "rs", "contact_pos", "vel_pre"}
    for fmt in ("match", "duel"):
        for level in ("high", "low"):
            c = cfg(fmt, level, seed=7)
            m = S.build(c)
            for _ in range(400):
                if m.phase().state == UNIT_OVER:
                    m.resume()
                m.step()
                if m.k % 40 == 0:
                    for s in c.seats:
                        bad = _keys(m.observe(s), set()) & forbidden
                        assert not bad, (fmt, level, bad)
            for s in c.seats:
                sp = seat_spec(S, c, s, plugin=str(SPORT_DIR / "plugin.py"))
                assert not (_keys(sp, set()) & forbidden), (fmt, level, _keys(sp, set()) & forbidden)


def test_exploit_never_moving_loses():
    m = play_out(S.build(cfg("match", "low")), ("A",))
    assert m.phase().state == FINISHED and m.scores()["A"] == 0 and m.tiebreaks()[0][1](m) == "opponent", m.scores()
    assert m.R.disqualified == 0                               # it never served: its 3rd undue delay lost it the match
    m = play_out(S.build(cfg("match", "high")), ("A",))       # the high level's rest: back to the ready pose
    assert m.scores()["A"] < m.scores()["opponent"] and m.tiebreaks()[0][1](m) == "opponent", m.scores()


def test_exploit_high_level_lowers_to_valid_low_level():
    """Every act / move of the high level is lowered to a low-level command that the low level's own validator accepts
    (same joint ranges, base travel and speed limits; the `move` and the rest `slew` are the base's rest position), and
    the episode of such commands replays bit for bit in a fresh match (core/local.py records the lowered commands)."""
    import tempfile as tf

    from core import local
    from core.episode import read_rows, replay
    P = str(SPORT_DIR / "plugin.py")
    for fmt in ("match", "duel"):
        c = cfg(fmt, "high", seed=3)
        pl = local.make_players(S, c, {s: "reference" for s in c.seats}, plugin=P)
        d = Path(tf.mkdtemp())
        live = local.play(S, c, pl, plugin=P, data=d, max_steps=1500)
        m = S.build(c)
        low = S.action_layer("low")
        n = 0
        for line in (d / "episode.jsonl").read_text().splitlines():
            ev = json.loads(line)
            if ev.get("ev") != "cmd":
                continue
            cmd = ev["cmd"]
            if cmd["type"] in ("traj", "slew"):
                body = {k: v for k, v in cmd.items() if k in ("type", "k0", "q", "speed")}
                assert isinstance(low.parse(ev["seat"], body, m, {}), dict), (fmt, cmd["type"])
                n += 1
        assert n >= 3, (fmt, n)
        again = replay(S, read_rows(d / "episode.jsonl"))
        assert again["state_sha"] == live["digest"] and again["k"] == live["k"], (fmt, again["k"], live["k"])


def _poses(mj, m, d, q, qadr, suffix=""):
    d.qpos[qadr] = q
    mj.mj_kinematics(m, d)
    P, R = [], []
    for name in AL.CHAIN:
        b = mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, name + suffix)
        P.append(d.xpos[b].copy())
        R.append(d.xmat[b].copy())
    site = mj.mj_name2id(m, mj.mjtObj.mjOBJ_SITE, "paddle_center" + suffix)
    return np.array(P + [d.site_xpos[site].copy()]), np.array(R + [d.site_xmat[site].copy()])


def test_exploit_robot_model_is_the_simulator_arm():
    """The robot's own model (/spec robot_model.robot_only_xml) and the kinematic chain of /spec (as the reference
    player's arm model reads it) reproduce the simulator's arm; the model holds nothing of the world."""
    import mujoco as mj
    from core.local import seat_spec
    c = cfg("match", "low")
    sp = seat_spec(S, c, "A", plugin=str(SPORT_DIR / "plugin.py"))
    xml = sp["robot_model"]["robot_only_xml"]
    tmp = Path(tempfile.mkdtemp())
    shutil.copytree(SPORT_DIR / "sim" / "scene" / "assets", tmp / "assets")
    (tmp / "robot_only.xml").write_text(xml)
    rm = mj.MjModel.from_xml_path(str(tmp / "robot_only.xml"))
    rd = mj.MjData(rm)
    s = S.build(c).R.s
    sd = mj.MjData(s.m)
    radr = np.array([rm.jnt_qposadr[mj.mj_name2id(rm, mj.mjtObj.mjOBJ_JOINT, j)] for j in rcb_tt.JOINTS])
    arm = BP.Arm(sp)
    rng = np.random.default_rng(11)
    worst = arm_err = 0.0
    for _ in range(200):
        q = rng.uniform(s.JOINT_LOW, s.JOINT_HIGH)
        P1, R1 = _poses(mj, s.m, sd, q, s._qadr)
        P2, R2 = _poses(mj, rm, rd, q, radr)
        worst = max(worst, np.abs(P1 - P2).max(), np.abs(R1 - R2).max())
        p, R = arm.fk(q)
        arm_err = max(arm_err, np.abs(p - P1[-1]).max(), np.abs(R.ravel() - R1[-1]).max())
    assert worst < 1e-9 and arm_err < 1e-6, (worst, arm_err)
    names = {mj.mj_id2name(rm, mj.mjtObj.mjOBJ_BODY, i) for i in range(rm.nbody)} - {"world"}
    assert names == set(AL.CHAIN), names
    assert (rm.njnt, rm.nu, rm.nlight, rm.ncam) == (9, 9, 0, 0)
    pairs = [mj.mj_id2name(rm, mj.mjtObj.mjOBJ_PAIR, i) for i in range(rm.npair)]   # only its self-collision pairs
    scene_pairs = [mj.mj_id2name(s.m, mj.mjtObj.mjOBJ_PAIR, i) for i in range(s.m.npair)]
    assert pairs and all(n.startswith("selfcol_") for n in pairs) and set(pairs) <= set(scene_pairs), pairs
    import xml.etree.ElementTree as ET
    root = ET.fromstring(xml)
    words = {el.get("name", "") for el in root.iter()} | {el.get("file", "") for el in root.iter()}
    assert not any(w in x for x in words for w in ("ball", "table", "net", "floor", "target", "pedestal", "rail"))
    assert not {el.tag for el in root.iter()} & {"freejoint", "light", "camera", "texture", "material"}
    # the low-level seat image ships the same model (sport.toml [seat.low], seat/robot_only.xml) and the sim's MuJoCo
    import tomllib
    assert (SPORT_DIR / "seat" / "robot_only.xml").read_text() == xml
    pins = tomllib.loads((SPORT_DIR / "sport.toml").read_text())["seat"]["low"]["pip"]
    assert f"mujoco=={mj.__version__}" in pins, (pins, mj.__version__)
    d = S.build(cfg("duel", "low", seed=1))                  # the duel's seats: the same chain in their own frames
    sp_b = seat_spec(S, cfg("duel", "low", seed=1), "B", plugin=str(SPORT_DIR / "plugin.py"))
    assert sp_b["kinematics"] == sp["kinematics"] and sp_b["limits"] == sp["limits"]
    dd = mj.MjData(d.g.m)
    for _ in range(50):
        q = rng.uniform(s.JOINT_LOW, s.JOINT_HIGH)
        P2, _ = _poses(mj, rm, rd, q, radr)
        P1, _ = _poses(mj, d.g.m, dd, q, d.g.qadr["B"], "_b")
        assert np.abs(P1 * np.array([-1, -1, 1]) - P2).max() < 1e-9


def _toss_and_drop(m, seat):
    """seat tosses a legal ball at its first chance and never strikes it."""
    layer = S.action_layer("high")
    base = m.R.s._target[:2] if hasattr(m, "R") else m.g.target[seat][:2]
    body = {"type": "toss", "pos": [float(base[0]) + 0.3, -1.45, 0.25], "height": 0.2}
    cmd = layer.parse(seat, body, m, {})
    low, _ = layer.lower(seat, cmd, None, m)
    return m.apply(seat, low)


def test_exploit_never_serving_is_disqualified():
    """A player who never tosses loses each of its services (undue delay, serve.SERVE_CLOCK_S) and its third such fault
    loses it the match, even when it is ahead on points: B tosses and drops every service (A scores), A never serves."""
    import serve as SV
    m = S.build(cfg("duel", "high", seed=5, first="B"))
    layer = S.action_layer("high")
    while m.phase().state != FINISHED and m.k < 20000:
        if m.phase().state == UNIT_OVER:
            m.resume()
        g = m.g
        if g.rs["server"] == "B" and g.rs["serve"]["toss"] is None and g.k >= g.rs["serve"]["open_k"]:
            assert "toss" in _toss_and_drop(m, "B")
        m.step()
    g = m.g
    assert m.phase().state == FINISHED and g.disqualified == "A" and g.winner == "B", (g.score, g.disqualified)
    assert g.score["A"] > g.score["B"], g.score                  # A was ahead and still lost
    assert g.delay_faults["A"] == SV.DELAY_LIMIT and m.tiebreaks()[0][1](m) == "B"
    falls = [p["how"] for p in g.points if p["server"] == "B"]
    assert falls and all("Law 2.6.4" in h or "2.10.1.1" in h for h in falls), falls
    delays = [p["how"] for p in g.points if p["server"] == "A"]           # the reason codes of racket_serve_zh.md
    assert delays and all(h.startswith("serve_fault:undue_delay: ") for h in delays), delays


def test_exploit_serve_rules():
    """The service is judged by ITTF 2.6 / 2.9.1.1 / 2.10.1.6 / 2.13.3 only (serve.py): the order of serving, the toss
    limits (refused), a dropped toss = the receiver's point, the bounce sequence, lets, obstruction, the position."""
    import serve as SV
    # 2.13.3 scaled: 2 serves each, from 4:4 one each, lets do not count
    seq, a, b = [], 0, 0
    for w in "ABABABAB" + "ABAB":
        seq.append("AB"[SV.server_of((a, b), 0)]); a += w == "A"; b += w == "B"
    assert "".join(seq) == "AABBAABB" + "ABAB", seq
    # toss refusals: not your service / not yet / in front of the end line / below the surface / out of reach
    m = S.build(cfg("duel", "high", seed=2, first="A"))
    hi = S.action_layer("high")

    def code(seat, body):
        try:
            hi.parse(seat, body, m, {})
        except Refused as e:
            return e.code
        return 200
    ok = {"type": "toss", "pos": [0.3, -1.45, 0.25], "height": 0.2}
    assert code("A", ok) == 409                                   # the receiver is not ready yet (the serve opens later)
    while m.k < m.g.rs["serve"]["open_k"]:
        m.step()
    assert code("B", ok) == 409                                   # not B's service
    assert code("A", dict(ok, pos=[0.3, -1.30, 0.25])) == 400     # in front of the end line (Law 2.6.4)
    assert code("A", dict(ok, pos=[0.3, -1.45, 0.01])) == 400     # below the playing surface (Law 2.6.4)
    assert code("A", dict(ok, pos=[1.2, -1.45, 0.25])) == 400     # out of the toss device's reach
    assert code("A", ok) == 200
    # a dropped toss: the receiver's point (match: you serve first; the opponent scores)
    mm = S.build(cfg("match", "high", seed=3))
    while mm.R.k < mm.R.leg["open"]:
        mm.step()
    assert "toss" in _toss_and_drop(mm, "A")
    while not mm.R.points:
        mm.step()
    p = mm.R.points[0]
    assert p["winner"] == "opponent" and p["how"].startswith("serve_fault:dropped: ") and "2.10.1.1" in p["how"], p
    assert abs(SV.vz_for_height(0.16) - 1.79) < 0.03                               # the 16 cm toss of Law 2.6.2
    # the judge: bounce sequence, let, obstruction, rising strike, position
    def J(events, pos=None):
        j = SV.ServeJudge(0)
        j.projected = True
        for e in events:
            if e == "pos":
                j.position(pos)
                continue
            kind, vz = (e, -1.0) if isinstance(e, str) else e
            j.contact(kind, [0.0, -1.45, 0.2] if kind == "paddle_server" else [0.0, 0.5, 0.02], [0.0, 3.0, vz], 1)
        return j.verdict
    assert J(["paddle_server", "table_server", "table_receiver"]) == "good"
    assert J(["paddle_server", "table_server", "net", "table_receiver"]) == "let"
    assert J(["paddle_server", "table_receiver"]) == "fault"                       # not on the server's court first
    assert J(["paddle_server", "table_server", "table_server"]) == "fault"         # twice on the server's court
    assert J([("paddle_server", 0.5), "table_server", "table_receiver"]) == "fault"  # struck while rising (2.6.3)
    assert J(["robot_server"]) == "fault"                                           # touched the arm first (2.6.2)
    assert J(["paddle_server", "table_server", "robot_receiver"]) == "obstruction"  # the receiver obstructs (2.10.1.6)
    assert J(["paddle_server", "table_server", "net", "robot_receiver"]) == "let"   # ... after a net touch (2.9.1.1)
    assert J(["pos"], pos=[0.0, -1.30, 0.2]) == "fault"                             # in front of the end line (2.6.4)
    assert J(["pos"], pos=[0.0, -1.45, 0.01]) == "fault"                            # below the surface (2.6.4)
    j = SV.ServeJudge(0)
    j.contact("paddle_server", [0, -1.45, 0.2], [0, 0, -1.0], 1)
    assert j.verdict == "fault" and "projected" in j.why                           # struck while resting (2.6.2)
    # D13 (revision 14 / 11): the bounce on the receiver's court at least SERVE_MIN_DEPTH from the net, else a fault
    # ("short"), also after a net touch (a let needs an otherwise correct service)
    def depth(y, net=False):
        j = SV.ServeJudge(0)
        j.projected = True
        j.contact("paddle_server", [0.0, -1.45, 0.2], [0.0, 3.0, -1.0], 1)
        j.contact("table_server", [0.0, -0.6, 0.02], [0.0, 3.0, -1.0], 2)
        if net:
            j.contact("net", [0.0, 0.0, 0.17], [0.0, 3.0, -1.0], 3)
        j.contact("table_receiver", [0.0, y, 0.02], [0.0, 3.0, -1.0], 4)
        return j.verdict, j.code
    d = SV.SERVE_MIN_DEPTH
    assert d == 0.2                                                                 # sport.toml D13
    assert depth(d + 0.01) == ("good", "") and depth(d + 0.01, net=True) == ("let", "")
    assert depth(d - 0.01) == ("fault", "short") and depth(d - 0.01, net=True) == ("fault", "short")
    # the scripted opponent's serve library obeys D13 (its bounce on our court: opponent frame, receiver at y < 0)
    import match as MT
    assert all(-s["bounce_rcv"][1] >= d - 1e-3 for s in MT.SERVE_LIBRARY["serves"])
    # the receiver does not see the server's toss point (the server does)
    while m.g.rs["serve"]["toss"] is None:
        assert "toss" in _toss_and_drop(m, "A")
    m.step()
    assert "toss" in m.observe("A")["serve"] and "toss" not in m.observe("B")["serve"]


def _let_service(name="mid_wide_bh_top", clearance=-0.02):
    """A service of the reference's repertoire re-aimed (on the reference's own flight model, no simulator) so that the
    ball's centre passes the net `clearance` m below the height at which it would just clear the tape (NET_Z + BALL_R):
    it clips the net and still lands on the receiver's court, a let (Law 2.9.1.1). Same return value as
    builtin_players.serve_plan; the vertical speed family is the one whose clearance is nearest the target."""
    r = BP.SERVES[name]
    base = np.array(r["base"], float)
    P = np.array([base[0] + r["toss"][0], base[1] + r["toss"][1], r["toss"][2]])
    t = 0.0
    while True:
        t += 0.005
        p, v = BP.toss_flight(P, BP.toss_speed(r["height"]), t)
        if v[2] < 0 and p[2] <= r["zc"]:
            break

    def solve(vz):                                    # = BP.solve_serve, without its net-clearance requirement
        vo = np.array([(r["b2"][0] - p[0]) / 0.6, 3.5, float(vz)])
        for _ in range(30):
            L, _ = BP.serve_bounces(p, vo)
            if len(L) < 2 or L[0][1] >= 0 or abs(L[0][0]) > BP.TABLE_X or L[0][1] < -BP.TABLE_Y:
                vo[1] *= 0.92
                continue
            e = L[1] - np.asarray(r["b2"], float)
            if np.linalg.norm(e) < 0.01:
                return vo
            J = np.zeros((2, 2))
            for c in range(2):
                dv = np.zeros(3); dv[c] = 0.02
                L2, _ = BP.serve_bounces(p, vo + dv)
                if len(L2) < 2:
                    return None
                J[:, c] = (L2[1] - L[1]) / 0.02
            vo[:2] -= np.clip(np.linalg.solve(J, e), -1.0, 1.0)
        return None
    best = None
    for vz in np.arange(-2.5, 2.5, 0.05):
        vo = solve(vz)
        if vo is None:
            continue
        F, _, b = BP.rollout(p, vo, 0.002, 900, bounce=True)
        seg = F[b[0]:b[1] + 1]
        c = float(seg[int(np.argmin(np.abs(seg[:, 1]))), 2] - (BP.NET_Z + BP.BALL_R))
        p_c, n, v_p = BP.paddle_for(p, v, vo, tangential=True)
        if np.linalg.norm(v_p) <= r.get("vp_max", BP.VP_SERVE) and (best is None or abs(c - clearance) < abs(best[0] - clearance)):
            best = (c, p_c, n, v_p)
    _, p_c, n, v_p = best
    brush = np.array([0.0, 0.0, float(r.get("brush", 0.0))])
    v_p = v_p + (brush - (brush @ n) * n)
    return ({"pos": P.round(4).tolist(), "height": r["height"]},
            {"t_from_projection": round(t, 3), "pos": p_c, "normal": n, "vel": v_p, "base": base})


def test_exploit_let_stalling():
    """Let stalling (D12, E11). A let does not change the server (2.13.3) and the Laws set no limit on lets, so before
    revision 11 / 9 a server could replay a service that clips the net for ever: at the low level a program sends the
    same toss and joint trajectory from the same reset state (duel: both robots are reset before every point, D11) and
    the physics repeats the let exactly. Reproduced on revision 10 / 8 (duel, low level, seed 1, B idle): after
    winning point 1, A served this let 39 times in a row and won 1:0 at the 40-point guard (D8); served from 0:0 all
    40 points were lets and the guard crashed (no point won to decide it). Now the server's third service let since
    the last point scored is the receiver's point (serve_fault:repeated_let). Duel: the real let service; match: the
    referee's let counter for both servers (the match does not reset the robot between points, so the same commands
    do not repeat a let there)."""
    import serve as SV
    from core import local as L
    plan = _let_service(name="mid_wide_bh", clearance=-0.01)   # re-aimed for revision 15 / 12 (the realistic
    # restitution and the servo changed the flight: the old fixture, mid_wide_bh_top at -0.02, now clears the net or
    # falls back; a test fixture only, the rule is unchanged)
    c = cfg("duel", "low", seed=1, first="A")
    m = S.build(c)
    layer = S.action_layer("low")
    ref = BP.Reference(L.seat_spec(S, c, "A", plugin=str(SPORT_DIR / "plugin.py")), "A", 1)   # low-level serve, re-aimed
    orig = BP.serve_plan
    BP.serve_plan = lambda name, rest_steps=10, recipe=None: plan
    try:
        while m.phase().state != FINISHED and m.k < 12000:
            if m.phase().state == UNIT_OVER:
                m.resume()
            if m.k % 5 == 0:
                st = m.observe("A")
                if st.get("you_serve"):
                    assert st["serve"].get("let_limit") == SV.LET_LIMIT and "lets_in_row" in st["serve"], st["serve"]
                body = ref.act({"state": dict(st, step=m.k, world="running"), "observation": {}})
                if body is not None:
                    m.apply("A", layer.lower("A", layer.parse("A", body, m, {}), None, m)[0])
            m.step()
    finally:
        BP.serve_plan = orig
    pts = m.g.points
    how = [p["how"][:60] for p in pts]
    lets = [p for p in pts if p["server"] == "A" and p["how"].startswith("let: ")]
    faults = [p for p in pts if p["how"].startswith("serve_fault:repeated_let: ")]
    assert len(lets) >= 4 and len(faults) >= 2, how                  # the exploit is live: A's service is a let
    run = 0
    for p in pts:                                                    # never more than LET_LIMIT - 1 lets in a row
        run = run + 1 if p["how"].startswith("let: ") else 0
        assert run < SV.LET_LIMIT, how
    for p in faults:                                                 # the third let is the receiver's point
        assert p["server"] == "A" and p["winner"] == "B" and f"let {SV.LET_LIMIT} of {SV.LET_LIMIT}" in p["how"], p
    assert [x["code"] for x in m.g.serve_log if x.get("code") == "repeated_let"] == ["repeated_let"] * len(faults)
    assert m.phase().state == FINISHED and len(pts) < 40, how         # decided by the score, not by the guard (D8)
    # format match: the same counter for your services and the scripted opponent's; a point scored resets it
    m = S.build(cfg("match", "low", seed=1))
    R = m.R
    for server, winner in ((0, "opponent"), (1, "you")):
        for i in range(SV.LET_LIMIT):
            R._service_let(server=server, how="let: the service touched the net")
        p = R.points[-SV.LET_LIMIT:]
        assert [x["winner"] for x in p] == ["let"] * (SV.LET_LIMIT - 1) + [winner], p
        assert p[-1]["how"].startswith("serve_fault:repeated_let: ") and R.serve_lets == 0, p
    R._service_let(server=0, how="let: the service touched the net")
    assert R.points[-1]["winner"] == "let" and R.serve_lets == 1


# ------------------------------------------------------------------ the low level's skill API (Agent as Robotics Engineer)
def _toml() -> dict:
    import tomllib
    return tomllib.loads((SPORT_DIR / "sport.toml").read_text())


def _evaluator():
    """lowlevel/evaluator.py (the skilleval hooks), loaded once."""
    import importlib.util
    import sys
    if "tt_evaluator" not in sys.modules:
        spec = importlib.util.spec_from_file_location("tt_evaluator", SPORT_DIR / "lowlevel" / "evaluator.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules["tt_evaluator"] = mod
        spec.loader.exec_module(mod)
    return sys.modules["tt_evaluator"]


def test_b9_every_hit_request_is_feasible():
    """Robot and mode rule 2 (B9) for the hit skill: every API-test request is valid under the contract and feasible
    under the real limits: from each hidden case's start state (the recorded match replayed bit for bit: the robot as
    our reference player's match left it, the ball in flight) the platform's own rehearsal of it meets the contract
    with a margin (evaluator._delivers); every public request is valid."""
    from skilleval import cases as C
    from skill_api import DT, HIT
    ev = _evaluator()
    cs = _toml()["modes"]["low"]["cases"]
    hidden, public = C.load(SPORT_DIR / cs["hidden"]), C.load(SPORT_DIR / cs["public"])
    assert len(hidden) >= 50 and len(public) >= 20, (len(hidden), len(public))
    for c in hidden + public:
        assert HIT.valid_request(c["request"]) is None, (c["id"], HIT.valid_request(c["request"]))
    for c in hidden:
        R = ev.core(c["level"], c["seed"], True)
        ev.replay_to(R, ev.load_log(c["log"]), c["k"])
        r = c["request"]
        R._ik_seed = None
        _, info = R.plan_paddle((R.k - R.leg["launch"]) * DT + float(r["t"]), r["pos"], r["normal"], r["vel"])
        assert ev._delivers(info, r), (c["id"], info.get("stroke"))


def test_exploit_toolkit_is_the_manifest():
    """E12: the low-level toolkit (seat/toolkit.toml) is what the seat gets and what /spec names: the manifest passes
    skilleval (schema, sha256 of every tool, the seat ships exactly its tools, the practice file and the manifest);
    the libraries import nothing of the simulator; the kinematics library reproduces the simulator's paddle kinematics
    (9 joints) and its IK is deterministic; the servo model rolls joint targets out exactly as the simulator's robot
    (no world contact), bit for bit."""
    import ast
    import re
    import sys
    from core.local import seat_spec
    from skilleval import declare, toolkit
    seat = SPORT_DIR / "seat"
    assert toolkit.problems(seat / "toolkit.toml", "tabletennis", "low") == []
    rows = declare.check("tabletennis", _toml(), SPORT_DIR)
    assert all(ok for _, ok, _ in rows), [r for r in rows if not r[1]]
    tk = toolkit.load(seat / "toolkit.toml")
    tools = {t["name"]: t for t in tk["tools"]}
    files = _toml()["seat"]["low"]["files"]
    libs = {"ik_library": "tt_kinematics.py", "servo_model": "tt_servo.py"}
    for name, fn in libs.items():
        lib = seat / "lib" / fn
        assert hashlib.sha256(lib.read_bytes()).hexdigest() == tools[name]["sha256"]
        assert files[tools[name]["file"]] == f"seat/lib/{fn}"
        tree = ast.parse(lib.read_text())
        imports = {a.name.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names} | \
            {n.module.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
        assert imports <= {"__future__", "os", "re", "pathlib", "numpy", "mujoco"}, (fn, imports)
    assert sorted(p.name for p in (seat / "lib").iterdir() if p.suffix == ".py") == sorted(libs.values())
    assert hashlib.sha256((seat / "robot_only.xml").read_bytes()).hexdigest() == tools["robot_model"]["sha256"]
    assert files[tools["robot_model"]["file"]] == "seat/robot_only.xml"
    assert files["/opt/robot/toolkit.toml"] == "seat/toolkit.toml"
    if str(seat / "lib") not in sys.path:
        sys.path.insert(0, str(seat / "lib"))
    import tt_kinematics
    import tt_servo
    assert tt_kinematics.VERSION == tools["ik_library"]["version"] and tt_servo.VERSION == tools["servo_model"]["version"]
    tmp = Path(tempfile.mkdtemp())
    xml = re.sub(r'meshdir="[^"]*"', f'meshdir="{(SPORT_DIR / "sim" / "scene" / "assets").resolve()}"',
                 (seat / "robot_only.xml").read_text(), count=1)
    (tmp / "robot_only.xml").write_text(xml)
    K = tt_kinematics.PaddleKinematics(tmp / "robot_only.xml")
    s = S.build(cfg("match", "low")).R.s
    rng = np.random.default_rng(5)
    for _ in range(50):
        q = rng.uniform(s.JOINT_LOW, s.JOINT_HIGH)
        p1, R1 = s.fk(q)
        p2, R2 = K.fk(q)
        assert np.abs(p1 - p2).max() < 1e-12 and np.abs(R1 - R2).max() < 1e-12
        assert np.abs(s.jacobian(q) - K.jacobian(q)).max() < 1e-12
    q0 = s.inst["q0"]
    for _ in range(8):
        pos = np.array([rng.uniform(-0.4, 0.4), rng.uniform(-1.5, -1.1), rng.uniform(0.2, 0.45)])
        n = np.array([rng.uniform(-0.3, 0.3), 1.0, rng.uniform(0.0, 0.4)])
        n /= np.linalg.norm(n)
        b = rcb_tt.HOME
        qa, ea = K.solve_near(pos, normal=n, q_ref=q0, base=b, vel_limit=rcb_tt.VEL_LIMIT)
        qb, eb = K.solve_near(pos, normal=n, q_ref=q0, base=b, vel_limit=rcb_tt.VEL_LIMIT)
        assert np.array_equal(qa, qb) and ea == eb                      # deterministic, bit for bit
        p, R = s.fk(np.r_[b, qa])
        assert ea < 1e-4 and np.linalg.norm(p - pos) < 1e-4 and R[:, 0] @ n > 1 - 1e-6, (ea, pos)
    # the servo model = the simulator's robot (the ball parked away, motions that touch nothing of the world)
    sim = tt_servo.RobotSim(tmp / "robot_only.xml")
    for seed in range(3):
        s2 = rcb_tt.Session(instance=str(SPORT_DIR / "sim" / "instance.npz"))
        sim.reset(s2.Q0)
        r2 = np.random.default_rng(seed)
        goal = np.clip(s2.Q0 + np.r_[r2.normal(0, 0.2, 2), r2.normal(0, 0.4, 7)], s2.JOINT_LOW + 0.05, s2.JOINT_HIGH - 0.05)
        q, v = s2.Q0.copy(), np.zeros(9)
        for _ in range(120):
            q, v = rcb_tt.seek(q, v, goal, rcb_tt.VEL_LIMIT, rcb_tt.ACC_LIMIT, s2.JOINT_LOW, s2.JOINT_HIGH)
            s2.d.qpos[0:3] = [3.0, 3.0, 5.0]; s2.d.qvel[0:6] = 0.0
            s2.step(q)
            sim.step(q)
            if s2.d.ncon == 0:
                assert np.array_equal(s2.d.qpos[s2._qadr], sim.d.qpos[sim.qadr]), seed
    sp = seat_spec(S, cfg("match", "low"), "A", plugin=str(SPORT_DIR / "plugin.py"))
    assert sp["toolkit"]["ik_library"]["sha256"] == tools["ik_library"]["sha256"]
    assert sp["toolkit"]["servo_model"]["sha256"] == tools["servo_model"]["sha256"]
    assert sp["toolkit"]["version"] == tk["toolkit"]["version"]
    assert sp["skill_contract"]["skill"] == "hit" and "deliverable" in sp["skill_contract"]
    hi = seat_spec(S, cfg("match", "high"), "A", plugin=str(SPORT_DIR / "plugin.py"))
    assert "toolkit" not in hi and "deliverable" not in hi["skill_contract"]


def test_exploit_hit_cases_hidden_stay_in_the_simulator():
    """E13: the low level's API test: the seat gets only the public practice requests (lowlevel/cases/public.jsonl:
    the request, the robot's state and our plan; no world state, no seed), never the hidden evaluation cases or any
    case's start state (lowlevel/cases/hidden.jsonl.gz, which also carries the replay logs, stays on the simulator side
    and out of the generated tasks); hidden and public are disjoint, also by request."""
    import sys
    import tempfile as tf
    from skilleval import cases as C
    d = _toml()
    cs = d["modes"]["low"]["cases"]
    hidden, public = C.load(SPORT_DIR / cs["hidden"]), C.load(SPORT_DIR / cs["public"])
    assert C.problems(hidden, public) == []
    seat = d["seat"]["low"]["files"]
    assert seat[cs["practice_path"]] == cs["public"]
    assert not any(v.startswith("lowlevel") and v != cs["public"] for v in seat.values()), seat
    assert all(set(c) == {"id", "request", "state", "reference"} for c in public)
    assert not {json.dumps(c["request"], sort_keys=True) for c in hidden} & \
        {json.dumps(c["request"], sort_keys=True) for c in public}
    repo = SPORT_DIR.parents[1]
    if (repo / "tools" / "sportlib.py").is_file():                    # in the repository (not in the sim image)
        sys.path.insert(0, str(repo / "tools"))
        import sportlib
        with tf.TemporaryDirectory() as tmp:
            sim = sportlib.assemble_sim("tabletennis", Path(tmp) / "sim")
            assert not (sim / "sport" / cs["hidden"]).exists() and (sim / "sport" / cs["public"]).exists()


def test_exploit_hit_skill_runs_isolated():
    """E14: the post-match evaluation hosts the player's hit skill in its own process (skilleval/sandbox.py: python -I,
    a scrubbed environment, its own directory): the skill imports nothing of the simulator and sees only what a call
    passes; its prints do not reach the protocol; a call that raises gives no command (counted) and the case goes on
    without it; one that overruns its time limit is stopped (counted)."""
    import os
    from skilleval import cases as C
    from skilleval.sandbox import SkillError, SkillProcess
    ev = _evaluator()
    tmp = Path(tempfile.mkdtemp())
    (tmp / "hit.py").write_text(
        "import os, sys, time\n"
        "def hit(t, pos, normal, vel, *, state, spec):\n"
        "    print('noise'); os.write(1, b'raw noise\\n')\n"
        "    if t > 2: raise RuntimeError('boom')\n"
        "    if t > 1: time.sleep(5)\n"
        "    return {'type': 'hold', 'env': sorted(os.environ), 'spec': sorted(spec),\n"
        "            'mods': sorted(m for m in sys.modules if m in ('match', 'rcb_tt', 'plugin', 'core', 'arm_commands', 'skilleval'))}\n")
    os.environ["RO_TEST_SECRET"] = "s3cr3t"
    try:
        with SkillProcess(tmp, "hit", context={"spec": ev.skill_spec()}, timeouts={"hit": 0.5}) as s:
            out = s.call("hit", 0.5, [0, -1.3, 0.3], [0, 1, 0], [0, 1, 0], state={"step": 0})
            assert out["type"] == "hold" and out["mods"] == [], out
            assert "RO_TEST_SECRET" not in out["env"], out["env"]
            assert out["spec"] == ["kinematics", "limits", "skill_contract", "toolkit"], out["spec"]
            try:
                s.call("hit", 2.5, [0, -1.3, 0.3], [0, 1, 0], [0, 1, 0], state={})
                raise AssertionError("no SkillError")
            except SkillError as e:
                assert e.kind == "error" and s.counters()["skill_errors"] == 1
            try:
                s.call("hit", 1.5, [0, -1.3, 0.3], [0, 1, 0], [0, 1, 0], state={})
                raise AssertionError("no time-out")
            except SkillError as e:
                assert e.kind == "timeout" and s.counters()["skill_timeouts"] == 1 and not s.alive
    finally:
        os.environ.pop("RO_TEST_SECRET")
    (tmp / "hit.py").write_text("def hit(t, pos, normal, vel, *, state, spec):\n    raise RuntimeError('boom')\n")
    case = C.load(SPORT_DIR / _toml()["modes"]["low"]["cases"]["hidden"])[0]
    with SkillProcess(tmp, "hit", context={"spec": ev.skill_spec()}, timeouts={"hit": 5}) as s:
        m = ev.api_case(case, s)                                         # the case runs on without a command
        assert m["measured"] and m["refused"] == 0 and s.counters()["skill_errors"] == 1, m


def test_exploit_skill_in_the_loop():
    """E15: the skill in the loop (skilleval/inloop.py, sport.toml [modes.low.inloop]): with RCB_SKILL_INLOOP_<seat>
    set (a generated low task: the seat's skill container) the low level lowers `hit` with the player's own skill (its
    own process, a private copy of the files, every call logged with their sha256); joint commands are refused (409)
    while a ball is in play and accepted otherwise, `toss` at any time (its own rules); an invalid request is refused
    (400); a skill that gives nothing lowers to a no-op; step() runs every 2 control steps through the core's tick and
    its commands are logged as commands, so the episode replays without the skill. Without the variable the low level
    is the joint layer, unchanged."""
    import os
    import types
    P = types.SimpleNamespace(**type(S).action_layer.__globals__)   # the loaded plugin module's names
    P._INLOOP.clear()
    assert type(S.action_layer("low")).__name__ == "JointLayer"
    P._INLOOP.clear()
    tmp = Path(tempfile.mkdtemp())
    (tmp / "hit.py").write_text(
        "def hit(t, pos, normal, vel, *, state, spec):\n"
        "    assert spec['timing']['compute_counts'] is False and 'skill_contract' in spec\n"
        "    return {'type': 'hold'} if t < 1 else None\n"
        "def step(*, state, spec):\n    return {'type': 'hold'} if state['step'] % 20 == 0 else None\n")
    os.environ.update(RCB_SKILL_INLOOP_A=str(tmp), RCB_SKILL_INLOOP_TIMING="free")
    try:
        layer = S.action_layer("low")
        assert type(layer).__name__ == "InLoopLayer" and layer.free == {"toss"}
        m = S.build(cfg("match", "low"))
        hit = {"type": "hit", "t": 0.5, "pos": [0, -1.3, 0.3], "normal": [0, 1, 0], "vel": [0, 1, 0]}
        assert not refused(layer, m, hit) and refused(layer, m, dict(hit, t=9))
        hold = {"type": "hold"}
        assert not refused(layer, m, hold)                                # no ball in play yet
        while not P._ball_in_play(m, "A"):
            m.step()
            if m.phase().state == UNIT_OVER:
                m.resume()
        try:
            layer.parse("A", hold, m, {})
            raise AssertionError("a joint command accepted while a ball is in play")
        except Refused as e:
            assert e.code == 409 and "hit" in str(e)
        try:                                                              # toss: not refused by the skill rule
            layer.parse("A", {"type": "toss", "pos": [0.3, -1.45, 0.2], "height": 0.3}, m, {})
        except Refused as e:
            assert "skill in the loop" not in str(e), e
        low, info = layer.lower("A", layer.parse("A", hit, m, {}), None, m)
        assert low == {"type": "hold"} and info["sha256"] and info["ok"]
        low, info = layer.lower("A", layer.parse("A", dict(hit, t=1.5), m, {}), None, m)
        assert low == {"type": "noop"} and info["skill_failed"] and info["error"]
        assert m.apply("A", low) == {}                                     # the no-op changes nothing
        ticks = [layer.tick("A", m) for _ in range(1)]
        assert all(t is None or t == {"type": "hold"} for t in ticks)
        rec = layer.loops["A"].calls
        assert {c["fn"] for c in rec} >= {"hit"} and all(c["sha256"] == rec[0]["sha256"] for c in rec)
        from core.local import seat_spec
        sp = seat_spec(S, cfg("match", "low"), "A", plugin=str(SPORT_DIR / "plugin.py"))
        assert sp["skill_in_loop"]["skills"] == ["hit"] and sp["skill_in_loop"]["free"] == ["toss"], sp["skill_in_loop"]
        assert sp["calls"]["act"].startswith("{'type':'hit'") and "hit" in sp["actions"], sp["calls"]["act"]
    finally:
        for k in ("RCB_SKILL_INLOOP_A", "RCB_SKILL_INLOOP_TIMING"):
            os.environ.pop(k, None)
        P._INLOOP.get("layer") and P._INLOOP["layer"].close()
        P._INLOOP.clear()
