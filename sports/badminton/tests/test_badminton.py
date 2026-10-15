"""badminton's own tests: one per [[exploits]] entry of sport.toml (tools/check runs them through core/testkit.py, after
the generic suite; they draw no images). Run alone: python3 -m core.testkit tests sports/badminton/plugin.py (in the
sport's image)."""
import copy
import math
import tomllib
from pathlib import Path

import numpy as np

from core.episode import load_plugin, replay
from core.sport import FINISHED, UNIT_OVER, MatchConfig, Refused

PLUGIN = str(Path(__file__).resolve().parents[1] / "plugin.py")
SPORT = load_plugin(PLUGIN)
DECL = tomllib.loads((Path(PLUGIN).parent / "sport.toml").read_text())
import bduel as BD  # noqa: E402  (sim/, on sys.path once the plugin is loaded)
import bm as BM  # noqa: E402
import wc as WC  # noqa: E402
import builtin_players as BP  # noqa: E402
from core import local  # noqa: E402

ACT = {"type": "act", "t": 0.6, "pos": [0.4, -1.9, 1.3], "normal": [0.0, 0.95, 0.3], "vel": [0.0, 1.4, 0.5]}


def cfg(fmt="duel-wallclock", level="high", seed=1, first="A"):
    seats = ("A",) if fmt == "match-wallclock" else ("A", "B")
    return MatchConfig(sport="badminton", format=fmt, seats=seats, seed=seed, first=first if len(seats) > 1 else None,
                       action_level=level)


def run_until(m, pred, n_max=3000):
    for _ in range(n_max):
        if pred(m):
            return
        if m.phase().state == UNIT_OVER:
            m.resume()
        m.step()
    raise AssertionError("condition not reached")


def in_flight(m, after=30):
    if m.single:
        return m.g.in_play() and m.g.leg["state"] == "incoming" and m.k - m.g.leg["launch"] > after
    return m.g.rs["state"] == "live" and m.k - m.g.rs["played_k"] > after


def serve_by(m, seat, rows=None, kind="long"):
    """The reference player's serve for `seat` (its stroke through the high level, then the toss), until the shuttle
    is in flight towards the receiver. rows: the episode rows (header first) get the commands."""
    run_until(m, lambda m: m.serve_state(seat)["phase"] == "open" and m.serve_state(seat)["server"])
    st = m.observe(seat)
    sgn = -1.0 if st["serve"]["court"] == "right" else 1.0
    BP.configure(SPORT.spec(m.cfg, seat, m))
    pc, rel, v_in, opts = BP.serve_plan(m.serve_state(seat)["base"], kind, np.array([sgn * 0.4, 0.85 * BM.COURT_Y]))
    n, vel = opts[0]
    lead = SPORT.action_layer("high").lead
    k_c = m.k + lead + 130
    low, info, r = high_act(m, seat, {"type": "act", "t": 1.3, "pos": pc.tolist(), "normal": n.tolist(),
                                      "vel": vel.tolist()})
    if rows is not None:
        rows.append({"ev": "cmd", "k": m.k, "seat": seat, "cmd": copy.deepcopy(low)})
    while m.k < k_c - round(BP.SERVE_FALL_S / BM.CONTROL_DT):
        m.step()
    cmd = SPORT.action_layer("low").parse(seat, {"type": "toss", "pos": rel.tolist()}, m, {})
    if rows is not None:
        rows.append({"ev": "cmd", "k": m.k, "seat": seat, "cmd": copy.deepcopy(cmd)})
    assert "rejected" not in m.apply(seat, cmd)
    run_until(m, lambda m: m.g.rs["state"] == "live" or m.phase().state != "playing", 400)
    assert m.g.rs["state"] == "live" and m.g.rs["serve_flight"], m.g.points[-1:]


def refused(layer, seat, body, m, code=400):
    try:
        layer.parse(seat, body, m, {})
    except Refused as e:
        assert e.code == code, (body, e.code, str(e))
        return
    raise AssertionError(f"accepted {body}")


def high_act(m, seat, body):
    """What the core does with a high-level act (snapshot at the request step, plan, lower, apply): the low command."""
    layer = SPORT.action_layer("high")
    cmd = layer.parse(seat, body, m, {})
    prepared = layer.prepare(layer.snapshot(seat, cmd, m, 0))
    low, info = layer.lower(seat, cmd, prepared, m)
    return low, info, m.apply(seat, low)


def test_exploit_bad_commands_are_refused():
    for fmt in ("match-wallclock", "duel-wallclock"):
        for level in ("high", "low"):
            m, layer = SPORT.build(cfg(fmt, level)), SPORT.action_layer(level)
            d0 = m.state_digest()
            bad = [{"type": "jump"}, {"type": "none", "x": 1}, [1, 2], {"type": "act"}, {"t": 1.0},
                   {"type": "toss"}, {"type": "toss", "pos": [0.8, -3.5]}, {"type": "toss", "pos": "x"},
                   {"type": "toss", "pos": [0.8, -3.5, float("nan")]}, {"type": "toss", "pos": [0.8, -3.5, 1.5],
                                                                             "t": 1.0},
                   {"type": "toss", "pos": [0.8, -3.5, 3.0]}, {"type": "toss", "pos": [2.0, -2.0, 1.5]}]
            if level == "high":
                bad += [dict(ACT, t=0.0), dict(ACT, t=6.0), dict(ACT, t=float("inf")), dict(ACT, t="1"),
                        dict(ACT, pos=[0.4, 0.2, 1.7]), dict(ACT, pos=[2.5, -1.5, 1.7]), dict(ACT, pos=[0.4, -1.5, 3.5]),
                        dict(ACT, pos=[0.4, -3.8, 1.5]), dict(ACT, pos=[0.4, -1.5]), dict(ACT, normal=[0, 0, 0]), dict(ACT, vel=[0, 9, 0]),
                        dict(ACT, vel=[0, float("nan"), 0]), dict(ACT, spin=3), {"type": "traj", "q": [[0] * 9]}]
                layer.parse("A", ACT, m, {})
            else:
                q = [float(x) for x in m.robot("A").s.target]
                bad += [{"type": "traj"}, {"type": "traj", "q": []}, {"type": "traj", "q": [q[:8]]},
                        {"type": "traj", "q": [q] * 601}, {"type": "traj", "q": [q[:8] + [float("nan")]]},
                        {"type": "traj", "q": [q], "k0": 3}, dict(ACT)]
                layer.parse("A", {"type": "traj", "q": [q]}, m, {})
            for body in bad:
                refused(layer, "A", body, m)
            assert m.state_digest() == d0


def test_exploit_joint_limits():
    for fmt in ("match-wallclock", "duel-wallclock"):
        m = SPORT.build(cfg(fmt, "low"))
        rob = m.robot("A")
        q0 = np.asarray(rob.s.target, float)
        d0 = m.state_digest()
        far = q0.copy()
        far[3] += 0.2                            # 20x the joint's speed limit in one control step
        out = q0.copy()
        out[2] = rob.s.hi[2] + 0.5               # outside the joint range
        for q in (far, out):
            low = SPORT.action_layer("low").parse("A", {"type": "traj", "q": [q.tolist()]}, m, {})
            r = m.apply("A", low)
            assert "rejected" in r, r
            assert m.state_digest() == d0 and rob.plan_j >= len(rob.plan)
        ok = q0.copy()
        ok[3] += 0.5 * BM.ARM_VEL[1] * BM.CONTROL_DT
        low = SPORT.action_layer("low").parse("A", {"type": "traj", "q": [ok.tolist()]}, m, {})
        assert "rejected" not in m.apply("A", low)
        m.step()
        assert np.array_equal(np.asarray(rob.s.target), ok)


def test_exploit_no_stacked_acts():
    for fmt in ("match-wallclock", "duel-wallclock"):
        m = SPORT.build(cfg(fmt, first="B"))
        if not m.single:
            serve_by(m, "B")
        run_until(m, in_flight)
        low, info, r = high_act(m, "A", ACT)
        k0 = info["effect_step"]
        assert k0 == m.k + SPORT.action_layer("high").lead and r.get("effect_step") == k0, (info, r)
        layer = SPORT.action_layer("high")
        refused(layer, "A", ACT, m, 409)
        refused(layer, "A", {"type": "none"}, m, 409)
        assert m.observe("A")["pending_commands"][0]["takes_effect_at_step"] == k0
        while m.k < k0:
            m.step()
        m.step()
        assert "pending_commands" not in m.observe("A")
        layer.parse("A", ACT, m, {})             # accepted again once it took effect


def test_exploit_no_shuttle_state_in_observation():
    allowed = set(DECL["observation"]["fields"])
    forbidden = {"sh", "shv", "sh_p", "sh_v", "shuttle", "opp", "opp_xy", "oracle", "advice", "leg", "rs", "plan"}
    for fmt in ("match-wallclock", "duel-wallclock"):
        m = SPORT.build(cfg(fmt, first="B"))
        if not m.single:
            serve_by(m, "B")
        run_until(m, in_flight)
        sh = np.asarray(m.g.s.sh_p if m.single else m.g.sh_p, float)
        for seat in m.cfg.seats:
            o = m.observe(seat)
            assert set(o) <= allowed, set(o) - allowed
            assert not set(o) & forbidden
            flat = []

            def walk(v):
                if isinstance(v, dict):
                    for x in v.values():
                        walk(x)
                elif isinstance(v, (list, tuple)):
                    if len(v) == 3 and all(isinstance(x, (int, float)) for x in v):
                        flat.append(np.asarray(v, float))
                    for x in v:
                        walk(x)
            walk(o)
            assert all(np.linalg.norm(x - sh) > 0.01 for x in flat), "the shuttle's position is in the observation"
        routes = {"start", "ready", "act", "wait", "observe", "say", "spec", "status"}
        assert not {"oracle"} & routes                 # the core's realtime routes have no privileged endpoint


class Idle:
    def __init__(self, spec, seat, seed=0):
        pass

    def act(self, view):
        return None


def test_exploit_idle_loses():
    m = SPORT.build(cfg("match-wallclock"))
    run_until(m, lambda m: m.g.done, 20000)
    assert m.g.score[0] == 0 and m.g.winner == 1
    assert all(p["winner"] == "opponent" for p in m.g.points)
    assert m.g.points[0]["how"].startswith("serve_fault:undue_delay: you did not toss the shuttle")   # you serve first
    # duel: a player who never serves (nor hits) loses to the reference, which serves (no images needed for that)
    for first in ("A", "B"):
        c = cfg("duel-wallclock", first=first)
        ps = {"A": Idle(None, "A"), "B": BP.Reference(local.seat_spec(SPORT, c, "B", plugin=PLUGIN), "B", 1)}
        r = local.play(SPORT, c, ps, render=False, plugin=PLUGIN)
        assert r["winner"] == "B", r["scores"]
        pts = r["final_fields"]["points"]
        assert all(p["how"].startswith("serve_fault:undue_delay: A did not toss") for p in pts if p["server"] == "A")
        # A never hits: the only points it can get are B's own service faults (the reference's serves are legal
        # about 99 % of the time, so with the seed's scatter stream B may fault one)
        assert all(p["winner"] == "B" or p["how"].startswith("serve_fault:") for p in pts if p["server"] == "B"), pts
        assert r["scores"]["A"] == sum(p["winner"] == "A" for p in pts) <= 1, r["scores"]


def test_exploit_never_serving_faults_every_serve():
    """Undue delay (no toss within 3 s of the receiver being ready) is a service fault every time, never a
    disqualification (user decision 2026-10-08, as in tennis): two players who never serve fault on each of their
    serves, the serve passes to the receiver each time, and the match ends on points (15-14 at the cap) after 14 or
    more undue delays by each player. A player who never serves gives back each point it wins at its next serve."""
    m = SPORT.build(cfg("duel-wallclock", first="A"))
    run_until(m, lambda m: m.g.done, 40000)
    pts = m.g.points
    assert len(pts) == 29 and all(p["how"].startswith("serve_fault:undue_delay: ") for p in pts)
    assert all(p["winner"] == BD.OTHER[p["server"]] and not p["let"] for p in pts)
    assert [p["server"] for p in pts] == ["A", "B"] * 14 + ["A"]
    assert all("disqualif" not in p["how"] and "disqualified" not in p for p in pts)
    assert m.g.score == {"A": 14, "B": 15} and m.g.winner == "B"
    assert m.phase().state == FINISHED and m.phase().reason == "match_over"
    assert [n for n, rule in m.tiebreaks()] == ["score", "last_point", "first_receiver"]
    assert [n for n, rule in m.tiebreaks() if rule(m)][0] == "score" and dict(m.tiebreaks())["score"](m) == "B"
    # format match-wallclock: you serve at 4-0 and never toss: a fault (the opponent's point and serve), no more
    m = SPORT.build(cfg("match-wallclock"))
    m.g.score[:] = [4, 0]
    m.g._start_point(m.k)
    run_until(m, lambda m: m.g.points, 1000)
    p = m.g.points[-1]
    assert p["how"].startswith("serve_fault:undue_delay: you did not toss") and p["winner"] == "opponent"
    assert not m.g.done and list(m.g.score) == [4, 1] and "disqualified" not in p


def test_exploit_serve_rules():
    """The serve (Laws 9, 10, 16): only the server tosses, once, from the step the receiver is ready, within reach;
    the serve is a fault if it is hit above 1.15 m, not upwards, missed, tossed late; both bases stay on their
    spots until the serve; the rally winner serves next from the court of its score's parity."""
    layer = SPORT.action_layer("high")
    m = SPORT.build(cfg("duel-wallclock", first="A"))
    rel = (m.serve_state("A")["base"] + np.array([0.5, 0.5])).tolist() + [1.5]
    refused(layer, "A", {"type": "toss", "pos": rel}, m, 409)          # the receiver is not ready yet
    refused(layer, "B", {"type": "toss", "pos": rel}, m, 409)          # not B's serve
    refused(layer, "A", {"type": "toss", "pos": rel[:2] + [2.5]}, m, 400)
    run_until(m, lambda m: m.serve_state("A")["phase"] == "open")
    low = SPORT.action_layer("low").parse("A", {"type": "toss", "pos": rel}, m, {})
    assert "rejected" not in m.apply("A", low) and m.serve_state("A")["phase"] == "tossed"
    refused(layer, "A", {"type": "toss", "pos": rel}, m, 409)          # one toss per serve
    run_until(m, lambda m: m.g.points, 300)                                  # nobody hits it: the server missed
    assert m.g.points[0]["how"] == "serve_fault:missed: A missed the shuttle (Law 9.1.8)" and m.g.score == {"A": 0, "B": 1}
    assert m.g.rs["server"] == "B" and not m.g.rs["service_court_right"]   # the rally winner serves, score 1: left
    assert m.observe("B")["you_serve"] and m.observe("B")["serve"]["court"] == "left"
    # the receiver's base moving before the serve is its fault (Law 9.1.4)
    if m.phase().state == UNIT_OVER:
        m.resume()
    rob = m.robot("A")
    q = np.asarray(rob.s.target, float)
    qs = [(q + np.r_[0.0, -0.0005 * i * i, np.zeros(7)]).tolist() for i in range(1, 21)]   # 10 m/s^2 to 1.95 m/s
    assert "rejected" not in m.apply("A", SPORT.action_layer("low").parse("A", {"type": "traj", "q": qs}, m, {}))
    run_until(m, lambda m: len(m.g.points) >= 2, 300)
    assert m.g.points[1]["how"].startswith("receive_fault:moved: A's base moved") and m.g.points[1]["winner"] == "B"
    # a toss may rise (height, at most 0.5 m): it rises that far and falls back (the serve is still B's to hit)
    refused(layer, "B", {"type": "toss", "pos": [0.0, -1.5, 1.2], "height": 0.6}, m, 400)
    refused(layer, "B", {"type": "toss", "pos": [0.0, -1.5, 1.2], "height": -0.1}, m, 400)
    refused(layer, "B", {"type": "toss", "pos": [0.0, -3.0, 1.2]}, m, 400)            # beyond the toss reach
    if m.phase().state == UNIT_OVER:
        m.resume()
    run_until(m, lambda m: m.serve_state("B")["phase"] == "open")
    low = SPORT.action_layer("low").parse("B", {"type": "toss", "pos": [-0.2, -1.5, 1.2], "height": 0.3}, m, {})
    assert "rejected" not in m.apply("B", low)
    top = 0.0
    while m.g.rs["state"] == "tossed":
        m.step()
        top = max(top, m.g.sh_p[2])
    assert abs(top - 1.5) < 0.01, top
    assert m.g.points[-1]["how"].startswith("serve_fault:missed: B missed the shuttle")
    # the serve hit above 1.15 m / not upwards (the referee's check at the server's hit, World._check_hit)
    for z, vz, want in ((1.30, 5.0, "above 1.15 m"), (1.00, -5.0, "did not go upwards"), (1.00, 5.0, None)):
        m = SPORT.build(cfg("duel-wallclock", first="A"))
        w = m.g
        run_until(m, lambda m: m.serve_state("A")["phase"] == "open")
        c0 = np.r_[m.serve_state("A")["base"] + np.array([0.4, 0.4]), z]
        w.toss("A", (c0 + np.array([0.0, 0.0, 0.002 * np.sign(vz)])).tolist())   # just beyond the string plane
        n = np.array([0.0, 0.0, 1.0])
        w.prev["A"] = (c0, n, w.sh_p)
        w._check_hit("A", c0 + np.array([0.0, 0.0, vz * BM.PHYS_DT]), n)
        assert w.rs["state"] == "live" and abs(w.rs["serve"]["contact_z"] - z) < 0.003
        if want:
            assert w._pending_point and w._pending_point[0] == "A" and want in w._pending_point[1], w._pending_point
        else:
            assert w._pending_point is None and w.rs["serve_flight"]


def _racket_pose(ph, centre, normal, base):
    q, err = BM.ik(ph, np.asarray(centre, float), np.asarray(normal, float), np.asarray(base, float), BM.Q0_ARM,
                   iters=400)
    assert err < 5e-3, err
    return np.r_[base, q]


def _shot_through(ph, q9, live=True):
    """Hold the racket still at q9 and send the shuttle through the string plane (moving against the normal)."""
    ph.reset(q9)
    c, R = ph.fk(q9)
    n = R[:, 0]
    ph.sh_p = tuple(float(x) for x in c + 0.03 * n)
    ph.sh_v = tuple(float(x) for x in -3.0 * n)
    ph.sh_live, ph.hit_enabled = live, True
    for _ in range(3):
        ph.step(q9)
    return [e for e in ph.events if e["kind"] == "hit"]


def test_exploit_no_reach_over_or_double_hit():
    ph = BM.Phys()
    normal = [0.0, -0.3, 1.0]
    own = _racket_pose(ph, [0.3, -0.35, 1.9], normal, [0.0, -0.9])
    over = _racket_pose(ph, [0.3, 0.35, 1.9], normal, [0.0, -0.5])
    assert _shot_through(ph, own), "control: the racket meets the shuttle on its own half"
    assert not _shot_through(ph, over), "the racket met the shuttle over the net (opponent's half)"
    hits = _shot_through(ph, own)                # after the hit the racket no longer touches it
    c, R = ph.fk(own)
    ph.sh_p = tuple(float(x) for x in c + 0.03 * R[:, 0])
    ph.sh_v = tuple(float(x) for x in -3.0 * R[:, 0])
    for _ in range(3):
        ph.step(own)
    assert len([e for e in ph.events if e["kind"] == "hit"]) == len(hits) == 1
    # duel: only the player whose turn it is meets the shuttle
    for expect, want in (("B", 0), ("A", 1)):
        m = SPORT.build(cfg("duel-wallclock"))
        w = m.g
        sd = w.sides["A"]
        sd.s.d.qpos[sd.s.qa] = own
        sd.s.d.qvel[sd.s.va] = 0.0
        BM.mj.mj_forward(sd.s.m, sd.s.d)
        sd.s.target = own.copy()
        sd.plan, sd.plan_j = [own.copy() for _ in range(10)], 0
        c, R = sd.s.fk(own)
        w.rs.update(state="live", expect=expect, last_hitter=BD.OTHER[expect], played_k=w.k)
        w.sh_p = tuple(float(x) for x in c + 0.03 * R[:, 0])
        w.sh_v = tuple(float(x) for x in -3.0 * R[:, 0])
        w.sh_live, w.hit_enabled = True, True
        for _ in range(3):
            w.advance()
        assert len([e for e in w.events if e["kind"] == "hit" and e["side"] == "A"]) == want, (expect, w.events)


def test_exploit_spec_is_the_simulator():
    sp = DECL["spec"]
    S = BM.COURT_SCALE
    assert S == 0.4 and (BM.NET_TOP, BM.COURT_X, BM.COURT_Y, BM.POST_X, BM.SHORT_SERVICE_Y) == (
        1.524, 2.59 * S, 6.70 * S, 3.05 * S, 1.98 * S)
    for s in ("1.524", "1.036", "2.68", "1.22", "0.792", "x 0.4"):
        assert s in sp["world"], s
    assert round(BM.K_DRAG, 3) == 0.212 and BM.V_TERM == 6.8 and BM.GRAV == 9.81
    assert "K = 0.212" in sp["shuttle"] and "6.8 m/s" in sp["shuttle"] and "9.81" in sp["shuttle"]
    assert (BM.R_HEAD, BM.RACKET_GAIN, BM.E_SHUTTLE, BM.OFFCENTRE_LOSS) == (0.14, 4.0, 0.15, 0.3)
    assert "4 max(0, v_head.n) (1 - 0.3 (r/0.14)^2) + 0.15 max(0, -v_in.n)" in sp["racket_contact"]
    assert (BM.BASE_VMAX, BM.BASE_AMAX, BM.BASE_JMAX, BM.BASE_FORCE) == (2.0, 15.0, 375.0, 3000.0)
    assert np.allclose(BM.HOME, [0.0, -1.6]) and np.allclose(BM.BASE_LO, [-1.5, -3.53])
    assert np.allclose(BM.BASE_HI, [1.5, -0.5]) and (BM.ACT_X, BM.ACT_Y) == (2.136, 3.68)
    assert "|bx| <= 1.5, -3.53 <= by <= -0.5, up to 2 m/s and 15 m/s^2 per axis" in sp["robot"]
    assert np.allclose(BM.serve_spot(True), [0.12, -1.6]) and np.allclose(BM.serve_spot(False), [-0.12, -1.6])
    assert "x = +0.12 in the right service court or -0.12 in the left one, y = -1.6" in sp["play"]
    assert "|x| <= 2.136, -3.68 <= y <= -0.3" in sp["act"]
    lim = BM.ARM.table()
    assert lim["velocity_rad_s"] == [2.175] * 4 + [2.61] * 3 and lim["acceleration_rad_s2"] == [15, 7.5, 10, 12.5, 15, 20, 20]
    assert lim["torque_nm"] == [87] * 4 + [12] * 3 and lim["torque_rate_nm_s"] == [1000.0] * 7
    assert DECL["arm"]["body"] == "franka_panda"
    assert (BM.SERVE_MAX_Z, BM.SERVE_WINDOW_STEPS * BM.CONTROL_DT, BM.TOSS_Z, BM.TOSS_REACH, BM.TOSS_H_MAX,
            BM.BASE_TOL) == (1.15, 3.0, (0.5, 1.8), 1.0, 0.5, 0.05)
    assert not hasattr(BM, "MAX_DELAY_FAULTS")
    for txt in ("below 1.15 m", "has 3 s to toss", "0.5 <= Z <= 1.8, within 1.0 m horizontally of your base",
                "within 5 cm of their spots", "every time, never a disqualification"):
        assert txt in sp["play"], txt
    assert (BM.WIN_POINTS, BM.CAP_POINTS, BM.MAX_HITS, BM.MAX_POINTS) == (11, 15, 400, 40)
    assert (BM.SINGLE_WIN_POINTS, BM.SINGLE_CAP_POINTS) == (11, 15)
    assert "400 hits" in sp["match"] and "40 points" in sp["match"] and "first to 11" in sp["match"] \
        and "at most 15" in sp["match"]
    sc = BM.Match.RACKET_SCATTER
    assert (sc["pos"], sc["ang_deg"], sc["speed"]) == (0.006, 0.8, 0.03) and "6 mm, 0.8 degrees and 3 %" in sp["act"]
    assert BM.READY_STEPS == BD.READY_STEPS == 100 and "1 s after play resumes" in sp["play"]
    for fmt in ("match-wallclock", "duel-wallclock"):
        c = cfg(fmt)
        m = SPORT.build(c)
        s = SPORT.spec(c, "A", m)
        assert s["shuttle"]["K_per_m"] == round(BM.K_DRAG, 4)
        assert s["match"]["win_points"] == 11
        assert s["act"]["lead_steps"] == 2 and np.allclose(s["base"]["home_xy"], [0.0, -1.6])
        assert s["court"]["half_length_m"] == BM.COURT_Y and s["racket"]["gain"] == 4.0
        assert s["arm_limits"]["acceleration_rad_s2"] == lim["acceleration_rad_s2"]
        for cam in s["cameras"]:                 # pinhole intrinsics: 45 degree vertical field of view, 360 px high
            assert abs(cam["focal_px"] - 180.0 / math.tan(math.radians(22.5))) < 0.1
    # the scene draws the simulator's court and gantry (sim/scene/mk_court.py)
    ph = BM.Phys()
    gid = lambda n: BM.mj.mj_name2id(ph.m, BM.mj.mjtObj.mjOBJ_GEOM, n)
    assert abs(ph.m.geom_pos[gid("side_sR")][0] - BM.COURT_X) < 1e-6
    assert abs(ph.m.geom_pos[gid("base_line1")][1] - BM.COURT_Y) < 1e-6
    assert abs(ph.m.geom_pos[gid("short_service1")][1] - BM.SHORT_SERVICE_Y) < 1e-6
    assert abs(ph.m.geom_pos[gid("post_R")][0] - BM.POST_X) < 1e-6
    assert abs(ph.m.geom_pos[gid("net_tape")][2] + ph.m.geom_size[gid("net_tape")][2] - BM.NET_TOP) < 1e-6
    assert np.allclose(ph.lo[:2], BM.BASE_LO - 0.05) and np.allclose(ph.hi[:2], BM.BASE_HI + 0.05)
    assert np.allclose(ph.m.actuator_ctrlrange[:2], [[-BM.BASE_FORCE, BM.BASE_FORCE]] * 2)


def test_exploit_real_robot_limits():
    """The arm is held to the Franka Panda's limits whatever it is commanded (the 1 kHz setpoint generator and the
    torque stage), and a low-level traj that would drive the gantry beyond its speed or acceleration is refused."""
    m = SPORT.build(cfg("duel-wallclock", "low"))
    rob = m.robot("A")
    q0 = np.asarray(rob.s.target, float)
    d0 = m.state_digest()
    rows = []                                     # the base: from rest to 0.8 m/s in one step (80 m/s^2)
    for i in range(1, 6):
        rows.append((q0 + np.r_[0.008 * i, 0.0, np.zeros(7)]).tolist())
    r = m.apply("A", SPORT.action_layer("low").parse("A", {"type": "traj", "q": rows}, m, {}))
    assert "rejected" in r and "gantry" in r["rejected"] and m.state_digest() == d0, r
    rows = [(q0 + np.r_[0.0004 * i * i, 0.0, np.zeros(7)]).tolist() for i in range(1, 6)]   # 8 m/s^2: accepted
    assert "rejected" not in m.apply("A", SPORT.action_layer("low").parse("A", {"type": "traj", "q": rows}, m, {}))
    # the arm: a jump of every joint at its speed limit each step (accelerations far beyond the limits)
    m = SPORT.build(cfg("duel-wallclock", "low"))
    rob = m.robot("A")
    q = np.asarray(rob.s.target, float)
    rows, sgn = [], 1.0
    for i in range(60):
        sgn = -sgn if i % 6 == 0 else sgn
        q = q + np.r_[0.0, 0.0, sgn * BM.ARM_VEL * BM.CONTROL_DT * 0.99]
        q = np.clip(q, rob.s.lo + 0.05, rob.s.hi - 0.05)
        rows.append(q.tolist())
    assert "rejected" not in m.apply("A", SPORT.action_layer("low").parse("A", {"type": "traj", "q": rows}, m, {}))
    ph = rob.s
    qs = []
    for _ in range(60):
        m.step()
        qs.append(ph.jl.q.copy())                 # the setpoint the servo tracks, every 10 ms
        assert np.all(np.abs(ph.jl.qd) <= BM.ARM.vel + 1e-9) and np.all(np.abs(ph.jl.qdd) <= BM.ARM.acc + 1e-9)
        assert np.all(np.abs(ph.trl.tau) <= BM.ARM.tau + 1e-9)
        # the joint-velocity rule (revision wallclock-12 / duel-14): the setpoints at 0.9 of the velocity limits
        assert np.all(np.abs(ph.jl.qd) <= BM.ARM.vel * BM.VMARGIN + 1e-9)
    # the velocity reflex: a physical arm joint faster than 1.05 x its limit (e.g. dragged by the base) stops the robot
    # at that pose (its targets ignored; the gantry brakes within its limits) for 1 s, then it follows its targets again
    assert (BM.VMARGIN, BM.REFLEX, BM.REFLEX_RECOVER_S) == (0.9, 1.05, 1.0)
    ph = BM.Phys()
    ph.reset(np.r_[BM.HOME, BM.Q0_ARM])
    t, v = ph.target.copy(), 0.0
    for _ in range(30):                           # the base moving at 1.2 m/s (targets within the gantry's limits)
        v = min(v + 0.1, 1.2); t = t.copy(); t[0] += v * BM.CONTROL_DT; ph.step(t)
    ph.d.qvel[ph.va[3]] = 1.2 * BM.ARM_VEL[1]    # joint 2 driven past its limit
    away = t.copy(); away[0] += 0.012
    ph.step(away)
    assert ph.reflex_at is not None and ph.n_reflex == 1
    pose = ph.reflex_at[1].copy()
    bv = []
    for _ in range(95):                           # targets the stopped robot ignores
        away = away.copy(); away[2] = min(away[2] + 0.01, ph.hi[2] - 0.1); away[0] = min(away[0] + 0.012, 1.4)
        ph.step(away)
        bv.append(ph.d.qvel[ph.va[0]])
    assert ph.reflex_at is not None and np.max(np.abs(ph.d.qpos[ph.qa[2:]] - pose[2:])) < 0.05
    assert abs(bv[-1]) < 1e-3 and np.max(np.abs(np.diff(bv))) / BM.CONTROL_DT < 1.2 * BM.BASE_AMAX   # braked, within limits
    for _ in range(10):
        ph.step(away)
    assert ph.reflex_at is None                   # recovered after 1 s
    for _ in range(150):
        ph.step(away)
    assert np.max(np.abs(ph.d.qpos[ph.qa] - away)) < 1e-3 and ph.n_reflex == 1     # caught up with its targets


def test_exploit_self_collision():
    """The arm and racket cannot pass through the robot's own pedestal, platform and links (sport revision duel-11):
    joint targets that would fold the forearm 8 cm deep into the 1.0 m pedestal leave the arm stopped at its surface,
    and the contact is recorded (bodies/selfcollision.Monitor, the final fields' self_collision). The stroke
    controller keeps clear of the stand: for a low forehand whose nearest IK solution folds the forearm into the
    pedestal it takes a clear one, and the swing it plays has no self-collision contact."""
    m = SPORT.build(cfg("duel-wallclock", "low"))
    rob = m.robot("A")
    mon = rob.s.selfcol["A"]
    q0 = np.asarray(rob.s.target, float)
    bad = np.r_[q0[:2], 1.77, 1.76, -0.34, -2.24, -0.39, 2.82, -0.55]      # forearm in the pedestal (IK: 7.9 cm deep)
    assert rob.s.clearance(bad) < -0.05
    n = int(np.ceil(np.max(np.abs(bad - q0)[2:] / (BM.ARM_VEL * BM.CONTROL_DT * 0.9))))
    rows = [(q0 + (bad - q0) * min(1.0, (i + 1) / n)).tolist() for i in range(n + 60)]
    assert "rejected" not in m.apply("A", SPORT.action_layer("low").parse("A", {"type": "traj", "q": rows}, m, {}))
    for _ in range(n + 60):
        m.step()
    s = mon.summary()
    assert s["events"] >= 1 and s["warnings"] >= 1 and s["max_depth_m"] < 0.01, s
    assert any("pedestal" in w["geoms"] for w in s["worst_pairs"]), s
    assert m.final_fields()["self_collision"]["A"]["events"] == s["events"]
    # the stroke controller: a low forehand (contact 0.6 m high at the controller's stance, face 55 deg up). Of the
    # IK solutions from the controller's starts the nearest (in move time from the ready pose) has the forearm in the
    # pedestal; ik_multi takes a clear one
    m = SPORT.build(cfg("duel-wallclock", "high"))
    rob = m.robot("A")
    ph = rob.s
    base = np.asarray(ph.target[:2], float)
    nrm = np.array([0.0, math.cos(math.radians(55)), math.sin(math.radians(55))])
    pos = np.r_[base + BM.stance_offset(0.6), 0.6]
    sols = [BM.ik(ph, pos, sg * nrm, base, np.clip(st, ph.lo[2:] + 1e-3, ph.hi[2:] - 1e-3))
            for st in [BM.Q0_ARM] + [BM.Q0_ARM + d for d in BM.IK_SEED_OFFSETS] for sg in (1.0, -1.0)]
    near = min((q for q, res in sols if res < BM.IK_OK), key=lambda q: BM.move_time(q - BM.Q0_ARM))
    assert ph.clearance(np.r_[base, near]) < -0.05
    q, res, _, _ = BM.ik_multi(ph, pos, nrm, base, [BM.Q0_ARM], BM.Q0_ARM)
    assert res < BM.IK_OK and ph.clearance(np.r_[base, q]) >= BM.CLEAR_MIN
    # and the stroke itself (1 s ahead, through the high level) is played without self-collision
    low, info, r = high_act(m, "A", {"type": "act", "t": 1.0, "pos": pos.tolist(), "normal": nrm.tolist(),
                                     "vel": (1.5 * nrm).tolist()})
    assert "self-collision" not in info["stroke"]["stroke"], info
    for _ in range(len(low["q"]) + 150):
        m.step()
    s = rob.s.selfcol["A"].summary()
    assert s["events"] == 0, s


def _legacy_reference(c, rows, k_req, lead, body, n_after):
    """The legacy semantics: the act queued at k_req is planned and applied inline at k_req + lead (wc.apply_inline /
    bduel.apply_inline, what the legacy replay does), then the world runs on. rows: the commands before k_req."""
    m = SPORT.build(c)
    at = {}
    for r in rows:
        if r["ev"] == "cmd":
            at.setdefault(r["k"], []).append(r)
    while m.k < k_req:
        for r in at.pop(m.k, []):
            m.apply(r["seat"], copy.deepcopy(r["cmd"]))
        if m.phase().state == UNIT_OVER:
            m.resume()
        m.step()
    while m.k < k_req + lead:
        m.step()
    dec = {k: body[k] for k in ("type", "t", "pos", "normal", "vel")}
    if m.single:
        WC.apply_inline(m.g, dec)
    else:
        m._ctx()
        BD.apply_inline(m.g.sides["A"], dec)
    for _ in range(n_after):
        if m.phase().state != "playing":
            break
        m.step()
    return m


def test_exploit_lowered_act_replays():
    for fmt in ("match-wallclock", "duel-wallclock"):
        c = cfg(fmt, first="B")
        m = SPORT.build(c)
        rows = [{"ev": "header", "cfg": c.to_json()}]
        if not m.single:
            serve_by(m, "B", rows)
        for _ in range(3000):
            if in_flight(m, 25):
                break
            if m.phase().state == UNIT_OVER:
                rows.append({"ev": "resume", "k": m.k})
                m.resume()
            m.step()
        k_req = m.k
        low, info, r = high_act(m, "A", ACT)
        assert low["type"] == "traj" and len(low["q"]) > 20 and low["k0"] == k_req + 2, (low.get("k0"), info)
        before = list(rows)
        rows.append({"ev": "cmd", "k": m.k, "seat": "A", "cmd": copy.deepcopy(low)})
        n = 150
        for _ in range(n + 2):
            if m.phase().state != "playing":
                break
            m.step()
        rows.append({"ev": "end", "k": m.k, "reason": "test"})
        ref = _legacy_reference(c, before, k_req, 2, ACT, m.k - k_req - 2)
        assert ref.k == m.k and ref.state_digest() == m.state_digest(), "the lowered act differs from the legacy plan"
        rob, rr = m.robot("A"), ref.robot("A")
        assert np.array_equal(np.asarray(rob.s.actions()), np.asarray(rr.s.actions()))
        got = replay(SPORT, rows)
        assert got["state_sha"] == m.state_digest() and got["k"] == m.k


def _racket_dev(side, n):
    """The execution scatter (position offset, m) that `side` (a bm.Match / bduel.Side) draws for its n-th act."""
    rng = BM.scatter_rng(side.lv["seed"], 13, side.lv.get("stream", 1), n)
    return tuple(float(x) for x in rng.normal(0.0, BM.Match.RACKET_SCATTER["pos"], 3))


def test_exploit_independent_scatter_streams():
    """Each robot draws its own scatter stream: in a duel the first server's (1) and the other's (2), so the n-th act
    of the two robots scatter independently; the other first order on the same seed swaps the streams. In the match
    format the player's stream starts with bm.SCATTER_TAG, which no draw of the scripted opponent does."""
    devs = {}
    for first in ("A", "B"):
        m = SPORT.build(cfg(first=first, seed=7))
        sides = m.g.sides
        assert sides[first].lv["stream"] == 1 and sides[{"A": "B", "B": "A"}[first]].lv["stream"] == 2
        devs[first] = {s: [_racket_dev(sides[s], n) for n in range(6)] for s in ("A", "B")}
    assert all(x != y for x, y in zip(devs["A"]["A"], devs["A"]["B"]))      # independent within a match
    assert devs["A"]["A"] == devs["B"]["B"] and devs["A"]["B"] == devs["B"]["A"]   # the pair swaps the streams
    ws = [SPORT.build(cfg(first="A", seed=s)).g.sides for s in (1, 2, 3, 4)]
    a = np.array([_racket_dev(w["A"], n)[0] for w in ws for n in range(50)])
    b = np.array([_racket_dev(w["B"], n)[0] for w in ws for n in range(50)])
    assert abs(float(np.corrcoef(a, b)[0, 1])) < 0.2
    single = SPORT.build(cfg("match-wallclock", seed=13))
    assert single.g.lv.get("stream", 1) == 1
    key = np.random.SeedSequence([single.g.lv["seed"], BM.SCATTER_TAG, 13, 1, 0]).generate_state(4).tolist()
    others = [np.random.SeedSequence([single.g.lv["seed"], p, l]).generate_state(4).tolist()
              for p in range(41) for l in range(301)]                          # the opponent's return choices
    others += [np.random.SeedSequence([single.g.lv["seed"], 11, n]).generate_state(4).tolist() for n in range(41)]
    assert key not in others


def test_b9_serves_returnable_by_an_informed_receiver():
    """Robot and mode rule 2 (B9), a regression test of the feasibility study (docs/benchmarks/
    badminton-minicourt-20261008.md, oracle.py; re-run in docs/screening/badminton.md: 80 of 81 legal grid serves
    returned with an instant reaction): the reference's long and short serves (seeds 1, 2) are each returned, under the
    real limits and with self-collision, by an informed receiver: the true shuttle state 0.2 s after the serve, a
    search of at most 12 strokes (forehand stances the base reaches in time, the documented contact law, aims deep and
    short in the server's court), each planned by the platform's stroke controller (with its scatter) on a copy."""
    lead = SPORT.action_layer("high").lead
    aims = [(-0.76, 2.36), (0.76, 2.36), (0.0, 1.8), (0.0, 2.48), (-0.76, 0.92), (0.76, 0.92)]
    for seed in (1, 2):
        for kind in ("long", "short"):
            m = SPORT.build(cfg("duel-wallclock", seed=seed, first="A"))
            serve_by(m, "A", kind=kind)
            k_hit = m.g.rs["played_k"]
            while m.k < k_hit + 20:
                m.step()
            g = m.g
            p_b, v_b = BD.to_side("B", np.array(g.sh_p)), BD.to_side("B", np.array(g.sh_v))
            P, V = BM.predict(p_b, v_b, n_max=8000)
            b0 = g.sides["B"].s.d.qpos[g.sides["B"].s.qa[:2]].copy()
            cands = []
            for j in range(lead + 30, len(P) // 10, 4):
                p, v = P[10 * j - 1], V[10 * j - 1]
                if not (0.65 <= p[2] <= 2.2 and p[1] <= -0.45):
                    continue
                if (j - lead) * BM.CONTROL_DT - BM.base_time(float(np.linalg.norm(BM.stance(p) - b0))) < 0.0:
                    continue
                for aim in aims:
                    for th, sp, v_out in BP.solve_launch(p, np.array(aim), (45, 35, 55, 25), vmax=4 * 2.6 + 2):
                        n, vel = BP.racket_state(v, v_out)
                        if np.linalg.norm(vel) <= 2.6:
                            cands.append((j, p, n, vel))
                            break
            assert cands, (seed, kind, "no stroke candidate")
            ok = False
            for j, p, n, vel in cands[::max(1, len(cands) // 12)][:12]:
                m2 = copy.deepcopy(m)
                k0 = m2.k
                high_act(m2, "B", {"type": "act", "t": round((j - lead) * BM.CONTROL_DT, 3), "pos": p.tolist(),
                                   "normal": n.tolist(), "vel": vel.tolist()})
                hit = None
                while m2.k < k0 + j + 25 and not m2.g.points and hit is None:
                    m2.step()
                    hit = next((e for e in m2.g.events if e["kind"] == "hit" and e.get("side") == "B"), None)
                if hit is None:
                    continue
                L, _ = BM.predict(np.array(hit["pos"]), np.array(hit["v_out"]), n_max=8000)
                if L[-1][2] <= 1e-9 and BM.in_court(L[-1][0], L[-1][1], -1):
                    ok = True
                    break
            assert ok, (seed, kind, "no stroke returned the serve")


def test_skill_contract_is_the_simulator():
    """skill_api.py (the high level's contract, P6) is the act and toss schemas' single source and its ranges are the
    simulator's; the contract's request check accepts the act the tests send and refuses one out of range."""
    import skill_api as SA
    PL = __import__("plugin")
    assert (SA.ACT_X, SA.ACT_Y, SA.T_MAX_S) == (BM.ACT_X, BM.ACT_Y, PL.T_MAX_S)
    assert (SA.TOSS_Z, SA.TOSS_REACH, SA.TOSS_H_MAX) == (BM.TOSS_Z, BM.TOSS_REACH, BM.TOSS_H_MAX)
    assert PL.HIGH_SCHEMA["act"] == SA.ACT.request_schema()["act"]
    assert PL.HIGH_SCHEMA["toss"] == SA.TOSS.request_schema()["toss"]
    req = {**{k: v for k, v in ACT.items() if k != "type"}, "base": [0.1, -2.0]}   # (the contract lists base too)
    assert SA.ACT.valid_request(req) is None
    assert SA.ACT.valid_request({**req, "vel": [0.0, 9.0, 0.0]}) is not None
    assert set(DECL["modes"]["high"]["skills"]) == set(SA.CONTRACTS)
    for z in (0.5, 0.9, 1.333, 1.9, 2.6):                 # the published stance formula is the controller's
        p = np.array([0.3, -1.2, z])
        r = max(0.30, 0.95 - 0.35 * abs(z - 1.333))
        assert np.allclose(BM.stance(p), np.clip(p[:2] - 0.62 * r, BM.BASE_LO, BM.BASE_HI)), z


def test_opponent_levels_are_placement_sensitive():
    """Revision wallclock-10 (docs/screening/badminton.md): the scripted opponent's levels are ordered (level 3's reach
    >= level 2's, level 2's attack >= level 1's), and the same shot is harder for it far from where it stands: a
    shuttle that lands near the opponent is returned, the same flight to the far corner of a robot standing in the
    other corner is not (its base needs time to get within reach, then its arm `swing` s)."""
    L = BM.LEVELS
    assert sorted(L) == [1, 2, 3] == DECL["formats"]["match-wallclock"]["levels"]
    assert L[3]["reach"] >= L[2]["reach"] >= L[1]["reach"] and L[2]["pace"] >= L[1]["pace"]
    assert all(lv["min_t"] == 0.0 and lv["swing"] > 0 for lv in L.values())
    m = BM.Match(1, lv_override={"seed": 1})
    p, v = np.array([0.3, -0.8, 1.5]), np.array([0.7, 4.5, 4.5])       # a lofted shot to the right of the far court
    Pl, _ = BM.predict(p, v)
    land = Pl[-1]
    assert land[2] <= 1e-9 and BM.in_court(land[0], land[1], +1), land
    res = {}
    for name, opp in (("near", np.array([land[0], land[1] - 0.2])), ("far", np.array([-0.9, 0.9]))):
        m.leg = {"kind": "return", "state": "incoming"}
        m.s.sh_p, m.s.sh_v = tuple(p), tuple(v)
        m.opp = {"xy0": opp, "k0": m.k, "txy": opp.copy(), "speed": 0.0}
        m._opp_plan()
        res[name] = (m.leg.get("opp_leave", False), m.leg.get("opp_pressure"))
    assert res["near"][0] is False and res["near"][1] is not None and res["near"][1] <= 1.0, res
    assert res["far"][0] or res["far"][1] > res["near"][1], res


def test_opponent_reads_habits_blocks_and_lobs():
    """Revision wallclock-11 (docs/screening/badminton.md): the scripted opponent reads a habit of your placement
    (alternating or repeating sides, all short or all deep) and commits to the next shot's spot; a shuttle it meets
    too late for a full stroke but with `block` s of arm time is blocked (a weak return), not missed; its clears pass
    above any racket on your side near the net when they can (so standing at the net is not a free interception)."""
    alt = [[0.8, 1.0], [-0.8, 1.0], [0.75, 1.1], [-0.8, 0.9], [0.8, 1.0]]
    rp = BM.read_pattern(alt)
    assert rp is not None and rp[0] < -0.5 and rp[1] < BM.READ_SHORT_Y, rp           # the other front corner next
    rp = BM.read_pattern([[0.7, 2.0], [0.6, 2.1], [0.7, 2.0], [0.65, 2.2], [0.7, 2.1]])
    assert rp is not None and rp[0] > 0.5 and rp[1] > BM.READ_SHORT_Y, rp           # the same deep corner again
    assert BM.read_pattern([[0.8, 1.0], [0.7, 2.0], [-0.8, 1.0], [-0.7, 2.2], [0.8, 1.0], [0.8, 2.1]]) is None
    assert BM.read_pattern(alt[:3]) is None                                           # too few shots to read
    p0 = np.array([0.2, 1.5, 1.0])
    over = [BM.over_the_net_player(p0, c[2]) for c in BM.solve_shots(p0, (0.5, -2.36), BM.SHOT_THETAS["clear"], 18.0)]
    assert any(over) and not all(over), over
    p, v = np.array([0.3, -0.2, 1.9]), np.array([0.4, 2.8, -1.5])                   # a quick push, about 0.54 s
    land = BM.predict(p, v)[0][-1]
    res = {}
    for b in (0.0, 0.45):
        m = BM.Match(1, lv_override={"seed": 1, "block": b, "reach": 0.7})
        o = np.array([land[0], land[1] - 0.1])                                          # it stands at the landing
        m.leg = {"kind": "return", "state": "incoming"}
        m.s.sh_p, m.s.sh_v = tuple(p), tuple(v)
        m.opp = {"xy0": o, "k0": m.k, "txy": o.copy(), "speed": 0.0}
        m._opp_plan()
        res[b] = m.leg.get("opp_pressure")
    assert res[0.0] > 1.0 and BM.PRESSURE_WEAK < res[0.45] <= 1.0, res               # missed / blocked


def test_exploit_flat_short_serve():
    """Revision wallclock-11 (screening, 10-12): a fast flat short serve to the front corner of the receiver's service
    court (a model scored 10 points in a row with one at about 5.6 m/s on the old opponent) is returned at every
    level: the receiver stands in its service court and blocks or returns it, it is not a winner."""
    import builtin_players as BPL
    nm = BPL.NET_MARGIN
    BPL.NET_MARGIN = 0.12
    try:
        for lv in sorted(BM.LEVELS):
            for right in (True, False):
                base = BM.serve_spot(right)
                pc = np.r_[base + 0.39, 1.05]                                     # the forehand serve contact
                tgt = np.array([(-0.95 if right else 0.95), 1.17])                # the receiver's front corner
                sols = BPL.solve_launch(pc, tgt, (45, 50, 55), vmax=13.4)
                assert sols, (lv, right)
                v = sols[0][2]
                assert 4.5 <= float(np.linalg.norm(v)) <= 7.0
                m = BM.Match(lv, lv_override={"seed": 1})
                n = int(round(m.lv["reaction_s"] / BM.PHYS_DT))
                P, V = BM.predict(pc, v)
                m.leg = {"kind": "our_serve", "state": "incoming", "service_court_right": right}
                m.s.sh_p, m.s.sh_v = tuple(P[n]), tuple(V[n])
                o = np.array([(-0.45 if right else 0.45) * BM.COURT_SCALE, BM.OPP_HOME[1]])
                m.opp = {"xy0": o, "k0": m.k, "txy": o.copy(), "speed": 0.0}
                m._opp_plan()
                assert not m.leg.get("opp_leave") and m.leg.get("opp_pressure") is not None, (lv, right, m.leg)
                assert m.leg["opp_pressure"] <= 1.0, (lv, right, m.leg["opp_pressure"], m.leg.get("opp_gap"))
    finally:
        BPL.NET_MARGIN = nm


def test_base_moves_only_when_told():
    """The base study (revision wallclock-11 / duel-12): the base is the Festo EGC-185-BS-KF-40P (2 m/s, 15 m/s^2 per
    axis, a 3 m x stroke); there is no automatic return: `none` leaves the base where it is, `move` sets where it waits
    and the ready target drives the base towards it within the axes' limits; the planned stroke's base stays at its stance afterwards (home_after);
    `act` and `move` parse an optional / required base within the axes' range."""
    assert BM.BASE_NAME == "festo_egc_bs" and BM.BASE_VMAX == 2.0 and list(BM.BASE_AXIS_AMAX) == [15.0, 15.0]
    assert BM.BASE_HI[0] == 1.5 and BM.BASE_LO[0] == -1.5
    m = SPORT.build(cfg("match-wallclock"))
    g = m.g
    b0 = g.s.d.qpos[g.s.qa[:2]].copy()
    g.rt_clear()
    assert np.allclose(g.home, g.s.target[:2])
    goal = b0 + np.array([0.6, -0.4])
    assert np.allclose(g.move(goal), goal, atol=1e-3)
    q = g._ready_target()                          # the ready target now drives the base towards the move point
    assert float(np.hypot(*(q[:2] - g.s.target[:2]))) <= 2.0 * BM.CONTROL_DT * 1.001
    assert np.dot(q[:2] - g.s.target[:2], goal - g.s.target[:2]) > 0
    layer = SPORT.action_layer("high")
    assert layer.parse("A", {"type": "move", "base": [0.2, -1.8]}, m, {})["base"] == [0.2, -1.8]
    refused(layer, "A", {"type": "move", "base": [1.9, -1.8]}, m)
    a = layer.parse("A", {**ACT, "base": [0.1, -2.0]}, m, {})
    assert a["base"] == [0.1, -2.0]
    plan, info = g.plan_racket(1.0, np.array(ACT["pos"]), np.array(ACT["normal"]), np.array(ACT["vel"]),
                               base=[0.1, -2.0])
    assert info["home_after"] == [0.1, -2.0] and np.allclose(g.home, [0.1, -2.0])


# ---------------------------------------------------------------------------------------------- the low level
SPORT_DIR = Path(PLUGIN).parent


def _evaluator():
    import importlib.util
    import sys
    if "bm_lowlevel_evaluator" not in sys.modules:
        spec = importlib.util.spec_from_file_location("bm_lowlevel_evaluator", SPORT_DIR / "lowlevel" / "evaluator.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
    return sys.modules["bm_lowlevel_evaluator"]


def _seat_xml(tmp):
    """seat/robot_only.xml with its meshdir pointing at the scene's assets (the seat has them at /opt/robot/assets)."""
    import re
    xml = re.sub(r'meshdir="[^"]*"', f'meshdir="{(SPORT_DIR / "sim" / "scene" / "assets").resolve()}"',
                 (SPORT_DIR / "seat" / "robot_only.xml").read_text(), count=1)
    (Path(tmp) / "robot_only.xml").write_text(xml)
    return Path(tmp) / "robot_only.xml"


def test_exploit_toolkit_is_the_manifest():
    """E15: the low-level toolkit (seat/toolkit.toml) is what the seat gets and what /spec names: the manifest passes
    skilleval (schema, sha256 of every tool, the seat ships exactly its tools, the practice file and the manifest);
    the libraries import nothing of the simulator; the kinematics library reproduces the simulator's racket kinematics
    (9 joints) and its IK is deterministic; the servo model rolls joint targets out as the simulator's robot does
    (bit for bit from rest; within 0.2 mm of the racket head from a moving start given the observed servo state)."""
    import ast
    import hashlib
    import sys
    import tempfile
    from core.local import seat_spec
    from skilleval import declare, toolkit
    seat = SPORT_DIR / "seat"
    assert toolkit.problems(seat / "toolkit.toml", "badminton", "low") == []
    rows = declare.check("badminton", DECL, SPORT_DIR)
    assert all(ok for _, ok, _ in rows), [r for r in rows if not r[1]]
    tk = toolkit.load(seat / "toolkit.toml")
    tools = {t["name"]: t for t in tk["tools"]}
    files = DECL["seat"]["low"]["files"]
    libs = {"ik_library": "bm_kinematics.py", "servo_model": "bm_servo.py"}
    for name, fn in libs.items():
        lib = seat / "lib" / fn
        assert hashlib.sha256(lib.read_bytes()).hexdigest() == tools[name]["sha256"]
        assert files[tools[name]["file"]] == f"seat/lib/{fn}"
        tree = ast.parse(lib.read_text())
        imports = {a.name.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names} | \
            {n.module.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
        assert imports <= {"__future__", "os", "re", "math", "copy", "pathlib", "numpy", "mujoco"}, (fn, imports)
    assert sorted(p.name for p in (seat / "lib").iterdir() if p.suffix == ".py") == sorted(libs.values())
    assert hashlib.sha256((seat / "robot_only.xml").read_bytes()).hexdigest() == tools["robot_model"]["sha256"]
    assert files["/opt/robot/toolkit.toml"] == "seat/toolkit.toml"
    if str(seat / "lib") not in sys.path:
        sys.path.insert(0, str(seat / "lib"))
    import bm_kinematics
    import bm_servo
    assert bm_kinematics.VERSION == tools["ik_library"]["version"] and bm_servo.VERSION == tools["servo_model"]["version"]
    xml = _seat_xml(tempfile.mkdtemp())
    K = bm_kinematics.RacketKinematics(xml)
    ph = BM.Phys()
    ph.reset(np.r_[BM.HOME, BM.Q0_ARM])
    rng = np.random.default_rng(5)
    for _ in range(50):
        q = rng.uniform(ph.lo + 0.05, ph.hi - 0.05)
        p1, R1 = ph.fk(q)
        p2, R2 = K.fk(q)
        assert np.abs(p1 - p2).max() < 1e-12 and np.abs(R1 - R2).max() < 1e-12
        assert np.abs(ph.jac(q) - K.jacobian(q)[:, 2:]).max() < 1e-12
    for _ in range(8):                                                  # reachable targets: poses of random joints
        p0, R0 = ph.fk(np.r_[BM.HOME, np.clip(BM.Q0_ARM + rng.normal(0, 0.3, 7), ph.lo[2:] + 0.05, ph.hi[2:] - 0.05)])
        pos, n = p0, R0[:, 0]
        qa, ea = K.solve_near(pos, normal=n, q_ref=BM.Q0_ARM, base=BM.HOME, vel_limit=BM.ARM_VEL)
        qb, eb = K.solve_near(pos, normal=n, q_ref=BM.Q0_ARM, base=BM.HOME, vel_limit=BM.ARM_VEL)
        assert np.array_equal(qa, qb) and ea == eb                      # deterministic, bit for bit
        p, R = ph.fk(np.r_[BM.HOME, qa])
        assert ea < 1e-4 and np.linalg.norm(p - pos) < 1e-4 and R[:, 0] @ n > 1 - 1e-6, (ea, pos)
    # the servo model = the simulator's robot: drive the live robot, then roll the rest out from its observed state
    PL = __import__("plugin")
    sim = bm_servo.RobotSim(xml, base_force=BM.BASE_FORCE)
    for seed in range(2):
        g = BM.Match(1, lv_override={"seed": 3})
        s = g.s
        r2 = np.random.default_rng(seed)
        goal = np.clip(s.target + np.r_[r2.normal(0, 0.2, 2), r2.normal(0, 0.5, 7)], s.lo + 0.1, s.hi - 0.1)
        rows, prev = [], s.target.copy()
        for i in range(1, 91):
            q = s.target + (goal - s.target) * (0.5 - 0.5 * math.cos(math.pi * min(i, 60) / 60))
            q = prev + np.clip(q - prev, -BM.VEL_LIMIT * BM.CONTROL_DT * 0.999, BM.VEL_LIMIT * BM.CONTROL_DT * 0.999)
            rows.append(q)
            prev = q
        for q in rows[:25]:
            s.step(q)
        st = PL.low_state(g)
        sim.reset(st["q"], st["qd"], [float(x) for x in s.target], st["servo"])
        for q in rows[25:]:
            s.step(q)
            sim.step(q)
            # from a moving start the match's servo reads the inertia and bias of the previous 1 ms step (MuJoCo's
            # forward pass precedes the integration), the copy those of the observed state: < 1e-3 rad (0.1 mm)
            assert np.abs(s.d.qpos[s.qa] - sim.d.qpos[sim.qadr]).max() < 1e-3, seed
            p_sim, p_cp = s.racket_state()[0], sim.racket()[0]
            assert np.linalg.norm(p_sim - p_cp) < 2e-4, seed
    # the joint-velocity rule (revision wallclock-12 / duel-14, servo model 1.2.0): the copy's setpoints run at 0.9 of
    # the limits, and its reflex brakes the arm and the gantry as the robot's does, whether the reflex is in progress at
    # the observation (servo.reflex, servo.base_sp) or fires within the rollout; both recover after 1 s
    sim = bm_servo.RobotSim(xml, base_force=BM.BASE_FORCE, base_limits=PL.low_spec(BM.Phys())["limits"]["base"])
    assert np.array_equal(sim.jl.V, BM.AL.JerkLimiter(BM.ARM_OP, BM.PHYS_DT).V)
    assert np.array_equal(bm_servo.base_setpoints().V, BM.AL.JerkLimiter(BM.base_limits(), BM.PHYS_DT).V)
    for where in ("observed", "rollout"):
        g = BM.Match(1, lv_override={"seed": 3})
        s = g.s
        rows, prev, v = [], s.target.copy(), 0.0
        goal = np.clip(s.target + np.r_[0.0, 0.0, np.random.default_rng(7).normal(0, 0.5, 7)], s.lo + 0.1, s.hi - 0.1)
        for i in range(1, 221):                     # the base moving at up to 1.2 m/s, the arm towards a goal
            v = min(v + 0.1, 1.2) if i < 60 else max(v - 0.1, 0.0)
            q = s.target + (goal - s.target) * (0.5 - 0.5 * math.cos(math.pi * min(i, 60) / 60))
            q[0] = prev[0] + v * BM.CONTROL_DT
            q = prev + np.clip(q - prev, -BM.VEL_LIMIT * BM.CONTROL_DT * 0.999, BM.VEL_LIMIT * BM.CONTROL_DT * 0.999)
            rows.append(q)
            prev = q
        for q in rows[:25]:
            s.step(q)
        k0 = 25
        if where == "observed":
            s.d.qvel[s.va[3]] = 1.2 * BM.ARM_VEL[1]   # joint 2 driven past its limit
            s.step(rows[25])
            k0 = 26
            assert s.reflex_at is not None and s.n_reflex == 1
        st = PL.low_state(g)
        assert (st["servo"]["reflex"] is not None) == (where == "observed") and st["servo"]["n_reflex"] == s.n_reflex
        sim.reset(st["q"], st["qd"], [float(x) for x in s.target], st["servo"])
        if where == "rollout":
            s.d.qvel[s.va[3]] = sim.d.qvel[sim.vadr[3]] = 1.2 * BM.ARM_VEL[1]
        for i, q in enumerate(rows[k0:]):
            s.step(q)
            sim.step(q)
            assert np.abs(s.d.qpos[s.qa] - sim.d.qpos[sim.qadr]).max() < 1e-3, (where, i)
            assert np.linalg.norm(s.racket_state()[0] - sim.racket()[0]) < 2e-4, (where, i)
            assert (s.reflex_at is None) == (sim.reflex_at is None), (where, i)
        assert s.n_reflex == 1 and sim.reflexes == (where == "rollout") and s.reflex_at is None and s.bl is not None
        assert np.abs(sim.bl.q - s.bl.q).max() < 1e-3
    sp = local.seat_spec(SPORT, cfg("match-wallclock", "low"), "A", plugin=PLUGIN)
    assert sp["toolkit"]["ik_library"]["sha256"] == tools["ik_library"]["sha256"]
    assert sp["toolkit"]["servo_model"]["sha256"] == tools["servo_model"]["sha256"]
    assert sp["toolkit"]["version"] == tk["toolkit"]["version"]
    assert sp["skill_contract"]["act"]["skill"] == "act" and "deliverable" in sp["skill_contract"]["act"]
    hi = local.seat_spec(SPORT, cfg("match-wallclock", "high"), "A", plugin=PLUGIN)
    assert "toolkit" not in hi and "deliverable" not in hi["skill_contract"]["act"]


def test_exploit_robot_model_is_the_simulator_robot():
    """E16: the robot's own model (/spec robot_model.robot_only_xml = seat/robot_only.xml) reproduces the simulator's
    robot (every body's pose at random joints) and holds nothing of the world; the seat image pins the sim's MuJoCo."""
    import tempfile
    import xml.etree.ElementTree as ET
    import mujoco as mj
    sp = local.seat_spec(SPORT, cfg("match-wallclock", "low"), "A", plugin=PLUGIN)
    xml = sp["robot_model"]["robot_only_xml"]
    assert (SPORT_DIR / "seat" / "robot_only.xml").read_text() == xml
    rm = mj.MjModel.from_xml_path(str(_seat_xml(tempfile.mkdtemp())))
    rd, ph = mj.MjData(rm), BM.Phys()
    radr = np.array([rm.jnt_qposadr[mj.mj_name2id(rm, mj.mjtObj.mjOBJ_JOINT, j)] for j in BM.JOINTS])
    names = [mj.mj_id2name(rm, mj.mjtObj.mjOBJ_BODY, i) for i in range(1, rm.nbody)]
    ids = [mj.mj_name2id(ph.m, mj.mjtObj.mjOBJ_BODY, n) for n in names]
    rng = np.random.default_rng(11)
    for _ in range(100):
        q = rng.uniform(ph.lo, ph.hi)
        ph.kin_set(q)
        rd.qpos[radr] = q
        mj.mj_kinematics(rm, rd)
        assert np.abs(ph.kin.xpos[ids] - rd.xpos[1:]).max() < 1e-12
        assert np.abs(ph.kin.xmat[ids] - rd.xmat[1:]).max() < 1e-12
    assert (rm.njnt, rm.nu, rm.nlight, rm.ncam, rm.nmocap) == (9, 9, 0, 0, 0)
    pairs = [mj.mj_id2name(rm, mj.mjtObj.mjOBJ_PAIR, i) for i in range(rm.npair)]
    scene_pairs = {mj.mj_id2name(ph.m, mj.mjtObj.mjOBJ_PAIR, i) for i in range(ph.m.npair)}
    assert pairs and all(n.startswith("selfcol") for n in pairs) and set(pairs) <= scene_pairs, pairs
    root = ET.fromstring(xml)
    words = {el.get("name", "") for el in root.iter()} | {el.get("file", "") for el in root.iter()}
    assert not any(w in x for x in words for w in ("shuttle", "court", "net", "floor", "line", "post", "hall"))
    assert not {el.tag for el in root.iter()} & {"freejoint", "light", "camera", "texture", "material"}
    assert f"mujoco=={mj.__version__}" in DECL["seat"]["low"]["pip"]


def test_exploit_act_cases_hidden_stay_in_the_simulator():
    """E17: the low level's API test: the seat gets only the public practice requests (lowlevel/cases/public.jsonl:
    the request, the robot's state and our plan; no world state, no seed), never the hidden evaluation cases or any
    case's start state (lowlevel/cases/hidden.jsonl.gz, which also carries the replay logs, stays on the simulator side
    and out of the generated tasks); hidden and public are disjoint, also by request; every request is valid and
    feasible under the real limits (from each hidden case's start state, replayed bit for bit, the platform's own
    rehearsal without execution scatter meets the contract with a margin)."""
    import json
    import sys
    import tempfile
    from skilleval import cases as C
    from skill_api import ACT as ACT_C
    cs = DECL["modes"]["low"]["cases"]
    hidden, public = C.load(SPORT_DIR / cs["hidden"]), C.load(SPORT_DIR / cs["public"])
    assert C.problems(hidden, public, ACT_C) == []
    assert len(hidden) >= 50 and len(public) >= 15, (len(hidden), len(public))
    seat = DECL["seat"]["low"]["files"]
    assert seat[cs["practice_path"]] == cs["public"]
    assert not any(v.startswith("lowlevel") and v != cs["public"] for v in seat.values()), seat
    assert all(set(c) == {"id", "request", "state", "reference"} for c in public)
    assert not {json.dumps(c["request"], sort_keys=True) for c in hidden} & \
        {json.dumps(c["request"], sort_keys=True) for c in public}
    ev = _evaluator()
    for c in hidden:
        st, info = ev._plan(dict(c), ev.load_log(c["log"]))
        assert ev._delivers(info, c["request"]), (c["id"], info.get("stroke"))
    repo = SPORT_DIR.parents[1]
    if (repo / "tools" / "sportlib.py").is_file():                    # in the repository (not in the sim image)
        sys.path.insert(0, str(repo / "tools"))
        import sportlib
        with tempfile.TemporaryDirectory() as tmp:
            sim = sportlib.assemble_sim("badminton", Path(tmp) / "sim")
            assert not (sim / "sport" / cs["hidden"]).exists() and (sim / "sport" / cs["public"]).exists()


def test_exploit_act_skill_runs_isolated():
    """E18: the post-match evaluation hosts the player's act skill in its own process (skilleval/sandbox.py: python -I,
    a scrubbed environment, its own directory): the skill imports nothing of the simulator and sees only what a call
    passes; its prints do not reach the protocol; a call that raises gives no command (counted) and the case goes on
    without it; one that overruns its time limit is stopped (counted)."""
    import os
    import tempfile
    from skilleval import cases as C
    from skilleval.sandbox import SkillError, SkillProcess
    ev = _evaluator()
    tmp = Path(tempfile.mkdtemp())
    (tmp / "act.py").write_text(
        "import os, sys, time\n"
        "def act(t, pos, normal, vel, *, state, spec):\n"
        "    print('noise'); os.write(1, b'raw noise\\n')\n"
        "    if t > 2: raise RuntimeError('boom')\n"
        "    if t > 1: time.sleep(5)\n"
        "    return {'type': 'none', 'env': sorted(os.environ), 'spec': sorted(spec),\n"
        "            'mods': sorted(m for m in sys.modules if m in ('bm', 'wc', 'bduel', 'plugin', 'core', 'skilleval'))}\n")
    os.environ["RO_TEST_SECRET"] = "s3cr3t"
    try:
        with SkillProcess(tmp, "act", context={"spec": ev.skill_spec()}, timeouts={"act": 0.5}) as s:
            out = s.call("act", 0.5, [0.4, -1.2, 1.3], [0, 1, 0], [0, 1, 0], state={"step": 0})
            assert out["type"] == "none" and out["mods"] == [], out
            assert "RO_TEST_SECRET" not in out["env"], out["env"]
            assert out["spec"] == ["joints", "limits", "skill_contract", "toolkit"], out["spec"]
            try:
                s.call("act", 2.5, [0.4, -1.2, 1.3], [0, 1, 0], [0, 1, 0], state={})
                raise AssertionError("no SkillError")
            except SkillError as e:
                assert e.kind == "error" and s.counters()["skill_errors"] == 1
            try:
                s.call("act", 1.5, [0.4, -1.2, 1.3], [0, 1, 0], [0, 1, 0], state={})
                raise AssertionError("no time-out")
            except SkillError as e:
                assert e.kind == "timeout" and s.counters()["skill_timeouts"] == 1 and not s.alive
    finally:
        os.environ.pop("RO_TEST_SECRET")
    (tmp / "act.py").write_text("def act(t, pos, normal, vel, *, state, spec):\n    raise RuntimeError('boom')\n")
    case = C.load(SPORT_DIR / DECL["modes"]["low"]["cases"]["hidden"])[0]
    with SkillProcess(tmp, "act", context={"spec": ev.skill_spec()}, timeouts={"act": 5}) as s:
        m = ev.api_case(case, s)                                         # the case runs on without a command
        assert m["measured"] and m["refused"] == 0 and s.counters()["skill_errors"] == 1, m


def test_exploit_skill_in_the_loop():
    """E19: the skill in the loop (skilleval/inloop.py): with RCB_SKILL_INLOOP set the low level lowers `act` with the
    player's own skill (its own process, a private copy of the files, every call logged with their sha256); joint
    commands are refused (409) while a shuttle the robot may meet is in play and accepted otherwise; an invalid
    request is refused (400); a skill that gives nothing lowers to a no-op (an empty traj: nothing changes); step()
    runs every 2 control steps through the core's tick and its commands are logged as commands, so the episode
    replays without the skill."""
    import os
    import tempfile
    import types
    P = types.SimpleNamespace(**type(SPORT).action_layer.__globals__)   # the loaded plugin module's names
    tmp = Path(tempfile.mkdtemp())
    (tmp / "act.py").write_text(
        "def act(t, pos, normal, vel, *, state, spec):\n    return {'type': 'none'} if t < 1 else None\n"
        "def step(*, state, spec):\n    return None\n")
    os.environ["RCB_SKILL_INLOOP"] = str(tmp)
    try:
        P._INLOOP.clear()
        layer = SPORT.action_layer("low")
        assert "InLoopLayer" in [c.__name__ for c in type(layer).__mro__]
        m = SPORT.build(cfg("match-wallclock", "low"))
        act = dict(ACT)
        assert layer.parse("A", act, m, {})["type"] == "act"
        refused(layer, "A", dict(act, t=9.0), m)
        none = {"type": "none"}
        layer.parse("A", none, m, {})                                     # no shuttle in play yet: accepted
        run_until(m, in_flight)
        assert P._ball_in_play(m, "A")
        refused(layer, "A", none, m, 409)                                 # joint commands refused in play
        refused(layer, "A", {"type": "traj", "q": [list(m.g.s.target)]}, m, 409)
        low, info = layer.lower("A", layer.parse("A", act, m, {}), None, m)
        assert low == {"type": "none"} and info["sha256"] and info["ok"]
        low, info = layer.lower("A", layer.parse("A", dict(act, t=1.5), m, {}), None, m)
        assert low == {"type": "traj", "k0": None, "q": []} and info["skill_failed"]
        assert "rejected" not in m.apply("A", low) and m.g.plan_j <= len(m.g.plan)
        assert layer.tick("A", m) is None
        rec = layer.loops["A"].calls
        assert {c["fn"] for c in rec} >= {"act"} and all(c["sha256"] == rec[0]["sha256"] for c in rec)
    finally:
        os.environ.pop("RCB_SKILL_INLOOP")
        P._INLOOP.get("layer") and [lp.close() for lp in P._INLOOP["layer"].loops.values()]
        P._INLOOP.clear()


class _RefPhys(BM.Phys):
    """bm.Phys with the servo and step of revision wallclock-12 as written before the planner speed-up (2026-10-13):
    bodies/limits.JerkLimiter.step, limits.jerk_caps and limits.TorqueRateLimiter called as they were."""

    def servo(self, q_des, qd_des, base_acc=None):
        mj, AL = BM.mj, BM.AL
        m, d, qa, va = self.m, self.d, self.qa, self.va
        mj.mj_fullM(m, self._M, d.qM)
        M = self._M
        lo, hi = m.actuator_ctrlrange[:, 0], m.actuator_ctrlrange[:, 1]
        if self.reflex_at is None and np.any(np.abs(d.qvel[va[2:]]) > BM.REFLEX * BM.ARM_VEL):
            if self.bl is None:
                self.bl = AL.JerkLimiter(BM.base_limits(), BM.PHYS_DT)
                bl0 = self.bl
                self.bl.reset(q_des[:2], np.clip(qd_des[:2], -bl0.V, bl0.V),
                              np.zeros(2) if base_acc is None else np.clip(base_acc, -bl0.A, bl0.A))
            bl = self.bl
            stop = bl.q + bl.qd * np.abs(bl.qd) / (2.0 * bl.A) + bl.qd * (bl.A / bl.J)
            self.reflex_at = (float(d.time), d.qpos[qa].copy(), np.clip(stop, BM.BASE_LO, BM.BASE_HI))
            self.n_reflex += 1
        if self.reflex_at is not None and d.time - self.reflex_at[0] >= BM.REFLEX_RECOVER_S - 1e-9:
            self.reflex_at = None
        if self.reflex_at is not None:
            q_des = np.r_[self.reflex_at[2], self.reflex_at[1][2:]]
            qd_des = np.zeros(9)
        if self.bl is not None:
            b_sp, bd_sp, base_acc = self.bl.step(q_des[:2], qd_des[:2])
        else:
            b_sp, bd_sp = q_des[:2], qd_des[:2]
        tau_b = BM.KP[:2] * (b_sp - d.qpos[qa[:2]]) + BM.KD[:2] * (bd_sp - d.qvel[va[:2]]) + d.qfrc_bias[va[:2]]
        d.ctrl[:2] = np.clip(tau_b, lo[:2], hi[:2])
        Ma = M[np.ix_(va[2:], va[2:])]
        q_sp, qd_sp, qdd_sp = self.jl.step(q_des[2:], qd_des[2:], AL.jerk_caps(Ma, BM.ARM))
        e = q_sp - d.qpos[qa[2:]]
        tau = Ma @ qdd_sp + BM.KP[2:] * e + BM.KD[2:] * (qd_sp - d.qvel[va[2:]]) + d.qfrc_bias[va[2:]]
        if BM.COUPLING_FF and base_acc is not None:
            tau = tau + M[np.ix_(va[2:], va[:2])] @ base_acc
        d.ctrl[2:] = self.trl(tau)
        if np.max(np.abs(e)) > BM.RESYNC:
            jl = self.jl
            big = np.abs(e) > BM.RESYNC
            jl.q[big] = d.qpos[qa[2:]][big] + np.sign(e[big]) * BM.RESYNC
            jl.qd[big] = np.clip(d.qvel[va[2:]], -jl.V, jl.V)[big]
            jl.qdd[big] = 0.0

    def step(self, q):
        mj = BM.mj
        q = np.asarray(q, dtype=np.float64).reshape(-1)
        self.check_target(q)
        m, d = self.m, self.d
        q_prev = self.target
        qd_des = (q - q_prev) / BM.CONTROL_DT
        b_acc = self.base_acc(q)
        for j in range(1, BM.SUBSTEPS + 1):
            q_des = q_prev + (q - q_prev) * (j / BM.SUBSTEPS)
            self.servo(q_des, qd_des, b_acc)
            mj.mj_step(m, d)
            self.selfcol["A"].update(d)
            c = d.site_xpos[self.site].copy(); n = d.site_xmat[self.site].reshape(3, 3)[:, 0].copy()
            if self.sh_live:
                if self.hit_enabled and self._prev is not None and self.sh_p[1] < 0:
                    self._check_hit(c, n)
                self._prev = (c, n, self.sh_p)
                self._advance_shuttle()
            else:
                self._prev = (c, n, self.sh_p)
        self.target = q.copy()
        self._actions.append(q.copy())
        self.k += 1
        self.n_steady += 1
        self.steps_total += BM.SUBSTEPS


def _fast_vs_ref_rows(seed, n=160):
    """Joint targets of a fast swing (within the speed limits) with base moves; seeded."""
    rng = np.random.default_rng(seed)
    q0 = np.r_[BM.HOME, BM.Q0_ARM]
    ph = BM.Phys()
    goal = np.clip(np.r_[BM.HOME + rng.uniform(-0.6, 0.6, 2), BM.Q0_ARM + rng.uniform(-1.0, 1.0, 7)],
                   ph.lo + 0.05, ph.hi - 0.05)
    lim = BM.VEL_LIMIT * BM.CONTROL_DT * 0.99
    rows, q = [], q0.copy()
    for i in range(n):
        tgt = goal if i < n // 2 else q0
        q = q + np.clip(tgt - q, -lim, lim) * (1.0 if i % 7 else 0.3)
        rows.append(q.copy())
    return q0, rows


def test_planner_fast_paths_bit_identical():
    """The stroke planner's speed-up (2026-10-13) changes no number: (1) bm._jl_step / bm._jerk_caps against
    limits.JerkLimiter.step / limits.jerk_caps on random states (with the velocity bound's hard branch), bit for bit;
    (2) bm.Phys's servo and step against the code before the speed-up (_RefPhys): the same state after fast swings
    (state_sha, the servo state, the self-collision record), including the joint-velocity reflex; (3) a rollout
    continued from Phys.checkpoint / restore is the uninterrupted one (the rehearsals' shared start)."""
    AL = BM.AL
    c = BM._arm_const()
    rng = np.random.default_rng(7)
    ph = BM.Phys()
    ph.reset(np.r_[BM.HOME, BM.Q0_ARM])
    for i in range(3000):
        ref = AL.JerkLimiter(BM.ARM_OP, BM.PHYS_DT)
        ref.reset(rng.uniform(-2, 2, 7), rng.uniform(-1.2, 1.2, 7) * BM.ARM_VEL * (1.1 if i % 3 == 0 else 0.8),
                  rng.uniform(-1, 1, 7) * BM.ARM.acc * (2.0 if i % 5 == 0 else 0.7))
        fast = AL.JerkLimiter(BM.ARM_OP, BM.PHYS_DT)
        fast.reset(ref.q, ref.qd, ref.qdd)
        ph.kin_set(np.r_[BM.HOME, rng.uniform(ph.lo[2:], ph.hi[2:])])
        M = np.zeros((ph.m.nv, ph.m.nv))
        BM.mj.mj_fullM(ph.m, M, ph.kin.qM)
        Ma = M[np.ix_(ph.va[2:], ph.va[2:])]
        r = ref.q + rng.normal(0, [0.001, 0.05, 1.0][i % 3], 7)
        rd = rng.normal(0, 1.0, 7) * (i % 2)
        cap = AL.jerk_caps(Ma, BM.ARM)
        assert BM._jerk_caps(c, Ma).tobytes() == cap.tobytes()
        want = ref.step(r, rd, cap)
        got = BM._jl_step(c, fast, r, rd, BM._jerk_caps(c, Ma))
        assert all(a.tobytes() == b.tobytes() for a, b in zip(want, got)), i
    for seed in range(6):
        q0, rows = _fast_vs_ref_rows(seed)
        a, b = _RefPhys(), BM.Phys()
        a.reset(q0); b.reset(q0)
        b.sh_live = a.sh_live = False
        for q in rows:
            a.step(q); b.step(q)
        assert a.state_sha() == b.state_sha(), seed
        assert a.servo_state().tobytes() == b.servo_state().tobytes()
        assert a.selfcol["A"].summary() == b.selfcol["A"].summary()
        assert all(x.tobytes() == y.tobytes() for x, y in zip(a._prev[:2], b._prev[:2]))
    q0, rows = _fast_vs_ref_rows(3, 150)               # the reflex: a shoulder kicked beyond 1.05 x its speed limit
    a, b = _RefPhys(), BM.Phys()
    for ph_ in (a, b):
        ph_.reset(q0)
        ph_.d.qvel[ph_.va[2]] = 1.2 * BM.ARM_VEL[0]
        for q in rows:
            ph_.step(q)
    assert a.n_reflex >= 1 and a.state_sha() == b.state_sha() and a.n_reflex == b.n_reflex
    q0, rows = _fast_vs_ref_rows(11)
    a, b = BM.Phys(), BM.Phys()
    a.reset(q0); b.reset(q0)
    for q in rows[:40]:
        a.step(q); b.step(q)
    ck = b.checkpoint()
    for q in rows[40:90]:
        b.step(q)
    b.restore(ck)
    for q in rows[40:]:
        a.step(q); b.step(q)
    assert a.state_sha() == b.state_sha() and a.servo_state().tobytes() == b.servo_state().tobytes()
    assert a.selfcol["A"].summary() == b.selfcol["A"].summary()
