"""tennis's own tests: one per [[exploits]] entry of sport.toml (tools/check runs them through core/testkit.py, after
the generic suite; they draw no images). Run alone: python3 -m core.testkit tests sports/tennis/plugin.py (in the
sport's image)."""
import math
import tomllib
from pathlib import Path

import numpy as np

from core.episode import load_plugin, replay
from core.sport import MatchConfig, Refused

PLUGIN = str(Path(__file__).resolve().parents[1] / "plugin.py")
SPORT = load_plugin(PLUGIN)
DECL = tomllib.loads((Path(PLUGIN).parent / "sport.toml").read_text())
import duel as DU  # noqa: E402  (sim/, on sys.path once the plugin is loaded)
import tn as TN  # noqa: E402


def cfg(fmt="duel", level="high", seed=1, first="A"):
    seats = ("A",) if fmt == "match" else ("A", "B")
    return MatchConfig(sport="tennis", format=fmt, seats=seats, seed=seed, first=first if len(seats) > 1 else None,
                       action_level=level)


def run_until(m, pred, n_max=2000):
    for _ in range(n_max):
        if pred(m):
            return
        m.step()
    raise AssertionError("condition not reached")


def in_flight(m):
    if m.single:
        return m.g.leg["kind"] == "serve" and m.g.in_play() and m.g.leg["state"] == "incoming" and \
            m.k - m.g.leg["launch"] > 30
    return m.g.rs["phase"] in ("toss", "flight", "rally") and m.k - m.g.rs["played_k"] > 30


SPOT = [0.5, -12.2]                                           # a legal serve spot (right half, behind the base line)


def to_spot(m, seat="A"):
    """The server's base to SPOT, at rest, and the window open."""
    m.g.rt_move(seat, *SPOT)
    run_until(m, lambda m: m.k >= m.g.rs["open_k"] and
              np.allclose(m.g.d.qpos[m.g.qa[seat][:2]], SPOT, atol=1e-3) and
              np.hypot(*m.g.d.qvel[m.g.va[seat][:2]]) < 0.01)


def toss_now(m, seat="A", height=0.5):
    to_spot(m, seat)
    m.apply(seat, {"type": "toss", "pos": [SPOT[0] + 0.62, SPOT[1] + 0.3, 1.6], "height": height, "k0": None})
    m.step()


def refused(layer, seat, body, m, code=400):
    try:
        layer.parse(seat, body, m, {})
    except Refused as e:
        assert e.code == code, (body, e.code, str(e))
        return
    raise AssertionError(f"accepted {body}")


def test_exploit_bad_commands_are_refused():
    for level in ("high", "low"):
        m, layer = SPORT.build(cfg(level=level)), SPORT.action_layer(level)
        d0 = m.state_digest()
        bad = [{"type": "move", "base": [7.0, -11.0]}, {"type": "move", "base": [0.0, -0.2]}, {"type": "move"},
               {"type": "toss", "pos": [0.0, -12.0]}, {"type": "toss", "pos": [0.0, -12.0, float("nan")]},
               {"type": "toss", "pos": [0.0, -12.0, 1.5], "height": "1"},
               {"type": "move", "base": [0.0, float("nan")]}, {"type": "jump"}, {"type": "none", "x": 1}, [1, 2]]
        if level == "high":
            ok = {"type": "act", "t": 1.0, "pos": [1.0, -10.5, 1.0], "normal": [0, 1, 0], "vel": [0, 2, 0]}
            bad += [dict(ok, t=0.0), dict(ok, t=6.0), dict(ok, t=float("inf")), dict(ok, pos=[1.0, 0.5, 1.0]),
                    dict(ok, pos=[1.0, -10.0]), dict(ok, normal=[0, 0, 0]), dict(ok, vel=[0, 9, 0]),
                    dict(ok, base=[9.0, -11.0]), dict(ok, spin=3), dict(ok, t="1"), {"type": "traj", "q": [[0] * 9]}]
            layer.parse("A", ok, m, {})
        else:
            q = [float(x) for x in m.target("A")]
            bad += [{"type": "traj", "q": []}, {"type": "traj", "q": [q[:8]]}, {"type": "traj", "q": [q] * 601},
                    {"type": "traj", "q": [q[:8] + [float("nan")]]}, {"type": "act", "t": 1.0}]
            layer.parse("A", {"type": "traj", "q": [q]}, m, {})
        for body in bad:
            refused(layer, "A", body, m)
        assert m.state_digest() == d0 and m.k == 0


def test_exploit_joint_limits():
    m = SPORT.build(cfg(level="low"))
    layer = SPORT.action_layer("low")
    q = np.array(m.target("A"), float)
    d0, plan0 = m.state_digest(), list(m.g.sides["A"].plan)
    jump = q.copy()
    jump[2] += 0.5                                            # far beyond one step of joint 1's speed limit
    out_of_range = q.copy()
    out_of_range[0] = 99.0
    for traj in ([jump.tolist()], [q.tolist(), out_of_range.tolist()]):
        cmd = layer.parse("A", {"type": "traj", "q": traj}, m, {})
        info = m.apply("A", cmd)
        assert "rejected" in info, info
    assert m.state_digest() == d0 and m.g.sides["A"].plan == plan0
    small = q.copy()
    small[2] += 0.5 * TN.VEL_LIMIT[2] * TN.CONTROL_DT
    info = m.apply("A", layer.parse("A", {"type": "traj", "q": [small.tolist()]}, m, {}))
    assert "rejected" not in info and len(m.g.sides["A"].plan) == 1


def lowered_act(m, layer, seat, body):
    """What the core does with a high-level act: snapshot, plan (in-process here), lower; the low command."""
    cmd = layer.parse(seat, body, m, {})
    job = layer.snapshot(seat, cmd, m, 1)
    prepared = SPORT.worker_call(SPORT.worker_init("plan", seat, m.cfg), "plan", job) if job else None
    low, info = layer.lower(seat, cmd, prepared, m)
    return low, info


ACT = {"type": "act", "t": 0.8, "pos": [0.8, -10.9, 1.0], "normal": [0.0, 1.0, 0.1], "vel": [0.0, 2.0, 0.0]}


def test_exploit_no_stacked_acts():
    m = SPORT.build(cfg())
    layer = SPORT.action_layer("high")
    run_until(m, lambda m: m.k >= 120)
    low, _ = lowered_act(m, layer, "A", ACT)
    assert low["type"] == "traj" and low["k0"] == m.k + layer.lead
    assert m.apply("A", low) == {"effect_step": low["k0"]}
    for body in (ACT, {"type": "move", "base": [1.0, -11.0]}, {"type": "none"}):
        refused(layer, "A", body, m, 409)
    layer.parse("B", {"type": "none"}, m, {})                  # the other seat is not blocked
    while m.k < low["k0"]:
        m.step()
    m.step()
    layer.parse("A", {"type": "none"}, m, {})                  # after the effect the next command is accepted
    assert m.g.sides["A"].plan, "the lowered stroke took effect"


def test_exploit_no_ball_state_in_observation():
    for c in (cfg(), cfg("match")):
        m = SPORT.build(c)
        if not m.single:
            toss_now(m, "A")                                  # the toss is in the air
        run_until(m, in_flight, 3000)
        p, v, w = (m.g.s.b_p, m.g.s.b_v, m.g.s.b_w) if m.single else (m.g.b_p, m.g.b_v, m.g.b_w)
        allowed = set(DECL["observation"]["fields"])
        for seat in c.seats:
            o = m.observe(seat)
            assert set(o) <= allowed, set(o) - allowed
            sv = dict(o.get("serve") or {})
            toss = sv.pop("toss", None) or {}         # the declared toss point and height (a command, shown to both)
            declared = [float(d) for d in toss.get("pos", [])]
            text = repr({**o, "serve": sv})
            for x in (*p, *v, *w):
                if any(abs(x - d) < 1e-6 for d in declared):
                    continue                          # a tossed ball still above its declared point
                for r in (repr(x), f"{x:.3f}", f"{x:.2f}", f"{-x:.3f}"):
                    if abs(x) > 0.05:
                        assert r not in text, (seat, r)
            if not m.single:
                assert "opp_x" not in text and "b_p" not in text


def test_exploit_idle_loses():
    m = SPORT.build(cfg("match"))                             # you serve point 1, the opponent point 2
    run_until(m, lambda m: len(m.g.points) >= 2, 4000)
    p1, p2 = m.g.points[:2]
    assert p1["winner"] == "opponent" and p1["how"].startswith("serve_fault:undue_delay (double fault"), p1
    assert p2["server"] == "opponent" and p2["winner"] == "opponent" and "bounced twice" in p2["how"], p2
    m = SPORT.build(cfg(first="B"))                           # B serves point 1, A points 2 and 3
    run_until(m, lambda m: len(m.g.points) >= 2, 6000)
    p1, p2 = m.g.points[:2]
    assert p1["server"] == "B" and p1["winner"] == "A" and p1["how"].startswith("serve_fault:undue_delay"), p1
    assert p2["server"] == "A" and p2["winner"] == "B", p2


def _clip_shot(sign=+1.0):
    p0 = (0.3, -3.0 * sign, 1.1)
    for vz in np.arange(-1.5, 0.5, 0.005):
        v = (0.0, 18.0 * sign, float(vz))
        P, V, W, ev = TN.predict(p0, v, (0.0, 0.0, 0.0))
        kinds = [e[1] for e in ev]
        if kinds[:1] == ["cord"] and "bounce" in kinds:
            b = ev[kinds.index("bounce")]
            if TN.in_court(sign * b[2], sign * b[3], +1):
                return p0, v
    raise AssertionError("no net-cord launch found")


def test_exploit_net_cord():
    """scripts/olympiad/tennis/test_net_cord.py, both engines."""
    m = SPORT.build(cfg("match")).g
    p0, v = _clip_shot()
    m.leg.update(state="to_opp", hit={"speed": float(np.linalg.norm(v)), "offcentre": 0.0}, launch=m.k)
    m.s.b_p, m.s.b_v, m.s.b_w, m.s.b_live, m.s.hit_enabled = p0, v, (0.0, 0.0, 0.0), True, False
    m.s.events = []
    m._opp_plan_k = m.k + int(round(m.lv["reaction_s"] / TN.CONTROL_DT))
    seen, n_res = [], len(m.results)
    for _ in range(600):
        m.step(m._arm_target())
        seen += [e["kind"] for e in m.s.events if e["kind"] not in seen]
        if len(m.results) > n_res:
            break
    assert "cord" in seen and m.results[-1]["returned"], (seen, m.results[-1])
    t = SPORT.build(cfg())
    g = t.g
    g.rs.update(phase="rally", hitter="A", expect="B", bounces=0, played_k=g.k)
    g.sides["A"].hit_enabled, g.sides["B"].hit_enabled = False, True
    g.b_p, g.b_v, g.b_w, g.b_live = p0, v, (0.0, 0.0, 0.0), True
    g.events = []
    n0, kinds = len(g.points), []
    while len(g.points) == n0:
        kinds = [e["kind"] for e in g.events]
        t.step()
    assert kinds[:2] == ["cord", "bounce"] and g.points[-1]["winner"] == "A" and "bounce twice" in g.points[-1]["how"]


def test_exploit_spec_is_the_simulator():
    sp = DECL["spec"]
    assert (TN.WIN_POINTS, TN.WIN_BY, TN.CAP_POINTS, TN.MAX_HITS, TN.MAX_POINTS) == (5, 2, None, 400, 40)
    assert not TN.match_won(6, 5) and TN.match_won(7, 5) and not TN.match_won(9, 8) and TN.match_won(5, 3)   # no cap
    assert (DU.WIN_POINTS, DU.MAX_HITS, DU.MAX_POINTS) == (TN.WIN_POINTS, TN.MAX_HITS, TN.MAX_POINTS)
    assert "first to 5" in sp["match"] and "no cap" in sp["match"] and "400 racket hits or 40 points" in sp["match"]
    assert f"{TN.K_DRAG:.4f}" in sp["ball"] and f"{TN.K_LIFT:.4f}" in sp["ball"] and str(TN.BALL_R) in sp["ball"]
    assert f"restitution {TN.E_COURT}" in sp["court_bounce"] and f"coefficient {TN.MU_COURT}" in sp["court_bounce"]
    assert f"{TN.ALPHA} m r^2" in sp["court_bounce"]
    assert f"restitution {TN.E_NET}" in sp["net"] and f"friction {TN.MU_NET}" in sp["net"]
    assert f"restitution {TN.E_RACKET}" in sp["racket_contact"] and f"{int(TN.RACKET_GAIN)} times" in sp["racket_contact"]
    assert f"{int(TN.OFFCENTRE_LOSS * 100)} %" in sp["racket_contact"]
    assert f"{TN.COURT_X}" in sp["world"] and f"{TN.COURT_Y}" in sp["world"] and f"{TN.NET_C}" in sp["world"]
    assert f"({TN.HOME[0]:g}, {TN.HOME[1]})" in sp["robot"] and f"{TN.BASE_VMAX:g} m/s" in sp["robot"]
    assert f"-15.4 <= by <= {TN.BASE_HI[1]:g}" in sp["robot"]
    assert f"{TN.SERVE_OPEN_S} s after the point starts and closes {TN.SERVE_CLOCK_S:g} s later" in sp["serve"]
    assert f"within {TN.TOSS_REACH} m horizontally" in sp["serve"] and f"0 <= H <= {TN.TOSS_H_MAX}" in sp["serve"]
    assert f"{TN.TOSS_Z[0]} <= Z <= {TN.TOSS_Z[1]}" in sp["serve"] and f"more than {TN.BASE_STILL_M:.2f} m" in sp["serve"]
    assert f"< {TN.BASE_REST_V} m/s" in sp["serve"] and f"{TN.SECOND_CLOCK_S:g} s for a second serve" in \
        " ".join(d["text"] for d in DECL["deviations"])
    for c in (cfg(), cfg("match")):
        m = SPORT.build(c)
        s = SPORT.spec(c, c.seats[0], m)
        assert s["act"]["lead_steps"] == max(1, math.ceil(0.5 / (TN.CONTROL_DT * 20) - 1e-9)) == 3
        assert s["match"]["win_points"] == TN.WIN_POINTS and len(s["cameras"]) == 2
        assert s["cameras"][0]["focal_px"] == round(240 / math.tan(math.radians(TN.Match.FOVY) / 2), 1)


def test_exploit_lowered_act_replays():
    c = cfg()
    m = SPORT.build(c)
    layer = SPORT.action_layer("high")
    rows = [{"ev": "header", "cfg": c.to_json()}]
    run_until(m, lambda m: m.k >= 130)
    low, _ = lowered_act(m, layer, "A", ACT)
    rows.append({"ev": "cmd", "k": m.k, "seat": "A", "cmd": low})
    m.apply("A", low)
    rows.append({"ev": "cmd", "k": m.k, "seat": "B", "cmd": {"type": "move", "base": [1.0, -11.0]}})
    m.apply("B", rows[-1]["cmd"])
    for _ in range(80):
        m.step()
    rows.append({"ev": "end", "k": m.k, "reason": "test"})
    r = replay(SPORT, rows)
    assert r["state_sha"] == m.state_digest() and r["k"] == m.k and m.g.sides["A"].plan


def test_exploit_serve_rules():
    m, layer = SPORT.build(cfg()), SPORT.action_layer("high")
    g = m.g
    good = {"type": "toss", "pos": [SPOT[0] + 0.62, SPOT[1] + 0.3, 1.6], "height": 0.5}
    refused(layer, "A", good, m, 409)                         # the window is not open yet
    refused(layer, "B", good, m, 409)                         # not B's serve
    run_until(m, lambda m: m.k >= g.rs["open_k"] + 5)
    refused(layer, "A", good, m, 400)                         # the base at home: in front of the base line
    g.rt_move("A", -0.5, -12.2)                               # behind the base line but in the left half
    run_until(m, lambda m: m.k >= g.rs["open_k"] + 150)
    refused(layer, "A", dict(good, pos=[0.1, -11.9, 1.6]), m, 400)
    to_spot(m, "A")
    for bad in (dict(good, height=1.5), dict(good, pos=[SPOT[0] + 2.0, SPOT[1], 1.6]), dict(good, pos=[1.1, -11.9, 3.0])):
        refused(layer, "A", bad, m, 400)
    cmd = layer.parse("A", good, m, {})
    assert cmd["k0"] == m.k + layer.lead
    m.apply("A", cmd)
    refused(layer, "A", good, m, 409)                         # one toss per serve (pending)
    run_until(m, lambda m: g.rs["phase"] == "toss")
    tz = g.rs["toss"]                                         # released at the declared point, straight up, no spin
    assert np.allclose(tz["pos"], good["pos"], atol=1e-4) and tz["height"] == 0.5 and g.b_w == (0.0, 0.0, 0.0)
    assert g.b_v[0] == g.b_v[1] == 0.0 and abs(g.b_v[2] - (TN.toss_vz(0.5) - TN.GRAV * TN.CONTROL_DT)) < 0.01
    refused(layer, "A", good, m, 409)                         # already tossed
    # the referee on injected events: a foot fault, then a serve out of the box: a double fault
    g.rs["base_at_release"] = g.rs["base_at_release"] + np.array([0.2, 0.0])
    g._on_event({"kind": "hit", "side": "A", "k": m.k, "pos": [1.1, -11.6, 1.2], "speed": 25.0})
    assert g.serves[-1]["why"].startswith("serve_fault:foot_fault") and g.rs["faults"] == 1 and g.rs["phase"] == "serve"
    to_spot(m, "A")
    m.apply("A", layer.parse("A", good, m, {}))
    run_until(m, lambda m: g.rs["phase"] == "toss")
    g._on_event({"kind": "hit", "side": "A", "k": m.k, "pos": [1.1, -11.6, 1.2], "speed": 25.0})
    assert g.rs["phase"] == "flight"
    g._on_event({"kind": "bounce", "k": m.k, "x": 1.0, "y": 3.0, "half": "B"})   # B's frame (-1, -3): the wrong box
    m.step()
    p = g.points[-1]
    assert p["winner"] == "B" and p["how"].startswith("serve_fault:out_of_box (double fault"), p
    # a serve touching the net and landing in is a let (that serve again; the fault count stays)
    m2 = SPORT.build(cfg())
    g2 = m2.g
    toss_now(m2, "A")
    g2._on_event({"kind": "hit", "side": "A", "k": m2.k, "pos": [1.1, -11.6, 1.2], "speed": 25.0})
    g2._on_event({"kind": "cord", "k": m2.k, "x": 0.0, "z": 0.95})
    g2._on_event({"kind": "bounce", "k": m2.k, "x": -2.0, "y": 4.0, "half": "B"})  # B's frame (2, -4): in the box
    assert g2.serves[-1]["outcome"] == "let" and g2.rs["faults"] == 0 and g2.rs["phase"] == "serve"
    # format match: the same window and stance rules for your serve
    m3, layer3 = SPORT.build(cfg("match")), SPORT.action_layer("high")
    refused(layer3, "A", good, m3, 409)
    run_until(m3, lambda m: m.k >= m.g.leg["open"] + 5)
    refused(layer3, "A", good, m3, 400)


def test_exploit_reach_over_net():
    assert TN.BASE_HI[1] == -0.4
    m, layer = SPORT.build(cfg()), SPORT.action_layer("high")
    layer.parse("A", {"type": "move", "base": [0.0, -0.5]}, m, {})          # the rails reach the net
    g = m.g
    g.rs.update(phase="rally", hitter="B", expect="A", bounces=0, played_k=g.k)
    g.b_live = True
    g._on_event({"kind": "hit", "side": "A", "k": g.k, "pos": [0.3, 0.2, 1.0], "speed": 10.0})
    m.step()
    p = g.points[-1]
    assert p["winner"] == "B" and "Rule 24 i" in p["how"], p


def _dev(seed, stream, n):
    rng = TN.scatter_rng(seed, 13, stream, n)
    return rng.normal(0.0, 1.0, 3).round(12).tolist()


def test_exploit_independent_scatter_streams():
    """Each robot draws its own scatter stream (rule A6): in a duel the first server's (1) and the other's (2), so the
    n-th acts of the two robots scatter independently; the other first order on the same seed swaps the streams. In the
    match format the player's stream starts with tn.SCATTER_TAG, which no draw of the scripted opponent does."""
    snaps = {}
    for first in ("A", "B"):
        m = SPORT.build(cfg(first=first, seed=7))
        snaps[first] = {s: m.g.plan_snapshot(s) for s in ("A", "B")}
        other = {"A": "B", "B": "A"}[first]
        assert snaps[first][first]["stream"] == 1 and snaps[first][other]["stream"] == 2
    dev = {f: {s: [_dev(sn["seed"], sn["stream"], n) for n in range(6)] for s, sn in snaps[f].items()} for f in "AB"}
    assert all(x != y for x, y in zip(dev["A"]["A"], dev["A"]["B"]))           # independent within a match
    assert dev["A"]["A"] == dev["B"]["B"] and dev["A"]["B"] == dev["B"]["A"]   # the pair swaps the streams
    a = np.array([_dev(s * 1000, 1, n)[0] for s in (1, 2, 3, 4) for n in range(50)])
    b = np.array([_dev(s * 1000, 2, n)[0] for s in (1, 2, 3, 4) for n in range(50)])
    assert abs(float(np.corrcoef(a, b)[0, 1])) < 0.2
    single = SPORT.build(cfg("match", seed=13))
    seed = single.g.lv["seed"]
    assert TN.plan_snapshot(single.g)["stream"] == 1
    key = np.random.SeedSequence([seed, TN.SCATTER_TAG, 13, 1, 0]).generate_state(4).tolist()
    others = [np.random.SeedSequence([seed, p, l]).generate_state(4).tolist() for p in range(41) for l in range(401)]
    others += [np.random.SeedSequence([seed, 11, n]).generate_state(4).tolist() for n in range(41)]
    assert key not in others
