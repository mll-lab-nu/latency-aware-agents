"""fencing's own tests: one per [[exploits]] entry of sport.toml (tools/check runs them through core/testkit.py after
the generic suite; ported from the legacy scripts/olympiad/fencing/test_fencing.py). They need MuJoCo: run them in the
sport's image, e.g. python3 -m core.testkit tests sports/fencing/plugin.py (on an assembled sim directory)."""
import json
import sys
from pathlib import Path

import numpy as np

from core.episode import load_plugin
from core.sport import UNIT_OVER, FINISHED, MatchConfig, Refused

HERE = Path(__file__).resolve().parent
PLUGIN = str(HERE.parents[0] / "plugin.py")
SPORT = load_plugin(PLUGIN)
sys.path.insert(0, str(HERE))                    # feasibility.py (the entry-gate study)
import bout as BT  # noqa: E402
import fencing as F  # noqa: E402
import strategies as S  # noqa: E402


def cfg(fmt="duel", seed=1, level=None, action_level="high"):
    seats = ("A",) if fmt == "bout" else ("A", "B")
    return MatchConfig(sport="fencing", format=fmt, seats=seats, seed=seed, first="A", level=level,
                       action_level=action_level)


def play_vs_level(strategy, level, seed, max_s=90.0, latency=0.06):
    """a privileged scripted strategy as seat A against the scripted opponent, through the v0.2 match"""
    m = SPORT.build(cfg("bout", seed, level))
    lag = int(round(latency / F.CONTROL_DT))
    while m.phase().state != FINISHED and m.k * F.CONTROL_DT < max_s:
        if m.phase().state == UNIT_OVER:
            m.resume()
        me = strategy(m.ctl["A"], np.random.default_rng([seed, 9]))
        hist, t, n0 = [], 0.0, len(m.bout.halts)
        while len(m.bout.halts) == n0 and m.k * F.CONTROL_DT < max_s:
            hist.append(S.snapshot(m.w))
            o = S.view(hist[max(0, len(hist) - 1 - lag)], "A")
            o["me"] = S.view(hist[-1], "A")["me"]
            me.act(t, o)
            m.step()
            t += F.CONTROL_DT
    return m


def test_exploit_commands_out_of_range_refused():
    high = [dict(type="act", t=0.1, tip=[2.5, 0, 1], dir=[1, 0, 0]), dict(type="act", t=-1, tip=[1, 0, 1], dir=[1, 0, 0]),
            dict(type="act", t=0.1, tip=[1, 0, 1], dir=[0, 0, 0]), dict(type="move", rail=[9.0]),
            dict(type="move", rail=[0.0, 5.0]), dict(type="teleport"),
            dict(type="act", t=0.1, tip=[1, 0, float("nan")], dir=[1, 0, 0]),
            dict(type="act", t="0.1", tip=[1, 0, 1], dir=[1, 0, 0]), dict(type="guard", rail=[1.0]),
            dict(type="act", t=0.1, tip=[1, 0, 1], dir=[1, 0, 0], spin=1)]
    low = [dict(type="joints", q=[0] * 7, t=0.1), dict(type="joints", q=[0.9, 0.7, -1.5, -2.5, 1.3, 0.8], t=0.1),
           dict(type="joints", q=[0.9, 0.7, -1.5, -2.5, 1.3, 0.8, -1.2], t=4.0), dict(type="move", x=8.0),
           dict(type="move", x=0.0, vmax=3.0), dict(type="hold", x=1), dict(type="traj", q=[]),
           dict(type="traj", q=[[0.0] * 7]), dict(type="traj", q=[[-2.0, 0, 0, 0, -1, 0, 1, 0]] * 601),
           dict(type="act", t=0.1, tip=[1, 0, 1], dir=[1, 0, 0])]
    for level, bodies in (("high", high), ("low", low)):
        m, layer = SPORT.build(cfg(action_level=level)), SPORT.action_layer(level)
        d0 = m.state_digest()
        for body in bodies:
            try:
                layer.parse("A", body, m, {})
            except Refused as e:
                assert e.code == 400
                continue
            raise AssertionError(f"{level}: accepted {body}")
        assert m.state_digest() == d0 and m.k == 0


def test_exploit_traj_speed_limits():
    m, layer = SPORT.build(cfg(action_level="low")), SPORT.action_layer("low")
    c = m.ctl["A"]
    here = [c.x_t, *c.q_t.tolist()]
    fast = [here, [here[0] + 0.05, *here[1:]]]                      # 5 m/s on the rail: refused
    try:
        layer.parse("A", {"type": "traj", "q": fast}, m, {})
        raise AssertionError("accepted a traj beyond the rail speed")
    except Refused:
        pass
    jump = [[here[0], here[1] + 0.2, *here[2:]]]                    # one row 0.2 rad away from the current target
    cmd = layer.parse("A", {"type": "traj", "q": jump}, m, {})
    d0 = m.state_digest()
    info = m.apply("A", cmd)
    assert "rejected" in info and not m.traj["A"] and m.state_digest() == d0
    jerky = [here, here, [here[0], here[1] + 0.02, *here[2:]]]
    try:                                                            # 0 -> 2 rad/s between rows: 200 rad/s^2
        layer.parse("A", {"type": "traj", "q": jerky}, m, {})
        raise AssertionError("accepted a traj beyond the arm's acceleration limit")
    except Refused as e:
        assert "acceleration" in str(e), e
    start = [[here[0], here[1] + 0.02, *here[2:]], [here[0], here[1] + 0.04, *here[2:]]]
    info = m.apply("A", layer.parse("A", {"type": "traj", "q": start}, m, {}))   # from rest to 2 rad/s: at the effect
    assert "rejected" in info and "acceleration" in info["rejected"] and not m.traj["A"], info
    for xs, what in (([0.0, 0.0, 0.001], "acceleration"),          # 0 -> 0.1 m/s in one step: 10 m/s^2
                     ([0.0, 0.0, 0.0, 0.0003], "jerk")):           # 0 -> 3 m/s^2 in one step: 300 m/s^3
        rows = [[here[0] + x, *here[1:]] for x in xs]
        try:
            layer.parse("A", {"type": "traj", "q": rows}, m, {})
            raise AssertionError(f"accepted a traj beyond the rail's {what} limit")
        except Refused as e:
            assert what in str(e), e
    info = m.apply("A", layer.parse("A", {"type": "traj", "q": [[here[0] + 0.001, *here[1:]]]}, m, {}))
    assert "rejected" in info and "rail" in info["rejected"] and not m.traj["A"], info   # from rest: at the effect
    x, v, a, ok = here[0], 0.0, 0.0, []                             # a jerk-limited rail move: accepted and executed
    for _ in range(50):
        x, v, a, _j = F.rail_goto(x, v, a, here[0] + 0.2, F.RAIL_VMAX)
        ok.append([x, *here[1:]])
    info = m.apply("A", layer.parse("A", {"type": "traj", "q": ok}, m, {}))
    assert "rejected" not in info, info
    for _ in range(80):
        m.step()
    assert abs(m.w.cart_x("A") - here[0] - 0.2) < 0.01, m.w.cart_x("A")
    c = m.ctl["A"]                                                  # rows that end at speed: the rail brakes to rest
    x, v, a, fast = c.x_t, 0.0, 0.0, []                              # at the last row within its limits
    for _ in range(40):
        x, v, a, _j = F.rail_goto(x, v, a, c.x_t + 1.5, F.RAIL_VMAX)
        fast.append([x, *c.q_t.tolist()])
    info = m.apply("A", layer.parse("A", {"type": "traj", "q": fast}, m, {}))
    assert "rejected" not in info, info
    xs = list(c.x_hist)
    for _ in range(200):
        m.step()
        xs.append(c.x_t)
    assert F.rail_rows_why(xs) is None, F.rail_rows_why(xs)
    assert abs(c.v_t) < 1e-9 and abs(c.x_t - fast[-1][0]) < 0.5, (c.v_t, c.x_t, fast[-1][0])
    m.apply("A", layer.parse("A", {"type": "move", "x": c.x_t - 1.0}, m, {}))   # an arm-only traj while the cart
    for _ in range(20):                                                         # runs would stop it at once
        m.step()
    hold = [[c.x_t, *c.q_t.tolist()]] * 3
    info = m.apply("A", layer.parse("A", {"type": "traj", "q": hold}, m, {}))
    assert "rejected" in info and "rail" in info["rejected"], info
    test_rail_goto_within_limits()


def test_rail_goto_within_limits():
    """the rail's go-to (fencing-4): every move, retarget and speed-cap change keeps the speed cap, 6 m/s^2 and
    150 m/s^3 (continuous and by differences of the 10 ms targets), arrives (overshoot under 0.5 mm) and rests exactly at the goal"""
    for goal, vmax, changes in ((0.003, 2.0, ()), (0.3, 2.0, ()), (4.0, 2.0, ()), (2.0, 0.5, ()),
                                (1.0, 2.0, ((30, -1.0, 2.0),)), (5.0, 2.0, ((100, 5.0, 0.5),))):
        x, v, a, xs, js = 0.0, 0.0, 0.0, [0.0], []
        g, vm = goal, vmax
        for k in range(800):
            for kk, gg, vv in changes:
                if k == kk:
                    g, vm = gg, vv
            x1, v1, a1, j = F.rail_goto(x, v, a, g, vm)
            peak = max(abs(v), abs(v1), *(abs(v + a * t + 0.5 * j * t * t) for t in np.linspace(0, F.CONTROL_DT, 11)))
            assert peak <= max(vm, abs(v)) + 1e-6 and abs(a1) <= F.RAIL_AMAX + 1e-9 and abs(j) <= F.RAIL_JMAX + 1e-9
            x, v, a = x1, v1, a1
            xs.append(x)
            js.append(j)
        assert F.rail_rows_why(xs) is None, (goal, F.rail_rows_why(xs))
        assert x == g and v == 0.0 and a == 0.0, (goal, x, v, a)
        if not changes:
            assert max(xs) <= goal + 5e-4, (goal, max(xs))      # under half a millimetre


def test_exploit_stalling_loses_against_level_1():
    """never attacking, point in line: the level-1 opponent breaks the waiting game and scores"""
    m = play_vs_level(S.BY_NAME["wall"], 1, 101, max_s=60.0)
    assert m.bout.score["B"] >= 2 and m.bout.score["A"] == 0, m.bout.score


def test_exploit_ramming_is_stop_hit():
    """charging the cart into the opponent with the arm on guard is no defence: it runs onto the point"""
    class Ram(S.Strategy):
        name = "ram"

        def act(self, t, o):
            self.c.rail(o["op"]["x"] - 0.4)
    import feasibility as FZ
    res = [FZ.play_phrase(Ram, S.BY_NAME["stop"], seed, 0.0 if seed == 0 else 1.0)["result"] for seed in range(12)]
    assert res.count("B") > res.count("A"), res      # fencing-4: B 4, A 2, 6 corps-a-corps (fencing-3: B 5, A 1)


def test_exploit_rear_limit_is_a_touch_against():
    m = SPORT.build(cfg(action_level="low"))
    m.apply("A", {"type": "move", "x": -7.6, "vmax": 2.0})
    for _ in range(600):
        m.step()
        if m.phase().state != "playing":
            break
    h = m.bout.halts[-1]
    assert h["why"] == "rear_limit" and h["awarded"] == ["B"] and m.scores() == {"A": 0, "B": 1}


ALLOWED_ROBOT = {"cart_x", "cart_v", "q", "point", "blade_dir", "guard", "rail_goal", "rail_vmax", "arm_moving",
                 "arm_arrive_in_s"}


def test_exploit_no_opponent_state_or_seed():
    from core import local
    for fmt in ("bout", "duel"):
        c = cfg(fmt, seed=97531, level=1 if fmt == "bout" else None)
        m = SPORT.build(c)
        for _ in range(30):
            m.step()
        st = m.observe("A")
        assert set(st["robot"]) == ALLOWED_ROBOT
        for key, v in st.items():
            if isinstance(v, dict) and "opponent" in v:
                assert key in ("score", "lamps"), key
        flat = json.dumps(st)
        for word in ("97531", "strateg", "B_", "opponent_x", "opp\""):
            assert word not in flat, word
        sp = json.dumps(local.seat_spec(SPORT, c, "A", m, plugin=PLUGIN))
        assert "97531" not in sp and '"seed"' not in sp and "strateg" not in sp and "repertoire" not in sp


def test_exploit_no_draws():
    for seed in range(1, 9):
        b = BT.Bout(seed)
        b.fence()
        b.k = BT.BOUT_STEPS                       # time is up at 0-0
        b._judge(dict(why="time", k=0, touches={}))
        assert b.phase == "halt" and b.extra and b.priority in ("A", "B")
        b.resume(); b.fence()
        b.k = BT.BOUT_STEPS + BT.EXTRA_STEPS       # the extra minute ends level
        b._judge(dict(why="time", k=0, touches={}))
        assert b.phase == "over" and b.winner == b.priority and b.how == "priority"
    b = BT.Bout(1); b.fence()
    b.score = {"A": 4, "B": 4}
    b._judge(dict(why="touch", k=0, touches={"A": 1, "B": 20}))
    assert b.score == {"A": 4, "B": 4} and b.halts[-1]["annulled"] and b.phase == "halt"
    b.resume(); b.fence()
    b._judge(dict(why="touch", k=0, touches={"B": 3}))
    assert b.phase == "over" and b.winner == "B" and b.how == "touches"
    m = SPORT.build(cfg())                        # the match's tiebreak chain decides every finished bout (fenced)
    m.bout.k = BT.BOUT_STEPS
    m.bout._judge(dict(why="time", k=0, touches={}))
    m.bout.resume(); m.bout.fence()
    m.bout.k = BT.BOUT_STEPS + BT.EXTRA_STEPS
    m.bout._judge(dict(why="time", k=0, touches={}))
    from core.outcome import decide
    out = decide(m, sides=["A", "B"])
    assert m.phase().state == FINISHED and out.winner == m.bout.priority and out.decided_by == "priority"


def _run_to_halt(m, retreat_x=None, max_k=None):
    """step the duel until the next halt; seat A (if retreat_x) keeps its cart at retreat_x, both arms hold on guard"""
    if retreat_x is not None:
        m.apply("A", {"type": "move", "x": retreat_x, "vmax": 2.0})
    n0 = len(m.bout.halts)
    while len(m.bout.halts) == n0 and (max_k is None or m.k < max_k):
        m.step()
    return m.bout.halts[-1] if len(m.bout.halts) > n0 else None


def test_exploit_passivity_cards():
    """E8 (t.124, D9): seat A retreats to just in front of its rear limit and waits, seat B waits on guard. Up to
    fencing-4 (no t.124) that runs the 3 minutes out at 0-0 into the extra minute (decided by the lot); now the first
    minute without a touch is a P-red card for both (1-1), the second a P-black card (equal scores: by the seeded lot)
    and the bout is over at 2 minutes of fencing time."""
    from core.outcome import decide
    back = -F.PISTE_HALF + F.CART_HALF + 0.05                      # the cart entirely in front of the rear limit
    old = SPORT.build(cfg(action_level="low"))
    old.bout.passive_steps = None                                   # the rules up to fencing-4
    h = _run_to_halt(old, back)
    assert h["why"] == "time" and h["k"] == BT.BOUT_STEPS and old.scores() == {"A": 0, "B": 0}, h
    assert old.bout.extra and old.bout.priority in ("A", "B") and old.phase().state == UNIT_OVER
    m = SPORT.build(cfg(action_level="low"))
    h = _run_to_halt(m, back)
    assert h["why"] == "passivity" and h["k"] == BT.PASSIVE_STEPS and h["cards"] == {"A": "P-red", "B": "P-red"}, h
    assert m.scores() == {"A": 1, "B": 1} and h["next_on_guard"] == BT.ON_GUARD_XS and h["no_hit_s"] == BT.PASSIVE_S
    v = m.observe("B")
    assert v["last_halt"]["cards"] == {"you": "P-red", "opponent": "P-red"} and v["passivity"]["p_cards"] == 1
    m.resume()
    assert m.observe("A")["passivity"] == {"no_hit_s": 0.0, "limit_s": BT.PASSIVE_S, "p_cards": 1}
    h = _run_to_halt(m, back)
    lot = "A" if np.random.default_rng([m.bout.seed, 124]).random() < 0.5 else "B"
    assert h["why"] == "passivity" and h["k"] == 2 * BT.PASSIVE_STEPS and h["cards"] == {lot: "P-black"}, h
    out = decide(m, sides=["A", "B"])
    assert m.phase().state == FINISHED and out.winner == BT.OTHER[lot] and out.decided_by == "passivity", out
    assert m.scores() == {"A": 1, "B": 1} and m.final_fields()["p_cards"] == 2

    b = BT.Bout(3); b.fence()                                       # the leader runs the clock: P-black to the lower
    b.score = {"A": 2, "B": 1}
    b.k = BT.PASSIVE_STEPS
    b._judge(dict(why="passivity", k=0, touches={}))
    assert b.score == {"A": 3, "B": 2} and b.phase == "halt" and b.minute_k0 == b.k and b.p_calls == 1
    b.resume(); b.fence()
    b.k += BT.PASSIVE_STEPS
    b._judge(dict(why="passivity", k=0, touches={}))
    assert b.winner == "A" and b.how == "passivity" and b.halts[-1]["cards"] == {"B": "P-black"}
    b = BT.Bout(3); b.fence()                                       # a P-red at 4-3 ends the bout 5-4 by touches
    b.score = {"A": 3, "B": 4}
    b._judge(dict(why="passivity", k=0, touches={}))
    assert b.winner == "B" and b.how == "touches" and b.score == {"A": 4, "B": 5}
    b = BT.Bout(3); b.fence()                                       # no P-card at 4-4 (t.124.3.a): the minute restarts
    b.score = {"A": 4, "B": 4}
    b.k = 9000
    b._judge(dict(why="passivity", k=0, touches={}))
    assert b.score == {"A": 4, "B": 4} and b.halts[-1]["cards"] == {} and b.p_calls == 0 and b.minute_k0 == 9000
    b = BT.Bout(3); b.fence()                                       # t.124.3.d: a touch restarts the minute,
    b.k = 2000                                                      # a corps-a-corps does not
    b._judge(dict(why="corps_a_corps", k=0, touches={}))
    assert b.minute_k0 == 0 and b.no_hit_s() == 20.0
    b.resume(); b.fence(); b.k = 3000
    b._judge(dict(why="touch", k=0, touches={"A": 5}))
    assert b.minute_k0 == 3000 and b.no_hit_s() == 0.0
    b = BT.Bout(3); b.fence()                                       # t.124.3.f: not in the extra minute
    b.k = BT.BOUT_STEPS
    b._judge(dict(why="time", k=0, touches={}))
    assert b.extra and b.no_hit_s() is None
