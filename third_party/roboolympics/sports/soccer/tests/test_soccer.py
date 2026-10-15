"""soccer's exploit tests (sport.toml [[exploits]], one test each; core/testkit.py runs them in tools/check stage 2),
ported from the legacy suite scripts/olympiad/soccer/test_soccer.py and test_identity.py. World-level tests drive
soccer.World directly (as the legacy ones did); the others go through the plugin and the core's own pieces (the
action layer, the outcome rule, the team channel). Run alone in the sim image: python3 tests/test_soccer.py."""
from __future__ import annotations

import contextlib
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import mujoco  # noqa: E402
import soccer as S  # noqa: E402
import soccer_scene as SC  # noqa: E402

from core.channel import TeamChannel  # noqa: E402
from core.outcome import decide  # noqa: E402
from core.sport import MatchConfig, Refused  # noqa: E402


def plugin():
    import importlib.util
    spec = importlib.util.spec_from_file_location("soccer_plugin_test", os.path.join(HERE, "plugin.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


P = plugin()


def cfg(fmt="duel", seed=1, first="A", level=None):
    return MatchConfig(sport="soccer", format=fmt, seats=P.FORMAT_SEATS[fmt], seed=seed, first=first,
                       action_level="low", level=level, sides=P.FORMAT_SIDES[fmt])


@contextlib.contextmanager
def half(seconds):
    """A shorter half for a test (soccer.HALF_S is read at every control step)."""
    old = S.HALF_S
    S.HALF_S = float(seconds)
    try:
        yield
    finally:
        S.HALF_S = old


def world(level=0, seed=1, **kw):
    return S.World(scripted={"B": level} if level else None, seed=seed, **kw)


def run_steps(w, n, cmds=None):
    for i in range(n):
        if w.done or w.stopped:
            return
        if cmds:
            cmds(w, i)
        w.advance()


def refused(layer, m, seat, body, code=400):
    try:
        layer.parse(seat, body, m, {})
    except Refused as e:
        assert e.code == code, (body, e.code, str(e))
        return
    raise AssertionError(f"accepted {body}")


# ------------------------------------------------------------------------------------------------ E1, E2
def test_exploit_out_of_range_commands_refused():
    m = P.SPORT.build(cfg("duel"))
    layer = P.SPORT.action_layer("low")
    d0 = m.state_digest()
    for b in (dict(entity="1", type="walk", v=[1.5, 0, 0]), dict(entity="1", type="walk", v=[0, 0, float("nan")]),
              dict(entity="3", type="stand"), dict(robot=3, type="stand"), dict(entity="1", type="kick", power=1.2),
              dict(entity="1", type="teleport"), dict(entity="1", type="walk", v=[0, 0]),
              dict(entity="1", type="walk", v=[0, 0, 0], x=1), dict(robot=1, entity="2", type="stand"), [1, 2],
              dict(entity="1", type="kick", aim=0.7), dict(entity="1", type="kick", style="volley"),
              dict(entity="1", type="kick", at=[1.0, 2.0]), dict(entity="1", type="kick", at=[1.0, 2.0, 11.0]),
              dict(entity="1", type="kick", at=[1.0, float("nan"), 0.0]), dict(entity="1", type="kick", at=1.0),
              dict(entity="1", type="kick", aim=float("inf")), dict(entity="1", type="kick", foot="both")):
        refused(layer, m, "A", b)
    assert layer.parse("B", dict(robot=2, type="kick", power=0.5), m, {}) == {"type": "kick", "power": 0.5, "aim": 0.0,
                                                                             "at": None, "foot": "right", "v": None,
                                                                             "style": "swing", "entity": "2"}
    assert layer.parse("B", dict(robot=2, type="kick", style="step", v=[0.4, 0.0, 0.1]), m, {})["v"] == [0.4, 0.0, 0.1]
    assert layer.parse("B", dict(robot=2, type="kick", at=[2.5, -1.0, 0.25]), m, {})["at"] == [2.5, -1.0, 0.25]
    assert layer.parse("B", dict(robot=2, type="kick", foot="left"), m, {})["foot"] == "left"
    assert layer.parse("B", dict(robot=2, type="kick", style="push"), m, {})["style"] == "push"
    assert layer.parse("A", dict(type="walk", v=[0.123456, 0, 1]), m, {})["v"] == [0.1235, 0.0, 1.0]  # legacy rounding
    t = P.SPORT.build(cfg("team"))
    refused(layer, t, "A2", dict(entity="1", type="stand"))         # a robot seat commands only its own robot
    assert layer.parse("A2", dict(type="stand"), t, {})["entity"] == "2"
    assert m.state_digest() == d0 and not m.w.acts


def test_exploit_penalized_robot_cannot_act():
    m = P.SPORT.build(cfg("duel"))
    layer = P.SPORT.action_layer("low")
    m.w._penalize("A1", "test", counts=False)
    refused(layer, m, "A", dict(entity="1", type="stand"), code=409)
    d0 = m.state_digest()
    info = m.apply("A", {"type": "walk", "v": [1.0, 0.0, 0.0], "entity": "1"})   # the boundary refuses it too
    assert "rejected" in info and m.state_digest() == d0 and not m.w.acts
    m.w.fallen["A2"] = m.w.k
    refused(layer, m, "A", dict(entity="2", type="stand"), code=409)
    assert m.refused["A"] == 2


# ------------------------------------------------------------------------------------------------ E3 - E9
def test_exploit_ball_holding_is_penalized():
    w = world()
    w.state = "play"
    w.ko = None
    for _ in range(int(6.0 / S.CONTROL_DT)):
        x, y = w.rob["A1"].xy(w.d)
        w.place_ball(x + 0.13, y)                # the ball parked at the robot's shins, under the trunk (soccer-2v2-8:
                                                 # 0.05 put it inside the legs, and the bouncy ball is pushed out)
        w.advance()
        if w.penalty["A1"]:
            break
    assert w.penalty["A1"] and w.penalty["A1"]["why"] == "ball holding"


def _ram(v):
    w = world()
    w.state = "play"
    w.ko = None
    w.place_ball(0, 3.0)
    w.place("A1", -0.6, 0.0, 0.0)
    w.place("B1", 0.4, 0.0, math.pi)
    w.place("A2", -4.0, -3.0, 0.0)
    w.place("B2", 4.0, 3.0, math.pi)
    mujoco.mj_forward(w.m, w.d)
    w.command("A1", dict(type="walk", v=[v, 0.0, 0.0]))
    for _ in range(int(12 / S.CONTROL_DT)):
        if w.stopped:
            w.resume()
        w.advance()
        if w.penalty["A1"]:
            break
    return w


def test_exploit_ramming_is_pushing():
    w = _ram(0.6)
    assert w.penalty["A1"] and "pushing" in w.penalty["A1"]["why"] and w.penalty["B1"] is None
    w = _ram(1.0)
    assert (w.penalty["A1"] or (w.stats["A1"]["falls"] and not w.stats["B1"]["falls"])) and w.penalty["B1"] is None


def test_exploit_stalling_loses_to_the_scripted_team():
    m = P.SPORT.build(cfg("match", seed=101, level=1))
    w = m.w
    for _ in range(int(S.HALF_S / S.CONTROL_DT)):
        if w.done or w.period != 1:
            break
        if w.stopped:
            m.resume()
        m.step()
    assert w.score["B"] > 0 and w.score["A"] == 0, w.score
    assert m.scores() == {"A": 0, "opponent": w.score["B"]}


def test_exploit_parking_both_robots_in_goal_is_illegal_defence():
    w = world(level=1, seed=111)
    w.place("A1", -4.75, 0.55, 0.0)
    w.place("A2", -4.75, -0.55, 0.0)
    mujoco.mj_forward(w.m, w.d)
    for _ in range(int(7 / S.CONTROL_DT)):
        if w.stopped or w.done:
            break
        w.advance()
        if w.penalty["A1"] or w.penalty["A2"]:
            break
    why = [(w.penalty[r] or {}).get("why", "") for r in ("A1", "A2")]
    assert sum("illegal defence" in x for x in why) == 1, why
    assert abs(w.t() - S.CROWD_S) < 0.1
    w = world(level=1, seed=111)
    w.place("A1", -4.75, 0.0, 0.0)
    w.place("A2", -2.0, -1.5, 0.0)               # one keeper: no foul
    mujoco.mj_forward(w.m, w.d)
    for _ in range(int(20 / S.CONTROL_DT)):
        if w.stopped or w.done:
            break
        w.advance()
    assert not any("illegal defence" in o for st in w.stats.values() for o in st["offenses"])


def test_exploit_kickoff_goal_without_touch_outside_circle():
    w = world()
    assert w.ko["team"] == "A"                   # the first kick-off has the kick-off conditions too
    w.ko["in_play"] = True
    w.state = "play"
    w.place_ball(S.HALF_L + 0.3, 0.0)            # straight into the goal from the kick-off
    w.last_touch = "A1"
    w.advance()
    assert w.score == {"A": 0, "B": 0} and w.stops[-1]["kind"] == "no_goal" and w.kickoff_team == "B"


def test_exploit_game_stuck_and_ball_out():
    w = world()
    w.ko = None
    w.state = "play"
    w.place_ball(0.0, S.HALF_W + 0.6)
    w.advance()
    assert w.stops[-1]["kind"] == "ball_out"
    np.testing.assert_allclose(w.ball_xy(), [0, 0], atol=1e-6)
    w = world()
    w.ko = None
    w.state = "play"
    w.place_ball(0.0, 3.0)
    run_steps(w, int(31 / S.CONTROL_DT))
    assert w.stops and w.stops[-1]["kind"] == "global_game_stuck"
    w = world()
    w.ko = None
    w.state = "play"
    w.place("A1", -0.6, 0.0, 0.0)
    w.place_ball(0.3, 0.0)
    mujoco.mj_forward(w.m, w.d)
    run_steps(w, int((S.LOCAL_STUCK_S + 1) / S.CONTROL_DT))
    assert (w.penalty["A1"] or {}).get("why") == "local game stuck"


def test_exploit_keeper_off_line_concedes():
    with half(1):
        m = P.SPORT.build(cfg("duel"))
        w = m.w
        while w.state != "pso_kick":
            if w.stopped:
                m.resume()
            m.step()
        if w.stopped:
            m.resume()
        cur = w.pso["cur"]
        keeper = cur["keeper"]
        m.apply(keeper[0], {"type": "walk", "v": [1.0, 0.0, 0.0], "entity": keeper[1]})   # out before the kick
        for _ in range(int(5 / S.CONTROL_DT)):
            if w.stopped or w.done:
                break
            m.step()
        assert w.pso["kicks"][0]["result"] == "goal"


# ------------------------------------------------------------------------------------------------ E10
def test_exploit_no_draw_shootout():
    w = world()
    w.pso = dict(kicks=[], first="A")
    for i in range(6):                           # 3 kicks each, all missed: still level
        w.pso["kicks"].append(dict(team="AB"[i % 2], result="wide", category=S.PSO_CATEGORIES.index("wide")))
    assert w._pso_winner() is None
    w.pso["kicks"] += [dict(team="A", result="saved", category=1), dict(team="B", result="post", category=2)]
    assert w._pso_winner() == ("A", "pso_sudden_death")            # saved ranks above post (HSL Law 14)
    w.pso["kicks"] = w.pso["kicks"][:6] + [dict(team=t, result="no_kick", category=5) for _ in range(3) for t in "AB"]
    win = w._pso_winner()
    assert win[1] == "pso_coin_toss" and win[0] in ("A", "B") and win == w._pso_winner()
    tosses = set()
    for seed in range(1, 12):
        w.seed = seed
        tosses.add(w._pso_winner()[0])
    assert tosses == {"A", "B"}
    m = P.SPORT.build(cfg("duel"))                # the tiebreak chain decides every finished match
    m.w.done, m.w.winner, m.w.end_reason = True, "B", "pso_coin_toss"
    assert decide(m, None, None, ["A", "B"]).winner == "B"
    assert decide(m, None, None, ["A", "B"]).decided_by == "coin_toss"


# ------------------------------------------------------------------------------------------------ E11
def walk_json(x, keys, values, path=""):
    if isinstance(x, dict):
        for k, v in x.items():
            keys.add(k)
            walk_json(v, keys, values, path + "/" + str(k))
    elif isinstance(x, list):
        for v in x:
            walk_json(v, keys, values, path)
    elif isinstance(x, (int, float)) and not isinstance(x, bool):
        values.append((path, float(x)))


def test_exploit_no_world_state_in_observation():
    seed = 987654
    for fmt in ("match", "duel", "team"):
        m = P.SPORT.build(cfg(fmt, seed=seed, level=1 if fmt == "match" else None))
        w = m.w
        w.place_ball(1.2345, -0.6789)
        marks = {"A1": (-2.1357, 1.9753), "A2": (-3.2468, -1.8642), "B1": (2.4681, 0.9135), "B2": (3.5791, -2.2468)}
        for r, (x, y) in marks.items():
            w.place(r, x, y, 0.4321 if r[0] == "A" else -2.7654)
        mujoco.mj_forward(w.m, w.d)
        w.state, w.ko = "play", None
        w.event("fall", robot="B1")
        w.event("pass", robot="B1", to="B2")
        for seat in m.cfg.seats:
            team, robots = seat[0], P.robots_of(seat)
            st = m.observe(seat)
            keys, values = set(), []
            walk_json(st, keys, values)
            assert not keys & {"ball", "ball_v", "truth", "qpos", "opponents", "xy", "seed"}, (fmt, seat)
            assert set(st["robots"]) == {r[1] for r in robots}
            secret = [S.to_team(team, (1.2345, -0.6789))]
            secret += [S.to_team(team, xy) for r, xy in marks.items() if r not in robots]
            for path, v in values:
                for p in secret:
                    for c in p:
                        assert not (abs(v - c) < 2e-3 and abs(c) > 0.5), (fmt, seat, path, v)
            assert not any(e["ev"] in ("pass", "fall") and e["robot"].startswith("them") for e in st["events"])
            sp = json.dumps(P.SPORT.spec(m.cfg, seat, m))
            assert "1.234" not in sp and "0.678" not in sp and str(seed) not in sp and str(seed) not in json.dumps(st)


# ------------------------------------------------------------------------------------------------ E12
def test_exploit_mirror_symmetric_world():
    w = world()
    m = w.m
    for a, b in (("A1", "B1"), ("A2", "B2"), ("A1", "A2")):
        ba = [i for i in range(m.nbody) if m.body(i).name.startswith(a + ":")]
        bb = [m.body(b + m.body(i).name[2:]).id for i in ba]
        for key in ("body_mass", "body_inertia", "body_ipos", "body_iquat", "body_pos", "body_quat"):
            np.testing.assert_array_equal(getattr(m, key)[ba], getattr(m, key)[bb], err_msg=f"{key} {a} {b}")
        phys = lambda g: bool(m.geom_contype[g] or m.geom_conaffinity[g])   # noqa: E731 (jersey numbers are visual)
        ga = [g for i in ba for g in range(m.ngeom) if m.geom_bodyid[g] == i and phys(g)]
        gb = [g for i in bb for g in range(m.ngeom) if m.geom_bodyid[g] == i and phys(g)]
        assert len(ga) == len(gb) and m.geom_contype[ga].any()
        for key in ("geom_size", "geom_friction", "geom_solref", "geom_solimp", "geom_contype", "geom_conaffinity",
                    "geom_pos", "geom_quat", "geom_type", "geom_condim", "geom_margin"):
            np.testing.assert_array_equal(getattr(m, key)[ga], getattr(m, key)[gb], err_msg=f"{key} {a} {b}")
        aa, ab = w.rob[a].act, w.rob[b].act
        for key in ("actuator_ctrlrange", "actuator_gainprm", "actuator_biasprm", "actuator_gear"):
            np.testing.assert_array_equal(getattr(m, key)[aa], getattr(m, key)[ab])
        assert m.cam_fovy[w.rob[a].cam] == m.cam_fovy[w.rob[b].cam]
    labels = [i for i in range(m.ngeom) if ":jersey_" in (m.geom(i).name or "")]
    assert labels and not m.geom_contype[labels].any() and not m.geom_conaffinity[labels].any()
    swap = {"A": "B", "B": "A", "W": "E", "E": "W", "N": "S", "S": "N"}
    n = 0
    for i in range(m.ngeom):
        name = m.geom(i).name or ""
        if name.startswith(("post_", "bar_", "net_", "board_", "goal_floor_")):
            if name.startswith("net_side_"):
                other = "net_side_" + swap[name[9]] + "_" + str(-int(name[11:]))
            elif name.startswith("post_"):
                parts = name.split("_")
                other = f"post_{swap[parts[1]]}_{parts[2]}"
            else:
                other = "_".join(swap.get(p, p) if j > 0 else p for j, p in enumerate(name.split("_")))
            j = m.geom(other).id
            p, q = m.geom_pos[i], m.geom_pos[j]
            np.testing.assert_allclose([p[0], p[1], p[2]], [-q[0], -q[1], q[2]], atol=1e-9, err_msg=f"{name} {other}")
            np.testing.assert_array_equal(m.geom_size[i], m.geom_size[j])
            n += 1
    assert n > 15

    def kick(team):
        w = world()
        w.prog = {r: dict(type="stand") for r in S.ROBOTS}
        spots = {"1": (-3.0 - np.mean(S.SWING_ZONE["ahead"]), np.mean(S.SWING_ZONE["right"])), "2": (-4.0, 2.5)}
        for r in S.ROBOTS:
            x, y = spots[r[1]] if r[0] == team else ((-spots[r[1]][0] if r[1] == "2" else 3.5), -2.8)
            p = S.from_team(r[0], (x, y))
            w.place(r, p[0], p[1], 0.0 if r[0] == "A" else math.pi)
        w.place_ball(*S.from_team(team, (-3.0, 0.0)))
        w.ko, w.state = None, "play"
        mujoco.mj_forward(w.m, w.d)
        w.command(team + "1", dict(type="kick", power=0.25, aim=0.1))   # a soft kick: no goal within the test
        traj = []
        for _ in range(350):                     # the swing kick (~3.5 s) and the ball's roll
            w.advance()
            traj.append(S.to_team(team, w.ball_xy()))
        return np.array(traj), w

    ta, wa = kick("A")
    tb, wb = kick("B")
    assert float(np.hypot(*(ta[-1] - ta[0]))) > 1.0, "the kick did not move the ball"
    np.testing.assert_allclose(ta, tb, atol=2e-3)
    assert wa.stats["A1"]["touches"] == wb.stats["B1"]["touches"]
    disp = {}
    for team in "AB":                            # the same body-frame walk moves an A and a B robot alike
        w = world()
        p0 = w.self_state(team + "2")["pose"]
        w.command(team + "2", dict(type="walk", v=[0.6, 0.15, 0.3]))
        for _ in range(200):
            if w.stopped:
                w.resume()
            w.advance()
        p1 = w.self_state(team + "2")["pose"]
        c, s = math.cos(-p0["yaw"]), math.sin(-p0["yaw"])
        dx, dy = p1["x"] - p0["x"], p1["y"] - p0["y"]
        disp[team] = np.array([c * dx - s * dy, s * dx + c * dy, S._wrap(p1["yaw"] - p0["yaw"])])
    np.testing.assert_allclose(disp["A"], disp["B"], atol=0.02)
    a, b = (P.SPORT.spec(cfg("duel"), s, P.SPORT.build(cfg("duel"))) for s in "AB")
    for x in (a, b):
        x.pop("team")
        x["match"].pop("first_kickoff")
    assert a == b


# ------------------------------------------------------------------------------------------------ E13, E14
def test_exploit_review_stall_forfeits():
    for fmt, seat, winner in (("duel", "B", "A"), ("team", "B2", "A"), ("match", "A", None)):
        m = P.SPORT.build(cfg(fmt, level=1 if fmt == "match" else None))
        c = m.cfg
        d0 = m.state_digest()
        m.forfeit(seat, "ready_timeout")
        assert m.state_digest() == d0                     # a core decision: the world is not changed
        out = decide(m, seat, "ready_timeout", c.side_list(), c.side_of)
        assert out.winner == winner and out.decided_by == "forfeit:ready_timeout", (fmt, out)
        assert m.final_fields()["forfeit"]["seat"] == seat


def test_exploit_team_channel_teammates_only():
    import tomllib
    conf = tomllib.loads(open(os.path.join(HERE, "sport.toml")).read())["team_channel"]
    ch = TeamChannel(cfg("team"), conf)
    assert ch.enabled
    assert ch.say("A1", "x" * 513, 0, S.CONTROL_DT)[0] == 400
    assert ch.say("A1", "m0", 0, S.CONTROL_DT)[0] == 200
    assert ch.say("A1", "m1", 5, S.CONTROL_DT)[0] == 429           # 0.1 game s later: too soon
    assert ch.say("A1", "m1", 10, S.CONTROL_DT)[0] == 200
    assert ch.say("B1", "theirs", 10, S.CONTROL_DT)[0] == 200
    assert [m["text"] for m in ch.messages("A2", 10, S.CONTROL_DT)[0]] == ["m0", "m1"]
    assert [m["text"] for m in ch.messages("B2", 10, S.CONTROL_DT)[0]] == ["theirs"]
    assert ch.messages("A1", 10, S.CONTROL_DT)[0] == [] and ch.messages("B1", 10, S.CONTROL_DT)[0] == []
    assert not TeamChannel(cfg("duel"), conf).enabled              # one seat per team: no channel



# ------------------------------------------------------------------------------------------------ E15
def test_exploit_real_limits_hold():
    """The legs are held to the real T1's limits (bodies/booster_t1): the walking command a policy sees moves at most
    2 / 2 / 4 per s toward the request (a full reverse takes a second), the motor torques never exceed the URDF's, and
    a motor never drives a joint beyond its speed limit (the torque-speed envelope); also during a swing kick."""
    L, Wk = S.T1["limits"]["leg"], S.T1["limits"]["walk"]
    assert L["torque_nm"] == [45.0, 30.0, 30.0, 60.0, 24.0, 15.0] and Wk["command_accel"] == [2.0, 2.0, 4.0]
    rate = np.asarray(Wk["command_accel"], float)
    w = world()
    w.prog = {r: dict(type="stand") for r in S.ROBOTS}
    for r in S.ROBOTS:
        if r != "A1":
            w._penalize(r, "test", counts=False)               # off the pitch for the test
    w.place("A1", -2.0, 0.0, 0.0)
    w.place_ball(2.0, 2.0)
    w.ko, w.state = None, "play"
    mujoco.mj_forward(w.m, w.d)
    tau_max = np.tile(L["torque_nm"], 2)
    vmax = np.tile(L["speed_rad_s"], 2)
    r = w.rob["A1"]
    seen = []
    worst_tau, worst_drive = 0.0, 0.0
    for i in range(500):
        if i == 0:
            w.command("A1", dict(type="walk", v=[1.0, 0.0, 0.0]))
        if i == 60:
            w.command("A1", dict(type="walk", v=[-1.0, 0.0, 0.0]))     # reverse: ramps at 2 m/s^2
        if i == 120:
            w.command("A1", dict(type="stand"))
        if i == 160:                                             # to the kick spot behind the ball
            w.place("A1", -4.0, 2.0, 0.0)
            w.place_ball(-4.0 + np.mean(S.SWING_ZONE["ahead"]), 2.0 - np.mean(S.SWING_ZONE["right"]))
            mujoco.mj_forward(w.m, w.d)
        if i == 170:
            w.command("A1", dict(type="kick", power=1.0))
        if w.stopped:
            break
        w.advance()
        seen.append(w.cmd_lim["A1"].cmd.copy())
        tau = w.d.ctrl[r.act]
        qd = w.d.qvel[r.jv]
        worst_tau = max(worst_tau, float(np.max(np.abs(tau) / tau_max)))
        drive = np.sign(tau) == np.sign(qd)                     # a motor driving its joint beyond 80 % of the limit
        frac = np.where(drive & (np.abs(qd) > 0.8 * vmax), np.abs(tau) / (tau_max * np.clip(
            (vmax - np.abs(qd)) / (0.2 * vmax), 1e-9, 1.0)), 0.0)
        worst_drive = max(worst_drive, float(np.max(frac)))
    steps = np.abs(np.diff(np.array(seen), axis=0)).max(axis=0)
    assert np.all(steps <= rate * S.CONTROL_DT + 1e-9), steps
    assert abs(seen[59][0] - 1.0) < 1e-9 and seen[60][0] < 0.97 and seen[60 + 48][0] > -0.97 and \
        abs(seen[60 + 49][0] + 1.0) < 1e-9                       # a full reverse takes 50 control steps (1 s)
    assert worst_tau <= 1.0 + 1e-9 and worst_drive <= 1.0 + 1e-6, (worst_tau, worst_drive)
    assert w.stats["A1"]["kicks"] == 1 and w.stats["A1"]["touches"] >= 1, w.stats["A1"]


# ------------------------------------------------------------------------------------------------ E17
def _alone(x=0.0, y=0.0, yaw=0.0):
    w = world()
    for r in S.ROBOTS:
        if r != "A1":
            w._penalize(r, "test", counts=False)
    w.ko, w.state = None, "play"
    w.place("A1", x, y, yaw)
    w.place_ball(x - 2.0, y + 2.0)
    mujoco.mj_forward(w.m, w.d)
    run_steps(w, 25)
    return w


def _roll(w, xy, v):
    d, bv = w.d, w._bv
    w.place_ball(float(xy[0]), float(xy[1]))
    d.qvel[bv:bv + 3] = [v[0], v[1], 0.0]
    d.qvel[bv + 3:bv + 6] = [-v[1] / S.BALL_R, v[0] / S.BALL_R, 0.0]


def test_exploit_timed_kick_reads_only_the_robot():
    """The timed kick (`at`: the player's point and time) never reads the ball: with the ball elsewhere (or rolling
    past), the kicking leg moves exactly the same way until the ball is touched, the kick never waits for a still ball,
    and a rolling ball that arrives where and when the player said is struck; the across-the-body aim is clipped."""
    feet = []
    for ball in ((-2.0, 2.0), (-1.0, -2.5)):
        w = _alone()
        w.place_ball(*ball)
        mujoco.mj_forward(w.m, w.d)
        w.command("A1", dict(type="kick", power=1.0, aim=0.0, at=[2.8, 0.32, -0.06]))
        tr = []
        for _ in range(int(3.2 / S.CONTROL_DT)):
            w.advance()
            tr.append(w.d.site_xpos[w.rob["A1"].feet[1]].copy())
        feet.append(np.array(tr))
        assert w.stats["A1"]["kicks_cancelled"] == 0 and w.stats["A1"]["touches"] == 0
    assert np.array_equal(feet[0], feet[1])
    assert float(np.ptp(feet[0][:, 0])) > 0.2                     # the foot did swing (through the empty point)
    # a ball rolling at 1 m/s from the front, where and when the player said: struck, forward, no fall
    w = _alone()
    v, D = 1.0, 2.5
    v0 = math.sqrt(v * v + 2 * S.ROLL_DECEL * D)
    t_arr = (v0 - v) / S.ROLL_DECEL
    w.command("A1", dict(type="kick", power=1.0, aim=0.0, at=[3.0, 0.32, -0.06]))
    run_steps(w, int(round((3.0 - t_arr) / S.CONTROL_DT)))
    _roll(w, (0.32 + D, -0.06), (-v0, 0.0))
    vmax = 0.0
    for _ in range(int(4 / S.CONTROL_DT)):
        w.advance()
        vmax = max(vmax, float(w.ball_v()[0]))
    assert w.stats["A1"]["touches"] >= 1 and vmax > 1.5 and w.fallen["A1"] is None, (vmax, w.stats["A1"])
    # the aim across the support foot is clipped (right foot: + is across), away from it is not
    w = _alone()
    w.command("A1", dict(type="kick", power=0.5, aim=0.6))
    assert abs(w.prog["A1"]["aim"] - S.SWING["aim_in_max"]) < 1e-12
    w.command("A1", dict(type="kick", power=0.5, aim=-0.6))
    assert abs(w.prog["A1"]["aim"] + 0.6) < 1e-12
    w.command("A1", dict(type="kick", power=0.5, aim=0.6, foot="left"))
    assert abs(w.prog["A1"]["aim"] - 0.6) < 1e-12


def test_exploit_timed_kick_update_and_late():
    """The same timed kick sent again before its swing only moves its time and point (no new preparation, counted as
    kicks_retimed); a strike time shorter than the preparation is struck as soon as possible and the team is told
    (kick_late, own robots only); the timed kick obeys the real limits like every kick (E15 test's envelope)."""
    w = _alone()
    w.command("A1", dict(type="kick", power=1.0, at=[3.0, 0.32, -0.06]))
    run_steps(w, 60)
    kk = w.prog["A1"]["kicker"]
    w.command("A1", dict(type="kick", power=1.0, at=[2.0, 0.34, -0.05]))
    assert w.prog["A1"]["kicker"] is kk and w.stats["A1"]["kicks"] == 1 and w.stats["A1"]["kicks_retimed"] == 1
    run_steps(w, int(4 / S.CONTROL_DT))
    assert kk.late is not None and kk.late < 0.05 and w.stats["A1"]["kicks_late"] == 0, kk.late
    w = _alone()
    w.command("A1", dict(type="kick", power=1.0, at=[0.5, 0.32, -0.06]))
    run_steps(w, int(4 / S.CONTROL_DT))
    ev = [e for e in w.events if e["ev"] == "kick_late"]
    assert len(ev) == 1 and ev[0]["late_s"] > 1.5 and w.stats["A1"]["kicks_late"] == 1, ev
    m = P.SPORT.build(cfg("duel"))
    m.w.events.append(dict(ev="kick_late", k=m.w.k, robot="A1", late_s=2.0))
    assert any(e["ev"] == "kick_late" for e in m.events_for("A")) and \
        not any(e["ev"] == "kick_late" for e in m.events_for("B"))


def test_exploit_step_kick_needs_a_reachable_step():
    """The in-walk kick (style step, soccer-2v2-8) kicks only a ball its kicking foot's swing can reach from a step: a
    ball far off the foot's line is never kicked (the robot walks on, the kick is called off after the approach limit,
    kick_cancelled to its own team); a ball on the line is kicked within about 1.5 s, the robot does not fall; the kick
    obeys the real limits like every kick (the actuators are the only writer of the legs)."""
    w = _alone(x=-2.0)
    w.place_ball(-1.2, 1.0)                       # 1 m to the left of the walking line: out of reach
    mujoco.mj_forward(w.m, w.d)
    w.command("A1", dict(type="kick", style="step", power=1.0, v=[0.3, 0.0, 0.0]))
    run_steps(w, int(3.0 / S.CONTROL_DT))
    assert w.stats["A1"]["kicks_cancelled"] == 1 and w.ball_contact["A1"] < 0
    assert any(e["ev"] == "kick_cancelled" and e.get("robot") == "A1" for e in w.events)
    w = _alone(x=-2.0)
    b0 = np.array([0.0, 0.0])
    w.place_ball(*b0)
    mujoco.mj_forward(w.m, w.d)
    r = w.rob["A1"]
    while float(w.ball_xy()[0] - r.xy(w.d)[0]) > 0.6:      # walk up, the ball on the right foot's line, facing it
        yaw = r.yaw(w.d)
        rel = w.ball_xy() - r.xy(w.d)
        left = -math.sin(yaw) * rel[0] + math.cos(yaw) * rel[1]
        w.command("A1", dict(type="walk", v=[0.4, float(np.clip(1.5 * (left + S.STEP["line_m"]), -0.3, 0.3)),
                                             float(np.clip(-2.0 * yaw, -0.5, 0.5))]))
        w.advance()
    k0 = w.k
    w.command("A1", dict(type="kick", style="step", power=1.0))
    run_steps(w, int(3.0 / S.CONTROL_DT))
    assert w.ball_contact["A1"] > k0 and (w.ball_contact["A1"] - k0) * S.CONTROL_DT < 1.6
    assert float(np.hypot(*(w.ball_xy() - b0))) > 1.0      # the ball went
    assert w.fallen["A1"] is None and w.stats["A1"]["falls"] == 0


# ------------------------------------------------------------------------------------------------ team_match
def test_team_match_format():
    """team_match: one seat per robot of team A (A1, A2: one side) against the scripted team B of the level; each seat
    commands only its own robot, the team channel is on for the two seats, a level outside 1-3 is refused."""
    from core.channel import TeamChannel
    c = cfg("team_match", level=3)
    m = P.SPORT.build(c)
    assert m.single and m.level == 3 and set(m.w.coaches) == {"B"}
    assert m.entities("A1") == ("1",) and m.entities("A2") == ("2",)
    assert c.side_list() == ["A"] and m.side_of_team("B") == P.OPP
    assert TeamChannel(c, P.DECL.get("team_channel")).enabled
    try:
        P.SPORT.build(cfg("team_match", level=9))
        raise AssertionError("level 9 accepted")
    except ValueError:
        pass


# ------------------------------------------------------------------------------------------------ observation
def test_observe_during_step_kick():
    """A robot in the middle of an in-walk kick (style step) is observed like any other (soccer-2v2-9 raised KeyError
    'n0' in World.self_state, which read the push kick's fields): match, duel and team, through core.local's views
    (the whitelisted observation), at every phase of the kick."""
    from core import local
    from core import spec as spec_mod
    plugin_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "plugin.py")
    decl = spec_mod.load(plugin_file)
    for fmt in ("match", "duel", "team", "team_match"):
        c = cfg(fmt, level=2 if fmt in ("match", "team_match") else None)
        m = P.SPORT.build(c)
        views = local.Views(P.SPORT, c, m, decl, False, plugin_file)
        seat = "A1" if fmt.startswith("team") else "A"
        m.w.ko, m.w.state = None, "play"
        m.w.command("A1", dict(type="kick", style="step", power=1.0, v=[0.3, 0.0, 0.0]))
        phases = set()
        for _ in range(int(2.5 / S.CONTROL_DT)):
            m.step()
            v = views.view(seat, {})
            cmd = v["state"]["robots"]["1"]["command"]
            if cmd["type"] == "kick":
                assert cmd["style"] == "step" and len(cmd["v"]) == 3, cmd
                phases.add(cmd["phase"])
        assert phases, fmt
        assert not views.leaks, views.leaks


if __name__ == "__main__":
    import time
    for name, f in list(globals().items()):
        if name.startswith("test_") and (len(sys.argv) < 2 or name in sys.argv[1:]):
            t = time.time()
            f()
            print(f"ok   {name} ({time.time() - t:.1f} s)", flush=True)

