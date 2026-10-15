"""soccer_ssl plugin tests (line E): the skill contract against the robot and the field, the schema generated from the
contract, the observation (whitelist, no truth), the vision delay, the action layer's refusals, the scripted team's
checks, the units and the tiebreaks, the robot handler, and the [[exploits]] list of sport.toml. A first feasibility
probe (pass, receive, shoot with the platform's skills, truth-driven) is the [modes] b9_test.

Run: python3 sports/soccer_ssl/tests/test_plugin.py [name ...]   (in the sport's image: mujoco, numpy)
The generic tests (determinism, replay, fuzzing, frame budget) are core/testkit.py's (tools/check stage 2)."""

from __future__ import annotations

import importlib.util
import json
import math
import os
import sys
import time
import tomllib

HERE = os.path.dirname(os.path.abspath(__file__))
SPORT_DIR = os.path.dirname(HERE)
ROOT = next(
    (
        d
        for d in (os.path.dirname(os.path.dirname(SPORT_DIR)), os.path.dirname(SPORT_DIR))
        if os.path.isdir(os.path.join(d, "core"))
    ),
    None,
)
for d in (ROOT, SPORT_DIR):
    if d and d not in sys.path:
        sys.path.insert(0, d)

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from core.sport import MatchConfig, Refused, parse_action  # noqa: E402


def _plugin():
    if "ssl_plugin_under_test" not in sys.modules:
        spec = importlib.util.spec_from_file_location("ssl_plugin_under_test", os.path.join(SPORT_DIR, "plugin.py"))
        mod = importlib.util.module_from_spec(spec)
        sys.modules["ssl_plugin_under_test"] = mod
        spec.loader.exec_module(mod)
    return sys.modules["ssl_plugin_under_test"]


P = _plugin()
import skill_api as K  # noqa: E402
from sim import field as F  # noqa: E402

DECL = tomllib.loads(open(os.path.join(SPORT_DIR, "sport.toml"), "rb").read().decode())
SEATS = {"match": ("A",), "duel": ("A", "B"), "team": tuple(DECL["formats"]["team"]["seats"])}


def build(fmt="duel", seed=1, first="A", level=None, action_level="high"):
    f = DECL["formats"][fmt]
    cfg = MatchConfig(
        sport="soccer_ssl",
        format=fmt,
        seats=tuple(f["seats"]),
        seed=seed,
        first=first,
        level=level,
        action_level=action_level,
        sides=f.get("sides"),
    )
    return P.SPORT.build(cfg), P.SPORT.action_layer(action_level)


def act(m, layer, seat, body):
    """parse (the core's path) + apply at the boundary, as the service does."""
    return m.apply(seat, parse_action(layer, seat, body, m, {}))


def refused(m, layer, seat, body):
    try:
        parse_action(layer, seat, body, m, {})
    except Refused as e:
        return e.code
    return None


def steps(m, n):
    for _ in range(n):
        m.step()


def place(w, rid, x, y, yaw=0.0):
    w._reset_robot(rid, (x, y, yaw))
    mujoco.mj_forward(w.model, w.data)


def ball(w, x, y, vel=(0.0, 0.0, 0.0)):
    w.ball.place(x, y, F.BALL_R, vel=vel)
    mujoco.mj_forward(w.model, w.data)


def keys_of(o, out=None):
    out = set() if out is None else out
    if isinstance(o, dict):
        for k, v in o.items():
            out.add(k)
            keys_of(v, out)
    elif isinstance(o, list):
        for v in o:
            keys_of(v, out)
    return out


# ------------------------------------------------------------------------------------------------ the contract
def test_contract_ranges_match_the_robot_and_the_field():
    """skill_api.py's constants are bodies/ssl_robot/body.toml's limits and sim/field.py's walls."""
    body = tomllib.loads(open(os.path.join(ROOT, "bodies", "ssl_robot", "body.toml"), "rb").read().decode())
    L = body["limits"]
    assert K.BASE_SPEED == L["base"]["speed_m_s"] and K.BASE_ACCEL == L["base"]["accel_m_s2"]
    assert K.YAW_RATE == L["base"]["yaw_rate_rad_s"]
    assert K.FLAT_MAX == L["kicker"]["flat_speed_max_m_s"] and K.CHIP_MAX == L["kicker"]["chip_speed_max_m_s"]
    assert K.CHIP_ANGLE_DEG == L["kicker"]["chip_angle_deg"] and K.RPM_MAX == L["dribbler"]["speed_max_rpm"]
    assert (K.WALL_X, K.WALL_Y, K.ROBOT_R) == (F.WALL_X, F.WALL_Y, F.ROBOT_R)
    assert K.MOVE.notes["limits"]["jerk_m_s3"] == L["base"]["jerk_m_s3"]
    assert K.MOVE.notes["limits"]["yaw_accel_rad_s2"] == L["base"]["yaw_accel_rad_s2"]
    assert K.KICK.notes["limits"]["speed_sigma_rel"] == L["kicker"]["speed_sigma_rel"]
    assert K.KICK.notes["limits"]["direction_sigma_deg"] == L["kicker"]["direction_sigma_deg"]
    assert K.KICK.notes["limits"]["charge_full_s"] == L["kicker"]["charge_full_s"]
    assert K.DRIBBLE.notes["limits"]["torque_max_nm"] == L["dribbler"]["torque_max_nm"]
    assert K.ROBOTS_PER_TEAM == P.N == 5


def test_schema_is_generated_from_the_contract():
    """the high level's act schema IS the contract's request schema (single source); /spec shows the contract."""
    for name, c in K.CONTRACTS.items():
        assert P.SCHEMA[name] == c.request_schema()[name]
    assert P.SCHEMA["substitute"]["to"]["values"] == K.SUBSTITUTE["args"]["to"]["values"]
    assert set(DECL["modes"]["high"]["skills"]) == set(K.CONTRACTS)
    m, _ = build("duel")
    spec = P.SPORT.spec(m.cfg, "A", m)
    assert set(spec["skill_contract"]) >= {"move", "kick", "dribble", "substitute", "batch"}
    json.dumps(spec)


def test_low_level_is_the_servo_interface():
    """the low level has no timed or moving arrival (that planning is the skill)."""
    m, low = build("duel", action_level="low")
    assert refused(m, low, "A", {"type": "move", "entity": "1", "x": 1, "y": 0, "theta": 0, "t": 3.0}) == 400
    assert act(m, low, "A", {"type": "move", "entity": "1", "x": -1, "y": 0, "theta": 0})["status"] == "ok"


# ------------------------------------------------------------------------------------------------ observation
def test_observation_fields_and_frames():
    """the observation is the whitelist; own robots in the team frame (yellow attacks -x in the world in half 1)."""
    m, layer = build("duel")
    steps(m, 5)
    allowed = set(DECL["observation"]["fields"])
    for seat in ("A", "B"):
        o = m.observe(seat)
        assert set(o) <= allowed, set(o) - allowed
        json.dumps(o)
    w = m.w
    p, _ = w.robot_pose(("yellow", 0))
    oy = m.observe("B")
    assert oy["frame"]["attack_sign"] == -1
    assert abs(oy["robots"]["0"]["pose"][0] + p[0]) < 1e-4  # team x = -world x
    assert oy["robots"]["0"]["pose"][0] < -4.0  # the keeper defends -x in its own frame


def test_exploit_no_world_state_in_observation():
    """E3: no ball or opponent coordinates, no seed, nothing of the scripted team; own events hide the kick's truth."""
    m, layer = build("match", seed=5)
    w = m.w
    rid = ("blue", 2)
    place(w, rid, -1.0, 0.0, 0.0)
    ball(w, *w.kick_point(rid))
    w.referee.force("RUNNING", ball=w.ball.pos)
    act(m, layer, "A", {"type": "kick", "entity": "2", "speed": 3.0, "armed_for": 0.5})
    steps(m, 60)
    o = m.observe("A")
    text = json.dumps(o)
    ks = keys_of(o)
    assert not ks & {
        "ball",
        "them_robots",
        "seed",
        "opponent_robots",
        "vel_ball",
        "v_ball",
        "dir_err_deg",
        "ball_body",
        "dribbler_engaged",
        "dribbler_deflection_rad",
    }, ks
    assert str(m.cfg.seed * 7919) not in text
    kicked = [e for e in o["events"] if e.get("kind") == "kicked"]
    assert kicked and set(kicked[0]) == {"kind", "robot", "t", "mode", "speed_cmd", "ball_in_beam"}, kicked
    assert set(o["robots"]) == {"0", "1", "2", "3", "4"}  # own robots only (they are the blue ones)
    assert "Placeholder" not in text and 'opponent": {"class' not in text


def test_exploit_images_are_delayed():
    """E4: render_state is the world 4 control steps ago (40 ms), right after a command too; both teams the same."""
    m, layer = build("duel")
    qs = []
    for i in range(12):
        if i == 6:
            act(m, layer, "A", {"type": "move", "entity": "2", "x": 0.0, "y": 2.0, "theta": 1.0})
        m.step()
        qs.append(np.array(m.w.data.qpos))
    shown = np.array(m.render_state()["qpos"])
    assert P.VISION_DELAY_STEPS == 4
    assert np.array_equal(shown, qs[-1 - 4]) and not np.array_equal(shown, qs[-1])
    assert m.observe("A")["vision"]["step_capture"] == m.k - 4
    assert m.cameras("A") == m.cameras("B")
    assert len(m.cameras("A")) == 4 and all(c["width"] == 800 and c["height"] == 600 for c in m.cameras("A"))


# ------------------------------------------------------------------------------------------------ actions
def test_exploit_bad_commands_are_refused():
    """E1: malformed, out of range, foreign robot, unknown skill, bad batch: HTTP 400, nothing changes."""
    m, layer = build("duel")
    d0 = m.state_digest()
    good = {"type": "move", "entity": "1", "x": 0.0, "y": 0.0, "theta": 0.0}
    bad = [
        {**good, "x": 5.5},
        {**good, "theta": 7.0},
        {**good, "x": float("nan")},
        {**good, "entity": "5"},
        {**good, "entity": "B1"},
        {k: v for k, v in good.items() if k != "entity"},
        {k: v for k, v in good.items() if k != "theta"},
        {**good, "t": None},
        {**good, "v_end": [1.0]},
        {**good, "speed": 2.0},
        {"type": "kick", "entity": "1", "speed": 9.0},
        {"type": "kick", "entity": "1", "speed": 3.0, "mode": "lob"},
        {"type": "kick", "entity": "1", "speed": 3.0, "armed_for": 11.0},
        {"type": "dribble", "entity": "1", "rpm": 20000},
        {"type": "substitute", "entity": "1", "to": "bench"},
        {"type": "pass", "entity": "1", "to": "2"},
        {"type": "batch", "commands": []},
        {"type": "batch", "commands": [good, good]},
        {"type": "batch", "commands": [good, {**good, "x": 9}]},
        {"type": "batch", "commands": [{k: v for k, v in good.items() if k != "entity"}]},
        {"type": "batch", "commands": [{"type": "batch", "commands": [good]}]},
        {"type": "batch", "commands": [dict(good, entity=str(i)) for i in range(5)] + [dict(good, entity="0")]},
        {"type": "batch", "commands": good},
    ]
    for b in bad:
        assert refused(m, layer, "A", b) == 400, b
    assert m.state_digest() == d0
    ok = {"type": "batch", "commands": [good, {"type": "kick", "entity": "1", "speed": 2.0, "armed_for": 1.0}]}
    assert refused(m, layer, "A", ok) is None


def test_exploit_off_field_robot_cannot_act():
    """E2: a robot off the field: refused (409); if it leaves between the request and the boundary, not carried out."""
    m, layer = build("duel")
    w = m.w
    w.remove_robot(("blue", 3))
    assert refused(m, layer, "A", {"type": "move", "entity": "3", "x": 0, "y": 0, "theta": 0}) == 409
    assert refused(m, layer, "A", {"type": "substitute", "entity": "3", "to": "on"}) is None
    cmd = parse_action(layer, "A", {"type": "dribble", "entity": "4", "rpm": 1000}, m, {})
    w.remove_robot(("blue", 4))
    assert "rejected" in m.apply("A", cmd)
    assert w.rigs[("blue", 4)].w_target == 0.0


def test_exploit_skills_are_actions_at_the_real_limits():
    """E5: a kick moves the ball only from the robot's own beam, along its heading; a target beyond the walls is
    clipped; the base keeps the real speed limit."""
    m, layer = build("duel")
    w = m.w
    w.referee.force("RUNNING", ball=w.ball.pos)
    rid = ("blue", 2)
    place(w, rid, -2.0, 1.0, 0.0)
    ball(w, -1.0, -1.0)
    act(m, layer, "A", {"type": "kick", "entity": "2", "speed": 6.0})  # no ball in the beam: nothing moves
    steps(m, 5)
    assert np.linalg.norm(w.ball.vel[:2]) < 1e-3
    ball(w, *w.kick_point(rid))
    m.step()  # recharge a little; the capacitor still has enough for 4 m/s
    heading = 0.7
    place(w, rid, -2.0, 1.0, heading)
    ball(w, *w.kick_point(rid))
    r = act(m, layer, "A", {"type": "kick", "entity": "2", "speed": 4.0})
    assert r["status"] == "ok", r
    steps(m, 3)
    v = w.ball.vel
    sp = float(np.hypot(v[0], v[1]))
    assert 3.4 < sp < 4.6, sp
    assert abs(math.atan2(v[1], v[0]) - heading) < math.radians(4.0)
    # a target beyond the walls is clipped; the robot never leaves the field; the speed limit holds
    r = act(m, layer, "A", {"type": "move", "entity": "1", "x": 4.95, "y": 3.45, "theta": 0.0})
    assert r["status"] == "clipped" and r["target"][:2] == [K.REACH_X, K.REACH_Y], r
    vmax = 0.0
    for _ in range(400):
        m.step()
        _, vel = w.robot_pose(("blue", 1))
        vmax = max(vmax, math.hypot(vel[0], vel[1]))
    p, _ = w.robot_pose(("blue", 1))
    assert abs(p[0]) <= F.WALL_X - F.ROBOT_R + 0.01 and abs(p[1]) <= F.WALL_Y - F.ROBOT_R + 0.01, p
    assert vmax <= K.BASE_SPEED * 1.02, vmax


def test_exploit_scripted_team_uses_the_same_checks():
    """E6: the scripted team's commands go through check_act: bad ones are refused and counted, it commands only its
    own (yellow) robots."""

    class Bad:
        def act(self, truth):
            assert truth["team"] == "yellow" and "ball" in truth
            return [
                {"type": "move", "entity": "1", "x": 99, "y": 0, "theta": 0},
                {"type": "move", "entity": "7", "x": 0, "y": 0, "theta": 0},
                {"type": "kick", "entity": "2", "speed": 3.0, "aim": 0.3},
                {"type": "move", "entity": "2", "x": 1.0, "y": 1.0, "theta": 0.0},
            ]

    m, _ = build("match")
    m.opp = Bad()
    blue = [m.w.robot_pose(("blue", n))[0] for n in range(5)]
    m.step()
    assert m.opp_refused == 3, m.opp_refused
    steps(m, 50)
    assert all(np.allclose(m.w.robot_pose(("blue", n))[0], blue[n], atol=1e-3) for n in range(5))
    tg = m.w.rigs[("yellow", 2)].state()["target"]
    assert abs(tg["x"] + 1.0) < 1e-9 and abs(tg["y"] + 1.0) < 1e-9  # yellow's frame: world = -team in half 1
    assert m.final_fields()["opponent"]["refused"] >= 3


def test_exploit_unit_end_freezes_the_world():
    """E7: a goal ends the unit: the phase is UNIT_OVER for every seat, an act at the boundary is not carried out,
    the world goes on only after resume()."""
    m, layer = build("duel")
    w = m.w
    for rid in w.ids:
        if rid != ("blue", 2):
            w.remove_robot(rid)
    rid = ("blue", 2)
    place(w, rid, 3.0, 0.0, 0.0)
    ball(w, *w.kick_point(rid))
    w.referee.force("RUNNING", ball=w.ball.pos)
    act(m, layer, "A", {"type": "kick", "entity": "2", "speed": 3.0, "armed_for": 0.3})
    for _ in range(300):
        m.step()
        if m.phase().state != "playing":
            break
    ph = m.phase()
    assert ph.state == "unit_over" and ph.reason == "goal" and set(ph.review) == {"A", "B"}, ph
    assert m.observe("A")["stoppage"]["kind"] == "goal" and m.observe("B")["stoppage"]["score"]["them"] == 1
    cmd = parse_action(layer, "A", {"type": "move", "entity": "2", "x": 0, "y": 0, "theta": 0}, m, {})
    assert "rejected" in m.apply("A", cmd)
    assert m.unit() == 1
    m.resume()
    assert m.phase().state == "playing" and m.observe("A")["events"] == []


def test_exploit_seats_command_only_their_robots():
    """E8: the team format's robot seat commands only its own robot (the default entity), a team seat its team's."""
    m, layer = build("team")
    assert m.entities("A2") == ("2",) and m.entities("B0") == ("0",)
    assert refused(m, layer, "A2", {"type": "move", "entity": "3", "x": 0, "y": 0, "theta": 0}) == 400
    r = act(m, layer, "A2", {"type": "move", "x": -0.5, "y": 0.3, "theta": 0})
    assert r["robot"] == 2 and m.w.rigs[("blue", 2)].target is not None
    r = act(m, layer, "B3", {"type": "dribble", "rpm": 500})
    assert r["robot"] == 3 and m.w.rigs[("yellow", 3)].w_target > 0
    assert set(m.observe("A2")["robots"]) == {"2"}
    m2, layer2 = build("duel")
    assert m2.entities("B") == ("0", "1", "2", "3", "4")
    act(m2, layer2, "B", {"type": "dribble", "entity": "1", "rpm": 800})
    assert m2.w.rigs[("yellow", 1)].w_target > 0 and m2.w.rigs[("blue", 1)].w_target == 0


def test_exploit_no_draw_tiebreaks():
    """E9: the tiebreak names are sport.toml's order; regular time, the shoot-out and the coin toss each decide."""
    from core.outcome import decide

    for how, score, so, want, name in (
        ("REGULAR_TIME", {"blue": 2, "yellow": 1}, None, "A", "goals"),
        ("SHOOTOUT", {"blue": 1, "yellow": 1}, {"blue": 3, "yellow": 4}, "B", "shootout"),
        ("SHOOTOUT", {"blue": 0, "yellow": 0}, {"blue": 7, "yellow": 7}, None, "coin_toss"),
    ):
        m, _ = build("duel")
        assert [n for n, _ in m.tiebreaks()] == DECL["tiebreaks"]["order"]
        ref = m.w.referee
        ref.score = dict(score)
        if so is not None:
            ref.shootout = dict(
                order=["blue", "yellow"],
                kicks={t: [] for t in ("blue", "yellow")},
                goals=dict(so),
                coin="yellow" if so["blue"] == so["yellow"] else None,
            )
        ref._end_match(how)
        m.step()
        out = decide(m, sides=["A", "B"], side_of=m.cfg.side_of)
        assert out.decided_by == name, (out, how)
        assert out.winner == (want or "B")
        assert m.observe("A")["result"]["winner"] == ("us" if out.winner == "A" else "them")


def test_exploit_substitution_rules():
    """E10: after a card any robot may be named (taken off at once); otherwise only from the substitution area; a
    robot comes back only when the team is below the allowed number."""
    m, layer = build("duel")
    w, ref = m.w, m.w.referee
    assert (
        act(m, layer, "A", {"type": "substitute", "entity": "3", "to": "off"})["status"] == "not_in_substitution_area"
    )
    ref.card("blue", "yellow", reason="TEST")
    m.step()
    assert m.observe("A")["referee"]["allowed_robots"]["us"] == 4
    assert "remove_robot_within_s" in m.observe("A")["referee"]
    r = act(m, layer, "A", {"type": "substitute", "entity": "3", "to": "off"})
    assert r["status"] == "ok" and not w.on_field[("blue", 3)], r
    m.step()
    assert act(m, layer, "A", {"type": "substitute", "entity": "3", "to": "on"})["status"] == "too_many_robots"
    assert "remove_robot_within_s" not in m.observe("A")["referee"]
    ref.yellow["blue"] = []
    m.step()
    assert act(m, layer, "A", {"type": "substitute", "entity": "3", "to": "on"})["status"] == "ok"
    assert w.on_field[("blue", 3)]
    m.step()
    ev = [e["kind"] for e in m.observe("A")["events"]]
    assert "ROBOT_SUBSTITUTED_OUT" in ev and "ROBOT_SUBSTITUTED_IN" in ev, ev


def test_nonstop_foul_cool_down_per_robot():
    """A robot jiggling the ball at the edge of the opponent defense area (a new contact every 0.1 s) is called once
    per 2 s (TIGERs AutoReferee BotInDefenseAreaDetector coolDownTime 2.0 s), not at every new contact: before the fix
    13 calls in 0.6 s (four cards in a second)."""
    m, layer = build("duel")
    w = m.w
    rid = ("blue", 2)
    for r in w.ids:
        if r != rid:
            w.remove_robot(r)
    place(w, rid, 4.5 - 1.0 - 0.05, 0.3, 0.0)  # partly inside the yellow defense area (x >= 3.5), facing it
    w.referee.force("RUNNING", ball=w.ball.pos)
    kx, ky = w.kick_point(rid)
    calls = []
    for i in range(250):
        if i % 10 == 0:
            ball(w, kx, ky)  # touch
        elif i % 10 == 5:
            ball(w, kx + 0.1, ky)  # apart
        w.step()
        calls = [e for e in w.referee.events if e["type"] == "ATTACKER_TOUCHED_BALL_IN_DEFENSE_AREA"]
        if i == 60:
            assert len(calls) == 1, [e["t"] for e in calls]
    assert 1 <= len(calls) <= 2, [e["t"] for e in calls]
    assert all(b["t"] - a["t"] >= 2.0 - 1e-9 for a, b in zip(calls, calls[1:]))
    assert w.referee.cards_given["blue"]["yellow"] == 0 and w.referee.cards_given["blue"]["red"] == 0


def test_stop_resumes_with_a_robot_stuck_near_the_defense_area():
    """A robot held in the 0.2 m zone around the opponent defense area during a stop: its fouls change the next
    command but do not restart the stop (game controller: change_gameevent.go), after the team's second such foul the
    robot handler moves it out (the GC halts for the handlers then), and the game resumes within the stop's limit
    (placement 1 s + STOP_MAX_S 10 s) instead of never (before the fix: 284 fouls, 500 s frozen)."""
    from sim import referee as R

    m, layer = build("duel")
    w, ref = m.w, m.w.referee
    rid = ("blue", 2)
    zone_x = F.HALF_L - F.DA_DEPTH - 0.1  # 0.1 m in front of the yellow defense area: inside the 0.2 m zone
    place(w, rid, zone_x, 0.0, 0.0)
    ref.force("RUNNING", ball=w.ball.pos)
    ref.stop(("FORCE_START", None), (0.0, 0.0))
    t0, left = w.data.time, None
    for _ in range(2000):
        w.rigs[rid].set_base_target(zone_x, 0.0, 0.0)  # the team keeps driving it back into the zone
        w.step()
        if ref.state != "STOP":
            left = w.data.time - t0
            break
    assert left is not None and left <= R.PLACE_DELAY_S + R.STOP_MAX_S + 0.5, left
    ev = [e["type"] for e in ref.events]
    assert "ROBOT_MOVED_BY_HANDLER" in ev, ev
    assert ev.count("ATTACKER_TOO_CLOSE_TO_DEFENSE_AREA") <= 6, ev.count("ATTACKER_TOO_CLOSE_TO_DEFENSE_AREA")


def test_cards_beyond_the_robots_do_not_crash():
    """More active cards than robots: the allowed number stops at 0, every robot is taken off after 10 s, the match
    goes on (no forced forfeit: the rulebook's is for a team that cannot field a legal robot, by committee)."""
    m, layer = build("duel")
    w, ref = m.w, m.w.referee
    ref.force("RUNNING", ball=w.ball.pos)
    ref.red["blue"] = 6
    for _ in range(1200):
        w.step()
    assert ref.allowed("blue") == 0 and not any(w.on_field[("blue", n)] for n in range(5))
    for _ in range(300):
        w.step()
    assert m.observe("A")["referee"]["on_field"]["us"] == 0 and ref.stage == "FIRST_HALF"


def test_exploit_onboard_mode_gives_no_global_pose():
    """E11, the onboard mode (formats *_onboard, S17): no global pose anywhere a player can read: no camera definitions
    in /spec (they hold world poses), no pose / vel / target in the robots' state (odometry and the told pose only),
    a move's reply echoes the commanded target; the targets are taken in the odometry frame, so an odometry error
    carries into where the robot goes; a referee move tells the robot its new pose; the front cameras are posed from
    the shown (40 ms old) frame."""
    from core import local

    m, layer = build("match_onboard")
    w = m.w
    assert type(m).__name__ == "SSLOnboardMatch" and m.cameras("A") == []
    spec = local.seat_spec(P.SPORT, m.cfg, "A", plugin=os.path.join(SPORT_DIR, "plugin.py"))
    assert "cameras" not in spec and spec["perception"] == "onboard" and "onboard_camera" in spec
    steps(m, 30)
    o = m.observe("A")
    for n, r in o["robots"].items():
        assert not set(r) & {"pose", "vel", "vel_body", "target"}, r
        assert {"odom", "odom_vel_body", "told"} <= set(r)
    # an odometry error of 0.3 m in x: the robot goes 0.3 m short of the commanded point (in the field)
    rid = ("blue", 2)
    p0, _ = w.robot_pose(rid)
    w.odo[rid].pose[0] += 0.3
    r = act(m, layer, "A", {"type": "move", "entity": "2", "x": -1.0, "y": 1.0, "theta": 0.0})
    assert r["target"] == [-1.0, 1.0, 0.0], r
    steps(m, 300)
    p1, _ = w.robot_pose(rid)
    assert abs(p1[0] - (-1.0 - 0.3)) < 0.03 and abs(p1[1] - 1.0) < 0.03, p1
    assert "target" not in m.observe("A")["robots"]["2"]
    # the referee moves a robot: its odometry and told pose are reset to the new pose
    w._reset_robot(rid, (0.5, -2.0, 1.0))
    st = w.own_state(rid)
    assert st["told"]["pose"] == [0.5, -2.0, 1.0] and abs(st["odom"][0] - 0.5) < 1e-9
    # cameras: one per robot on the field, from the frame the images show
    job, _ = m.camera_job("A")
    k, q = w.vision_frame()
    assert len(job["cams"]) == 5 and job["state"]["qpos"] == [float(x) for x in q] and k <= m.k - 4
    assert k % P.ONBOARD_PERIOD == 0


def test_onboard_scripted_team_commands_true_poses():
    """The onboard mode puts the AGENT's move targets into each robot's odometry frame; the scripted team reads the
    truth, so its targets are true team-frame poses: an odometry error of a scripted robot does not carry into where
    it goes (before this fix the scripted levels drifted with their unread odometry: 0.3-1 m after collisions)."""
    m, layer = build("match_onboard")
    w = m.w
    assert w.truth_frame_teams == {"yellow"}
    m.opp = None  # this test commands the scripted team's robot itself
    steps(m, 30)
    rid = ("yellow", 2)
    w.odo[rid].pose[0] += 0.3
    m.execute("yellow", {"type": "move", "entity": "2", "x": -1.0, "y": 1.0, "theta": 0.0})
    steps(m, 300)
    p1, _ = w.robot_pose(rid)
    s = w.sign("yellow")
    assert abs(p1[0] - s * -1.0) < 0.03 and abs(p1[1] - s * 1.0) < 0.03, p1


def test_overhead_formats_unchanged_by_the_onboard_mode():
    """The onboard machinery runs only in the *_onboard formats: the overhead formats draw no odometry random numbers
    and keep their vision frame (the goldens replay unchanged)."""
    m, _ = build("duel")
    assert not m.w.onboard and m.w.odo == {} and m.w.vision_period == 1
    steps(m, 10)
    assert m.w.vision_frame()[0] == m.k - 4


# ------------------------------------------------------------------------------------------------ feasibility
def test_b9_pass_receive_and_shoot_feasible():
    """[modes] b9_test, a first probe with the platform's own skills (an upper-bound player that knows the truth; the
    full B9 of proposal §7 is step ②): a 2.5 m flat pass at 3 m/s is received and held by a robot whose dribbler
    spins and that steps onto the ball's line once it sees the pass (the kick's direction scatter moves the ball a
    few cm); the receiver turns to the goal with the ball and its 5 m/s shot from 3 m scores."""
    m, layer = build("duel", seed=3)
    w = m.w
    keep = {("blue", 2), ("blue", 3)}
    for rid in w.ids:
        if rid not in keep:
            w.remove_robot(rid)
    place(w, ("blue", 2), -1.0, 0.0, 0.0)
    place(w, ("blue", 3), 1.5, 0.0, math.pi)
    ball(w, *w.kick_point(("blue", 2)))
    w.referee.force("RUNNING", ball=w.ball.pos)
    act(
        m,
        layer,
        "A",
        {
            "type": "batch",
            "commands": [
                {"type": "dribble", "entity": "3", "rpm": 3000},
                {"type": "kick", "entity": "2", "speed": 3.0, "armed_for": 0.3},
            ],
        },
    )
    held = 0
    for i in range(300):
        m.step()
        if i == 10:  # the upper-bound receiver reads the ball's line (truth) and steps onto it
            (bx, by, _), (vx, vy, _) = w.ball.pos, w.ball.vel
            y = float(by + vy / vx * (1.5 - K.ROBOT_R - bx))
            act(m, layer, "A", {"type": "move", "entity": "3", "x": 1.5, "y": y, "theta": math.pi})
        st = w.own_state(("blue", 3))
        held = held + 1 if st["ball_in_beam"] else 0
        if held >= 50:
            break
    assert held >= 50, "the pass was not received and held for 0.5 s"
    # turn to the goal with the ball in steps of 0.5 rad (a single half turn at the full 20 rad/s, 50 rad/s^2 would
    # pass the grip's 9.1 rad/s and lose the ball: the move skill has no turn-rate argument), then shoot
    p, _ = w.robot_pose(("blue", 3))
    for th in np.linspace(math.pi, 0.0, 8)[1:]:
        act(m, layer, "A", {"type": "move", "entity": "3", "x": p[0], "y": p[1], "theta": float(th)})
        steps(m, 30)
    steps(m, 50)
    assert w.own_state(("blue", 3))["ball_in_beam"], "the ball was lost while turning"
    act(m, layer, "A", {"type": "kick", "entity": "3", "speed": 5.0})
    for _ in range(200):
        m.step()
        if w.referee.score["blue"]:
            break
    assert w.referee.score["blue"] == 1, [e for e in w.events if e.get("type") not in ("skill", "robot_event")][-5:]


# ------------------------------------------------------------------------------------------------ runner
if __name__ == "__main__":
    names = [n for n in sorted(globals()) if n.startswith("test_") and callable(globals()[n])]
    pick = sys.argv[1:] or names
    bad = 0
    for n in pick:
        t = time.time()
        try:
            globals()[n]()
            print(f"PASS {n} ({time.time() - t:.1f} s)")
        except Exception as e:  # noqa: BLE001
            import traceback

            bad += 1
            print(f"FAIL {n}: {type(e).__name__}: {e}")
            traceback.print_exc()
    print(f"{len(pick) - bad} passed, {bad} failed")
    sys.exit(1 if bad else 0)
