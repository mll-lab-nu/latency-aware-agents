"""racing's own tests: one per [[exploits]] entry of sport.toml (tools/check runs them through core/testkit.py after
the generic suite), from the legacy scripts/olympiad/racing/test_racing.py. They drive race.py with a trusted test
driver (not the reference: it knows the poses) and the plugin's interface. Run alone (in the sport's image):
python3 -m core.testkit tests sports/racing/plugin.py."""

import json
import math
from pathlib import Path

import numpy as np

from core.episode import load_plugin
from core.sport import FINISHED, UNIT_OVER, MatchConfig, Refused

PLUGIN = str(Path(__file__).resolve().parents[1] / "plugin.py")
SPORT = load_plugin(PLUGIN)
import race as R  # noqa: E402  (the plugin's directory is on sys.path now)
from race import LANE, MAX_STEER, WHEELBASE, Race  # noqa: E402


def cfg(fmt, seed=1, first="A", level=None):
    seats = {"match": ("A",), "duel": ("A", "B"), "race": ("A", "B", "C", "D")}[fmt]
    return MatchConfig(sport="racing", format=fmt, seats=seats, seed=seed, first=first, level=level, action_level="low")


def follow(r, s, speed, off=0.0, target=None, avoid=True):
    """A trusted test driver: pure pursuit on the centre line at lateral offset `off`, detouring around the obstacles
    (circuit.line) unless avoid=False."""
    p = r.position(s)[:2]
    yaw = r.heading(s)
    if target is None:
        sv, _ = r.project(p)
        c, t = r.point(sv + 1.05)
        lat = r.circuit.line(sv + 1.05, off) if avoid else off
        target = c + lat * np.array([-t[1], t[0]])
    d = target - p
    err = math.atan2(math.sin(math.atan2(d[1], d[0]) - yaw), math.cos(math.atan2(d[1], d[0]) - yaw))
    return [
        speed,
        float(np.clip(math.atan2(2 * WHEELBASE * math.sin(err), max(0.5, np.linalg.norm(d))), -MAX_STEER, MAX_STEER)),
    ]


def lane(r, s):
    return LANE if r.grid_side[s] == "inner" else -LANE


def run(r, policy, max_steps=None):
    while not r.done and (max_steps is None or r.k < max_steps):
        if r.awaiting_review:
            r.ready()
        for s, c in policy(r).items():
            if s not in r.retired:
                r.set_command(s, c)
        r.step()
    return r


def keys(x):
    if isinstance(x, dict):
        for k, v in x.items():
            yield k
            yield from keys(v)
    elif isinstance(x, list):
        for v in x:
            yield from keys(v)


def test_exploit_bad_commands_are_refused():
    m = SPORT.build(cfg("duel"))
    layer = SPORT.action_layer("low")
    d0 = m.state_digest()
    for body in (
        {"speed": float("nan"), "steering": 0},
        {"speed": 1, "steering": float("inf")},
        {"speed": 5, "steering": 0},
        {"speed": -1.6, "steering": 0},
        {"speed": 1, "steering": 0.49},
        {"speed": True, "steering": 0},
        {"speed": "1", "steering": 0},
        {"type": "drive", "speed": 1},
        {"type": "turbo"},
        {"speed": 1, "steering": 0, "boost": 1},
        {"type": "drive", "speed": 1, "steering": 0, "entity": "x"},
    ):
        try:
            layer.parse("A", body, m, {})
            raise AssertionError(f"accepted {body}")
        except Refused as e:
            assert e.code == 400, body
    assert m.state_digest() == d0
    assert layer.parse("A", {"speed": 2.0, "steering": -0.1}, m, {}) == {
        "type": "drive",
        "speed": 2.0,
        "steering": -0.1,
    }
    assert layer.parse("A", {"type": "none"}, m, {}) == {"type": "none"}
    rm = SPORT.build(cfg("race"))  # a retired car cannot act (409)
    assert rm.retire("D", "no_show")
    try:
        layer.parse("D", {"speed": 1.0, "steering": 0.0}, rm, {})
        raise AssertionError("a retired car acted")
    except Refused as e:
        assert e.code == 409


def test_exploit_reverse_and_wrong_way_earn_no_checkpoint():
    r = Race(duel=True, first="B", seed=1)
    r.set_command("A", [-1.0, 0.0])
    while not r.awaiting_review:
        r.step()
    assert r.next_gate["A"] == 1 and r.finished["A"] is None
    assert r.results[-1]["cars"]["A"]["checkpoints"] == 0


def test_exploit_cutting_through_obstacles_is_slower():
    times = {}
    for avoid in (True, False):
        r = Race(seed=404, duel=True)
        r.park("B")
        while not r.awaiting_review:
            s, _ = r.track_position("A")
            off = 0.0
            if not avoid:  # aim straight through the next obstacle
                ahead = [o for o in r.circuit.obstacles if 0 < (o["s"] - s) % r.length < 6]
                if ahead:
                    off = ahead[0]["lat"]
            r.set_command("A", follow(r, "A", 2.6, off, avoid=avoid))
            r.step()
        times[avoid] = (r.results[-1]["cars"]["A"]["finish_time"] or 1e9, r.obstacle_crashes["A"])
    assert times[True][1] == 0 and times[False][1] > 0
    assert times[True][0] < times[False][0]


def test_exploit_ramming_is_penalised_and_disqualifies():
    """The reviewer's experiment: B drives its line at 3.0 m/s, A chases it at 4.0 m/s and drives into it: every ram
    is A's crash, A is placed 2 m behind each time, and the third warning disqualifies A."""

    def rammer(r):
        ram = r.heat_k > 50
        return dict(
            A=follow(r, "A", 4.0, target=r.position("B")[:2]) if ram else follow(r, "A", 4.0, lane(r, "A")),
            B=follow(r, "B", 3.0, lane(r, "B")),
        )

    r = run(Race(seed=1, duel=True), rammer)
    assert r.winner() == "B" and r.warnings == {"A": 3, "B": 0} and r.end_reason == "disqualification"
    assert all(c["at_fault"] == "A" for c in r.crashes if c["fault_rule"] != "obstacle")
    m = SPORT.build(cfg("duel"))
    m.r = r
    assert dict(m.tiebreaks())["disqualification"](m) == "B"


def test_exploit_blocking_or_stalling_never_beats_the_scripted_car():
    for level in (1, 4):
        for name in ("stop_in_its_lane", "brake_check", "never_move", "reverse"):

            def agent(r, name=name):
                sa, _ = r.track_position("A")
                t = r.heat_k * R.DT
                if name == "stop_in_its_lane":
                    return dict(A=follow(r, "A", 3.0 if sa < 6 else 0.0, lane(r, "B")))
                if name == "brake_check":
                    return dict(A=follow(r, "A", 3.0 if int(t) % 3 else 0.0, lane(r, "B")))
                if name == "never_move":
                    return dict(A=[0.0, 0.0])
                return dict(A=[-1.5, 0.0])

            r = run(Race(seed=101, level=level), agent)
            assert r.winner() == "B" and r.warnings["B"] == 0, (level, name)


def test_exploit_world_state_hidden():
    for fmt in ("match", "duel", "race"):
        m = SPORT.build(cfg(fmt, seed=2))
        for _ in range(60):
            for s in m.seats:
                m.apply(s, {"type": "drive", "speed": 2.0, "steering": 0.0})
            m.step()
        for s in m.seats:
            obs = m.observe(s)
            bad = {
                k
                for k in keys(obs)
                if any(w in k.lower() for w in ("progress", "distance", "pos", "xy", "qpos", "gate", "order", "seed"))
            }
            assert not bad, bad
            sp = SPORT.spec(m.cfg, s, m)
            assert "seed" not in set(keys(sp)), fmt
            assert len(m.cameras(s)) == 1 and m.cameras(s)[0]["name"] == "front"
            assert m.cameras(s, {"camera": "chase"})[0]["name"] == "chase"
        try:
            m.observe("A", {"camera": "overview"})
            raise AssertionError("a camera outside the car was granted")
        except Refused:
            pass
    m = SPORT.build(cfg("duel"))  # another car's obstacle call is not shown during the heat
    m.r.heat_crashes.append(dict(t=1.0, at_fault="B", kind="impact", fault_rule="obstacle", obstacle=0))
    assert m.observe("A")["crashes_this_heat"] == [] and len(m.observe("B")["crashes_this_heat"]) == 1


def test_exploit_wedged_cars_are_separated():
    r = Race(seed=2, duel=True)
    r.place("A", 1.0, -LANE)
    r.place("B", 1.62, -LANE)
    r.set_command("A", [0.15, 0.0])
    r.set_command("B", [-0.15, 0.0])  # slow pushing against each other
    for _ in range(400):
        r.step()
        if r.crashes:
            break
    assert r.crashes and r.crashes[0]["kind"] in ("stuck", "impact")


def test_exploit_no_draws():
    r = Race(seed=5, duel=True)
    r.finished = {"A": 3.21, "B": 3.21}
    r.check_heat_end()
    assert r.results[-1]["decided_by"] == "lot" and r.results[-1]["winner"] == R.lot(5, 0, "finish")
    m = SPORT.build(
        MatchConfig(sport="racing", format="race", seats=("A", "B", "C"), seed=1, first="A", action_level="low")
    )
    r = m.r  # equal points and equal count-back: the seed's draw (not the last heat's place, D10)
    r.results = [
        dict(positions={"A": 1, "B": 2, "C": 3}),
        dict(positions={"A": 3, "B": 1, "C": 2}),
        dict(positions={"A": 2, "B": 3, "C": 1}),
    ]
    r.points = {"A": 58, "B": 58, "C": 58}
    want = sorted("ABC", key=lambda s: R.lot_key(1, 0, "classification", s))
    assert r.classification() == want
    r.done = True
    rules = dict(m.tiebreaks())
    assert rules["points"](m) is None and rules["count_back"](m) is None and rules["lot"](m) == want[0]
    assert m.ranking() == want
    r.points = {"A": 58, "B": 50, "C": 58}
    r.results[2]["positions"] = {"A": 1, "B": 3, "C": 2}  # A: two first places
    assert r.classification()[0] == "A" and rules["count_back"](m) == "A"
    m = SPORT.build(cfg("duel", seed=7))  # idle cars: every heat goes to progress or the lot
    while m.phase().state != FINISHED:
        if m.phase().state == UNIT_OVER:
            m.resume()
        m.step()
    assert m.r.winner() in ("A", "B") and all(x["winner"] in ("A", "B") for x in m.r.results)
    assert sum(1 for _, rule in m.tiebreaks() if rule(m) is not None) >= 1


def test_exploit_retirement_and_racing_on():
    m = SPORT.build(cfg("race"))
    assert m.retire("D", "no_show") and m.r.retired["D"]["why"] == "did_not_start"  # the legacy reason (hash)
    assert m.observe("D")["own"]["retired"] and m.phase().state != FINISHED
    assert m.apply("D", {"type": "drive", "speed": 2.0, "steering": 0.0}).get("rejected")
    k = m.k
    m.step()
    assert m.k == k + 1
    m.r.awaiting_review = True  # a review: B and C miss it
    m.r.results.append(
        dict(
            heat=1,
            decided_by="finish",
            positions={"A": 1, "B": 2, "C": 3, "D": 4},
            cars={s: dict(checkpoints=0, finish_time=None, obstacle_crashes=0) for s in "ABCD"},
        )
    )
    assert m.phase().review == ("A", "B", "C")
    assert m.retire("B", "ready_timeout") and m.r.retired["B"]["why"] == "review_forfeit"
    assert m.retire("C", "ready_timeout") and m.r.done  # one car left: the race is over
    k = m.k
    assert m.step() == [] and m.k == k and m.phase().state == FINISHED
    assert m.ranking()[0] == "A" and set(m.ranking()) == {"A", "B", "C", "D"}
    assert not SPORT.build(cfg("duel")).retire("B", "no_show")  # two cars: the core decides (forfeit)
    json.dumps(m.final_fields())
