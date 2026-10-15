"""basketball's own tests: one per [[exploits]] entry of sport.toml (tools/check runs them through core/testkit.py,
after the generic suite). Run alone: python3 -m core.testkit tests sports/basketball/plugin.py (in the sport's image)."""
import math
import os
import re
import tempfile
import time

import numpy as np
from pathlib import Path

from core import local
from core.episode import load_plugin
from core.spec import load
from core.sport import MatchConfig, Refused

PLUGIN = str(Path(__file__).resolve().parents[1] / "plugin.py")
SPORT = load_plugin(PLUGIN)
DECL = load(PLUGIN)
import bball  # noqa: E402  (the plugin's directory is on sys.path now)
import court  # noqa: E402


def cfg(seed=1):
    return MatchConfig(sport="basketball", format="single", seats=("A",), seed=seed, first="A", action_level="low")


def shot(angle=54.0, speed=7.2, spin=1.5, yaw=0.0):
    return {"type": "shoot", "angle_deg": angle, "speed": speed, "spin_hz": spin, "yaw_deg": yaw}


def parse(m, seat, body):
    layer = SPORT.action_layer("low")
    return layer.lower(seat, layer.parse(seat, body, m, {"clock_left_s": {"A": 3600.0}}), None, m)[0]


def test_exploit_bad_actions_are_refused():
    m, layer = SPORT.build(cfg()), SPORT.action_layer("low")
    d0 = m.state_digest()
    for body in (shot(angle=80.5), shot(angle=19.9), shot(speed=2.9), shot(speed=14.1), shot(spin=5.5),
                 shot(yaw=-35.1), {"type": "shoot", "angle_deg": 54.0}, shot(speed=float("nan")), shot(speed="7"),
                 shot(spin=True), {"type": "throw", "speed": 7}, {**shot(), "power": 1}, [1, 2, 3]):
        try:
            layer.parse("A", body, m, {})
            raise AssertionError(f"accepted {body}")
        except Refused as e:
            assert e.code == 400
    assert m.state_digest() == d0 and m.k == 0
    assert layer.parse("A", shot(speed=7.123456789), m, {})["speed"] == 7.1235     # 4 decimals, as the legacy service
    assert "clock_left_s" not in layer.parse("A", shot(), m, {"clock_left_s": {"A": 1.0}})


def test_exploit_out_of_turn():
    from core.turn import TurnService
    os.environ["RCB_PLAYER_MODE"] = "oracle"
    with tempfile.TemporaryDirectory() as tmp:
        svc = TurnService(SPORT, PLUGIN, cfg(), Path(tmp))
        post = svc.routes(True)
        assert post["act"]("A", shot())[0] == 409                    # before the start: refused
        post["start"]("A", {})
        code, out = post["act"]("A", shot())
        assert code == 200 and svc.m.k == 1 and out["result"] in ("made", "rim", "backboard", "blocked", "airball")
        ob = out["observation"]                                      # the legacy picture strip (core 0.2.8 reply)
        assert ob.get("image_png_b64") and [(t["camera"], t["label"]) for t in ob["tiles"]] == [
            (c, f"{o:+d} ms") for c in ("rim", "shooter_zoom") for o in (-200, -100, 0, 100)], ob.get("tiles")
        assert {(t["width"], t["height"]) for t in ob["tiles"]} == {(400, 300)} and ob["tiles"][4]["y"] == 300
        assert post["act"]("A", shot())[0] == 409                    # the review comes first
        assert post["ready"]("A", {"why": "w", "plan": "p"})[0] == 200
        assert post["act"]("A", shot())[0] == 200 and svc.m.k == 2
        svc.collect()


def test_exploit_no_positions_in_observation_or_spec():
    for c in (cfg(seed=7), cfg(seed=2)):
        m = SPORT.build(c)
        for _ in range(5):
            m.apply(m.to_move(), parse(m, m.to_move(), shot(yaw=3.0)))
        text = "".join(repr(m.observe(s)) + repr(local.seat_spec(SPORT, c, s, m, PLUGIN)) for s in c.seats)
        secret = []
        for sp in bball.make_spots(c.seed):
            secret += [*sp["pos"], sp["r"], sp["heading"], sp["heading_offset"], math.degrees(sp["heading_offset"])]
        g = m.g
        secret += [v for r in g.results for v in r["release_point"][:2]]         # (its height is public: [spec])
        secret += [r["nearest_hoop_dist_m"] for r in g.results]
        secret += [r["defender_distance_m"] for r in g.results]
        for x in secret:
            if abs(x - round(x, 1)) < 1e-9:                          # 3.0 (the defender's cap) ...: public constants
                continue
            for form in (repr(x), f"{x:.4f}", f"{x:.3f}"):
                if form in ("0.000", "-0.000", "0.0000", "-0.0000"):
                    continue
                assert not re.search(rf"(?<![\d.]){re.escape(form)}(?!\d)", text), (x, form)
        for word in ("release_point", "heading", "nearest", "defender_distance", "pos_world"):
            assert word not in repr([m.observe(s) for s in c.seats]), word


def test_exploit_stalling_loses():
    from core.turn import TurnService
    os.environ["RCB_PLAYER_MODE"] = "oracle"
    old = os.environ.get("CLOCK_S")
    os.environ["CLOCK_S"] = "0.2"
    try:
        for c, want in ((cfg(), None),):
            with tempfile.TemporaryDirectory() as tmp:
                svc = TurnService(SPORT, PLUGIN, c, Path(tmp))
                post = svc.routes(True)
                for s in c.seats:
                    post["start"](s, {})
                time.sleep(0.5)
                code, out = post["act"]("A", shot())
                assert code == 200 and "clock_forfeit" in out and svc.m.k == 0
                o = svc.outcome()
                assert svc.reason == "clock" and o.winner == want, (svc.reason, o)
                t = time.time()
                while not svc.finalized and time.time() - t < 120:
                    time.sleep(0.2)
                assert svc.finalized
                f = svc.m.final_fields()
                assert f["clock_forfeit"] is True
    finally:
        if old is None:
            os.environ.pop("CLOCK_S", None)
        else:
            os.environ["CLOCK_S"] = old


def test_exploit_spec_matches_the_simulator():
    """Not a loophole but the same kind of guard: the prompt's rule constants ([spec]) are the simulator's own."""
    sp = DECL["spec"]
    assert {k: list(v) for k, v in bball.LIMITS.items()} == sp["limits"]
    assert sp["execution_scatter_1sigma"] == bball.NOISE
    assert sp["release_point_in_shooter_frame_m"] == [bball.REL_FWD, 0.0, bball.REL_H]
    assert sp["release_point_in_shooter_frame_m"] == [court.MUZZLE_FWD, 0.0, court.MUZZLE_H]
    assert sp["throw_to_release_s"] == bball.THROW_RELEASE_S
    assert sp["ball_radius_m"] == court.BALL_R and sp["ball_mass_kg"] == court.BALL_M
    assert sp["drag_coef_n_s2_m2"] == round(bball.C_D, 7) and sp["magnus_coef"] == round(bball.C_M, 8)
    assert sp["hoop_centre_m"] == list(court.HOOP) and sp["rim_inner_diameter_m"] == 2 * court.RIM_R_IN
    assert DECL["formats"]["single"]["success"]["score_at_least"] == bball.MAKES_NEEDED
    assert len(bball.Game(1).plan) == 10 and len(set(bball.Game(1).plan)) == 5


def test_exploit_defender_blocks():
    m = SPORT.build(cfg(seed=7))
    g = m.g
    yaw = -math.degrees(g.spot()["heading_offset"])                # straight at the hoop, flat: into the hands
    m.apply("A", parse(m, "A", shot(angle=22.0, speed=8.0, spin=0.0, yaw=round(yaw, 4))))
    r = g.results[-1]
    assert r["result"] == "blocked" and r["touched"][0] == "defender" and m.scores() == {"A": 0}


def test_exploit_machine_never_touches_the_ball():
    """D7: the shooting machine's motion is display only. No machine part has a contact pair (or collides at all),
    and a shot's flight and result do not depend on the machine's display path."""
    g = bball.Game(7)
    m = g.m
    machine = {i for i in range(m.ngeom) if m.geom_bodyid[i] in {m.body(n).id for n in ("shooter", "turret", "barrel")}}
    assert machine and all(m.geom_contype[i] == 0 and m.geom_conaffinity[i] == 0 for i in machine)
    assert not machine & ({int(m.pair_geom1[k]) for k in range(m.npair)} | {int(m.pair_geom2[k]) for k in range(m.npair)})
    dec = {"angle_deg": 54.0, "speed": 7.4, "spin_hz": 1.5, "yaw_deg": -6.0}

    def ball_track(game):
        out = []
        game.frame_cb = lambda gg, mode: out.append(gg.d.qpos[gg.bq:gg.bq + 7].copy())
        res = game.shoot(dict(dec))
        return res, out

    r1, t1 = ball_track(g)
    real = bball.machine_pose
    try:
        bball.machine_pose = lambda t, yaw, elev: (0.6 * math.sin(7 * t), -0.4 - 0.9 * abs(math.cos(5 * t)))  # any path
        g2 = bball.Game(7)
        r2, t2 = ball_track(g2)
    finally:
        bball.machine_pose = real
    rel = int(round(bball.THROW_RELEASE_S / bball.CONTROL_DT))
    assert r1 == r2, (r1, r2)
    assert len(t1) == len(t2) and all(np.allclose(a, b, rtol=0, atol=1e-9) for a, b in zip(t1[rel:], t2[rel:]))
