"""hello: the minimal example sport and the core's test fixture (copy this directory to start a new sport).

Target throwing, turn-based. A ball is thrown from a fixed release point at a target painted on the ground 4-6 m away
(placed by the seed) and rolls to rest (MuJoCo). Points by the ring the ball rests in. Single: 5 balls alone against
the success line; duel: 5 balls each, alternating, the higher total wins (tiebreaks below). The rule constants are
data in sport.toml [spec] (also what /spec shows). The four groups of contract ② (core/sport.py), and nothing else:
  world        Hello (resting balls; each throw is simulated alone in a fresh MuJoCo data: balls never touch)
  actions      low: {"type": "throw", yaw, pitch, speed} (schema); high: {"type": "aim", x, y} lowered to a throw
               (the skill API `aim`: its contract is skill_api.py, the two modes are sport.toml [modes.high] / [modes.low])
  referee      phase / scores / tiebreaks / to_move
  observation  observe (referee information only: no ball or target position) + cameras (definitions; the task's
               renderer draws them from scene() and render_state()); optional reply images: reply_views shows the
               ball just thrown at rest (the overhead camera) in the reply to the throw
"""
from __future__ import annotations

import math
import os

import mujoco
import numpy as np

from core.rng import stream
from core.spec import load
from core.sport import FINISHED, PLAYING, MatchBase, Phase, SchemaLayer, check_command, sha_json
from skill_api import AIM as AIM_CONTRACT

HERE = os.path.dirname(os.path.abspath(__file__))
DECL = load(HERE)
S = DECL["spec"]
RELEASE, R_BALL, BALLS, EXTRA = S["release_point_m"], S["ball_radius_m"], S["balls_per_side"], S["sudden_death_pairs"]
RINGS, SCATTER = [(r["radius_m"], r["points"]) for r in S["rings"]], S["execution_scatter_1sigma"]
NUM = {"type": "number"}
SCHEMA = {"throw": {"yaw": {**NUM, "min": -20, "max": 20}, "pitch": {**NUM, "min": 0, "max": 60},
                    "speed": {**NUM, "min": 1, "max": 12}}}
AIM = AIM_CONTRACT.request_schema()                                    # the skill API's request = the high level's act
FRICTION = 'friction="0.9 0.05 0.1" condim="6"'
GROUND = f'<geom type="plane" size="20 20 .1" {FRICTION} rgba="0.33 0.5 0.3 1"/>'
FLIGHT = mujoco.MjModel.from_xml_string(
    f'<mujoco><option timestep="0.002"/><worldbody>{GROUND}<body pos="{" ".join(map(str, RELEASE))}"><freejoint/>'
    f'<geom type="sphere" size="{R_BALL}" mass="0.4" {FRICTION}/></body></worldbody></mujoco>')
CAMERAS = [{"name": "thrower", "pos": [-1.2, 0, 1.9], "lookat": [5, 0, 0], "fovy_deg": 40, "width": 320, "height": 240},
           {"name": "overhead", "pos": [5, 0, 7], "lookat": [5, 0, 0], "up": [1, 0, 0], "fovy_deg": 34, "width": 320,
            "height": 240}]


def roll(yaw, pitch, speed):
    """One throw simulated to rest in a fresh MjData -> (x, y) of the resting ball."""
    d = mujoco.MjData(FLIGHT)
    y, p = math.radians(yaw), math.radians(pitch)
    d.qvel[:3] = [speed * math.cos(p) * math.cos(y), speed * math.cos(p) * math.sin(y), speed * math.sin(p)]
    still = 0
    for _ in range(4000):
        mujoco.mj_step(FLIGHT, d)
        still = still + 1 if np.linalg.norm(d.qvel[:3]) < 0.01 else 0
        if still > 25:
            break
    return float(d.qpos[0]), float(d.qpos[1])


class Hello(MatchBase):
    def __init__(self, cfg):
        self.cfg, self.sides, self.k = cfg, cfg.side_list(), 0
        lay = stream(cfg, "layout")
        self.target = (float(lay.uniform(4.0, 6.0)), float(lay.uniform(-0.5, 0.5)))
        first = cfg.first if cfg.first in self.sides else self.sides[0]
        self.order = [first] + [s for s in self.sides if s != first]       # throwing order = ball colour
        self.throws: list[dict] = []                                        # {side, n, rest, r, points}

    def state_digest(self):
        return sha_json({"target": self.target, "order": self.order, "throws": self.throws})

    def schema(self, seat):
        return SCHEMA

    def apply(self, seat, cmd):
        side = self.cfg.side_of(seat)
        n, g = self.thrown(side), stream(self.cfg, f"scatter/{self.order.index(side)}/{self.thrown(side)}")
        x, y = roll(cmd["yaw"] + g.normal(0, SCATTER["yaw_deg"]), cmd["pitch"] + g.normal(0, SCATTER["pitch_deg"]),
                    cmd["speed"] * (1 + g.normal(0, SCATTER["speed_rel"])))
        r = math.hypot(x - self.target[0], y - self.target[1])
        pts = next((p for edge, p in RINGS if r < edge), 0)
        self.throws.append({"side": side, "n": n, "rest": [x, y], "r": r, "points": pts})
        self.k += 1
        return {"points": pts}

    def thrown(self, side):
        return sum(t["side"] == side for t in self.throws)

    def pts(self, side, n0=0, n1=BALLS):
        return sum(t["points"] for t in self.throws if t["side"] == side and n0 <= t["n"] < n1)

    def to_move(self):
        if self.phase().state == FINISHED:
            return None
        return self.cfg.seats_of(min(self.order, key=lambda s: (self.thrown(s), self.order.index(s))))[0]

    def phase(self):
        n = [self.thrown(s) for s in self.order]
        if min(n) < BALLS or n[0] != n[-1]:
            return Phase(PLAYING)
        if len(n) == 1 or self.decided() or min(n) >= BALLS + EXTRA:
            return Phase(FINISHED, reason="match_over")
        return Phase(PLAYING)                                               # level: a sudden-death pair

    def decided(self):
        return any(self.pts(self.sides[0], a, b) != self.pts(self.sides[1], a, b) for a, b in self.units())

    def units(self):
        return [(0, BALLS)] + [(BALLS + j, BALLS + j + 1) for j in range(self.thrown(self.sides[0]) - BALLS)]

    def scores(self):
        return {s: self.pts(s) for s in self.sides}

    def tiebreaks(self):
        if len(self.sides) == 1:
            return []                                                        # solo: the score against the line
        a, b = self.sides

        def first_unequal(m):
            return next(((a if m.pts(a, x, y) > m.pts(b, x, y) else b) for x, y in m.units()
                         if m.pts(a, x, y) != m.pts(b, x, y)), None)

        def closest(m):
            ra, rb = (min(t["r"] for t in m.throws if t["side"] == s) for s in (a, b))
            return None if ra == rb else (a if ra < rb else b)
        return [("total_then_sudden_death", first_unequal), ("closest_ball", closest),
                ("second_thrower", lambda m: m.order[1])]

    def observe(self, seat, query=None):
        me = self.cfg.side_of(seat)
        who = {s: "you" if s == me else "opponent" for s in self.sides}
        return {"ball": self.thrown(me) + 1, "points": {who[s]: self.pts(s, 0, 99) for s in self.sides},
                "colour": {who[s]: ("yellow", "blue")[self.order.index(s)] for s in self.sides},
                "throws": [{"by": who[t["side"]], "ball": t["n"] + 1, "points": t["points"]} for t in self.throws]}

    def cameras(self, seat, query=None):
        return CAMERAS

    def reply_views(self, seat, cmd, info):
        """Optional (core 0.2.8): the images of a throw's result in its reply. The ball is at rest now, so the
        state is render_state() (the core's default); a sport whose state moves on inside apply() (arrows pulled,
        the next visit) passes the state of the result per frame instead."""
        return {"frames": [{"cameras": [CAMERAS[1]]}], "note": "your ball at rest (overhead camera)"}

    def render_state(self):
        q = []
        for i in range(2):
            rest = [t["rest"] for t in self.throws if i < len(self.order) and t["side"] == self.order[i]]
            for j in range(BALLS + EXTRA):                                  # unused balls wait under the ground
                q += [*(rest[j] + [R_BALL] if j < len(rest) else [-3 - j * 0.2, 2.0 - 4 * i, -1.0]), 1, 0, 0, 0]
        return {"qpos": q}

    def final_fields(self):
        return {"target": list(self.target), "order": self.order, "throws": self.throws}

    def continuous_score(self):
        """Optional (core 0.2.14, sport.toml [score] from = "hook"): the mean points per ball of each side, 0-10
        (0 when a side threw nothing). The template of a continuous score the scores alone do not give."""
        out = {}
        for s in self.sides:
            mine = [t["points"] for t in self.throws if t["side"] == s]
            out[s] = sum(mine) / len(mine) if mine else 0.0
        return out


class AimLayer(SchemaLayer):
    """The high level (the template of a `lower` plugin): aim at a ground point; lowered to the throw of pitch 35 deg
    that lands there (the ball still rolls a little after landing)."""
    level = "high"

    def schema(self, match, seat):
        return AIM

    def parse(self, seat, body, match, ctx):
        return check_command(AIM, body)

    def lower(self, seat, cmd, prepared, match):
        dx, dy, h, p = cmd["x"] - RELEASE[0], cmd["y"] - RELEASE[1], RELEASE[2] - R_BALL, math.radians(35)
        dist = math.hypot(dx, dy)
        v = dist / math.cos(p) * math.sqrt(9.81 / (2 * (dist * math.tan(p) + h)))
        low = {"type": "throw", "yaw": round(math.degrees(math.atan2(dy, dx)), 6), "pitch": 35.0,
               "speed": round(min(12.0, v), 6)}
        return low, {"planned": low}


class HelloSport:
    name, protocol, revision, source_dirs, workers = "hello", "turn", DECL["sport"]["revision"], [HERE], ()

    def build(self, cfg):
        return Hello(cfg)

    def action_layer(self, level):
        return AimLayer() if level == "high" else SchemaLayer()

    def spec(self, cfg, seat, match):
        """/spec: the skill API's contract at both levels; at the low level also the deliverable and the toolkit
        (sport.toml [modes.low]: implement aim yourself, evaluated after the match)."""
        if cfg.action_level != "low":
            return {"skill_contract": AIM_CONTRACT.describe()}
        lo = DECL["modes"]["low"]
        return {"skill_contract": AIM_CONTRACT.describe(lo["deliverable"]),
                "toolkit": {"manifest": lo["toolkit_path"], "practice_requests": lo["cases"]["practice_path"],
                            "tools": lo["tools"], "forbidden": lo["forbidden"]}}

    def scene(self, cfg):
        from renderers.base import Scene
        tx, ty = Hello(cfg).target
        col = ["0.85 0.1 0.1 1"] + ["0.95 0.95 0.95 1", "0.25 0.25 0.25 1"] * 3
        rings = "".join(f'<geom type="cylinder" size="{r} {0.001 * (len(RINGS) - i)}" pos="{tx} {ty} 0" contype="0" '
                        f'conaffinity="0" rgba="{col[i]}"/>' for i, (r, _) in enumerate(RINGS))
        balls = "".join(f'<body><freejoint/><geom type="sphere" size="{R_BALL}" contype="0" conaffinity="0" '
                        f'rgba="{("1 0.85 0.1 1", "0.15 0.45 1 1")[i]}"/></body>' for i in range(2)
                        for _ in range(BALLS + EXTRA))
        return Scene(xml=f'<mujoco><visual><headlight ambient=".35 .35 .35" diffuse=".3 .3 .3"/></visual><worldbody>'
                         f'{GROUND}<light pos="2 0 8" dir="0 0 -1" diffuse=".4 .4 .4"/>{rings}{balls}</worldbody>'
                         '</mujoco>')

    def players(self):
        import builtin_players as bp
        return {"reference": bp.Reference, "random": bp.Random}


SPORT = HelloSport()
