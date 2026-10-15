"""drone: drone racing after FAI F9U on the v0.2 core (docs/framework_zh.md §4.2). Migrated from the legacy tree
(scripts/olympiad/drone, scripts/olympiad/shared/drone/sim) in stage 2; the simulator race.py (the quadrotor, its
attitude controller and rotors, the seeded course, wind and shutters, the gate judge, Heat and Event) is unchanged,
so the legacy recordings replay bit-exactly (golden/convert.py). The four groups of contract ② (core/sport.py):

  world        Drone: one race.Event per match. The Event holds the current Heat, and a Heat holds one race.Race per
               seat, each in its own MuJoCo world (the drones cannot touch and no seat sees the other's drone); one
               control step = race.Heat.step() (every seat's world 10 ms, 5 physics substeps)
  actions      low only: {"type": "fly", vx, vy, vz, yaw_rate, duration} -> race.Race.act (world-frame target
               velocity and yaw rate through the onboard attitude controller), applied at the next control step
  referee      a unit = a round (format racing: 3 qualifying rounds, solo) or a heat (format duel:
               the first seat with 2 heat wins, at most 5 heats, then the stated tiebreak chain of race.Event);
               phase / scores / tiebreaks; the world freezes after every unit for the reviews
  observation  observe = the seat's own telemetry and the referee's calls (no gate, panel or shutter coordinate, no
               seed); cameras = the FPV camera on the seat's own drone, as a world-frame camera definition at this
               instant, drawn by the task's renderer from scene() (the course without any drone) and render_state()
"""
from __future__ import annotations

import os
import re

import numpy as np

import race as R
from core.sport import FINISHED, PLAYING, UNIT_OVER, LowLayer, MatchBase, Phase, Refused, check_command, sha_json

HERE = os.path.dirname(os.path.abspath(__file__))
NUM = {"type": "number"}
SCHEMA = {"fly": {
    "vx": {**NUM, "min": -4.0, "max": 4.0, "unit": "m/s", "doc": "world-frame target velocity x (forward)"},
    "vy": {**NUM, "min": -4.0, "max": 4.0, "unit": "m/s", "doc": "world-frame target velocity y (left); |(vx, vy)| <= 4"},
    "vz": {**NUM, "min": -2.0, "max": 2.0, "unit": "m/s", "doc": "world-frame target velocity z (up)"},
    "yaw_rate": {**NUM, "min": -1.5, "max": 1.5, "unit": "rad/s", "doc": "heading rate (+ = to the left)"},
    "duration": {**NUM, "min": 0.05, "max": 1.0, "unit": "s",
                 "doc": "game seconds; then the target is zero velocity and the heading is held"}}}
FPV = {"name": "fpv", "body_position": [0.12, 0.0, 0.035], "fovy_deg": 70, "width": 320, "height": 240}
TIEBREAKS = ("heat_wins", "most_heat_wins", "best_results", "last_heat_winner", "lot")


def level_of(cfg) -> int:
    """The course level: the duel always flies level 2 (race.DUEL_LEVEL); the single format the task's level (1-3)."""
    if len(cfg.seats) == 2:
        return R.DUEL_LEVEL
    lv = int(cfg.level or 1)
    if lv not in R.LEVELS:
        raise ValueError("level must be 1, 2 or 3")
    return lv


def low_action(cmd) -> dict:
    return {"velocity": [cmd["vx"], cmd["vy"], cmd["vz"]], "yaw_rate": cmd["yaw_rate"], "duration": cmd["duration"]}


class Drone(MatchBase):
    def __init__(self, cfg):
        self.cfg, self.seats = cfg, tuple(cfg.seats)
        if len(self.seats) not in (1, 2) or (len(self.seats) == 2 and self.seats != ("A", "B")):
            raise ValueError("drone: one seat (A) or two (A, B)")
        self.duel = len(self.seats) == 2
        self.level = level_of(cfg)
        self.ev = R.Event(int(cfg.seed), self.level, self.seats)
        self.k, self.dt = 0, R.DT
        self.review = False                       # a unit just closed: the reviews are due before the next one

    # ------------------------------------------------------------------ world
    def step(self):
        if self.review or self.ev.over:
            raise RuntimeError("the world is frozen")
        heat = self.ev.heat
        before = {s: (r.passed, r.done) for s, r in heat.races.items()}
        heat.step()
        self.k += 1
        events = []
        for s, r in heat.races.items():
            events += [{"kind": "gate", "seat": s, "gate": g + 1, "t": r.gate_times[g]} for g in range(before[s][0],
                                                                                              r.passed)]
            if r.done and not before[s][1]:
                events.append({"kind": "flight_end", "seat": s, "reason": r.reason, "gates": r.passed})
        if heat.done:
            summary = self.ev.close_heat()
            self.review = not self.ev.over
            events.append({"kind": "heat", "number": summary["number"], "winner": summary["winner"]})
        return events

    def last_heat(self):
        """The Heat whose worlds hold the final state (the legacy record's final_hash): the last closed heat once the
        event is over, else the current one (a fresh heat during a review)."""
        return self.ev.closed_heats[-1] if self.ev.over else self.ev.heat

    def state_digest(self):
        """Every seat's world (race.Race.state_hash: MuJoCo state, rotors, command, judge) and every closed heat's
        result. The legacy record's final_hash is the "races" part (golden/convert.py)."""
        races = {s: r.state_hash() for s, r in self.last_heat().races.items()}
        return sha_json({"races": races, "heats": self.ev.closed})

    # ------------------------------------------------------------------ actions
    def schema(self, seat):
        return SCHEMA

    def apply(self, seat, cmd):
        heat = self.ev.heat
        if self.review or self.ev.over:
            return {"rejected": "the world is frozen"}
        race = heat.races[seat]
        if race.done:
            return {"rejected": "your flight of this unit is over"}
        heat.act(seat, low_action(cmd))
        return {"applied_tick": heat.k, "heat": heat.number}

    # ------------------------------------------------------------------ referee
    def phase(self):
        if self.ev.over:
            return Phase(FINISHED, reason="match_over")
        if self.review:
            last = self.ev.closed[-1]
            return Phase(UNIT_OVER, review=self.seats, reason=f"heat {last['number']} over ({last['end']})")
        return Phase(PLAYING)

    def resume(self):
        self.review = False

    def unit(self):
        return len(self.ev.closed)

    def scores(self):
        if self.duel:
            return self.ev.wins()
        return {"A": self.ev.standing("A")["finished"]}

    def tiebreaks(self):
        """race.Event.duel_winner as a chain (F9U C.6.5.3, C.6.5, C.6.3 a; the lot is the last resort)."""
        if not self.duel:
            return []

        def by(how):
            def rule(m):
                w, h = m.ev.duel_winner()
                return w if h == how else None
            return rule
        return [(name, by(name)) for name in TIEBREAKS]

    # ------------------------------------------------------------------ observation
    def observe(self, seat, query=None):
        heat, ev = self.ev.heat, self.ev
        race = heat.races[seat]
        st = race.status()
        out = {"heat": heat.number, "heat_tick": heat.k, "tick": st["tick"], "time_s": st["time_s"],
               "gates_passed": st["gates_passed"], "gates_total": st["gates_total"], "gate_times": st["gate_times"],
               "flight_over": st["done"], "flight_reason": st["reason"], "finished": st["finished"],
               "time_limit_s": st["time_limit_s"], "robot": st["robot"], "command": st["command"],
               "command_until_tick": st["command_until_tick"], "event_over": ev.over, "results": ev.flights(seat),
               "standing": ev.standing(seat), "level": self.level}
        if ev.closed:
            last = ev.closed[-1]
            out["last_heat"] = {"number": last["number"], "result": last["results"][seat]}
            if self.duel:
                other = "B" if seat == "A" else "A"
                out["last_heat"].update(winner=last["winner"], opponent_result=last["results"][other])
        if self.duel:
            other = "B" if seat == "A" else "A"
            w = ev.wins()
            out["heat_wins"] = {"you": w[seat], "opponent": w[other]}
        return out

    def cameras(self, seat, query=None):
        """The seat's FPV camera (race.scene_xml: 0.12 m ahead of and 0.035 m above the centre, looking along the body
        x axis, image right = body -y) as a world-frame definition at this instant."""
        r = self.ev.heat.races[seat]
        rot = np.asarray(r.data.xmat[r.body]).reshape(3, 3)
        pos = np.asarray(r.data.xpos[r.body]) + rot @ np.asarray(FPV["body_position"])
        return [{"name": "fpv", "pos": pos.tolist(), "lookat": (pos + rot[:, 0]).tolist(), "up": rot[:, 2].tolist(),
                 "fovy_deg": FPV["fovy_deg"], "width": FPV["width"], "height": FPV["height"]}]

    def render_state(self):
        """The course scene has no drone (nq = 0); the moving shutters at the current time of the heat (the worlds that
        still fly share one clock; a seat whose flight is over sees the shutters move on)."""
        races = list(self.ev.heat.races.values())
        r = max(races, key=lambda x: float(x.data.time))
        return {"qpos": [], "mocap_pos": np.asarray(r.data.mocap_pos).ravel().tolist()}

    def broadcast_frame(self):
        """The legacy spectator frame (broadcast_log: q, h = heat, m = shutters; the duel's qA / qB)."""
        races = self.ev.heat.races
        first = races[self.seats[0]]
        extra = {"h": self.ev.heat.number, "m": np.round(first.data.mocap_pos, 4).ravel().tolist()}
        if self.duel:
            return {"q": [], "qA": np.round(races["A"].data.qpos, 5).tolist(),
                    "qB": np.round(races["B"].data.qpos, 5).tolist(), **extra}
        return {"q": np.asarray(first.data.qpos).tolist(), **extra}

    def final_fields(self):
        summary = self.ev.summary()
        return {"version": R.VERSION, "kind": self.ev.kind, "level": self.level, "fingerprint": self.ev.fingerprint,
                "event": summary, "event_over": self.ev.over,
                "final_hash": {s: r.state_hash() for s, r in self.last_heat().races.items()}}


class FlyLayer(LowLayer):
    """The only action level: the schema's bounds plus race.validate_action (horizontal speed <= 4 m/s); an act for a
    seat whose flight of this unit is over is refused (409), as in the legacy service."""

    def __init__(self):
        super().__init__(None)

    def parse(self, seat, body, match, ctx):
        cmd = check_command(SCHEMA, body)
        if "entity" in cmd:
            raise Refused("the drone takes no entity")
        try:
            R.validate_action(low_action(cmd))
        except (TypeError, ValueError) as exc:
            raise Refused(str(exc)) from exc
        if match.ev.heat.races[seat].done and not match.review:
            raise Refused("your flight of this unit is over; wait for the heat to end", 409)
        return cmd


def course_scene_xml(seed: int, level: int) -> str:
    """race.scene_xml of the event's course without the drone (its body, rotor sites, motors and the FPV camera): the
    players' and the spectators' renderer draws the course and the shutters (mocap) only, so no seat ever sees a
    drone (the duel's opponent flies in its own world)."""
    r = R.Race(seed, level, 1)
    xml = re.sub(r'<body name="drone".*?</body>', "", r.xml, count=1, flags=re.S)
    return re.sub(r"<actuator>.*?</actuator>", "", xml, count=1, flags=re.S)


class DroneSport:
    name, protocol, revision, source_dirs, workers = "drone", "realtime", "drone-race-4+core-0.2", [HERE], ("render",)

    def build(self, cfg):
        return Drone(cfg)

    def action_layer(self, level):
        return FlyLayer()

    def scene(self, cfg):
        from renderers.base import Scene
        return Scene(xml=course_scene_xml(int(cfg.seed), level_of(cfg)))

    def spec(self, cfg, seat, match):
        """race.Race.spec(): the public geometry and limits (no seed). Its "timing", "protocol" and "version" keys are
        renamed so that they do not shadow the core's /spec fields."""
        sp = match.ev.heat.races[seat].spec()
        sp.pop("protocol", None)
        out = {"simulator_version": sp.pop("version"), "race_timing": sp.pop("timing"), **sp}
        out["camera"]["world_definition"] = ("the `cameras` entry fpv is this camera's world pose when /spec was read; "
                                              "it moves with the drone (use your own pose from the state)")
        if match.duel:
            out["event_format"] = {"kind": "duel", "heats_to_win": R.HEATS_TO_WIN, "max_heats": R.MAX_HEATS,
                                   "opponent": "the other seat flies the same course at the same time in its own "
                                               "world: no contact, not visible, no progress during a heat; heat "
                                               "results after each heat"}
        else:
            out["event_format"] = {"kind": "single", "rounds": R.ROUNDS,
                                   "rule": "qualifying rounds: the best round ranks; second and third best split ties"}
        return out

    def players(self):
        import builtin_players as bp
        return {"reference": bp.Reference, "random": bp.Random}


SPORT = DroneSport()
