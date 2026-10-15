"""racing: 1:10-scale robot racing after the RoboRacer rules on the v0.2 core (docs/framework_zh.md §4.2, §10 stage 3).
Migrated from the legacy tree (scripts/olympiad/racing, scripts/olympiad/shared/racing/sim): the simulator and referee
race.py (MuJoCo cars on a seeded walled circuit with fixed obstacles, the crash and obstacle rules, heats, the
multi-car points classification, the scripted single-player car) is the legacy file (only its fingerprint helper and
docstring changed), so every legacy recording replays bit-exactly (golden/convert.py). The four groups of contract ②:

  world        RacingMatch: one race.Race per match; one control step = race.Race.step() (10 ms, 5 physics substeps)
  actions      low only: {"type": "drive", speed, steering} (the targets of the wheel-speed and steering servos, held
               until replaced) or {"type": "none"} (brake, straight); applied at the next control step
  referee      a unit = a heat. match: seat A against the scripted car (side "opponent"), up to 3 heats, two heat wins
               win; duel: seats A, B, the same rules; race: seats A.. (3-6 cars, the format declares 4), one heat per
               car with a rotating grid, F1 points and count-back (Match.ranking); a car that does not start, misses a
               review or is disqualified is retired (Match.retire) and the others race on. The world freezes after
               every heat for the reviews of the running cars.
  observation  observe = the seat's own car (speed, command, grid slot and side, held, retired) and the referee's
               information (race score, crash calls of this heat, the results of finished heats); no car pose, track
               progress or running order of any car during a heat, no seed. cameras = the car's front camera (or, with
               the observation query {"camera": "chase"}, the chase camera) as a world-frame definition at this
               instant, drawn by the task's renderer from scene() (race.scene_xml: the circuit and every car) and
               render_state() (qpos).
"""

from __future__ import annotations

import os

import numpy as np

import race as R
from core.sport import FINISHED, PLAYING, UNIT_OVER, LowLayer, MatchBase, Phase, Refused, check_command

HERE = os.path.dirname(os.path.abspath(__file__))
# Mesa llvmpipe (MUJOCO_GL=osmesa) starts about 2 x nproc threads per GL context: 65 per render worker on a 32-core
# host, so the 4 render workers of the race format exceeded the sim container's pids_limit (256) and the fresh-process
# replay could not be forked (Harbor: "Resource temporarily unavailable"). Four rasterizer threads per worker draw
# byte-identical images (checked for every car and both cameras) at about the same speed.
os.environ.setdefault("LP_NUM_THREADS", "4")
REVISION = "racing-v3+core-0.2"
OPP = "opponent"  # the scripted car's side in the match format (car B of race.py)
NUM = {"type": "number"}
SCHEMA = {
    "drive": {
        "speed": {
            **NUM,
            "min": R.MIN_SPEED,
            "max": R.MAX_SPEED,
            "unit": "m/s",
            "doc": "target forward speed of the rear wheels (negative: reverse); held until replaced",
        },
        "steering": {
            **NUM,
            "min": -R.MAX_STEER,
            "max": R.MAX_STEER,
            "unit": "rad",
            "doc": "front-wheel steering angle (positive turns left); held until replaced",
        },
    },
    "none": {},
}
# the cars' camera mounts (race.scene_xml): position in the car frame and MuJoCo xyaxes (image right, image up)
MOUNTS = {
    "front": ((0.22, 0.0, 0.24), (0.0, -1.0, 0.0), (0.342, 0.0, 0.940)),
    "chase": ((-1.5, 0.0, 1.35), (0.0, -1.0, 0.0), (0.58, 0.0, 0.815)),
}
CAM_FOVY = 70
# the core's retirement reasons -> the legacy record's (race.Race.retire keeps `why` in the state hash)
RETIRE_WHY = {"no_show": "did_not_start", "ready_timeout": "review_forfeit"}
TIEBREAKS = ("disqualification", "heat_wins", "points", "count_back", "lot")
CAMERA_SPEC = dict(
    width=R.CAM_W,
    height=R.CAM_H,
    fovy_deg=CAM_FOVY,
    cameras=["front", "chase"],
    front=dict(offset_m=0.22, height_m=0.395, pitch_down_deg=20),
    chase=dict(offset_m=-1.5, height_m=1.505, pitch_down_deg=35.4),
)


def config_of(cfg) -> dict:
    """race.Race's arguments for a match configuration: one seat = the match against the scripted car (level 1-4),
    two seats = the duel, 3-6 seats = the multi-car race (seats A, B, C, ... in that order)."""
    seats = tuple(cfg.seats)
    n = len(seats)
    if seats != R.ALL_SEATS[:n] or not (n in (1, 2) or R.RACE_CARS[0] <= n <= R.RACE_CARS[1]):
        raise ValueError("racing: seats A (match), A-B (duel) or A.. for 3-6 cars (race)")
    level = int(cfg.level or 1) if n == 1 else 1
    if level not in (1, 2, 3, 4):
        raise ValueError("level must be 1-4")
    first = cfg.first or "A"
    return dict(seed=int(cfg.seed), level=level, duel=n > 1, first=first, cars=max(2, n))


def mount_camera(race, seat, name):
    """A camera of race.scene_xml on `seat`'s car as a world-frame camera definition (MuJoCo orthonormalises the
    xyaxes: x = image right, y = image up, the camera looks along -z = y x x)."""
    off, x, y = (np.asarray(v, float) for v in MOUNTS[name])
    y = y - (y @ x) * x
    y /= np.linalg.norm(y)
    fwd = -np.cross(x, y)
    rot = np.asarray(race.data.xmat[race.body_id[seat]]).reshape(3, 3)
    pos = np.asarray(race.data.xpos[race.body_id[seat]]) + rot @ off
    return {
        "name": name,
        "pos": pos.tolist(),
        "lookat": (pos + rot @ fwd).tolist(),
        "up": (rot @ y).tolist(),
        "fovy_deg": CAM_FOVY,
        "width": R.CAM_W,
        "height": R.CAM_H,
    }


def public_crash(c):
    out = dict(t=c["t"], at_fault=c["at_fault"], kind=c["kind"], rule=c["fault_rule"])
    if "other" in c:
        out["other"] = c["other"]
    return out


def public_result(res, multi):
    """A finished heat as every seat may know it (the legacy service's public_result)."""
    out = dict(
        heat=res["heat"],
        winner=res["winner"],
        decided_by=res["decided_by"],
        heat_time=res["heat_time"],
        winner_finish_time=res["finish_time"],
        grid=res["grid"],
        cars={
            s: dict(checkpoints=v["checkpoints"], finish_time=v["finish_time"], obstacle_crashes=v["obstacle_crashes"])
            for s, v in res["cars"].items()
        },
        crashes=[public_crash(c) for c in res["crashes"]],
    )
    if multi:
        out.update(
            order=res["order"],
            positions=res["positions"],
            heat_points=res["heat_points"],
            points=res["points"],
            decided_by_car=res["decided_by_car"],
            retired=res["retired"],
        )
    return out


class Race(R.Race):
    """race.Race with two v0.2 tie rules (found by the fairness check with identical reference drivers): the heat-3
    grid of a head-to-head race (inner_seat, D3: the seed's draw) and the classification of the multi-car race (D10): points, then
    count-back, then a
    draw fixed by the seed. Legacy: the better place in the last heat after count-back. Every car starts once from
    every slot, so cars of equal pace tie on points and count-back, and the last heat's place then went to the car on
    pole in the last heat, a slot fixed by FIRST_PLAYER (the fairness check: the heat-1 pole car never won). Neither the
    classification nor the race winner is part of the state hash; the retired cars are ordered as before."""

    def inner_seat(self, heat):
        """Head-to-head heat 3 (deviation D3): the inner side goes to a draw fixed by the seed (the rulebook's random
        method). Legacy: the car with the faster winning lap of heats 1-2 (none or equal: the heat-1 grid). Between
        drivers of equal pace the faster lap is an artefact of when in its control period a heat starts, and in the
        fairness check it always went to the car that had started inside in heat 1, never to FIRST_PLAYER. Heats 1
        and 2 are the legacy rule; the draw for seed 1 is A, the heat-3 inner car of both legacy duel records."""
        if heat < 2:
            return super().inner_seat(heat)
        return R.lot(self.seed, heat, "inner")

    def classification(self):
        if not self.multi:
            return None
        pos = {s: [r["positions"].get(s) for r in self.results] for s in self.seats}

        def key(s):
            return (
                -self.points[s],
                *[-pos[s].count(i) for i in range(1, self.cars + 1)],
                R.lot_key(self.seed, 0, "classification", s),
            )

        run = sorted(self.active(), key=key)
        ret = sorted(
            self.retired,
            key=lambda s: (-self.retired[s]["step"], -self.points[s], R.lot_key(self.seed, 0, "retired", s)),
        )
        return run + ret


class RacingMatch(MatchBase):
    def __init__(self, cfg):
        self.cfg = cfg
        self.seats = tuple(cfg.seats)
        self.config = config_of(cfg)
        self.r = Race(**self.config)
        self.single = len(self.seats) == 1
        self.multi = self.r.multi
        self.dt = R.DT

    @property
    def k(self):
        return self.r.k

    def side(self, car):
        """race.py's car name -> the core's side (the scripted car B of the match format is "opponent")."""
        return OPP if self.single and car == "B" else car

    # ------------------------------------------------------------------ world
    def step(self):
        r = self.r
        if r.done:
            return []  # a review retirement ended the race (Match.retire): the core ends the match next
        r.step()
        return [dict(e) for e in r.events]

    def state_digest(self):
        """race.Race.digest(): the MuJoCo integration state and every referee variable (the legacy final hash)."""
        return self.r.digest()

    # ------------------------------------------------------------------ actions
    def schema(self, seat):
        return SCHEMA

    def apply(self, seat, cmd):
        r = self.r
        if seat in r.retired or r.done or r.awaiting_review:
            return {"rejected": "your car is retired" if seat in r.retired else "the world is frozen"}
        r.set_command(seat, [0.0, 0.0] if cmd["type"] == "none" else [float(cmd["speed"]), float(cmd["steering"])])
        return {}

    # ------------------------------------------------------------------ referee
    def phase(self):
        r = self.r
        if r.done:
            return Phase(FINISHED, reason="match_over")
        if r.awaiting_review:
            last = r.results[-1]
            return Phase(
                UNIT_OVER,
                review=tuple(s for s in self.seats if s not in r.retired),
                reason=f"heat {last['heat']} over ({last['decided_by']})",
            )
        return Phase(PLAYING)

    def resume(self):
        self.r.ready()

    def unit(self):
        return len(self.r.results)

    def scores(self):
        return {self.side(s): v for s, v in self.r.scores().items()}

    def tiebreaks(self):
        """Head-to-head (match, duel): a third warning loses the race (RoboRacer race violations), else the first to
        two heat wins (each heat is decided inside race.py: finish, progress, who reached it first, the seed's draw).
        Multi-car race (decided by Match.ranking; these rules give the same winner): points, count-back, the seed's
        draw."""

        def disqualification(m):
            r = m.r
            return m.side(r.winner_seat) if r.done and not r.multi and r.end_reason == "disqualification" else None

        def heat_wins(m):
            r = m.r
            return m.side(r.winner_seat) if r.done and not r.multi and r.end_reason != "disqualification" else None

        def leader(m, key):
            r = m.r
            if not (r.done and r.multi):
                return None
            run = r.active()
            if not run:
                return None
            best = min(key(m, s) for s in run)
            top = [s for s in run if key(m, s) == best]
            return top[0] if len(top) == 1 else None

        def pos(m, s):
            return [x["positions"].get(s) for x in m.r.results]

        rules = {
            "disqualification": disqualification,
            "heat_wins": heat_wins,
            "points": lambda m: leader(m, lambda m, s: -m.r.points[s]),
            "count_back": lambda m: leader(
                m, lambda m, s: (-m.r.points[s], *[-pos(m, s).count(i) for i in range(1, m.r.cars + 1)])
            ),
            "lot": lambda m: m.r.classification()[0] if m.r.done and m.r.multi else None,
        }
        return [(name, rules[name]) for name in TIEBREAKS]

    def ranking(self):
        """The multi-car classification (Race.classification: points, count-back, the seed's draw, then the retired
        cars, the later retirement first)."""
        return self.r.classification() if self.multi else None

    def retire(self, side, why):
        """N cars: a car whose seat never started (no_show) or missed a review (ready_timeout) is retired by race.py
        (parked off the track, classified after the running cars) and the others race on."""
        r = self.r
        if not self.multi or r.done or side not in r.seats:
            return False
        if side not in r.retired:
            r.retire(side, RETIRE_WHY.get(why, why))
        return True

    # ------------------------------------------------------------------ observation
    def observe(self, seat, query=None):
        camera_of(query)  # a bad observation query is refused (400)
        r = self.r
        race = dict(warnings={self.side(s): v for s, v in r.warnings.items()}, max_heats=r.max_heats)
        if self.multi:
            race.update(
                points=dict(r.points),
                points_per_place=list(R.POINTS[: r.cars]),
                cars=r.cars,
                retired={s: v["why"] for s, v in r.retired.items()},
            )
        else:
            race.update(heats_won={self.side(s): v for s, v in r.wins.items()}, heats_to_win=R.HEATS_TO_WIN)
        calls = []
        for c in r.heat_crashes:  # an obstacle call only to its own car (it would locate a car)
            if c["fault_rule"] == "obstacle" and c["at_fault"] != seat:
                continue
            x = public_crash(c)
            if c["fault_rule"] == "obstacle":
                x["obstacle"] = c["obstacle"]
            calls.append(x)
        own = dict(
            grid=r.grid[seat],
            grid_side=r.grid_side[seat],
            forward_speed=round(r.forward_speed(seat), 4),
            command=list(r.commands[seat]),
            held=bool(r.heat_k < r.hold[seat]),
            retired=seat in r.retired,
        )
        out = dict(
            heat=r.heat + 1,
            heat_time_s=round(r.heat_k * R.DT, 4),
            awaiting_review=r.awaiting_review,
            race=race,
            crashes_this_heat=calls,
            results=[public_result(x, self.multi) for x in r.results],
            own=own,
            race_over=r.done,
        )
        if r.done:
            out["race_winner"] = self.side(r.winner_seat) if r.winner_seat else None
            if self.multi:
                out["classification"] = r.classification()
        return out

    def cameras(self, seat, query=None):
        return [mount_camera(self.r, seat, camera_of(query))]

    def render_state(self):
        return {"qpos": np.asarray(self.r.data.qpos).tolist()}

    def final_fields(self):
        """The legacy final.json fields that the match core does not hold itself (reward keys, Season-1 readers)."""
        r = self.r
        return dict(
            config=self.config,
            level=self.config["level"],
            cars=r.cars,
            first_player=self.config["first"],
            results=r.results,
            crashes=r.crashes,
            classification=r.classification(),
            race_retired=r.retired,
            end_reason=r.end_reason,
            race_winner=self.side(r.winner_seat) if r.winner_seat else None,
            wall_steps=r.wall_steps,
            car_contact_steps=r.contact_steps,
            obstacle_contact_steps=r.obstacle_steps,
            heats=len(r.results),
            max_heats=r.max_heats,
            heat_seconds=r.heat_seconds,
            race_scores=r.scores(),
            opponent_car="B" if self.single else None,
        )


def camera_of(query) -> str:
    """The observation query: {"camera": "front"} (the default) or {"camera": "chase"}."""
    if not query:
        return "front"
    if set(query) - {"camera"} or query.get("camera", "front") not in MOUNTS:
        raise Refused('observation query: {"camera": "front"} or {"camera": "chase"}')
    return query.get("camera", "front")


class DriveLayer(LowLayer):
    """The only action level: the targets of the car's servos. A body without "type" is a drive command (the legacy
    `{"speed": S, "steering": D}` keeps working); the core's schema check bounds and rounds the numbers. A retired car
    cannot act (409)."""

    def __init__(self):
        super().__init__(None)

    def parse(self, seat, body, match, ctx):
        if isinstance(body, dict) and "type" not in body:
            body = {"type": "drive", **body}
        cmd = check_command(SCHEMA, body)
        if "entity" in cmd:
            raise Refused("a car takes no entity")
        if seat in match.r.retired:
            raise Refused("your car is retired", 409)
        return cmd


class Racing:
    name = "racing"
    protocol = "realtime"
    revision = REVISION
    source_dirs = [HERE]
    workers = ("render",)

    def build(self, cfg):
        return RacingMatch(cfg)

    def action_layer(self, level):
        return DriveLayer()

    def scene(self, cfg):
        from renderers.base import Scene

        c = config_of(cfg)
        return Scene(xml=R.scene_xml(c["seed"], c["cars"]))

    def spec(self, cfg, seat, match):
        """The legacy /spec content (no seed: it selects hidden state such as the draws; the circuit it selects is
        public geometry, the rulebook's map). Its timing part is `race_timing` (the core's `timing` is the clock)."""
        r = match.r
        c = r.circuit
        geo = c.summary()
        fmt = "race" if r.multi else "duel" if r.duel else "single"
        timing = dict(
            control_dt=R.DT, physics_dt=float(r.model.opt.timestep), heat_seconds=r.heat_seconds, max_heats=r.max_heats
        )
        if r.multi:
            timing["finish_window_s"] = R.FINISH_WINDOW_S
        else:
            timing["heats_to_win"] = R.HEATS_TO_WIN
        return dict(
            config=dict(
                level=r.level,
                format=fmt,
                cars=r.cars,
                first_player=r.first,
                seats=list(match.seats),
                opponent="scripted car B (side 'opponent')" if match.single else None,
            ),
            race_timing=timing,
            action=dict(
                type="drive",
                speed_m_s=[R.MIN_SPEED, R.MAX_SPEED],
                steering_rad=[-R.MAX_STEER, R.MAX_STEER],
                hold="until replaced; none brakes; one request in flight",
            ),
            camera=dict(CAMERA_SPEC, query='POST /observe {"query": {"camera": "chase"}} (default front)'),
            geometry=dict(
                track="circuit",
                length_m=geo["length_m"],
                width_m=geo["width_m"],
                direction="counter-clockwise",
                start_line="centreline[0]",
                centreline=geo["centreline"],
                centreline_columns=["x_m", "y_m", "width_m"],
                centreline_spacing_m=round(c.length / len(geo["centreline"]), 4),
                obstacles=[
                    dict(
                        id=o["id"],
                        kind=o["kind"],
                        x_m=o["x"],
                        y_m=o["y"],
                        yaw_rad=o["yaw"],
                        size_m=[0.34, 0.34, 0.4] if o["kind"] == "barrel" else [0.5, 0.4, 0.3],
                    )
                    for o in c.obstacles
                ],
                car_length_m=0.60,
                car_width_m=0.45,
                wheelbase_m=R.WHEELBASE,
                grid_lateral_m=R.LANE,
                grid_step_m=R.GRID_STEP if r.multi else 0.0,
                checkpoints=R.GATES,
            ),
            referee=dict(
                crash_speed_change_m_s=R.CRASH_DV,
                stuck_contact_s=R.STUCK_S,
                placed_behind_m=R.BEHIND_M,
                warnings_to_disqualify=R.WARNINGS_DQ,
                obstacle_hold_s=R.OBSTACLE_HOLD_S,
                points=list(R.POINTS[: r.cars]) if r.multi else None,
            ),
        )

    def players(self):
        import builtin_players as bp

        return {"reference": bp.Reference, "random": bp.Random}


SPORT = Racing()
