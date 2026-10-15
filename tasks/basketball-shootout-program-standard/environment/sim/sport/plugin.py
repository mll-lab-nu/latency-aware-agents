"""basketball: the robot basketball shootout on the v0.2 core, wrapping the simulator (bball.py: the MuJoCo court, the
shooting machine's release (D7; its motion is display only), the ball's flight with drag and Magnus force, the static
defender, the make rule; court.py: the MJCF). One format: the single shootout, scored against the success line (review item B10,
2026-10-07: the legacy duel had no interaction, identical courts, so a head-to-head is the single scores on the same
seeds; its module bb_duel.py is gone). The four groups of contract ② (core/sport.py):
  world        bball.Game, a pure function of the seed and the commands
  actions      one low level: {"type": "shoot", angle_deg, speed, spin_hz, yaw_deg} (schema, 4 decimals as the legacy
               service)
  referee      phase / scores (makes) / to_move; no tiebreaks (scored against the success line of sport.toml)
  observation  observe (makes and results only: no hoop, ball, spot or shooter coordinates) + cameras: the four views
               at the current spot (shooter_wide, shooter_zoom on the shooter; broadcast, baseline fixed), drawn by the
               task's renderer from scene() (court.py's MJCF with the shooter as a mocap body) and render_state(); reply
               images (core 0.2.8): reply_views gives the reply to a shot the legacy picture strip (rim camera +
               shooter_zoom, 4 frames around the ball's closest approach to the hoop) from the shot's snapshots,
               drawn by the core with the task's renderer off the service lock
"""
from __future__ import annotations

import math
import os

import bball
import court
from core.spec import load
from core.sport import FINISHED, PLAYING, MatchBase, Phase, SchemaLayer

HERE = os.path.dirname(os.path.abspath(__file__))
DECL = load(HERE)
REVISION = DECL["sport"]["revision"]
SUCCESS_LINE = DECL["formats"]["single"]["success"]["score_at_least"]
KEYS = ("angle_deg", "speed", "spin_hz", "yaw_deg")
TILE = (640, 480)
STRIP_TILE = (400, 300)
STRIP_OFFSETS = (-20, -10, 0, 10)                # control steps (10 ms) around the closest approach
NUM = {"type": "number", "digits": 4}
DOCS = {"angle_deg": ("deg", "release elevation above horizontal"), "speed": ("m/s", "release speed"),
        "spin_hz": ("rev/s", "backspin (negative = topspin)"),
        "yaw_deg": ("deg", "release direction relative to the shooter's heading, + = left")}
SCHEMA = {"shoot": {k: {**NUM, "min": lo, "max": hi, "unit": DOCS[k][0], "doc": DOCS[k][1]}
                    for k, (lo, hi) in bball.LIMITS.items()}}


# ------------------------------------------------------------------------------------------------ cameras and scene
def _cam(name, pos, lookat, fovy, size):
    return {"name": name, "pos": [float(v) for v in pos], "lookat": [float(v) for v in lookat], "up": [0.0, 0.0, 1.0],
            "fovy_deg": float(fovy), "width": size[0], "height": size[1]}


def shooter_cams(pos, heading, size=TILE, names=("shooter_wide", "shooter_zoom")) -> list[dict]:
    """The two cameras on the shooter (court.py: SHOOTER_CAM_POS in its frame, pitched up, no roll) in world terms."""
    sx, sy, sz = court.SHOOTER_CAM_POS
    c, s = math.cos(heading), math.sin(heading)
    p = [pos[0] + sx * c - sy * s, pos[1] + sx * s + sy * c, sz]
    pt = math.radians(court.SHOOTER_CAM_PITCH_DEG)
    fw = [math.cos(pt) * c, math.cos(pt) * s, math.sin(pt)]
    fovy = {"shooter_wide": court.SHOOTER_WIDE_FOVY, "shooter_zoom": court.SHOOTER_ZOOM_FOVY}
    return [_cam(n, p, [p[i] + fw[i] for i in range(3)], fovy[n], size) for n in names]


def world_cam(name, size=TILE) -> dict:
    c = court.WORLD_CAMS[name]
    return _cam(name, c["pos"], c["target"], c["fovy"], size)


def views(game) -> list[dict]:
    """The four decision cameras of `game` at its current spot (the legacy 2 x 2 grid, left to right, top to bottom)."""
    sp = game.spot()
    return shooter_cams(sp["pos"], sp["heading"]) + [world_cam("broadcast"), world_cam("baseline")]


def cam_spec() -> list[dict]:
    """The cameras as /spec describes them (the legacy server_common.cam_spec): the shooter's cameras in the shooter's
    frame (its position and heading are not given), the fixed ones in the world frame."""
    focal = lambda fovy, h: round((h / 2) / math.tan(math.radians(fovy) / 2), 3)  # noqa: E731
    out = []
    for name, off in (("shooter_wide", [0, 0]), ("shooter_zoom", [TILE[0], 0])):
        fovy = court.SHOOTER_WIDE_FOVY if name == "shooter_wide" else court.SHOOTER_ZOOM_FOVY
        out.append({"name": name, "mount": "on the shooter (moves with it)",
                    "pos_in_shooter_frame": list(court.SHOOTER_CAM_POS), "pitch_up_deg": court.SHOOTER_CAM_PITCH_DEG,
                    "orientation": f"optical axis along the shooter's heading (+x of its frame), pitched up "
                                   f"{court.SHOOTER_CAM_PITCH_DEG} deg, no roll; image right = -y of its frame",
                    "fovy_deg": fovy, "tile_px": list(TILE), "tile_offset_px": off, "focal_px": focal(fovy, TILE[1]),
                    "principal_point_px": [TILE[0] / 2, TILE[1] / 2]})
    for name, c in court.WORLD_CAMS.items():
        tile = TILE if name != "rim" else STRIP_TILE
        out.append({"name": name, "mount": "fixed in the world", "pos": list(c["pos"]), "look_at": list(c["target"]),
                    "up_hint": [0, 0, 1], "fovy_deg": c["fovy"], "tile_px": list(tile),
                    "tile_offset_px": {"broadcast": [0, TILE[1]], "baseline": [TILE[0], TILE[1]]}.get(name),
                    "focal_px": focal(c["fovy"], tile[1]), "principal_point_px": [tile[0] / 2, tile[1] / 2],
                    **({"use": "row 1 of the shot's picture strip"} if name == "rim" else {})})
    return out


SHOOTER_TAG = '<body name="shooter" pos='


def scene_xml() -> str:
    """court.py's MJCF with the shooter a mocap body (the renderer places it per spot; the physics model of bball.Game
    moves it through body_pos instead). Mocap order: defender, shooter."""
    xml = court.build_xml()
    assert xml.count(SHOOTER_TAG) == 1
    return xml.replace(SHOOTER_TAG, '<body name="shooter" mocap="true" pos=')


def scene_assets() -> dict:
    """No mesh files: the court and the shooting machine are primitives (the Panda meshes left with the arm)."""
    return {}


def game_state(game) -> dict:
    """The render state of a court: the ball and the machine's display joints, the defender and the shooter (mocap)."""
    m, d = game.m, game.d
    return {"qpos": [float(v) for v in d.qpos],
            "mocap_pos": [[float(v) for v in d.mocap_pos[game.defender_mocap]],
                          [float(v) for v in m.body_pos[game.shooter]]],
            "mocap_quat": [[float(v) for v in d.mocap_quat[game.defender_mocap]],
                           [float(v) for v in m.body_quat[game.shooter]]]}


def snap_state(game, snap) -> dict:
    _, q, mp, mq, bp, bq = snap
    i = game.defender_mocap
    return {"qpos": [float(v) for v in q], "mocap_pos": [[float(v) for v in mp[i]], [float(v) for v in bp]],
            "mocap_quat": [[float(v) for v in mq[i]], [float(v) for v in bq]]}


def heading_of(quat) -> float:
    return 2.0 * math.atan2(float(quat[3]), float(quat[0]))


# ------------------------------------------------------------------------------------------------ the match
class Basketball(MatchBase):
    def __init__(self, cfg):
        self.cfg, self.seed, self.k = cfg, int(cfg.seed), 0
        self.g = bball.Game(self.seed)
        self.shots: list[dict] = []                   # every shot: seat + the simulator's result
        self.forfeit_rec = None                       # the core's decision (the game's hash does not hold it)

    # ------------------------------------------------------------------ world
    def state_digest(self):
        return self.g.state_sha()

    def schema(self, seat):
        return SCHEMA

    def apply(self, seat, cmd):
        res = self.g.shoot({k: cmd[k] for k in KEYS})
        self.shots.append({"seat": seat, **{k: v for k, v in res.items() if k != "nearest_snap"}})
        self.k += 1
        return {"res": res}

    def forfeit(self, seat, why):
        self.forfeit_rec = {"seat": seat, "why": why}
        return True

    # ------------------------------------------------------------------ referee
    def phase(self):
        return Phase(FINISHED, reason="all_shots_taken") if self.g.done else Phase(PLAYING)

    def to_move(self):
        return None if self.g.done else self.cfg.seats[0]

    def scores(self):
        return {self.cfg.seats[0]: self.g.made}

    def tiebreaks(self):
        return []                                                       # scored against the success line

    # ------------------------------------------------------------------ observation
    def next_shot(self):
        """The next shot as (shot index, spot index), or None (done)."""
        g = self.g
        return None if g.done else (g.shot, g.plan[g.shot])

    def observe(self, seat, query=None):
        g = self.g
        out = {"made": g.made}
        nxt = self.next_shot()
        if nxt is None:
            out["stage"] = "over"
        else:
            out.update({"stage": "shootout", "shot": nxt[0] + 1, "spot": nxt[1] + 1,
                        "attempt_at_this_spot": g.attempt_at_spot()})
        out.update({"shots_total": len(g.plan), "shots_left": len(g.plan) - g.shot, "success_line": SUCCESS_LINE})
        out["shots"] = [{"shot": x["shot"], "spot": x["spot"], "result": x["result"]} for x in self.shots]
        if self.shots:
            L = self.shots[-1]
            out["last_shot"] = {k: L.get(k) for k in ("shot", "spot", "result", "how", "touched")}
        return out

    def cameras(self, seat, query=None):
        """The four views at the current spot until the shootout is over."""
        return [] if self.next_shot() is None else views(self.g)

    def render_state(self):
        return game_state(self.g)

    def reply_views(self, seat, cmd, info):
        """The legacy picture strip of the shot just taken (server_common.strip_b64), as reply images (core 0.2.8):
        4 frames 100 ms apart around the ball's closest approach to the hoop, from the shot's snapshots (part of the
        game's state after the shot, so a replay has them too); row 1 the rim camera, row 2 the shooter's zoom camera
        (at the shooter's pose in that frame), 400 x 300 tiles. None if the shot left no snapshots."""
        g = self.g
        snaps = g.last_snaps
        if not snaps:
            return None
        i0 = info["res"].get("nearest_snap", len(snaps) - 1)
        if i0 is None or i0 < 0:
            i0 = len(snaps) - 1
        frames = []
        for off in STRIP_OFFSETS:
            sn = snaps[min(max(i0 + off, 0), len(snaps) - 1)]
            frames.append({"state": snap_state(g, sn), "label": f"{off * 10:+d} ms",
                           "cameras": [world_cam("rim", STRIP_TILE),
                                       shooter_cams(sn[4][:2], heading_of(sn[5]), STRIP_TILE, ("shooter_zoom",))[0]]})
        return {"frames": frames, "order": "camera", "columns": len(STRIP_OFFSETS),
                "note": "2 rows x 4 columns, 400 x 300 tiles, 100 ms apart around the ball's closest approach to the "
                        "hoop (label: ms from it); row 1: rim camera (side view of the hoop); row 2: your shooter_zoom "
                        "camera"}

    def final_fields(self):
        """Sport fields of final.json, under the legacy names the Season-1 readers use (live_board, regrade)."""
        g = self.g
        return {"made": g.made, "shots_taken": g.shot, "shots_total": len(g.plan), "makes_needed": SUCCESS_LINE,
                "complete": g.done, "clock_forfeit": bool(self.forfeit_rec and self.forfeit_rec["why"] == "clock"),
                "counts": {c: sum(1 for r in g.results if r["result"] == c)
                           for c in ("made", "rim", "backboard", "blocked", "airball")},
                "results": [{k: v for k, v in r.items() if k != "nearest_snap"} for r in g.results],
                "decisions": [a["decision"] for a in g.actions]}


class ShootLayer(SchemaLayer):
    """The one action level: the request body is checked against SCHEMA (4 decimals). The reply gives the shot's
    result (the core attaches the picture strip of Basketball.reply_views)."""

    def reply(self, seat, cmd, info, match):
        res = info["res"]
        return {"accepted": {k: cmd[k] for k in KEYS}, "result": res["result"], "how": res.get("how"),
                "touched": res["touched"], "made": match.g.made, "shots_left": len(match.g.plan) - match.g.shot}


class BasketballSport:
    name, protocol, revision, source_dirs, workers = "basketball", "turn", REVISION, [HERE], ()

    def build(self, cfg):
        return Basketball(cfg)

    def action_layer(self, level):
        return ShootLayer()

    def scene(self, cfg):
        from renderers.base import Scene
        return Scene(xml=scene_xml(), assets=scene_assets())

    def spec(self, cfg, seat, match):
        """The structured part of /spec (sport.toml [spec] holds the rule constants): the cameras as the legacy
        service described them (the shooter's in its own frame: the world-frame definitions would give away the
        shooter's position), the shots and the success line."""
        return {"cameras": cam_spec(), "spots": len(set(match.g.plan)), "attempts_per_spot": 2,
                "shots": len(match.g.plan), "success_line": SUCCESS_LINE}

    def players(self):
        import builtin_players as bp
        return {"reference": bp.Reference, "random": bp.Random}


SPORT = BasketballSport()
