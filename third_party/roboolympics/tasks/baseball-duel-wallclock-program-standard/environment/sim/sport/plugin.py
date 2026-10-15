"""baseball: robot baseball in real time (docs/framework_zh.md §4.2). Migrated from the Season-1 tasks
baseball-batting-wallclock-program-standard and baseball-duel-wallclock-program-standard (docs §10 stage 4). The four
groups of contract ②, on the legacy simulator (bb.py: ball flight, swing rig, bat-ball impact, fielders,
the scripted pitcher, the count; bbrt.py: the 100 Hz pitch clock of the batting game; bbduel.py: the pitcher-against-
batter duel on the same clock):
  world + referee  format batting = bbrt.RT (seat A bats 5 plate appearances against the scripted pitcher of level
                   1-4, which plays inside the world), format duel = bbduel.Duel (plate appearances alternate between
                   seats A and B, one bats and the other pitches); step = one 10 ms control step; a pitch is judged
                   in the step that ends it (the batted ball, the count) and the world freezes for the reviews; the
                   next pitch starts at resume()
  actions          one level, the rig's and the pitcher's own commands (the legacy commands, unchanged): `swing` (the
                   bat square T s after the command takes effect, plane, height, speed, reach), `pitch` (duel: type,
                   speed, target at the plate; thrown at the release with a seeded scatter) and `none`
  referee          phase / scores / tiebreaks (batting: none, a solo score against the 4-base line; duel: more bases,
                   then the winner of the last pitch)
  observation      observe = the legacy state (the count, the bases, what was heard, the seat's command, the last
                   pitch), pure; camera_job = the legacy 2 cameras x 4 frames 30 ms apart, drawn by the sport itself
                   (bb.Renderer, MuJoCo) in the seat's render worker

The state digest is the legacy state_sha() (bb.Game: the judged pitches; bbduel.Duel: the judged pitches and the step
count), so it changes once per pitch and the legacy golden records replay bit-exactly (golden/convert.py; the duel
records with pitches are retired since revision duel-2, which gives each batter's plate appearances their own scatter
stream, bbduel.Duel.slot).
"""
from __future__ import annotations

import json
import math
import os
import struct
import sys
import zlib

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import bb  # noqa: E402
import bbduel  # noqa: E402
import bbrt  # noqa: E402

from core.sport import FINISHED, PLAYING, UNIT_OVER, MatchBase, Phase, Refused, check_command  # noqa: E402

REVISION = "baseball-wallclock-3+duel-2+core-0.2"
SEATS2 = bbduel.SIDES
DT = bbrt.DT
ON_BASE = ("walk", "single", "double", "home run")       # plate-appearance results that win the pitch for the batter
IMAGE_LAYOUT = ("2 rows x 4 columns. Row 1: camera behind home plate (catcher's view) looking at the pitcher; "
                "row 2: camera beside the first part of the pitch path near the pitcher, looking along -x (the "
                "pitcher is on the right, home plate off-image to the left). Columns: four frames 30 ms apart, "
                "oldest left, the latest (the state at the request) right; each tile is stamped with its time since "
                "release (negative: the windup, the ball still in the pitcher's hand). The ball is the small bright "
                "white sphere (7.3 cm). Camera poses and intrinsics: `game spec`.")


class Baseball(MatchBase):
    def __init__(self, cfg):
        self.cfg = cfg
        self.single = len(cfg.seats) == 1
        self.dt = DT
        if self.single:
            self.level_ = 1 if cfg.level is None else int(cfg.level)
            if self.level_ not in bb.LEVELS:
                raise ValueError(f"pitcher level must be one of {sorted(bb.LEVELS)}")
            self.g = bbrt.RT(self.level_, int(cfg.seed))
        else:
            self.level_ = None
            self.g = bbduel.Duel(int(cfg.seed), cfg.first if cfg.first in SEATS2 else "A")

    @property
    def k(self):
        return self.g.k

    def history(self):
        return self.g.G.history if self.single else self.g.history

    def done(self):
        return self.g.G.done if self.single else self.g.done

    # ------------------------------------------------------------------ world + referee
    def step(self):
        g = self.g
        if g.over:
            raise RuntimeError("step() during a review: the pitch is over (resume first)")
        g.advance(1)
        if g.over and not g.judged:
            rec = g.finish()                       # the batted ball, the fielders, the count (the world is frozen)
            return [{"kind": "pitch_over", "pitch": rec["pitch"] + 1, "outcome": rec["outcome"],
                     "plate_appearance_result": rec.get("pa_result")}]
        return []

    def state_digest(self):
        return self.g.G.state_sha() if self.single else self.g.state_sha()

    def phase(self):
        if self.done():
            return Phase(FINISHED, reason="match_over")
        if self.g.over:
            return Phase(UNIT_OVER, review=tuple(self.cfg.seats), reason=self.history()[-1]["outcome"])
        return Phase(PLAYING)

    def resume(self):
        g = self.g
        if g.over and g.judged and not self.done():
            g.next_pitch()                         # the next pitch's flight (single) / windup (duel), frozen time

    def unit(self):
        return len(self.history())

    def scores(self):
        if self.single:
            return {self.cfg.seats[0]: int(self.g.G.bases)}
        return {s: int(self.g.score[s]) for s in SEATS2}

    def tiebreaks(self):
        """batting: none (a solo score against the line). duel: bbduel.Duel.winner as two rules: more bases (after
        the regulation plate appearances and up to 5 extra pairs), else (last resort, not from the rulebook) the
        winner of the last pitch: the batter if it ended the plate appearance on base, else the pitcher."""
        if self.single:
            return []

        def bases(m):
            a, b = m.g.score["A"], m.g.score["B"]
            return None if a == b else ("A" if a > b else "B")

        def last_pitch(m):
            if not m.g.history:
                return None
            last = m.g.history[-1]
            return last["batter"] if last.get("pa_result") in ON_BASE else last["pitcher"]
        return [("bases", bases), ("last_pitch", last_pitch)]

    def forfeit(self, seat, why):
        return False                               # the core decides a forfeit; the world is left as it was

    # ------------------------------------------------------------------ actions (one level)
    def schema(self, seat):
        return SINGLE_SCHEMA if self.single else DUEL_SCHEMA

    def apply(self, seat, cmd):
        typ = cmd.get("type")
        if self.single:
            if typ not in ("swing", "none"):
                return {"accepted": False, "rejected": f"unknown command {typ!r}"}
            info = bbrt.apply(self.g, cmd)
        else:
            g = self.g
            if typ == "pitch":
                if seat != g.pitcher:
                    return {"accepted": False, "rejected": "you are batting: swing or none"}
                try:
                    p = pitch_for(g.seed, g.stream, g.j, cmd)
                except ValueError as exc:
                    return {"accepted": False, "rejected": str(exc)}
                info = g.apply(seat, cmd, p)
            elif typ == "swing":
                if seat != g.batter:
                    return {"accepted": False, "rejected": "you are pitching: pitch or none"}
                info = g.apply(seat, cmd)
            elif typ == "none":
                info = g.apply(seat, cmd)
            else:
                return {"accepted": False, "rejected": f"unknown command {typ!r}"}
        info = dict(info)
        if not info.get("accepted"):
            info["rejected"] = info.get("why", "not accepted")
        return info

    # ------------------------------------------------------------------ observation
    def _last_pitch(self):
        h = self.history()
        if not h:
            return None
        if self.single:
            pub = self.g.G.public_rec(h[-1])
            d = pub.pop("your_decision")
            pub["your_command"] = ("take (no swing started)" if d["type"] == "take" else
                                   {"swing_start_t": round(d["t"], 4),
                                    **{k: d[k] for k in ("plane", "height", "speed", "reach")}})
            return pub
        return self.g.public_rec(h[-1])

    def _pitch_phase(self):
        g = self.g
        if g.over:
            return "pitch over"
        return "ball released" if g.released() else "windup (the ball is still in the pitcher's hand)"

    def observe(self, seat, query=None):
        g = self.g
        if self.single:
            G = g.G
            o = {"pitch": g.idx + 1, "plate_appearance": min(G.pa + 1, G.lv["n_pa"]),
                 "n_plate_appearances": G.lv["n_pa"], "count": {"balls": G.balls, "strikes": G.strikes},
                 "bases": G.bases, "threshold": int(G.lv["threshold"]),
                 "pa_results": [r["result"] for r in G.pa_results]}
            o["t_since_release"] = g.t if g.released() else None
        else:
            o = {"role": g.role(seat), "batter": g.batter, "pitcher": g.pitcher, "plate_appearance": g.pa + 1,
                 "your_plate_appearances": sum(1 for r in g.pa_results if r["batter"] == seat),
                 "count": {"balls": g.balls, "strikes": g.strikes},
                 "score": {"you": g.score[seat], "opponent": g.score[bbduel.OTHER[seat]]}, "pitch": g.idx + 1}
            o["t_since_release"] = g.t if (g.released() and not g.over) else None
            if not g.released() and not g.over:
                o["release_in_s"] = round(-g.t, 4) if seat == g.pitcher else None
        o["pitch_phase"] = self._pitch_phase()
        o["heard"] = g.heard()
        o["command"] = g.command() if self.single else g.command(seat)
        o["last_pitch"] = self._last_pitch()
        return o

    def camera_job(self, seat):
        snaps = self.g.snapshots()
        meta = {"view_times": [s["t"] for s in snaps], "view_steps": [s["k"] for s in snaps],
                "view_sim_times": [round(s["k"] * DT, 4) for s in snaps], "image_layout": IMAGE_LAYOUT}
        return {"snaps": snaps}, meta

    # ------------------------------------------------------------------ spectators (never read back)
    def _ball_now(self):
        return [float(x) for x in self.g.ball_at(self.g.t)]

    def broadcast_frame(self):
        """The legacy broadcast frame (broadcast_log.pitch_frame): the newest snapshot's ball, swing and time since
        release, and the pitch index (scripts/olympiad/broadcast/adapters.py draws it with bb.Renderer.set)."""
        sn = self.g.snapshots()[-1]
        return {"q": [], "ball": [round(float(x), 4) for x in sn["ball"]],
                "sw": None if sn["swing"] is None else [round(float(x), 5) for x in sn["swing"]],
                "t": round(sn["t"], 4), "p": int(sn["pitch"])}

    def render_state(self):
        """The scene now for the spectator video (renderers/): the ball and the bat (mocap bodies of bb.scene_xml)."""
        g = self.g
        rec = g.rec
        sw = rec.get("swing_obj") if rec is not None else None
        if sw is None:
            th, ca, sa, px, H, ns = bb.TH0, 1.0, 0.0, bb.PIVOT_X0, 0.9, (0.0, 0.0, 1.0)
        else:
            th = sw.angle(g.t)[0]
            ca, sa, px, H, ns = sw.ca, sw.sa, sw.px, sw.H, sw.normal()
        u = (math.cos(th), math.sin(th) * ca, math.sin(th) * sa)
        q = _mat2quat(np.array([u, bb.cross(ns, u), ns]).T)
        bat = [float(px), float(bb.PIVOT_Y), float(H)]
        return {"qpos": [], "mocap_pos": [self._ball_now(), bat, bat], "mocap_quat": [[1.0, 0.0, 0.0, 0.0], q, q]}

    def final_fields(self):
        g = self.g
        if self.single:
            G = g.G
            s = G.summary()
            return {"level": self.level_, "bases": G.bases, "threshold": int(G.lv["threshold"]),
                    "passed": bool(G.bases >= G.lv["threshold"]),
                    **{k: s[k] for k in ("pa_done", "n_pa", "hits", "walks", "strikeouts", "home_runs", "pitches")},
                    "pa_results": s["pa_results"], "n_actions": len(G.decisions),
                    "pitch_log": [G.public_rec(r) | {"zone": r["zone"], "crossing": r["crossing"]} for r in G.history]}
        return {"score": dict(g.score), "first_batter": g.first, "pitches": len(g.history),
                "plate_appearances": g.pa, "pa_results": g.pa_results,
                "pitch_log": [g.public_rec(r) for r in g.history],
                "format": {"pa_per_seat": bbduel.PA_PER_SEAT, "extra_pairs": bbduel.EXTRA_PAIRS,
                           "pa_max_pitches": bbduel.PA_MAX_PITCHES}}


def _mat2quat(R):
    """Rotation matrix -> unit quaternion (w, x, y, z)."""
    t = R[0, 0] + R[1, 1] + R[2, 2]
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        q = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    else:
        i = int(np.argmax([R[0, 0], R[1, 1], R[2, 2]]))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = math.sqrt(1.0 + R[i, i] - R[j, j] - R[k, k]) * 2
        q = [0.0] * 4
        q[0] = (R[k, j] - R[j, k]) / s
        q[1 + i] = 0.25 * s
        q[1 + j] = (R[j, i] + R[i, j]) / s
        q[1 + k] = (R[k, i] + R[i, k]) / s
    n = math.sqrt(sum(x * x for x in q))
    return [float(x / n) for x in q]


# ---------------------------------------------------------------------------------------------- the pitches
_PITCHES: dict = {}


def pitch_for(seed, slot, j, cmd) -> bb.Pitch:
    """bbduel.make_pitch (the commanded pitch with its seeded execution scatter, a pure function), cached: the action
    layer checks that a pitch can be thrown when the command arrives, the world throws the same pitch at the release."""
    key = (int(seed), int(slot), int(j), cmd["pitch"], float(cmd["speed"]), float(cmd["x"]), float(cmd["z"]))
    p = _PITCHES.get(key)
    if p is None:
        p = bbduel.make_pitch(int(seed), int(slot), int(j), cmd)
        if len(_PITCHES) > 256:
            _PITCHES.clear()
        _PITCHES[key] = p
    return p


# ---------------------------------------------------------------------------------------------- the action layer
NUM = {"type": "number"}
SWING = {"swing": {
    "t": {**NUM, "min": 0.0, "max": 2.5, "unit": "s", "doc": "the bat square T s after the control step at which the "
                                                            "command takes effect (0 < T <= 2.5)"},
    "plane": {**NUM, "min": -15.0, "max": 35.0, "unit": "deg", "doc": "swing plane (attack angle)"},
    "height": {**NUM, "min": 0.30, "max": 1.40, "unit": "m", "doc": "height of the pivot (the knob)"},
    "speed": {**NUM, "min": 15.0, "max": 34.0, "unit": "m/s", "doc": "speed of the sweet spot"},
    "reach": {**NUM, "min": -0.20, "max": 0.20, "unit": "m", "default": 0.0,
              "doc": "the pivot's x offset from -0.66 m (optional, default 0)"}}}
PITCH = {"pitch": {
    "pitch": {"type": "enum", "values": sorted(bbduel.ARSENAL), "doc": "the pitch type (fixed spin, spec match.pitching)"},
    "speed": {**NUM, "min": min(v[0] for v in bbduel.ARSENAL.values()), "max": max(v[1] for v in bbduel.ARSENAL.values()),
              "unit": "m/s", "doc": "release speed, within the type's band (spec match.pitching.arsenal)"},
    "x": {**NUM, "min": bbduel.TARGET_X[0], "max": bbduel.TARGET_X[1], "unit": "m", "doc": "target at the plate, x"},
    "z": {**NUM, "min": bbduel.TARGET_Z[0], "max": bbduel.TARGET_Z[1], "unit": "m", "doc": "target at the plate, height"}}}
NONE = {"none": {}}
SINGLE_SCHEMA = {**SWING, **NONE}
DUEL_SCHEMA = {**SWING, **PITCH, **NONE}


class RigLayer:
    """The one action level: the swing rig's command (and, in the duel, the pitcher's), checked against the schema
    when it arrives (HTTP 400 if malformed or out of range, or the wrong role's command); it takes effect unchanged at
    the next control step. A swing before the release or after the swing has started is accepted by the core and
    refused by the rig when it takes effect (`rejected`, nothing changes), as in the legacy services."""
    level = "low"
    worker = None
    delay_steps = 0

    def schema(self, match, seat):
        return match.schema(seat)

    def parse(self, seat, body, match, ctx):
        cmd = check_command(match.schema(seat), body)
        typ = cmd["type"]
        if typ == "swing" and not cmd["t"] > 0.0:
            raise Refused("t must be in (0, 2.5] s from when the command takes effect")
        if not match.single and typ != "none":
            role = match.g.role(seat)
            if typ == "pitch" and role != "pitching":
                raise Refused("you are batting now: act with {'type': 'swing', ...} or {'type': 'none'}")
            if typ == "swing" and role != "batting":
                raise Refused("you are pitching now: act with {'type': 'pitch', ...} or {'type': 'none'}")
            if typ == "pitch":
                lo, hi = bbduel.ARSENAL[cmd["pitch"]]
                if not lo <= cmd["speed"] <= hi:
                    raise Refused(f"a {cmd['pitch']} is thrown at {lo:g}-{hi:g} m/s")
                try:
                    pitch_for(match.g.seed, match.g.stream, match.g.j, cmd)
                except ValueError as exc:
                    raise Refused(str(exc)) from None
        return cmd

    def snapshot(self, seat, cmd, match, ordinal):
        return None

    def prepare(self, job):
        return None

    def lower(self, seat, cmd, prepared, match):
        return cmd, {}

    def reply(self, seat, cmd, info, match):
        return info


# ---------------------------------------------------------------------------------------------- the images
class Views:
    """The legacy observation images (bbrt.render_snaps): 2 cameras x 4 frames, each tile stamped with its camera and
    time since release, one PNG. A tile's scene (ball position, bat) is drawn once and reused (the windup frames of a
    pitch are one scene); the same pixels as the legacy drawing."""

    def __init__(self):
        self.rd = bb.Renderer()
        self.tiles: dict = {}
        self.sheets: dict = {}

    def raw(self, cam, sn):
        key = (cam, tuple(sn["ball"]), sn["swing"], None if sn["swing"] is None else sn["t"])
        img = self.tiles.get(key)
        if img is None:
            sw = None if sn["swing"] is None else bb.Swing(*sn["swing"])
            self.rd.set(tuple(sn["ball"]), sw, sn["t"])
            img = np.array(self.rd.render(cam), copy=True)
            if len(self.tiles) > 96:
                self.tiles.clear()
            self.tiles[key] = img
        return img

    def draw(self, job):
        from PIL import Image, ImageDraw
        snaps = job["snaps"]
        key = json.dumps([[s["ball"], s["swing"], s["t"]] for s in snaps], default=float)
        if key in self.sheets:
            return self.sheets[key]
        rows = []
        for cam in ("catcher", "side"):
            tiles = []
            for sn in snaps:
                im = Image.fromarray(self.raw(cam, sn))
                lab = f"{cam}  t={sn['t']:+.2f}s" + ("  (windup)" if sn["t"] < 0 else "")
                ImageDraw.Draw(im).text((6, 4), lab, fill=(255, 255, 0))
                tiles.append(np.asarray(im))
            rows.append(np.concatenate(tiles, axis=1))
        out = png(np.concatenate(rows, axis=0))
        if len(self.sheets) > 8:
            self.sheets.clear()
        self.sheets[key] = out
        return out


def png(img: np.ndarray) -> bytes:
    """RGB8 PNG without row filters, zlib level 1 (the same pixels as the legacy imageio PNG, faster)."""
    h, w = img.shape[:2]
    raw = np.zeros((h, w * 3 + 1), np.uint8)
    raw[:, 1:] = np.ascontiguousarray(img[..., :3], np.uint8).reshape(h, -1)

    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw.tobytes(), 1)) + chunk(b"IEND", b""))


_LOCAL_VIEWS: list = []


# ---------------------------------------------------------------------------------------------- the sport
class BaseballSport:
    name = "baseball"
    protocol = "realtime"
    revision = REVISION
    source_dirs = [HERE]
    workers = ("render",)

    def build(self, cfg):
        return Baseball(cfg)

    def action_layer(self, level):
        return RigLayer()

    def worker_init(self, kind, seat, cfg):
        return Views()

    def worker_call(self, obj, kind, job):
        if obj is None:                            # in-process (core/local.py): one renderer for every seat
            if not _LOCAL_VIEWS:
                _LOCAL_VIEWS.append(Views())
            obj = _LOCAL_VIEWS[0]
        return {"png": obj.draw(job)}

    def scene(self, cfg):
        from renderers.base import Scene
        return Scene(xml=bb.scene_xml())

    def spec(self, cfg, seat, m):
        single = len(cfg.seats) == 1
        match = {"score": "bases: walk 1, single 1, double 2, home run 4 (no triples)",
                 "count": "3 strikes = strikeout, 4 balls = walk, a foul is a strike unless there are already 2 "
                          "strikes; a plate appearance reaching 15 pitches is an out",
                 "pa_max_pitches": bb.PA_MAX_PITCHES}
        if single:
            G = m.g.G
            match.update(plate_appearances=G.lv["n_pa"], opponent_level=m.level_,
                         success=f"bases >= {G.lv['threshold']}",
                         pitcher={"release": list(bb.REL),
                                  "arsenal": {n: {"speed_mps": list(G.lv["speeds"][n]),
                                                  "spin_rad_s": bb.PITCH_TYPES[n]["spin"],
                                                  "spin_axis": list(bb.PITCH_TYPES[n]["axis"])} for n in G.lv["speeds"]},
                                  "mix": G.lv["mix"]})
        else:
            match.update(
                format=(f"plate appearances alternate between the seats: one bats, the other pitches; each seat bats "
                        f"{bbduel.PA_PER_SEAT} times; level bases after that: one more each, up to "
                        f"{bbduel.EXTRA_PAIRS} times; still level: the winner of the last pitch wins (no draws)"),
                first_player=m.g.first, first_player_meaning="the seat that bats first (plate appearances 1, 3, 5, ...)",
                pa_per_seat=bbduel.PA_PER_SEAT, extra_pairs=bbduel.EXTRA_PAIRS,
                pitching={"release": list(bb.REL), "target_x_m": list(bbduel.TARGET_X),
                          "target_z_m": list(bbduel.TARGET_Z), "scatter": dict(bbduel.PITCH_SCATTER),
                          "arsenal": {n: {"speed_mps": list(bbduel.ARSENAL[n]), "spin_rad_s": bb.PITCH_TYPES[n]["spin"],
                                          "spin_axis": list(bb.PITCH_TYPES[n]["axis"])} for n in sorted(bbduel.ARSENAL)},
                          "command": "{'type':'pitch','pitch':TYPE,'speed':m/s,'x':target x,'z':target z} during the "
                                     "windup; at the release the last one is thrown; none = an automatic ball"})
        return {
            "robot": "4-parameter kinematic swing rig holding a 0.84 m / 0.90 kg bat",
            "match": match,
            "pitch_clock": {
                "control_dt_s": DT, "windup_s": list(bbrt.WINDUP), "frame_gap_s": bbrt.FRAME_GAP,
                "rule": "an act takes effect at the next control step after it arrives; the rig (and the pitcher) "
                        "keeps its current command until then; a command does not carry over to the next pitch",
                "pitch_resolved": f"a planned swing starts at its time; with no swing started {bbrt.TAKE_AFTER} s "
                                  "after the ball reached the plate, the pitch is taken; the pitch is over "
                                  f"{bbrt.END_AFTER} s after the contact or the plate, then the world freezes for "
                                  "the reviews"},
            "swing_rig": {"pivot": f"(-0.66 + REACH, {bb.PIVOT_Y}, HEIGHT) = the knob", "bat_length": bb.L_BAT,
                          "bat_axis": "u(theta) = (cos th, sin th cos a, sin th sin a), a = PLANE_DEG",
                          "profile": "theta = -180 deg at rest; from the swing start constant angular acceleration to "
                                     "omega = SPEED/0.69 at -30 deg (time 2*(150 deg in rad)/omega), constant omega to "
                                     "+90 deg, then it slows to rest at +150 deg",
                          "time_to_square_s": "(5*pi/3 + pi/6) / omega = 3.9741 / SPEED after the swing start",
                          "command": "swing T ...: the bat square T s after the control step at which the act takes "
                                     "effect; the swing starts 3.9741 / SPEED s before that (at once if that is "
                                     "already past); refused before the release; once started it cannot be changed",
                          "bat_radius_m": "knob 0.025 (s < 0.02), handle 0.0125 (s < 0.40), taper to 0.033 at s = "
                                          "0.60, barrel 0.033 to the end 0.84",
                          "sweet_spot_s": bb.S_SS, "scatter": dict(bb.SCATTER)},
            "ball": {"mass": bb.M_BALL, "radius": bb.R_BALL, "air_density": bb.RHO, "C_D": bb.CD,
                     "C_L": "1.5 S (S < 0.1), 0.09 + 0.6 S; S = R|w|/|v|", "spin_decay_s": bb.SPIN_TAU,
                     "bat_ball": {"cor": "max(0.10, 0.50 - 4.0 (s - 0.69)^2)", "bat_cm_s": bb.S_CM,
                                  "bat_I_cm": bb.I_CM, "friction": bb.MU}},
            "zone": {"plane": "y = 0", "x": [-bb.ZONE_X, bb.ZONE_X], "z": [bb.ZONE_LO, bb.ZONE_HI],
                     "rule": "strike if the ball's centre crosses y = 0 within the zone grown by the ball radius"},
            "field": {"foul_lines_deg": 45, "first_base_xy": [round(v, 2) for v in bb.B1], "fence_m": bb.FENCE_R,
                      "fence_height_m": bb.FENCE_H,
                      "fielders": {f[0]: {"xy": [round(v, 2) for v in bb.FIELDER_POS[f[0]]], "r0": f[3], "v_run": f[4],
                                          "t_react": f[5]} for f in bb.FIELDERS},
                      "catch_height_m": bb.REACH_H, "runner_to_first_s": bb.RUNNER_TO_1B, "throw_mps": bb.THROW_V},
            "cameras": bb.cam_info(),
            "image": "2 rows (catcher, side) x 4 columns (the newest frames, 30 ms apart, oldest left); tile 480 x "
                     "360; pinhole, image right = forward x up",
            "world": "home plate centre = origin, +y to the pitcher / centre field, +x to first base, z up; ground z = 0",
        }

    def players(self):
        import builtin_players as bp
        return {"reference": bp.Reference, "rival": bp.Rival, "random": bp.Random}


SPORT = BaseballSport()
