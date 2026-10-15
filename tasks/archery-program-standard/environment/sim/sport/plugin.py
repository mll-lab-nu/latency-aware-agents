"""archery: target archery in the wind on the v0.2 core, wrapping the legacy simulator archery.py (arrow ballistics
with quadratic drag in a seeded gusty wind, World Archery scoring, the Round). One format: a single round, scored
against the success line (review item B10, 2026-10-07: the legacy duel had no interaction between the archers,
identical lanes, so a head-to-head is the single scores on the same seeds; the unused archery.Duel was removed from
the simulator on 2026-10-08). The four groups of contract ② (core/sport.py):
  world        archery.Round; the simulator draws its winds and scatter from the module seed, which the plugin installs
               before every call into it (archery.SEED / WINDS), so one process can host many matches
  actions      one low level: {"type": "shoot", elev_deg, azim_deg, draw} (schema, 4 decimals as the legacy service)
  referee      phase / scores / to_move; no tiebreaks (a solo round, scored against the success line of sport.toml)
  observation  observe (scores and calls only: no wind numbers, no coordinates) + cameras: the four views (archer,
               scope, side, target), drawn by the task's renderer from scene() and render_state() (one scene for every
               distance: the distance-dependent parts are mocap bodies placed per end); reply images (core 0.2.8):
               reply_views gives the reply to each arrow the target close-up with this end's arrows, the one just shot
               included (also the third, whose end is over and whose arrows are pulled before the next observation)
"""
from __future__ import annotations

import math
import os

import archery as A
from core.spec import load
from core.sport import FINISHED, PLAYING, MatchBase, Phase, SchemaLayer

HERE = os.path.dirname(os.path.abspath(__file__))
REVISION = load(HERE)["sport"]["revision"]
SUCCESS_LINE = load(HERE)["formats"]["single"]["success"]["score_at_least"]
HIDDEN = [0.0, 0.0, -50.0]
LINE_XS = list(range(10, 100, 10))
NUM = {"type": "number", "digits": 4}
SCHEMA = {"shoot": {
    "elev_deg": {**NUM, "min": -10.0, "max": 30.0, "unit": "deg", "doc": "launch direction above horizontal"},
    "azim_deg": {**NUM, "min": -20.0, "max": 20.0, "unit": "deg", "doc": "heading: 0 = down the range, + = left"},
    "draw": {**NUM, "min": A.DRAW_MIN, "max": A.DRAW_MAX, "doc": f"draw fraction: launch speed = {A.V_FULL:g} m/s x draw"}}}


def use_seed(seed: int) -> None:
    """The simulator keeps its seed in module globals (the legacy service had one match per process)."""
    if A.SEED != seed:
        A.SEED = seed
        A.WINDS, A.WIND_MEANS = A.make_winds(seed)


# ------------------------------------------------------------------------------------------------ cameras and scene
def cameras_at(D: float) -> list[dict]:
    """The four decision cameras for an end at distance D (archery.Views poses, as camera definitions)."""
    def cam(name, w, h, fovy, pos, yaw, pitch):
        y, p = math.radians(yaw), math.radians(pitch)
        fw = (math.cos(p) * math.cos(y), math.cos(p) * math.sin(y), math.sin(p))
        return {"name": name, "pos": [float(v) for v in pos], "lookat": [float(pos[i]) + fw[i] for i in range(3)],
                "up": [0.0, 0.0, 1.0], "fovy_deg": float(fovy), "width": w, "height": h}
    return [cam("archer", A.ARCHER_W, A.ARCHER_H, A.FOVY_ARCHER, A.ARCHER_POS, 0.0, A.ARCHER_PITCH),
            cam("scope", A.SCOPE_W, A.SCOPE_H, A.scope_fovy(D), A.LAUNCH, 0.0, A.scope_pitch(D)),
            cam("side", A.SIDE_W, A.SIDE_H, A.FOVY_SIDE, (D / 2, -A.SIDE_OFF, A.SIDE_Z), 90.0, A.SIDE_PITCH),
            cam("target", A.CLOSE_W, A.CLOSE_H, A.close_fovy(), (D - A.CLOSE_BACK, 0.0, A.FACE_Z), 0.0, 0.0)]


def cameras_spec(D: float) -> list[dict]:
    """cameras_at(D) with the pinhole numbers a player needs (archery.cameras_numeric: axes, focal length)."""
    num = {c["name"]: c for c in A.cameras_numeric(D)}
    return [dict(c, **{k: num[c["name"]][k] for k in ("forward", "right", "up", "focal_px", "principal_point_px")}) for c in cameras_at(D)]


def scene_xml() -> str:
    """archery.scene_xml with every distance-dependent part (boss + face, poles, windsocks, distance lines) a mocap
    body: mocap order = sock0..3, arrow0..2, flying (as the legacy scene), boss, pole0..2, line10..90, line_D."""
    h = A.BOSS_H
    socks = [A._sock_body(f"sock{i}", HIDDEN, 1.0) for i in range(3)] + [A._sock_body("sock3", HIDDEN, A.TSOCK_SCALE)]
    arrows = [A._arrow_body(f"arrow{i}", A.ARROW_COLS[i]) for i in range(3)] + [A._arrow_body("flying", (1.0, 0.5, 0.0))]
    leg_z = (A.FACE_Z - h) / 2
    boss = (f'<body name="boss" mocap="true" pos="0 0 -50">'
            f'<geom type="box" pos="{A.BOSS_T / 2 + 0.008} 0 0" size="{A.BOSS_T / 2} {h} {h}" rgba="0.77 0.67 0.43 1"/>'
            f'<geom type="mesh" mesh="face" material="face" pos="-0.002 0 0" contype="0" conaffinity="0"/>'
            f'<geom type="box" pos="0.25 0.5 {leg_z - A.FACE_Z}" size="0.04 0.04 {leg_z}" rgba="0.45 0.3 0.15 1"/>'
            f'<geom type="box" pos="0.25 -0.5 {leg_z - A.FACE_Z}" size="0.04 0.04 {leg_z}" rgba="0.45 0.3 0.15 1"/>'
            f'<geom type="cylinder" fromto="0 {A.TPOLE_Y} {-A.FACE_Z} 0 {A.TPOLE_Y} {A.TPOLE_H - A.FACE_Z}" '
            f'size="0.02" rgba="0.85 0.85 0.85 1"/></body>')
    poles = "".join(f'<body name="pole{i}" mocap="true" pos="0 0 -50"><geom type="cylinder" fromto="0 0 0 0 0 '
                    f'{A.POLE_H}" size="0.03" rgba="0.85 0.85 0.85 1"/></body>' for i in range(3))
    lines = "".join(f'<body name="line{x}" mocap="true" pos="0 0 -50"><geom type="box" size="0.04 6 0.001" '
                    f'rgba="0.95 0.95 0.95 1"/></body>' for x in LINE_XS)
    lines += ('<body name="line_D" mocap="true" pos="0 0 -50"><geom type="box" size="0.08 6 0.001" '
              'rgba="0.95 0.95 0.95 1"/></body>')
    return f"""
<mujoco model="archery">
  <visual><global offwidth="1600" offheight="1000"/><quality shadowsize="4096"/>
    <headlight ambient="0.35 0.35 0.35" diffuse="0.25 0.25 0.25" specular="0 0 0"/><map znear="0.02" zfar="50"/></visual>
  <statistic extent="10" center="35 0 1"/>
  <asset>
    <texture name="sky" type="skybox" builtin="gradient" rgb1="0.5 0.7 0.95" rgb2="0.95 0.97 1.0" width="256" height="256"/>
    <texture name="grass" type="2d" file="grass.png"/>
    <texture name="face" type="2d" file="face.png"/>
    <material name="grass" texture="grass" texrepeat="120 120" specular="0"/>
    <material name="face" texture="face" specular="0.02"/>
    <mesh name="face" vertex="0 {h} {-h}  0 {-h} {-h}  0 {-h} {h}  0 {h} {h}  0.004 {h} {-h}  0.004 {-h} {-h}  0.004 {-h} {h}  0.004 {h} {h}"
          texcoord="0 0  1 0  1 1  0 1  0 0  1 0  1 1  0 1" face="0 1 2  0 2 3" inertia="shell"/>
  </asset>
  <worldbody>
    <light directional="true" pos="0 0 10" dir="0.35 0.3 -0.8" diffuse="0.8 0.8 0.75" castshadow="false"/>
    <geom type="plane" size="200 200 0.1" pos="35 0 0" material="grass"/>
    <geom type="box" pos="0 0 0.001" size="0.08 6 0.001" rgba="0.95 0.95 0.95 1"/>
    <geom type="box" pos="-0.4 -0.25 0.6" size="0.02 0.02 0.6" rgba="0.3 0.3 0.35 1"/>
    {''.join(socks)}
    {''.join(arrows)}
    {boss}
    {poles}
    {lines}
  </worldbody>
</mujoco>"""


def lane_state(D: float, wind, stuck) -> dict:
    """The mocap poses of one lane: the windsocks for `wind`, this end's arrows `stuck` [(hit, vel, kind)], the boss
    and the poles at distance D, the distance lines up to D (archery.Views.set_state, extended to the mocap scene)."""
    head, droop = A.sock_angles(wind)
    q = [float(v) for v in A._quat_yaw_pitch(head, droop)]
    one = [1.0, 0.0, 0.0, 0.0]
    pos = [[x, A.POLE_Y, A.POLE_H] for x in A.sock_x(D)] + [[D, A.TPOLE_Y, A.TPOLE_H]]
    quat = [q] * 4
    for i in range(3):
        if i < len(stuck):
            pos.append([float(v) for v in stuck[i][0]])
            quat.append([float(v) for v in A._quat_dir(stuck[i][1])])
        else:
            pos.append(HIDDEN)
            quat.append(one)
    pos.append(HIDDEN)                                                      # the flying arrow (spectators only)
    quat.append(one)
    pos.append([D, 0.0, A.FACE_Z])                                          # boss + face + the target's pole
    pos += [[x, A.POLE_Y, 0.0] for x in A.sock_x(D)]
    pos += [[float(x), 0.0, 0.001] if x < D else HIDDEN for x in LINE_XS]
    pos.append([D, 0.0, 0.001])
    quat += [one] * (len(pos) - len(quat))
    return {"qpos": [], "mocap_pos": pos, "mocap_quat": quat}


def card(r) -> list[dict]:
    return [{"end": e + 1, "distance_m": D, "arrows": [A.label(v, x) for v, x in zip(r.scores[e], r.xs[e])],
             "end_score": sum(r.scores[e])} for e, D in enumerate(A.DISTANCES)]


def clock_call(rec) -> str | None:
    """The spotter's call of an arrow on the face: its score and clock position ("9 at 2 o'clock"); misses get the
    simulator's words (archery.miss_words)."""
    if rec["score"] == 0:
        return rec.get("miss")
    dy, dz = rec["hit"][1], rec["hit"][2] - A.FACE_Z
    if math.hypot(dy, dz) < A.RING_W / 2:
        return f"{rec['call']} in the centre"
    ang = math.degrees(math.atan2(-dy, dz)) % 360.0                         # 12 o'clock = up, 3 o'clock = right (-y)
    hour = int(round(ang / 30.0)) % 12 or 12
    return f"{rec['call']} at {hour} o'clock"


# ------------------------------------------------------------------------------------------------ the match
class Archery(MatchBase):
    def __init__(self, cfg):
        self.cfg, self.seed, self.k = cfg, int(cfg.seed), 0
        use_seed(self.seed)
        self.r = A.Round()
        self.recs: list[dict] = []                    # every arrow: the simulator's record + seat
        self.forfeit_rec = None                       # the core's decision (the round sha does not hold it)
        self.face = None                              # (D, wind, stuck) the face as the last arrow left it (replies)

    # ------------------------------------------------------------------ world
    def state_digest(self):
        return self.r.state_sha()

    def schema(self, seat):
        return SCHEMA

    def apply(self, seat, cmd):
        use_seed(self.seed)
        before = list(self.r.stuck)                   # this end's arrows already on the face (pulled after the 3rd)
        rec, fl = self.r.shoot(cmd["elev_deg"], cmd["azim_deg"], cmd["draw"])
        self.face = (rec["distance_m"], rec["wind"], before + [(fl["hit"], fl["vel"], fl["kind"])])
        rec = dict(rec, seat=seat)
        self.recs.append(rec)
        self.k += 1
        return {"rec": rec}

    def forfeit(self, seat, why):
        self.forfeit_rec = {"seat": seat, "why": why}
        return True

    # ------------------------------------------------------------------ referee
    def phase(self):
        return Phase(FINISHED, reason="round_over") if self.r.done else Phase(PLAYING)

    def to_move(self):
        return None if self.r.done else self.cfg.seats[0]

    def scores(self):
        return {self.cfg.seats[0]: self.r.total()}

    def tiebreaks(self):
        return []                                                       # scored against the success line

    # ------------------------------------------------------------------ observation
    def next_arrow(self):
        """(end index, arrow index, distance) of the next arrow, or None."""
        r = self.r
        return None if r.done else (r.end, r.arrow, A.DISTANCES[r.end])

    def observe(self, seat, query=None):
        r = self.r
        out = {"total": r.total(), "scorecard": card(r)}
        nxt = self.next_arrow()
        if nxt is None:
            out["stage"] = "over"
        else:
            e, a, D = nxt
            out.update({"stage": "round", "distance_m": D, "end": e + 1, "of_ends": A.N_ENDS, "arrow": a + 1,
                        "arrows_per_end": A.ARROWS_PER_END,
                        "this_end": [A.label(v, x) for v, x in zip(r.scores[e], r.xs[e])]})
        out["arrows"] = [{"end": x["end"], "arrow": x["arrow"], "score": x["call"]} for x in self.recs]
        if self.recs:
            L = self.recs[-1]
            out["last_arrow"] = {"end": L["end"], "arrow": L["arrow"], "score": L["call"],
                                 "you_sent": {"elev_deg": L["elev_deg"], "azim_deg": L["azim_deg"], "draw": L["draw"]},
                                 "spotter": clock_call(L), "flight_time_s": round(L["flight_s"], 2)}
        return out

    def cameras(self, seat, query=None):
        """The four views of the lane (this end's arrows, the wind of the next arrow) until the round is over."""
        nxt = self.next_arrow()
        return [] if nxt is None else cameras_at(nxt[2])

    def reply_views(self, seat, cmd, info):
        """The reply to an arrow (core 0.2.8): the target close-up (the observation's `target` camera) of the face as
        this arrow left it, with this end's arrows, the third included before the arrows are pulled. Its windsocks
        show the wind this arrow flew in (none is in the close-up's view): nothing about the next arrow."""
        D, wind, stuck = self.face
        return {"frames": [{"cameras": [cameras_at(D)[3]], "state": lane_state(D, wind, stuck)}],
                "note": "target close-up after this arrow: this end's arrows (magenta 1, lime 2, cyan 3)"}

    def render_state(self):
        use_seed(self.seed)
        r = self.r
        nxt = self.next_arrow()
        if nxt is None:                                                 # over: the lane as the last arrow left it
            return lane_state(A.DISTANCES[r.end], A.WINDS[r.end][min(r.arrow, A.ARROWS_PER_END - 1)], r.stuck)
        e, a, D = nxt
        return lane_state(D, A.WINDS[e][a], r.stuck)

    def final_fields(self):
        """Sport fields of final.json, under the legacy names the Season-1 readers use (live_board, regrade)."""
        r = self.r
        return {"total": r.total(), "n_x": r.n_x(), "end_scores": [sum(s) for s in r.scores],
                "arrow_scores": r.scores, "arrows_shot": sum(len(s) for s in r.scores),
                "round_complete": r.done, "ends": A.N_ENDS, "arrows_per_end": A.ARROWS_PER_END,
                "distances_m": A.DISTANCES, "max_total": 10 * A.N_ENDS * A.ARROWS_PER_END,
                "success_min_total": SUCCESS_LINE,
                "clock_forfeit": bool(self.forfeit_rec and self.forfeit_rec["why"] == "clock"),
                "shots": [{k: v for k, v in x.items() if k != "trail"} for x in self.recs]}


class ShootLayer(SchemaLayer):
    """The one action level: the request body is checked against SCHEMA (4 decimals); the reply gives the arrow's
    score, the spotter's call and, at the end of an end, the end's scorecard."""

    def reply(self, seat, cmd, info, match):
        rec = info["rec"]
        out = {"accepted": {k: cmd[k] for k in ("elev_deg", "azim_deg", "draw")},
               "last_arrow": {"end": rec["end"], "arrow": rec["arrow"], "score": rec["call"],
                              "spotter": clock_call(rec), "flight_time_s": round(rec["flight_s"], 2)}}
        if rec.get("end_finished"):
            out["end_finished"] = card(match.r)[rec["end"] - 1]
        return out


class ArcherySport:
    name, protocol, revision, source_dirs, workers = "archery", "turn", REVISION, [HERE], ()

    def build(self, cfg):
        return Archery(cfg)

    def action_layer(self, level):
        return ShootLayer()

    def scene(self, cfg):
        from renderers.base import Scene
        return Scene(xml=scene_xml(), assets={"face.png": A._face_png(), "grass.png": A._grass_png()})

    def spec(self, cfg, seat, match):
        """The structured part of /spec (sport.toml [spec] holds the rule constants): the ends, the cameras of every
        distance (pose, axes, intrinsics), the windsocks' mouths."""
        dists = sorted(set(A.DISTANCES))
        nxt = match.next_arrow()
        return {"ends": A.spec_ends(), "cameras": cameras_spec(nxt[2] if nxt else A.DISTANCES[0]),
                "cameras_by_distance": {str(int(D)): cameras_spec(D) for D in dists},
                "windsocks_by_distance": {str(int(D)): A.sock_tops(D) for D in dists}}

    def players(self):
        import builtin_players as bp
        return {"reference": bp.Reference, "random": bp.Random}


SPORT = ArcherySport()
