"""Ten-pin bowling on the v0.2 core (USBC Playing Rules 2025-2026). Migrated from the Season-1 task
bowling-program-standard (docs/framework_zh.md §10, stage 1). One format: the single game (user decision of 2026-10-08,
as review item B10: the legacy duel bowling-duel-program-standard had no interaction between the players, two
identical lanes, so a head-to-head is the single scores on the same seeds; sim/sport.py keeps its duel code, unused).

The simulator is sim/, unchanged from the legacy tree but the release-speed cap (bowling-4, D5): sim/bowling.py
(MuJoCo lane with an oil pattern, 10 pins, a 7 kg ball; seeded execution scatter; scoring; the camera views) and
sim/sport.py (the legacy rules module of the shared turn service: turns, roll-off frames, decide). turnshot.py maps
the rules module onto contract ② (it is the pattern golf putting and disc golf copy); this file adds only what is
bowling's:
  world + referee  Bowling(TurnShotMatch): one 10-frame game on the seed's lane, scored against the success line
  actions          one level (low): {"type": "roll", board, angle_deg, speed, rpm, axis_deg} (SCHEMA, then the
                   legacy validate(): 4-decimal rounding), with the chess clock at the ball (ShotLayer)
  observation      observe(): the decision point, the scorecard, the last ball's pin count (no coordinates);
                   camera_job(): approach, pindeck, pindeck_after and the lane from above, one PNG (sim/bowling.py)
"""
from __future__ import annotations

import math
import os

from core.spec import load
from turnshot import ShotLayer, TurnShotMatch, load_rules

HERE = os.path.dirname(os.path.abspath(__file__))
SIM = os.path.join(HERE, "sim")
DECL = load(HERE)
NUM = {"type": "number", "digits": 4}
SCHEMA = {"roll": {"board": {**NUM, "min": 1.0, "max": 39.0, "unit": "board (1 = right gutter, 20 = centre)"},
                   "angle_deg": {**NUM, "min": -10.0, "max": 10.0, "unit": "deg, + = towards the left"},
                   "speed": {**NUM, "min": 4.47, "max": 10.73, "unit": "m/s"},
                   "rpm": {**NUM, "min": 0.0, "max": 700.0, "unit": "rev/min"},
                   "axis_deg": {**NUM, "min": -90.0, "max": 90.0, "unit": "deg, + hooks left"}}}
_VIEWS = []                     # one renderer per process: every seed's lane looks the same (the oil is invisible)
STATIC_PANEL_X = 14.0           # lane panels 1-3 (x < 14.5 m) show only the empty lane at a decision point


def cached_views(B, lane):
    """sim/bowling.py's Views, drawing the same images, with the lane panels that cannot change cached: a panel
    camera looking straight down at x < STATIC_PANEL_X while the ball and every pin lie beyond 15 m (parked, on the
    deck or stored). Saves 3 of the 7 renders of a decision point (the fairness check plays hundreds of games)."""
    movable = [lane.ball_q] + list(lane.pin_q)

    class Views(B.Views):
        static: dict = {}

        def shot(self, qpos, camera, w, h, fovy):
            key = (tuple(camera.lookat), camera.azimuth, camera.elevation, w, h, fovy)
            if camera.elevation == -90.0 and camera.lookat[0] < STATIC_PANEL_X and \
                    all(qpos[q] > 15.0 for q in movable):
                if key not in self.static:
                    self.static[key] = super().shot(qpos, camera, w, h, fovy)
                return self.static[key].copy()
            return super().shot(qpos, camera, w, h, fovy)
    return Views(lane)


def rules(seed):
    return load_rules(SIM, seed)


class Bowling(TurnShotMatch):
    def schema(self, seat):
        return SCHEMA

    def views(self, seat):
        g = self.M.cur(seat)
        if not _VIEWS:
            _VIEWS.append(cached_views(self.S.B, g.lane))
        return _VIEWS[0].decision_views(g)


class BowlingSport:
    name, protocol, revision = "bowling", "turn", DECL["sport"]["revision"]
    source_dirs = [HERE]
    workers = ()                # the views are drawn in-process (about 0.2 s per decision point)

    def build(self, cfg):
        return Bowling(cfg, rules(cfg.seed))

    def action_layer(self, level):
        return ShotLayer(lambda m: m.S)

    def worker_call(self, obj, kind, job):
        return job              # camera_job() already drew the PNG

    def spec(self, cfg, seat, match):
        S = match.S
        sp = S.spec(False)                       # the single game (no seed: rule O3)
        for c in sp["cameras"]:                  # the pinhole intrinsics the generated prompt names
            w, h = c["size_px"]
            if "panels" in c:
                h = c["panels"][0]["rows_px"][1] - c["panels"][0]["rows_px"][0]
            fovy = c.get("fovy_deg", c.get("fovy_deg_per_panel"))
            c["focal_px"] = round(h / 2 / math.tan(math.radians(fovy) / 2), 4)
            c["principal_point_px"] = [w / 2, h / 2] if "panels" not in c else "[width / 2, the panel's centre row]"
        sp["image"] = {"size_px": list(S.IMAGE_SIZE), "layout": S.IMAGE_LAYOUT,
                       "tiles": {k: {"x": x, "y": y, "w": w, "h": h} for k, (x, y, w, h) in S.TILES.items()}}
        return sp

    def players(self):
        import builtin_players as bp
        return {"reference": bp.Reference, "random": bp.Random}


SPORT = BowlingSport()
