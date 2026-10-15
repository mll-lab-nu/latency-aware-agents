"""Mini-golf putting on the v0.2 core (WMF Rulebook 2.3). Migrated from the Season-1 task golf-putting-program-standard
(docs/framework_zh.md §10, stage 1). One format: the single round (user decision of 2026-10-08, as review item B10:
the legacy duel golf-putting-duel-program-standard had no interaction between the players, the balls never met on
identical holes, so a head-to-head is the single scores on the same seeds; sim/sport.py keeps its duel code, unused).

The simulator is sim/, unchanged from the legacy tree: sim/golf.py (a deterministic rolling-ball model on a height
field with walls, blocks and a cup; seeded stroke scatter; the course of a seed; the MuJoCo scene that draws the
camera views) and sim/sport.py (the legacy rules module of the shared turn service: hole order, pick-ups, the
sudden-death play-off, decide). turnshot.py (byte-identical to sports/bowling/turnshot.py) maps the rules module onto
contract ②; this file adds only what is golf's:
  world + referee  Golf(TurnShotMatch): one round of the seed's 4 holes, scored against the success line (par)
  actions          one level (low): {"type": "putt", aim_deg, speed} (SCHEMA, then the legacy validate(): aim
                   rounded to 3 decimals, speed to 4), with the chess clock at the stroke (ShotLayer)
  observation      observe(): the decision point, the scorecard, the last stroke's result (no coordinates);
                   camera_job(): behind, side and overhead (with the previous stroke's path), one PNG (sim/golf.py)
"""

from __future__ import annotations

import os

from core.spec import load
from turnshot import ShotLayer, TurnShotMatch, load_rules

HERE = os.path.dirname(os.path.abspath(__file__))
SIM = os.path.join(HERE, "sim")
DECL = load(HERE)
SCHEMA = {
    "putt": {
        "aim_deg": {
            "type": "number",
            "digits": 3,
            "min": -180.0,
            "max": 180.0,
            "unit": "deg in the green's frame: 0 = +x, + = counter-clockwise seen from above",
        },
        "speed": {
            "type": "number",
            "digits": 4,
            "min": 0.05,
            "max": 6.0,
            "unit": "m/s, the ball's initial rolling speed",
        },
    }
}


def rules(seed):
    return load_rules(SIM, seed)


class Golf(TurnShotMatch):
    def schema(self, seat):
        return SCHEMA


class GolfSport:
    name, protocol, revision = "golf", "turn", DECL["sport"]["revision"]
    source_dirs = [HERE]
    workers = ()  # the views are drawn in-process (about 0.3 s per decision point)

    def build(self, cfg):
        return Golf(cfg, rules(cfg.seed))

    def action_layer(self, level):
        return ShotLayer(lambda m: m.S)

    def worker_call(self, obj, kind, job):
        return job  # camera_job() already drew the PNG

    def spec(self, cfg, seat, match):
        S = match.S
        sp = S.spec(False)  # the single round (no seed: rule O3)
        sp["image"] = {
            "size_px": list(S.IMAGE_SIZE),
            "layout": S.IMAGE_LAYOUT,
            "tiles": {k: {"x": x, "y": y, "w": w, "h": h} for k, (x, y, w, h) in S.TILES.items()},
        }
        return sp

    def players(self):
        import builtin_players as bp

        return {"reference": bp.Reference, "random": bp.Random}


SPORT = GolfSport()
