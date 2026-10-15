"""Disc golf on the v0.2 core (PDGA Official Rules of Disc Golf, rev. 2026-01-01). Migrated from the Season-1 task
discgolf-course-program-standard (docs/framework_zh.md §10, stage 1). One format: the single round (user decision of
2026-10-08, as review item B10: the legacy duel discgolf-duel-program-standard had no interaction between the players,
the discs never met on identical courses, so a head-to-head is the single scores on the same seeds; sim/sport.py keeps
its duel code, unused).

The simulator is sim/, unchanged from the legacy tree: sim/flight.py (the Hummel 2003 disc-flight model, RK4),
sim/discgolf.py (the 6-hole course per seed, trees, basket, ground, OB and mandatory rules, seeded wind and execution
scatter, the MuJoCo views) and sim/sport.py (the legacy rules module of the shared turn service: PDGA order of play,
sudden-death play-off holes, decide). turnshot.py (a byte-identical copy of sports/bowling/turnshot.py) maps the rules
module onto contract ②; this file adds only what is disc golf's:
  world + referee  DiscGolf(TurnShotMatch): one round of the seed's 6-hole course, scored against the success line
  actions          one level (low): {"type": "throw", disc, speed, launch_deg, nose_deg, bank_deg, spin_rps, aim_deg}
                   (SCHEMA, then the legacy validate(): |spin| >= 4, 4-decimal rounding, disc_id), with the chess
                   clock at the throw (ShotLayer)
  observation      observe(): the decision point (hole, par, throw, lie, basket, the last throw's result), the
                   scorecard; camera_job(): overview, tee and windsock views in one PNG (sim/discgolf.py)
"""
from __future__ import annotations

import os

from core.spec import load
from turnshot import ShotLayer, TurnShotMatch, load_rules

HERE = os.path.dirname(os.path.abspath(__file__))
SIM = os.path.join(HERE, "sim")
DECL = load(HERE)
NUM = {"type": "number", "digits": 4}
SCHEMA = {"throw": {"disc": {"type": "enum", "values": ["putter", "midrange", "driver"],
                             "doc": "the disc (aerodynamic coefficients in /spec discs)"},
                    "speed": {**NUM, "min": 5.0, "max": 30.0, "unit": "m/s, release speed"},
                    "launch_deg": {**NUM, "min": -5.0, "max": 35.0, "unit": "deg, the velocity above horizontal"},
                    "nose_deg": {**NUM, "min": -10.0, "max": 15.0, "unit": "deg, the disc's pitch above the velocity"},
                    "bank_deg": {**NUM, "min": -60.0, "max": 60.0, "unit": "deg, + = left edge down (seen from behind)"},
                    "spin_rps": {**NUM, "min": -20.0, "max": 20.0,
                                 "unit": "rev/s, + = clockwise seen from above; |spin| >= 4"},
                    "aim_deg": {**NUM, "min": -180.0, "max": 180.0,
                                "unit": "deg, heading in the hole's frame (0 = +x, 90 = +y)"}}}


def rules(seed):
    return load_rules(SIM, seed)


class DiscGolf(TurnShotMatch):
    def schema(self, seat):
        return SCHEMA


class DiscGolfSport:
    name, protocol, revision = "discgolf", "turn", DECL["sport"]["revision"]
    source_dirs = [HERE]
    workers = ()                # the views are drawn in-process (about 0.3 s per decision point)

    def build(self, cfg):
        return DiscGolf(cfg, rules(cfg.seed))

    def action_layer(self, level):
        return ShotLayer(lambda m: m.S)

    def worker_call(self, obj, kind, job):
        return job              # camera_job() already drew the PNG

    def spec(self, cfg, seat, match):
        S = match.S
        sp = S.spec(False)                       # the single round (no seed: rule O3)
        sp["image"] = {"size_px": list(S.IMAGE_SIZE), "layout": S.IMAGE_LAYOUT,
                       "tiles": {k: {"x": x, "y": y, "w": w, "h": h} for k, (x, y, w, h) in S.TILES.items()}}
        return sp

    def players(self):
        import builtin_players as bp
        return {"reference": bp.Reference, "random": bp.Random}


SPORT = DiscGolfSport()
