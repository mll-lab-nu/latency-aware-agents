"""Our HIGH-level reference player (deliverable form: /app/output/skill/player.py, class Player): the reference
strategy (ref_strategy.py, true state) sending the platform's skills move / strike."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ref_strategy import RefStrategy  # noqa: E402


class Player:
    def __init__(self, spec):
        g = ((spec or {}).get("geometry") or {}).get("goal_width_m", 0.35)
        self.s = RefStrategy(goal_width=g)

    def act(self, view):
        return self.s.tick(view["state"])
