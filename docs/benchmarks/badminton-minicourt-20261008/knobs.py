"""The official sport (revision wallclock-7 / duel-9: the small court, gain 4, the real Franka limits, the EGC
gantry) for the minicourt study scripts: no patches, only the import paths and a description of the physics."""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
BMDIR = ROOT / "sports" / "badminton"
for p in (str(ROOT), str(BMDIR), str(BMDIR / "sim")):
    if p not in sys.path:
        sys.path.insert(0, p)

import bm as BM  # noqa: E402
import bduel as BD  # noqa: E402
import builtin_players as BP  # noqa: E402
import plugin as PL  # noqa: E402

VARIANT = os.environ.get("TAG", "mini")
GAIN = BM.RACKET_GAIN
SERVE_CAPS = [float(x) for x in (os.environ.get("SERVE_CAPS") or "1.8,2.4,3.0,4.2").split(",")]


def describe() -> dict:
    return dict(variant=VARIANT, gain=BM.RACKET_GAIN, court_scale=BM.COURT_SCALE, base_v=BM.BASE_VMAX,
                base_a=BM.BASE_AMAX, base_j=BM.BASE_JMAX, base_f=BM.BASE_FORCE, arm="franka_panda FCI limits")
