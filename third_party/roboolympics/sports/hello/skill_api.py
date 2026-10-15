"""hello's skill API (docs/framework_zh.md §6.3; skilleval/README.md): one skill, `aim`, the same in both modes.

  high  Agent as Athlete: the act {type: aim, x, y} IS the skill; the platform implements it (plugin.AimLayer)
  low   Agent as Robotics Engineer: the player implements aim(x, y, *, state, spec) itself, from the toolkit only
        (seat/toolkit.toml: the launcher's throw interface and its flight model), at /app/output/skill/aim.py

This file is the contract's single source: the request ranges (the high level's act schema is built from them,
plugin.AIM), what is measured, the tolerance and what counts as success (skilleval/contract.py). Standard library only."""

from skilleval.contract import SkillContract, Tolerance

AIM = SkillContract(
    name="aim",
    call="aim(x, y)",
    args={
        "x": {"type": "number", "min": 1, "max": 9, "unit": "m", "doc": "landing point, metres ahead (+x)"},
        "y": {"type": "number", "min": -3, "max": 3, "unit": "m", "doc": "landing point, metres to the left (+y)"},
    },
    returns="one low-level command {type: throw, yaw, pitch, speed} (the throw schema of game spec at the low level: "
    "yaw -20..20 deg, pitch 0..60 deg, speed 1..12 m/s)",
    measured="the ball's first ground contact (its centre where the ball first touches the ground, interpolated "
    "inside the 2 ms physics step) when the command is thrown in the simulator WITHOUT execution scatter; "
    "landing_err_m = its distance from (x, y)",
    tolerances=(Tolerance("landing_err_m", 0.03, text="the ball first touches the ground within 3 cm of (x, y)"),),
    must_be_zero=("refused",),
    feasible="every API-test request comes from our reference player's own matches and the platform's "
    "implementation delivers it within the tolerance with a command inside the launcher's ranges",
)
CONTRACTS = {"aim": AIM}
