"""Screening hooks of wheeled 5 v 5 soccer for tools/screen (step 4 of docs/construction_zh.md; tools/screen/README.md).

The variants are built-in teams (sports/soccer_ssl/players) that play through the agent's skills. `VARIANTS` is the
one table: tools/screen (through make_player, once the plugin exists) and the world-level harness (worldrun.py, before
it) both read it. Every variant names a players.make team, its options and the effect delay in control steps.

Capabilities (proposal §6.1) and their impairments, weak / strong:
  A1 ball position     Gaussian noise on the ball position, 1 cm / 5 cm (velocity from 0.1 s differences)
  A2 ball dynamics     rolling deceleration 50 % too high / no sliding phase and 100 % too high
  A4 latency           acting on the world 0.1 s / 0.2 s old (instead of the vision's 0.04 s), no compensation
  B2 reading opponents support spots ignore the opponents / passes and spots ignore them
  B3 teamwork          the ball-getter drawn at random every 2 s / every field robot goes for the ball
Value-type skill necessity (decisions 10-11): `nopass` = the reference's decisions without ever passing.
"""
from __future__ import annotations

FORMAT = "match"
ACTION_LEVEL = "high"
DELAY = 2                   # control steps (10 ms) = sport.toml [timing] reaction_delay_steps (core 0.2.18; LATENCY 10-13)
LEVELS = [2, 3]             # the official opponent levels (L1 is the warm-up)
REFERENCE = "reference"
REPLICAS = ["reference_rep1", "reference_rep2"]
BASELINES = ["allchase", "fixed", "shootonly", "random", "idle"]
NECESSITY = []              # kick is the necessary skill (no program scores without it); see VALUE_NECESSITY
VALUE_NECESSITY = {"passing": "nopass"}
SENSITIVITY = {
    "A1": ["a1_weak", "a1_strong"],
    "A2": ["a2_weak", "a2_strong"],
    "A4": ["a4_weak", "a4_strong"],
    "B2": ["b2_weak", "b2_strong"],
    "B3": ["b3_weak", "b3_strong"],
}

VARIANTS = {
    "reference": dict(team="reference"),
    # replicas: move targets nudged by 0.1-0.2 mm (survives the 4-decimal rounding of commands) (nothing a decision sees; the match diverges chaotically).
    # (Until 10-12 the replicas added 0.1 mm ball noise, which also switched the velocity to 0.1 s differences of
    # positions: a real impairment on L2, -2.1 goals/match; replaced.)
    "reference_rep1": dict(team="reference", opts=dict(target_eps=1e-4)),
    "reference_rep2": dict(team="reference", opts=dict(target_eps=2e-4)),
    "nopass": dict(team="reference", opts=dict(passing=False)),
    "allchase": dict(team="allchase"),
    "fixed": dict(team="fixed"),
    "shootonly": dict(team="shootonly"),
    "random": dict(team="random"),
    "idle": dict(team="idle"),
    "ref_shoot0": dict(team="reference", opts=dict(shoot_range=0.0)),   # the reference before 10-12 (no shooting rule)
    "ref_shoot1": dict(team="reference", opts=dict(shoot_range=5.5, shoot_margin=-0.15)),   # calibration probes
    "ref_shoot2": dict(team="reference", opts=dict(shoot_range=6.5, shoot_margin=-0.4)),   # = the reference now
    "nopass_shoot2": dict(team="reference", opts=dict(passing=False, shoot_range=6.5, shoot_margin=-0.4)),
    "a1_weak": dict(team="reference", opts=dict(ball_noise=0.01)),
    "a1_strong": dict(team="reference", opts=dict(ball_noise=0.05)),
    "a2_weak": dict(team="reference", opts=dict(roll_dec=0.675)),
    "a2_strong": dict(team="reference", opts=dict(roll_dec=0.9, model_slide=False)),
    "a4_weak": dict(team="reference", opts=dict(lag=0.06, lookahead=0.0)),   # the world 0.1 s old, uncompensated
    "a4_strong": dict(team="reference", opts=dict(lag=0.16, lookahead=0.0)),  # 0.2 s old
    "b2_weak": dict(team="reference", opts=dict(read_opp="passes")),
    "b2_strong": dict(team="reference", opts=dict(read_opp=False)),
    "b3_weak": dict(team="reference", opts=dict(roles="random", role_period=2.0)),
    "b3_strong": dict(team="reference", opts=dict(roles="swarm")),
}


def make_player(variant, spec, seat, seed, delay, opts):
    """a built-in player for tools/screen/play.py (act(view) over the plugin's observation). Needs the plugin's
    observation format (line E, skill_api.py); builtin_players.TeamPlayer adapts a team to it."""
    import builtin_players as BP  # noqa: PLC0415

    v = VARIANTS[variant]
    return BP.TeamPlayer(v["team"], seed=seed, spec=spec, **dict(v.get("opts", {}), **(opts or {})))


def summary(result, players):
    sc = result.get("scores") or {}
    a, b = sc.get("A", 0), sc.get("opponent", sc.get("B", 0))
    tot = a + b
    return dict(won=result.get("winner") == "A", goal_diff=a - b, point_share=(a / tot) if tot else 0.5)
