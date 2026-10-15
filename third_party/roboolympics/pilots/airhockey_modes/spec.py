"""The /spec of the air-hockey two-mode pilot (what the player reads first), per level. `python3 spec.py` writes
spec_high.json and spec_low.json next to this file."""
from __future__ import annotations

import json
import sys
import tomllib
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SLOWDOWN = 10


def spec(level: str) -> dict:
    sys.path.insert(0, str(HERE))
    from contract import CONTRACT
    import planner as PL
    rules = tomllib.loads((ROOT / "sports" / "airhockey" / "sport.toml").read_text())["spec"]
    common = {
        "level": level,
        "pilot": "RoboOlympics two-mode pilot: air hockey with the real KUKA iiwa 14 limits (revision v5, 2026-10-08; v6, 2026-10-10: the mallet stays within 2 cm of the table)",
        "format": "one game against the scripted opponent (seat B; level from the task: 1 does not guard its goal, 2 guards "
                  "the 0.80 line at 0.6 m/s): first to 7 goals, or the 120 s cap of play then the tiebreak chain; you "
                  "are seat A (own half x < 0, you attack the goal at x = +0.974)",
        "rules": {k: rules[k] for k in ("rally", "goal", "seven_second_rule", "dead_puck", "review", "game", "table",
                                        "puck", "mallet", "rates")},
        "geometry": {"length_m": 1.948, "width_m": 1.038, "goal_width_m": 0.35, "puck_radius_m": PL.PUCK_R,
                     "mallet_radius_m": PL.MALLET_R, "contact_distance_m": round(PL.CONTACT, 5),
                     "workspace": "own half: 0.05 <= -x <= 0.92, |y| <= 0.47, within 1.45 m of the robot base at (-1.51, 0)",
                     "home": [-0.82, 0.0], "puck_spot_abs_x_m": 0.58},
        "robot": "a KUKA LBR iiwa 14 R820 holding the Air Hockey Challenge striker (2 striker joints + mallet); every "
                 "joint held to the real limits (0.95 x the data-sheet velocities, the AHC accelerations = 10 x the "
                 "velocities, estimated jerks; motor torques): a mallet reaches about 1.0-1.2 m/s on a typical line, "
                 "a puck leaves a hit at about 1.4 x the mallet's speed. The opponent's arm has the same limits",
        "timing": {"control_dt_s": 0.01, "slowdown": SLOWDOWN,
                   "real_time": f"the world runs on its own clock at 1/{SLOWDOWN} of real speed and never waits for "
                                "you; an act takes effect at the next 10 ms control step (reply took_effect); one act "
                                "in flight; after every rally (goal, foul, dead puck) the world freezes until you send "
                                "ready {why, plan}"},
        "observation": "TRUE state (pilot simplification; the official sport gives only a camera image): t (game s), k "
                       "(control step), rally, score [you, opponent], puck, puck_v, mallet, mallet_v, opp_mallet, "
                       "opp_mallet_v (table frame, m and m/s), q, qd (9 joints: 7 iiwa axes + 2 striker joints), "
                       "q_target, q_target_v (the servo's current joint target and its velocity), plan_rows_left, "
                       "control (path | rows), puck_half (own | opponent | null) and puck_half_s (the seven-second "
                       "clock), play_time_left_s, state, rallies",
        "toolkit": "/opt/robot/toolkit.toml",
    }
    if level == "high":
        common.update(
            commands={
                "move": {"x": "m", "y": "m", "does": "the mallet to (x, y) along the platform's path (at most 2 m/s and "
                         "10 m/s^2, retimed to the joint limits, IK, motors) and held there (the official sport's target)"},
                "strike": {"t": "game time of the contact (between the next step and 3 s ahead)",
                           "x": "the puck centre's x at t (your prediction)", "y": "the puck centre's y at t",
                           "dx": "drive direction x (any length, normalised)", "dy": "drive direction y",
                           "speed": f"mallet speed at contact, {PL.SPEED_MIN}-{PL.SPEED_MAX} m/s (capped to what the "
                                    "joints allow on that line: reply speed / speed_cap)",
                           "does": "the platform approaches the wind-up point behind the contact point (around the "
                                   "puck if needed; the puck assumed near (x, y)), settles there, drives along the "
                                   "line timed to pass the contact point (x, y) - contact_distance x d at t, follows "
                                   "through and stops. Reply: contact_t, speed, speed_cap, late_s (> 0: not enough "
                                   "time to wind up, the contact comes that much later), feasible, why. It does not "
                                   "watch the puck: aim, timing and clearing the way are yours. A new command "
                                   "replaces the strike under way"},
                "none": {"does": "hold where the mallet is"}},
            deliverable={"path": "/app/output/skill/player.py (other modules of yours in /app/output/skill/)",
                         "interface": "class Player: __init__(self, spec); act(self, view) -> a list of commands "
                                      "(move / strike / none, as above; [] keeps the last); called every 2 control "
                                      "steps (20 ms) with view = {'state': the observation, 'replies': the results "
                                      "of your previous commands}",
                         "rules": "numpy and your own files only; deterministic; no network; no state from the match "
                                  "process",
                         "evaluation": "after the match, in an isolated container: your FINAL player.py with the "
                                       "platform's skills plays three games of the official format against the "
                                       "scripted opponent at levels 1, 2 and 3 (fixed seeds)"})
    else:
        common.update(
            commands={
                "plan": {"q": "rows of the 9 joint targets, row 0 = the target at the end of the next 10 ms control "
                              "step, at most 300 rows",
                         "k0": "optional: the step your rows are timed from (row i = the target of step k0 + 1 + i); "
                               "rows already past when the plan takes effect are dropped (reply skipped)",
                         "servo": "between rows the target is interpolated linearly at 1 kHz and followed by the "
                                  "robot's setpoint generator (velocity / acceleration / jerk limits) and PD with "
                                  "inverse dynamics; torques within the motors'",
                         "refused": "a row outside the joint ranges; a row-to-row velocity beyond the velocity limit "
                                    "(the first row counted from q_target) or acceleration beyond the acceleration "
                                    "limit (counted from q_target_v); the mallet centre (forward kinematics of a row) "
                                    "outside the workspace (1 cm tolerance) or more than 2 cm off its table height; "
                                    "the mallet more than 2 cm off its table height at the setpoints the robot's "
                                    "setpoint generator would make of the rows (a row that asks a joint for more "
                                    "acceleration or jerk than it has is limited joint by joint, which bends the path)",
                         "does": "replaces the plan under way; after its last row the target holds"},
                "hold": {"does": "drop the plan, hold the current target"}},
            limits={"velocity_rad_s": [1.40935, 1.40935, 1.65806, 1.24355, 2.15548, 2.23838, 2.23838, 3.0, 3.0],
                    "acceleration_rad_s2": [14.8353, 14.8353, 17.4533, 13.09, 22.6893, 23.5619, 23.5619, 50.0, 50.0],
                    "jerk_rad_s3": [741.765, 741.765, 872.665, 654.5, 1134.465, 1178.095, 1178.095, None, None],
                    "range_rad": "the robot model's joint ranges (also ahkin.RANGE_LO / RANGE_HI)"},
            skill_api=CONTRACT,
            deliverable={"path": "/app/output/skill/skill.py (other modules of yours in /app/output/skill/)",
                         "interface": "class Impl: __init__(self, spec); move(self, view, x, y) -> commands; "
                                      "strike(self, view, req) -> commands; step(self, view) -> commands. req = {t, x, "
                                      "y, dx, dy, speed}: the high level's strike (skill_api); move: the mallet to (x, "
                                      "y) at rest, as fast as you can; step is called every 2 control steps (20 ms) "
                                      "when no request comes; commands = a list of plan / hold; view = {'state': the "
                                      "observation, 'replies': the results of your previous commands}",
                         "rules": "only the low-level toolkit (/opt/robot: the robot model, lib/ahkin.py) with numpy / "
                                  "mujoco and your own files; deterministic; no network; no state from the match "
                                  "process (evaluated fresh); the platform's skills are not available",
                         "evaluation": "after the match, in an isolated container: (1) API test: hidden strike requests "
                                       "from our reference player's games (feasible for the arm; public examples in "
                                       "/opt/robot/apitest_cases_public.json), each from the recorded state of the "
                                       "arm, its servo and the puck; success per skill_api; (2) competition test: our "
                                       "fixed reference strategy calls YOUR move / strike in games against the "
                                       "scripted opponent at levels 1, 2 and 3"})
    return common


if __name__ == "__main__":
    for lv in ("high", "low"):
        (HERE / f"spec_{lv}.json").write_text(json.dumps(spec(lv), indent=1))
        print(HERE / f"spec_{lv}.json")
