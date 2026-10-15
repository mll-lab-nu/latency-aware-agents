# RoboOlympics: robot air hockey

Tasks: `tasks/airhockey-match-wallclock-program-standard` (the agent against a scripted opponent, levels 1-4) and
`tasks/airhockey-duel-wallclock-program-standard` (two agents). Sources: `scripts/olympiad/airhockey/` and
`scripts/olympiad/shared/airhockey/sim/`; generate with `scripts/olympiad/generate.py`. Revision
`airhockey-program-v2` (2026-10-05). Results of `airhockey-program-v1` (four rallies, possession-timeout points,
draws, asymmetric mallets) are not comparable and must not be mixed with v2 results.

What it adds to the programme (S2): an invasion-style game with continuous possession changes, attack and defence
transitions and rebound geometry, played by articulated arms under physical torque limits.

## Rules

Rulebook: USAA (United States Air Hockey Association) Official Rules of Air Hockey,
<https://goldstandardairhockey.com/usaa-official-rules-of-air-hockey/> (retrieved 2026-10-05). Followed as written:
a goal is the puck wholly in the goal (I.2, V.1) and scores for the other player, also an own goal; the player scored
upon serves (I.4); the seven-second rule (I.10) is a foul whose penalty is forfeiture of the puck (VI.1), not a point;
a puck touching the centre line belongs to nobody (I.11); players stay in their own half (I.12).

Deviations (also in `spec.py` `DEVIATIONS`, the README and the prompt):

| Rule | Deviation | Why |
|---|---|---|
| I.1, III.1 | one game to 7 goals instead of a match of several games | session length |
| none (no time limit) | safety cap: 120 s of play or 40 rallies; then more goals, else the last goal, else less time with the puck in one's own half, else the player who did not have the first possession | no draws, a bounded match |
| III.1-2 | the game starts with the puck in `FIRST_PLAYER`'s half (the coin toss) instead of a face-off | a fair, run-time choice of who starts |
| III.2 | a face-off places the puck at rest at the centre, both mallets home | no referee hand |
| IV.6, I.11 | dead puck: out of both robots' reach for 2 s, or on the centre line for 7 s, gives a face-off | the arms' reach; play cannot stall |
| I.14-15 | no time-outs; after every goal, foul or dead puck the mallets go home and the world freezes for the review | program play (review after every rally, rule F5) |
| I.8, I.13, VI.6 | palming, goal-tending and topping cannot occur | only the mallets touch the puck; the puck is planar |
| I.12, VI.9 | the mallets cannot cross the centre line | the workspace ends 0.05 m before it |

Single player and duel use identical rules; only the opponent differs.

## World and interface

- MuJoCo 3.3.7; two seven-axis KUKA iiwa arms with passive two-axis strikers (AirHockeyChallenge, MIT, commit
  34729081d; `scene/ORIGIN.md`), the AirHockeyChallenge table (1.948 x 1.038 m, goal 0.25 m, puck r 0.03165 m,
  mallet r 0.04815 m). Physics 1 kHz, control 10 ms (the protocol's step).
- The robots are point-symmetric through the table centre with identical parameters (the v1 iiwa2 mallet lacked
  `class="robot"`: condim 3 / solref (.02, 1) / priority 0 instead of 4 / (.02, .3) / 2; fixed, and
  `test_mirror_symmetry` checks geometry, parameters and mirrored puck paths to 1e-6).
- Puck: 10 g, air-cushion drag about 10 %/s, rail restitution about 0.8 (v1: 50 %/s and 0.46).
- Action: a mallet target in the own workspace (0.05 <= |x| <= 0.92, |y| <= 0.47, at most 1.45 m from the base);
  outside = HTTP 400. Path limits 2 m/s, 10 m/s^2; damped least-squares IK; gravity-compensated torque-limited
  joint control. Every puck position away from the centre line is reachable by the mallet of its half (test); the
  only unreachable positions are near the centre line at the sides (|y| > ~0.28), which the dead-puck rule handles.
- Observation: one 320 x 240 RGB image per call (calibration public), own robot state, score, time of play, the
  rallies that ended (`news`, `last_rally`). No puck or opponent state, no seed (tests).
- Duel calls are under `/<seat>/`; identity is the seat container's network address; one render process and one
  metering proxy per seat; metering files byte-identical to table tennis (single-seat copy for the single task,
  per-seat copy for the duel).

## Opponent levels (single player)

| Level | Mallet speed | Reaction delay | Defence | Shot |
|---|---|---|---|---|
| 1 | 0.4 m/s | 150 ms | none (stays where it is) | straight at the goal centre, loose line-up |
| 2 | 0.6 m/s | 100 ms | guard line, follows the puck's y | straight, centre |
| 3 | 1.0 m/s | 60 ms | guard line, follows the puck's y | far corner |
| 4 | 2.0 m/s | 30 ms | guard line at the predicted crossing (rail bounces folded in) | far corner, tight line-up |

## Exploit review (R6; tests in `test_airhockey.py`, `test_exploit_*`)

| Strategy | How the rules stop it |
|---|---|
| hold the puck / never shoot | seven-second foul: the opponent gets the puck; no point (v1 gave a point) |
| park the puck where no mallet reaches it (rails, centre line) | the workspace reaches the rails; the centre-line side zones are dead pucks (face-off after 2 s); a puck resting on the centre line is a dead puck after 7 s |
| park the mallet in front of the own goal and never play | the parker commits seven-second fouls and loses the territory last resort; a playing opponent wins (test) |
| do nothing | loses to the level-1 opponent (test); a single-player session that ends early is a valid loss |
| ram the opponent | impossible: the two workspaces are 0.1 m apart (more than two mallet radii) |
| run out the clock when ahead | allowed, as in a timed game: the cap decides by goals first |

## Broadcast

`broadcast=True`: `environment/sim/broadcast_log.py` writes `/data/broadcast.jsonl` (qpos every 2 control steps,
events `run`, `point` (goal), `stoppage` (foul / dead puck), `pause` (review), `end`). The grader never reads it.
A broadcast adapter (`scripts/olympiad/broadcast/adapters.py`, not changed here) needs: the model from
`hockey.scene_xml()` (import `hockey` from the match's sim directory), the qpos layout (puck_x, puck_y, puck_yaw, then
arm A's 9 and arm B's 9 joints, as in the model), the `overview` camera, and no extra geometry (both robots are in the
scene in both tasks).

## Validation (2026-10-05, revision v2, jobs tagged `ahfix`)

Checks: `generate.py check` clean; `ci_checks/run_all.sh` (both tasks conform; only the repository-wide client-image
WARN); `check-olympiad.py airhockey`: 52 ok, 0 warnings, 0 failures; `task_conventions.py --check` rc 0;
`test_program_play.py` ALL OK on both tasks; `test_airhockey.py` 18 tests OK.

Harbor runs (all valid_trial = 1, deterministic = 1, the configuration as given, official_config = 1):

| Run | Result | Decided by |
|---|---|---|
| oracle L1 / 101 / x20 (default) | win 3:0 | cap, goals |
| oracle L1 / 111 / x1 | win 3:0 | cap, goals |
| oracle L2 / 202 / x10 | win 1:0 | cap, goals |
| oracle L3 / 303 / x10 | win 1:0 | cap, goals |
| oracle L4 / 404 / x10 | loss 0:0 | cap, territory |
| nop L1 / 101 / x20 | valid loss (no start) | - |
| oracle vs oracle x10, first A / B | A wins / B wins, 0:0 | cap, territory |
| oracle vs oracle x20, first A / B | A wins / B wins, 0:0 | cap, territory |
| luna L1 / 101 / x10 | loss 0:0 (4 fouls, 166 contact steps) | cap, territory |
| luna vs luna x10, first A | B wins 0:0 | cap, territory |

Timing: per-step simulator lateness p99 below 0.3 ms everywhere; maximum 0.003-0.07 s at x1/x10; three x20 runs had
one stall of about 3 s (0.1 % of steps late by more than one period), report-only.

Open: two reference players do not score against each other in 120 s of play, so oracle duels are decided by the
territory last resort; the reference does not beat level 4.
