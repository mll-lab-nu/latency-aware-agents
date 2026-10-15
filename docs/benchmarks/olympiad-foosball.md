# RoboOlympics: table football (foosball)

Tasks: `tasks/foosball-match-wallclock-program-standard` (the agent against a scripted team, levels 1-4) and
`tasks/foosball-duel-wallclock-program-standard` (two agents). Sources: `scripts/olympiad/foosball/` and
`scripts/olympiad/shared/foosball/sim/`; generate with `scripts/olympiad/generate.py`. Revision `foosball-program-v1`
(2026-10-05). Exhibition sport (rules §12).

What it adds to the programme (S2): an invasion game on a table where one agent drives **four rods at once** (11
figures): blocking with several rods, lifting own rods out of a shot, passing from rod to rod, and the goalkeeper and
2-rod sharing the goal. Partial observability: the figures and rods hide the ball from the overhead camera.

## Rules

Rulebook: ITSF Rules of Table Soccer, Standard Matchplay Rules (SMR), Version 2.0, December 2023,
<https://s3.tablesoccer.org/organization/ITSF/rules/Standard_Matchplay_Rules_2024_en.pdf>. The SMR name their rules
rather than number them; the references below are its sections.

Followed as written (in `foos.py` `Foosball.referee`, `touch`, `end_rally`):

| SMR | Rule | Implementation |
|---|---|---|
| 5.1, 7.1.1 | kick-off by the team that conceded; the first one by the initial flip | `next=('kickoff', conceder)`; `FIRST_PLAYER` is the flip |
| 5.2 | opening sequence: a move by the restart figure, a contact by another figure, then one by any figure but that one, before the ball may leave the restart rod; Incomplete Opening Sequence -> 5-rod Possession Award | `opening` (collapsed contact sequence of the restarting team's restart rod group, length >= 3); the reply's `restart.opening_complete` |
| 6 | possession clock while the ball is reachable from a rod; goalie rods (goalkeeper + 2-rod) 15 s, 5-rod 10 s (5-rod award), 3-rod 15 s (goalie rod award) | per (team, rod group) clock while the ball's centre is within `X_REACH` = 0.077 m along u of a rod of the group (the figures cover every v); reset when no rod of the group reaches it |
| 7.1.1 | a goal is the ball wholly over the goal line; own goals count | centre beyond 0.6175 m in the mouth |
| 7.2 | dead ball (stationary, unreachable): between the 5-rods kick-off, behind a 5-rod the nearest goalie rod | 1 s at rest and unreachable |
| 7.3 | ball off table: to the goalie rod of the team that did not play it off | implemented; the cover makes it practically impossible |
| 10 | Illegal Spin: a transition from a contact spin (more than 360 degrees before or after the contact) -> 5-rod award; a spin goal does not count, an own goal from a spin does | `sweep` per rod (rotation since the last reversal or stop), checked at contacts and transitions; the stops (+-2.8 rad) make it impossible, a unit test forces it |
| 18 | Goalie rod / 5-rod Possession Award: the ball at rest at the non-offending team's goalkeeper / central 5-rod figure | `restart(('goalie'|'midfield', team))`, the side of the figure and +-8 mm from the seed |

Deviations (also in `spec.py` `DEVIATIONS`, the README and the prompt):

| SMR | Deviation | Why |
|---|---|---|
| 3 | one game to 5 points instead of a match of several games | session length (points and games targets are event settings) |
| none | safety cap 240 s of play or 40 rallies; then more goals, else the last goal, else less time with the ball in one's own half, else the team without the first kick-off | no draws (R4), a bounded game |
| 5.1.2, 5.3 | the referee places the ball; play resumes when the review is handed in; no delay-of-game rule; the restart rod's clock already runs during the opening | program play (reviews), stalling closed by the clocks |
| 18 (Continue), 17, 7.1.2, 15.2 | possession awards only: no option to continue, no warnings, penalty shots, time-outs or reset calls | an automatic referee; R5 (fouls give possession, never points) |
| 15.2-15.3 | jarring and reset cannot occur | the table is fixed and rigid |
| 10 | rods have stops at +-2.8 rad | robot rods; spinning is impossible (the check stays) |
| 7.3 Illegal Aerial | a transparent cover 4 cm above the playfield | no aerials, no ball off table |
| 11-12, 7.2 Playing the Ball Dead | passing, wall and intentional dead-ball rules are not judged | they need a referee's judgement of stopped, pinned or intentional |
| - | figures of the two teams do not collide with each other | robot rods; only the ball connects the teams |

## World and interface

- MuJoCo 3.3.7; a procedural table (primitive geoms, no meshes; Foosball_CU, Columbia, Apache-2.0, was considered: it
  needs the `sdflib` mesh plugin and multi-MB meshes, and its scene is in centimetres). Playfield 1.20 x 0.68 m, goals
  0.20 m, side-wall fillets (r 0.02 m) so a ball cannot rest against a wall, a cover at 0.04 m. Physics 1 kHz, control
  10 ms; a control step costs about 0.5 ms.
- Rods 0.15 m apart, axes 0.09 m above the playfield; per team goalkeeper rod 3 figures (+-0.11 m travel), 2-rod 2
  (+-0.19), 5-rod 5 (+-0.06), 3-rod 3 (+-0.11). A figure is a torso box and a capsule foot along u (r 0.012 m, half
  length 0.014 m, 3 mm above the playfield). Every ball position on the playfield is within reach of a rod
  (`test_reach_is_physical_and_covers_the_playfield` checks X_REACH against MuJoCo's geometry).
- Ball: 35 mm, 24 g, rolling resistance about 0.1 m/s^2; walls bounce, figures are dead (a block stops the ball at the
  figure). A kick at 80 rad/s sends the ball at 4-6 m/s; an off-centre hit sends it at an angle (about 1.15 x
  asin(offset / 0.0295 m)).
- Action: per-rod targets `{rods: {rod: {slide, angle, omega?, speed?}}}` for any subset of the rods; a speed- and
  acceleration-limited path per joint (2.5 m/s, 40 m/s^2; 80 rad/s, 4000 rad/s^2) tracked by force/torque-limited
  motors (60 N, 5 N m); the command reaches the motors 20 ms (2 control steps) after the step that takes it. Out of
  range = 400; one act in flight per seat (429); 409 while frozen.
- Observation: one 384 x 240 RGB image from the seat's overhead camera (own goal on the left; calibration in `/spec`);
  own rods (slide, angle, velocities, commands), score, fouls, clocks, the restart and its opening state, `news` and
  `last_rally`. No ball or opponent state, no seed (tests). The reference player's camera errors (300 random states):
  ball 2.2 mm mean, 8 mm p95 (partly hidden by figures); opposing figures 3 mm mean.
- The two teams are point-symmetric through the table centre (`test_mirror_symmetry`: every figure's geometry and
  contact parameters, masses, inertias, joint ranges, damping, motor limits, and a mirrored kick's ball path to 1e-6).

## Opponent levels (single player)

The scripted team is the reference policy (`foos_policy.py`) on the true ball position a reaction delay ago (and, from
level 3, the agent's figure positions), in the trusted sim, with the same rods and motors.

| Level | Delay | Slide speed | Kick | Style |
|---|---|---|---|---|
| 1 | 140 ms, no extrapolation | 0.6 m/s | 20 rad/s | solo; never shoots from its 5-rod and 3-rod |
| 2 | 90 ms | 1.0 m/s | 35 rad/s | solo |
| 3 | 60 ms | 1.6 m/s | 55 rad/s | team (aims, passes, lifts, splits the goal) |
| 4 | 30 ms | 2.5 m/s | 80 rad/s | team |

## Does passing and multi-rod coordination pay off?

`scripts/olympiad/foosball/diag_passing.py` (results of the run below in `docs/benchmarks/olympiad-foosball-diag.json`).
Both players see the ball 40 ms late with 3 mm noise (the camera player's error) and play on the simulator in-process.

Full games (15 seeds x both first kick-offs x both seats = 60 games per pair; every game ended 5:x, none by the cap):

| Pair (test : base) | Games won by test | Goals | One-sided binomial p |
|---|---|---|---|
| `split` : `solo` (goalkeeper covers the 2-rod's gap vs both on the ball's line) | 40 / 60 (67 %) | 260 : 204 | 0.007 |
| `team` : `shooter` (reference vs "shoots from where it gets the ball, clears its lane") | 38 / 60 (63 %) | 244 : 222 | 0.026 |
| `team` : `nopass` (passing on vs off, all else equal) | 35 / 60 (58 %) | 234 : 221 | 0.12 |
| `team` : `solo` | 31 / 60 (52 %) | 237 : 230 | 0.45 |

Set pieces (200 possessions from rest each; outcome = goal before the opponent touches the ball):

| Possession | Goals |
|---|---|
| 5-rod, shot at the goal (`shooter`) | 10 / 200 (5 %) |
| 5-rod, pass to the 3-rod when that lane is clearly more open (`team`) | 4 / 200 (2 %) |
| 3-rod shot against a stacked defence (goalkeeper on the 2-rod's line) | 66 / 200 (33 %) |
| 3-rod shot against the split defence | 23 / 200 (12 %) |

Reading: **multi-rod coordination pays off** where it is about sharing a task between rods: the goalkeeper covering the
part of the goal the 2-rod leaves open cuts a 3-rod shot's conversion from 33 % to 12 % and wins two games in three.
A ball on the own 3-rod is worth about seven times one on the 5-rod (33 % vs 5 %), so getting it there is the point of
passing; but the reference's **pass is not yet good enough to show a robust gain on its own** (58 % against the same
player without passing, not significant at 60 games): three 5-rod plays in four are stopped by the opposing 5-rod,
and after a caught pass the 3-rod needs about half a second to get its foot behind the ball, by which time the
defence has re-formed. A better passer (lateral ball control, a one-timer) is exactly the skill the sport leaves to
the agents. The diagnostic is reproducible (`--seeds`, `--trials`) and runs in about 40 minutes on 8 cores.

## Exploit review (R6; tests in `test_foosball.py`, `test_exploit_*` and the rule tests)

| Strategy | How the rules stop it |
|---|---|
| hold or pin the ball on a rod (stalling) | possession clock: 10 s on the 5-rod, 15 s elsewhere; a foul gives the ball to the opponent, never a point |
| rock the ball across the border between two rods to restart the clock | possession is reachability, not a zone: the ball stays in both rods' reach and both clocks keep running |
| park the ball in a corner or against a wall | the fillets roll it off the wall; the goalkeeper rod's outer figures reach the corners; its clock runs out |
| play the ball dead / wedge it | every playfield position is reachable (test); a dead ball would be restarted after 1 s (SMR 7.2); a rally always ends (`test_exploit_ball_cannot_get_stuck`) |
| shoot straight from the kick-off | opening sequence (SMR 5.2): the opponent gets the ball |
| spin the rods | stops at +-2.8 rad, commands within +-2.6 rad (400 otherwise); the referee's spin check stays |
| chip the ball over the rods / off the table | the cover keeps it below 4 cm |
| only block (turtle) / do nothing | loses: the blocking team commits possession fouls and concedes (tests) |
| interfere with the other team's rods | figures of the two teams do not collide; only the ball connects them |
| sit on a lead | the clocks force play; the cap decides by goals first |
| a single-player session that ends early | a valid loss (`session_ended`), not a technical failure |

## Broadcast

`broadcast=True`: `environment/sim/broadcast_log.py` writes `/data/broadcast.jsonl` (qpos every 2 control steps;
events `run`, `point` (goal), `stoppage` (foul, dead ball, ball off table), `pause` (review), `end`). The grader never
reads it. A broadcast adapter needs the model from `foos.scene_xml()`, the camera `broadcast`, and the qpos layout
(ball free joint 7, then per rod slide and angle, seat A's goalkeeper, 2-rod, 5-rod, 3-rod first, then seat B's).

## Validation

Checks (2026-10-05): `generate.py check` clean; `ci_checks/run_all.sh tasks/foosball-*`: both tasks conform, the only
FAIL is the repository-wide `tasks/dataset.toml` manifest (entries added at merge); `check-olympiad.py --root .
foosball`: 52 ok, 0 warnings, 0 failures; `task_conventions.py --check` rc 0; `test_program_play.py` ALL OK on both
tasks; `test_foosball.py` 23 tests OK.

Harbor runs (tag `foos`; all valid_trial = 1, deterministic = 1, the configuration as given, official_config = 1):

| Run | Result | Rallies (goals / fouls) | sim lag max / p99 |
|---|---|---|---|
| oracle L1 / 101 / x20 (default) | win 5:0 | 9 (5 / 4) | 1.31 s (one stall) / 1.8 ms |
| oracle L1 / 111 / x1 | win 5:0 | 13 (5 / 8) | 0.16 s / 1.7 ms |
| oracle L2 / 202 / x10 | win 5:2 | 7 | 39 ms / 2.0 ms |
| oracle L3 / 303 / x5 | win 5:3 | 8 | 0.13 s / 0.1 ms |
| oracle L4 / 404 / x2 | loss 2:5 (valid) | 8 | 6 ms / 0.6 ms |
| nop L1 / 101 / x20 | valid loss (no start, `session_ended`) | 0 | - |
| oracle vs oracle x10, first A | B wins 5:1 | 6 | 0.16 s / 2.4 ms |
| oracle vs oracle x10, first B | B wins 5:1 | 6 | 6 ms / 2.0 ms |
| oracle vs oracle x20, first A | A wins 5:2 | 7 | 2.9 s (one stall) / 2.0 ms |
| oracle vs oracle x20, first B | B wins 5:1 | 6 | 4 ms / 2.6 ms |
| luna (gpt-6-luna, Codex) L1 / 101 / x10, `run_single.sh` | win 1:1 on the cap (last goal); 12 possession fouls (mostly the scripted level 1), 1 incomplete opening | 16 | 0.21 s / 0.3 ms |

The luna smoke used 275 metered requests, 28.9 M input and 0.12 M output tokens. Timing is report-only; the two
multi-second stalls happened on a host shared with a running season (load about 20).

Open: the reference passer does not yet show a significant gain from passing alone (see the diagnostic); the
reference loses to level 4.
