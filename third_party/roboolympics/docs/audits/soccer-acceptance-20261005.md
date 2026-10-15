# Soccer acceptance (soccer-2v2-3), 2026-10-05

Release check of RoboOlympics soccer against `docs/benchmarks/roboolympics-rules.md` (v1.0 draft) and the acceptance
questions of `SPORTS_RELEASE_SELECTION_2026-10-05_ZH.md` §4. Branch `olympiad-soccer`; all numbers use the default match
length (2 x 2.5 min of game time).

## Changes

| Area | Change | Rule |
|---|---|---|
| Metering | `think_proxy.py` / `think_clock.py` are no longer soccer's own copies: the generator copies table tennis's single-seat copy into the single task and its per-seat copy into the duel and team tasks (`metering=` in `spec.py`, new optional key of `generate.py`, a shared-file change in its own commit). `server.py` works with both. | I2 |
| task.toml | `category`, `timing_policy = "report_only"` (single); `final.json` reports `timing_policy`. | C2, T4 |
| Session end | A single-player session that ends mid-match (the collect hook) is a valid loss (`unfinished` = 1, `success` 0); a duel that ends unplayed stays invalid. | F6 |
| First kick-off | The match's first kick-off had no kick-off conditions (`ko` was only set at restarts, the state stayed `kickoff` until the first stoppage, so game-stuck never fired): now set at reset. | rules |
| Pushing | Walking an opponent across the pitch was never pushing (contacts broke every few steps and reset the 5 s count; nobody fell). Now gaps up to 0.5 s do not end a contact, and pushing an opponent 0.6 m back while moving into it faster is pushing. | R6 |
| Illegal defence | Both robots parked on their own goal line held level 1 off for a whole half (then the shoot-out). Now both robots of a team in their own penalty area for more than 5 s: the one farther from the ball is removed (not counted), after the SPL illegal-defender rule; documented deviation. | R6 |
| Ball at a board | The boards replace kick-ins/corners; a ball resting against a board (or beside the goal on the goal line) could not be kicked by anyone (the kick needs ~1 m of run-up behind the ball) and stalled play for 30-90 s (local-stuck penalties one after the other). Now a ball within 0.3 m of a board that moves less than 0.3 m in 3 s is moved 1 m inwards, play continuing (event `ball_replaced`, no coordinates). | R5, R6 |
| Team channel | `TEAM_CHANNEL=off` (exported before `run_team.sh`) switches the channel off for the match: `say` answers 403, `/spec` `team_channel.enabled` false, reward `team_channel_on` 0. | F7, control |
| Reference player | Opponents found by jersey colour (red/blue, ray to the trunk plane) and kept clear of; kick aims chosen so the kick spot is on the pitch (post, penalty-spot area, ahead of the ball near a board); tap-ins in front of the goal mouth; looser alignment when an opponent is near the ball; the support robot stays out of the penalty area while the striker is in it. Still camera-only, no privileged endpoint. | F1 |
| Tests | `scripts/olympiad/soccer/test_soccer.py` (27 tests): no draw (R4), mirror (F3), session end (F6), world state (O2), seed hidden (O3/O4), exploit review (R6), replay (G2). | T |

Revision `soccer-2v2-2` -> `soccer-2v2-3` (rules changed). The jersey/number commit 47fda510f was already committed in the
worktree when this check started (clean tree, nothing uncommitted).

## Exploit review (R6)

| Strategy tried | What happens | Test |
|---|---|---|
| Never play (stand still) | level 1 scores in the first half | `test_exploit_stalling_loses_to_the_scripted_team` |
| Park both robots in the own goal | illegal defence after 5 s, one robot removed | `test_exploit_parking_both_robots_in_goal_is_illegal_defence` |
| Ram / bulldoze an opponent | pushing penalty (bulldozing); a full-speed charge knocks the charger over | `test_exploit_ramming_is_pushing` |
| Hold the ball under the trunk | ball holding after 5 s | `test_exploit_ball_holding_is_penalized` |
| Park the ball at a board | moved 1 m inwards after 3 s | harness (`ball_replaced` events in every match) |
| Nobody plays the ball | dropped ball after 30 s | `test_exploit_nobody_plays_the_ball_dropped_ball` |
| Stand at the ball (stall) | local game stuck: the nearest robot removed | `test_exploit_local_stuck_removes_the_nearest` |
| Score straight from the kick-off | no goal, the other team kicks off | `test_exploit_kickoff_goal_without_touch_outside_circle` |
| Keeper charges out in the shoot-out | goal given | `test_exploit_keeper_off_line_concedes` |
| Stall a review | forfeit after the review limit | `test_exploit_review_stall_forfeits` |
| Out-of-range / unknown commands | HTTP 400 | `test_exploit_out_of_range_commands_refused` |
| Team channel abuse | 512 bytes, 5/s, teammate only; off switch | `test_exploit_team_channel_off_and_limits` |
| Leave the pitch | impossible (boards); a ball that leaves restarts with a dropped ball | `test_exploit_ball_out_restarts_with_dropped_ball` |

## Checks

`generate.py check` clean; `check-olympiad.py soccer`: 80 ok, 0 warnings, 0 failures; `ci_checks/run_all.sh` on the
three soccer tasks: all checks passed; `task_conventions.py --check` OK; `test_program_play.py` ALL OK on the three tasks;
`test_soccer.py` 27/27 and `test_identity.py` in the soccer image.

## Harbor runs (tag `socceracc`, x10 unless noted; every one valid with a deterministic replay)

Play statistics come from replaying each recorded match (`episode.jsonl`) and measuring it: shots = the ball leaving
towards a goal from within 4 m faster than 0.6 m/s after a touch of the attacking team; still ball = the ball slower than
0.05 m/s for more than 3 s, as a share of the time in play; falls = share of robot-time lying down.

| Run | Result | Shots (on target) A/B | Passes A/B | Still ball | Fallen | Lag max / p99 |
|---|---|---|---|---|---|---|
| single oracle L1 s101 (default) | won 1:1, shoot-out 2:0 (success 1) | 6 (5) / 7 (3) | 4 / 0 | 25 % | 0 % | 3.0 / 0.009 s |
| single oracle L1 s111 x1 | won 2:1 | 10 (6) / 5 (5) | 1 / 1 | 40 % | 0 % | 0.013 / 0.008 s |
| single oracle L2 s202 | lost 2:3 (valid loss) | | | | | |
| single nop L1 s101 | valid loss, success 0 (`unfinished`) | | | | | |
| duel oracle/oracle s1, first A | 3:3, shoot-out 6:6, coin toss B | 11 (11) / 11 (6) | 5 / 3 | 11 % | 0.3 % | 3.2 / 0.008 s |
| duel oracle/oracle s2, first B | A 6:2 | 9 (9) / 10 (4) | 2 / 0 | 14 % | 0.7 % | 0.35 / 0.004 s |
| team 4 x oracle s1, channel on | B 4:3 (4,554 messages) | 10 (9) / 12 (10) | 2 / 0 | 21 % | 0 % | 2.2 / 0.014 s |
| team 4 x oracle s1, channel off | B 5:1 (0 messages, ~1,145 `say` refused per seat) | 5 (3) / 7 (7) | 6 / 3 | 32 % | 0.2 % | 0.07 / 0.009 s |
| duel luna/luna s1 (smoke) | A 5:0 (79k / 145k output tokens) | 10 (9) / 0 | 2 / 1 | 30 % | 1.0 % | 0.39 / 0.004 s |
| team 4 x luna s1 (smoke) | B 1:0 (34 messages; 54k-143k output tokens per seat; 3 recovered transient API errors) | 4 (1) / 3 (3) | 1 / 0 | 73 % | 0.3 % | 2.4 / 0.011 s |

## Offline harness (deterministic; reference player at the x10 cadence: a decision every 4 control steps, acts 1 step later)

The task's own `server.Service` and `World`, the reference player's code, head cameras rendered at the decision times;
every match replayed in a fresh `World` from its acts (all bit-identical).

| Format | Matches | Results |
|---|---|---|
| single L1 (seeds 101, 111, 121) | 3 | won 3:0, 3:2, 6:0 |
| single L2 (seeds 202, 212, 222) | 3 | won 2:1, lost 0:1, 3:3 and won the coin toss |
| duel (seeds 1-3, both first kick-offs) | 6 | all decided in regulation: 4:3, 3:5, 6:3, 2:5, 3:4, 4:2 (A won 3, B won 3) |
| team, both teams with the channel on / off (seeds 1, 2) | 2 + 2 | on: 1:2, 1:6 (3,752 messages each); off: 4:7, 4:3 |

Across the 8 harness duels/teams: 5-14 shots per team per match, most on target; 0-8 completed passes per team; falls
0-0.6 % of robot-time; still ball 5-29 % of the time in play (before the rule fixes it was 35-60 %, much of it balls
stuck at the boards); no shoot-out or coin toss in a harness duel. In Harbor, 1 of 2 reference duels went to the
shoot-out and was decided by the coin toss (both strikers scored all six kicks: the reference keeper rarely saves).

Limitations: the comms-off switch applies to the whole match (both teams), so the comparison is between matches, not
between teams of one match; the reference shoot-out keeper is weak, so level duels between reference players often go to
the coin toss; level 2 is not reliably beaten by the reference (not required).
