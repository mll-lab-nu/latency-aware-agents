# fencing (epee)

Robot epee fencing on the v0.2 framework (docs/framework_zh.md §4.2, §10 stage 3), migrated from the legacy tree
(`scripts/olympiad/fencing/`, `scripts/olympiad/shared/fencing/sim/`). Two tasks, same names as in Season 1:
`fencing-bout-wallclock-program-standard` (format `bout`: seat A against the scripted fencer, levels 1-3)
and `fencing-duel-wallclock-program-standard` (format `duel`: seats A and B). Rules: FIE Technical and
Material Rules, August 2026, epee, with the deviations D1-D9 of `sport.toml` (`rules_check.md`: every article and
value checked against the PDFs).

| File | Part of contract ② | What it is |
|---|---|---|
| `sport.toml` | declarations | rulebook and deviations, formats, modes, timing, observation whitelist, rule constants, tiebreaks, exploits E1-E8, players, fairness, official configurations, tasks, capabilities |
| `plugin.py` | world, actions, referee, observation | `FencingMatch` (MatchBase) around the legacy `bout.Bout`; `JointLayer` (low: joints / move / hold / traj) and `PointLayer` (high: the legacy act / move / guard / none, IK in the seat's plan worker); `Fencing.scene()` for the spectator video |
| `fencing.py`, `bout.py`, `opponent.py`, `strategies.py`, `scene/` | world (legacy; fencing-3: the arm's planner and servo under the real Panda's limits, `bodies/limits.py`; fencing-4: the rail as a Festo EGC-TB-185 axis) | the MuJoCo world, the motion controllers, the apparatus and phrase referee, the pool-bout referee, the scripted opponent and its strategies, the MJCF scene |
| `cameras.py` | observation | the two cameras per seat (legacy poses) and the 2 x 4 tile image, with a tile cache |
| `builtin_players.py` | built-in players | `Reference` (the legacy camera-only fencer; its own IK for the low level), `Random` |
| `grade.py` | reward fields | the legacy reward keys of both tasks |
| `tests/test_fencing.py` | exploit list | one test per `[[exploits]]` entry |
| `tests/feasibility.py`, `tests/feasibility_results.json` | entry gate 5 | the legacy feasibility study (14 scripted strategies, round robins) |
| `golden/` | golden replays | `convert.py` (the legacy records of the golden-data store) + recorded v0.2 matches |

Run: `PYTHON=python3.12 tools/check fencing` (all stages; `RO_GOLDEN_DATA=<store>` adds the legacy records to stage 3).
Generate: `python3 tools/generate.py fencing` (publish = true: `tasks/fencing-*`).

## Entry gates (CONTRIBUTING.md)
1. Physics: two Franka arms with epees in MuJoCo (1 ms steps), blades, guards and bodies collide; the apparatus reads
   the contact force along the blade. Adds the `contact` physics type.
2. Decisions: distance, timing and the choice of action per phrase decide the bout (the feasibility study: no
   strategy wins every match-up).
3. Cameras only: the reference fencer plays from the side and top images and its own robot state.
4. Rulebook: FIE, August 2026; the referee is `bout.py` + `fencing.Phrase` (fencing-5: with unwillingness to fight, t.124).
5. The reference beats the level-1 fencer; the duel's fairness: `tools/check fencing` stage 4.

## Arm limits (fencing-3, 2026-10-07)
The arms are held to the real robot's limits (user decision 2026-10-07, review item C3). The scene's arm is MuJoCo
Menagerie's Franka Emika Panda (`panda_nohand`), so the Panda's (FER's) official numbers apply, from Franka's
"Control Interface Specification and Robot Limits" (`bodies/franka_panda/body.toml`, with the sources):

| joint | A1 | A2 | A3 | A4 | A5 | A6 | A7 |
|---|---|---|---|---|---|---|---|
| velocity (rad/s) | 2.175 | 2.175 | 2.175 | 2.175 | 2.61 | 2.61 | 2.61 |
| acceleration (rad/s^2) | 15 | 7.5 | 10 | 12.5 | 15 | 20 | 20 |
| jerk (rad/s^3) | 7500 | 3750 | 5000 | 6250 | 7500 | 10000 | 10000 |
| torque (N m) | 87 | 87 | 87 | 87 | 12 | 12 | 12 |
| torque rate (N m/s) | 1000 | 1000 | 1000 | 1000 | 1000 | 1000 | 1000 |

fencing-2 used 1.5 x the Panda's velocity and acceleration limits (deviation D7, accepted 2026-10-05) and limited
neither jerk nor torque rate. What enforces them (`fencing.py`, with the shared `bodies/limits.py`):
- **Planner** (`Controller.install`, every `act` / `joints` / `guard`): a straight joint-space move, time-optimal
  under velocity, acceleration and jerk (a 7-segment S-curve on the path parameter, the slowest joint setting the
  pace) at 95 % of the limits and of the jerk the torque-rate limit allows at the start, middle and goal poses
  (`jerk_caps`: share 0.7 of 1000 N m/s over the arm's inertia). A full extension from on guard now takes 0.55-0.75 s
  (fencing-2: 0.50-0.58 s).
- **Servo** (`World.step`, every 1 ms physics step): a jerk-limited setpoint generator (`JerkLimiter`) follows the
  planner's reference (or a `traj` row's cubic Hermite segment) and never exceeds the velocity, acceleration and jerk
  limits; the motor torque is the setpoint's inertia feed-forward M(q) qdd + the cart's measured acceleration through
  the arm's coupling inertia + PD on the setpoint + gravity/Coriolis, then limited to 1 N m per ms and to 87 / 12 N m
  (`TorqueRateLimiter`). A joint pushed more than 0.02 rad off its setpoint (a hit, a bind) makes the setpoint follow
  the arm (within the velocity limit), so the rate-limited servo cannot wind up.
- **Low level**: `traj` rows must keep the velocity limits (as before) and now also the acceleration limits (second
  differences / (10 ms)^2); rows beyond them are refused (400), and a first row that does not join the current targets
  within them is rejected at the effect (exploit E2). The rail keeps its own limits (see "Rail limits").

`python3 tools/arm_limits.py measure fencing` replays the golden matches and reports every joint against the limits
(docs/benchmarks/roboolympics-arm-limits-2026-10-07.md has the before / after tables): the commanded setpoints and
the motor torques and torque rates stay within the limits; the joints' actual motion exceeds them only when an
external force acts (a blade or point hitting the guard or the arm, a bind, the cart's footwork).

## Rail limits (fencing-4, 2026-10-09)
The footwork rail is a real product (deviation D8; real-limits audit §3.2, user approval 2026-10-09): a **Festo
EGC-TB-185** toothed-belt axis, catalogue limits 5 m/s, 50 m/s^2 and a 2500 N feed force (Festo, "Toothed belt axis
EGC-TB", operating instructions 8162896, 2023-05). The sport runs it inside that envelope (`fencing.py` RAIL_*,
`game spec` robot.rail):

| | fencing-3 | fencing-4 | Festo EGC-TB-185 |
|---|---|---|---|
| speed | 2 m/s | 2 m/s | 5 m/s |
| acceleration | 6 m/s^2 (the go-to's final step 8-11) | 6 m/s^2 | 50 m/s^2 |
| jerk | none | 150 m/s^3 (estimate: Festo publishes none; the B9 study's value) | - |
| motor force | 3000 N | 2500 N | 2500 N |

- **Go-to** (`rail_goto`, every `move` / `act` rail / `joints` rail_x and the scripted fencer): per 10 ms control step
  the largest constant jerk after which the rail can still brake to rest at the goal (the braking distance under the
  acceleration and jerk limits in closed form, bisection), its speed never above the command's vmax; the last few
  micrometres by deadbeat steps within the jerk limit, so it rests exactly on the goal (overshoot under 0.5 mm). A 4 m
  advance takes 2.6 s (fencing-3: 2.3 s).
- **Servo** (`World.step`, every 1 ms): the step's cubic is sampled at each physics step; the motor force is the
  rail's inertia times the reference acceleration (feed-forward) + PD + bias, clipped at 2500 N (fencing-3: PD on the
  10 ms targets interpolated linearly, no feed-forward). On a free 4 m move the cart itself measures 2.00 m/s and
  6.0 m/s^2 at most (10 ms averages; fencing-3: 2.03 m/s, 8.8 m/s^2); contacts (corps-a-corps, blade on body) still
  push it beyond.
- **Low level**: a `traj`'s rail column must keep 2 m/s, 6 m/s^2 and 150 m/s^3 (first, second and third differences
  of the rows), joined to the rail's last three targets: an arm-only `traj` while the cart runs would stop it at once
  and is rejected at the effect (exploit E2; fencing-3 checked the speed only and set the rail's velocity to zero at
  every row). After the last row the go-to brings the rail to rest at it within the limits.

## Unwillingness to fight (fencing-5, 2026-10-09)
Up to fencing-4 nothing stopped a bout without touches: a fencer who retreats to its rear limit (or just waits) against
one who waits runs the 3 minutes out at 0-0, and the extra minute ends on the lot (t.38.2.b). The FIE rule against
this is **t.124, unwillingness to fight** (Technical Rules, August 2026, p. 39-40; sanctions table t.170, offence 0.5):
"There is unwillingness to fight when there is one minute of fencing without a hit or without a hit scored off the
target. In the event of unwillingness to fight, the Referee will immediately call 'Halt!' as a decision of 'fact'".
Its sanctions are written for direct elimination (t.124.1) and team matches (t.124.2): the first time a P-red card to
both (a penalty hit each), the second time a P-black card to the lower score, who loses the bout (equal scores: the
lower seeding, lots for fencers not in the FIE ranking); no P-card at 14-14 (t.124.3.a); the minute restarts after
each hit, hit off the target, annulled hit and penalty hit (t.124.3.d); not at equal scores at the end of regulation
time (t.124.3.f). The pool bout (t.38) has no such clause.

Deviation D9 applies the direct-elimination version to our one pool bout (`bout.py`, PASSIVE_S):

| | FIE t.124 | fencing-5 |
|---|---|---|
| applies to | direct elimination, team matches | the pool bout (both formats) |
| the minute | 1 min of fencing (effective time, t.37) | 60 s of the bout clock `fencing_time_s` (6000 control steps of 10 ms; the clock runs only while the world runs; at slowdown 20 that is 20 min of wall time) |
| 1st time | P-red to both: a penalty hit each | the same (`awarded` both, on guard on the on-guard lines, t.22.6) |
| 2nd time | P-black to the lower score (equal: lower seeding / lots) | P-black to the lower score; equal: a seeded lot (`[seed, 124]`, none of our fencers is ranked) |
| no card | at 14-14 | at 4-4 (the halt is called, the minute restarts) |
| minute restarts | hit, hit off target (not at epee), annulled hit, penalty hit, new period | touch, double, annulled double (4-4), rear-limit touch (t.34), P-red; not a corps-a-corps |
| extra minute | t.124 does not apply (t.124.3.f) | the same |

The observation has `passivity` = {no_hit_s, limit_s, p_cards} (None in the extra minute); a passivity halt's
`last_halt` has `why` = "passivity", `no_hit_s` and `cards` {you, opponent}; `game spec` bout.passivity_s; the
tiebreak chain names the P-black win `passivity`; reward key `passivity_halts`. Exploit E8
(`test_exploit_passivity_cards`): seat A retreats to 5 cm in front of its rear limit and waits, seat B waits on guard;
under the fencing-4 rules the bout reaches the end of time at 0-0 (k 18000) and the extra minute, under fencing-5
the halt comes at 60.00 s (P-red, 1-1) and at 120.00 s (P-black by the lot, the bout over). The golden matches and the
fairness matches have no minute without a touch, so their results are unchanged (see the pull request).

## What changed from the legacy tasks
- **Actions.** The high level is the legacy interface (`act`, `move`, `guard`, `none`); numbers must be JSON numbers
  (the legacy service also took numeric strings). The episode logs low-level commands only: an `act` is logged as the
  `joints` command it lowers to (the IK goal), `guard` as `joints` to the on-guard pose, `none` as `hold`. The new low
  level (`joints`, `move`, `hold`, `traj`) is the joint space of the same controller.
- **Observation image.** The legacy image: 2 rows (side, top) x 4 columns (camera frames 20 ms apart) of 800 x 360
  tiles, the same camera poses, colours and labels (cameras.py; drawn in the seat's render worker, PNG encoded by the
  core). A tile cache draws only the camera frames that are new since the last view (the pixels are unchanged). One
  view costs 0.3 s of CPU at host load ~45 and 3.3 s at load ~700 (`frame_budget_ms` = 4000).
- **Scripted opponent.** Unchanged and inside the world (`step()`), so the replay re-runs it; its commands are not
  episode rows (the core has no world-side command source; see "Core hooks").
- **Duel reference styles.** By seat and player seed (seed 0, the live default: A attacks, B defends, as the legacy
  oracle); local matches alternate the styles with the seed.
- **Reviews in the bout.** The core's review timeout (`READY_TIMEOUT_S`, 300 s) also applies to the single bout (the
  legacy single service waited forever); a missed review is a valid loss. Core 0.2.2: a seat that never starts
  loses (`no_show`; in the bout once the start window, counted from the service start, closes, or since core 0.2.7
  at the collect if the session ends first).
- **Defaults.** `RCB_SEED` default 1 (legacy 101); `RCB_SLOWDOWN` default 20 as before (`[timing] slowdown`).
- **Reward keys.** The legacy keys are kept, plus the core's common ones (`match_over`, `direct_play`,
  `lowlevel_actions`, `solo`, `no_show`, `<seat>_no_show`, `<seat>_won`, `<seat>_transient_api_errors`,
  `<seat>_unrecovered_api_errors`; the duel already had the last three). Removed: `sim_lag_frac_late_over_1_period`
  (the core reports no per-step lateness fractions). `refused_busy` / `<seat>_refused_busy` stay but are always 0
  (the core's 429 for an act in flight is not counted). `official_config` follows the core: seed in the official list, level 1-3, slowdown in {1, 2, 5, 10, 20}
  (legacy: the seed tied to its level; the duel 10 or 20 only).

## Golden records
fencing-6 (the arms' setpoint generator, bodies/limits.JerkLimiter, brakes within the jerk in force: no limit cycle at
the torque-rate jerk caps, 2026-10-12) re-recorded the three v0.2 cases: `bout-high-s1-L1` 5:1 (k 3833),
`bout-low-s2-L1` 5:0 (k 2695), `duel-high-s1-A` 5:4 (k 5254). On fencing-5 with self-collision (main before the fix) the
reference played them 4:5 (k 8801), 4:5 (k 9518), 2:5 (k 5470); the fencing-5 records of the two high cases had not
been re-recorded after self-collision and no longer replayed.

fencing-4 (the rail) re-recorded the three v0.2 cases: `bout-high-s1-L1` 5:3 (k 6903), `bout-low-s2-L1` 4:5 (k 9518,
the reference loses this one now), `duel-high-s1-A` 5:4 (k 5198); fencing-3 had 5:2, 5:4, 5:0.

fencing-3 (the arm limits) re-recorded the three v0.2 cases (`bout-high-s1-L1` 5:2, `bout-low-s2-L1` 5:4,
`duel-high-s1-A` 5:0, built-in players, `tools/golden.py record`) and retired every legacy record of the golden-data
store (`convert.py` RETIRED_ALL: their joint targets no longer produce the recorded motion under the real limits; they
stay in the store as they are). What follows describes the converter as it was up to fencing-2.

`golden/convert.py` turns the legacy records (both fencers' joint targets per step, `raw/trajectory.json`) into
`traj` commands per phrase; the replay through the v0.2 core reproduces the legacy state hash, score, winner and
steps exactly (including the single bout, where the opponent is re-run). The duel no-show record (`nop-B`) is
tagged `rule_change` in its converted header: it replays the legacy decision as recorded (no winner; no forfeit row
or play is invented); under core 0.2.2 a live match in that situation is a valid win of the present side (`no_show`).
The bout's no-start record (`nop`) was already a valid loss and stays one.
