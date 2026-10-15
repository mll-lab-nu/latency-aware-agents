# soccer

Robot soccer 2 v 2 on the v0.2 framework (docs/framework_zh.md §4.2, §10 stage 4), migrated from the legacy tree
(`scripts/olympiad/soccer/`, `scripts/olympiad/shared/soccer/sim/`). Four Booster T1 lower bodies on the released
walking policy, one ball, a boarded 10 x 7 m pitch, two halves of 5 game minutes and a penalty shoot-out when level,
under the RoboCup HSL Rule Book 2026 with the deviations D1-D15 of `sport.toml`. Revision soccer-2v2-4 (2026-10-09)
holds the robots to the real Booster T1's limits and makes the platform kick a real swing-leg kick (section "Robot
limits and the kick"); soccer-2v2-6 (2026-10-11) is the screened configuration (section "Screening changes",
`docs/screening/soccer.md`). Three tasks, same names as in Season 1:

| task | format | seats |
|---|---|---|
| `soccer-match-wallclock-program-standard` | `match` | A (both robots of team A) against the scripted team B of `soccer.Coach` (levels 1-3: 2 and 3 official, default 2; 1 a warm-up) |
| `soccer-duel-wallclock-program-standard` | `duel` (format A) | A (team A) against B (team B) |
| `soccer-team-wallclock-program-standard` | `team` (format B) | A1, A2 (side A) against B1, B2 (side B): one robot each, the team channel between teammates |

| File | Part of contract ② | What it is |
|---|---|---|
| `sport.toml` | declarations | rule book and deviations D1-D13, `[body]` (booster_t1), formats (`team` with `sides`; `team_match`, preview: A1, A2 against the scripted team), `[team_channel]`, modes, timing, observation whitelist, rule constants, tiebreaks, exploits E1-E15, players, fairness, official configurations, tasks (legacy names), capabilities |
| `plugin.py` | world, actions, referee, observation | `SoccerMatch` (MatchBase) around `soccer.World`; `RobotLayer` (the one action level, one entity per robot); `Soccer.scene()` (soccer_scene's MJCF + the T1 meshes for the renderers) |
| `soccer.py`, `soccer_scene.py`, `models/` | world | the world (the real T1 limits through `bodies/servo.py`), the walking policy batch, the kick skill, the referee, the scripted coach; the scene and the head camera; the walking policy (MIT) |
| `swing_kick.py` | world | the swing-leg kick (the platform `kick`, style `swing`): stance, weight shift, swing through the ball, return to the policy's standing pose, stability check |
| `../../bodies/booster_t1/body.toml` | world (robot library) | the T1's leg limits with their sources, a limits-only legged entry (`bodies/__init__.py` LEGS, `sport.toml [legs] body`); enforced by the library's `bodies/servo.py` `Actuators` (motor curves `[motors]`, torque limit) and `WalkCommand` (the generator copies them into the sim) |
| `builtin_players.py` | built-in players | `Reference` (the Season-1 camera-only `cam_player.py` as an `act(view)` player), `Stand`, `Random` |
| `grade.py` | reward fields | `evaluate(final)` -> the legacy reward keys (below) |
| `tests/test_soccer.py` | exploit list | one test per `[[exploits]]` entry (ported from the legacy `test_soccer.py` / `test_identity.py`) |
| `golden/` | golden replays | soccer-2v2-4 recordings with the built-in players + `convert.py` (the legacy soccer-2v2-3 records of the golden-data store, retired: they no longer replay) |

Run: `PYTHON=python3.12 RO_GOLDEN_DATA=<store> tools/check soccer` (all stages; stage 5 needs `--harbor` at x10, see
below). Generate: `python3 tools/generate.py soccer` (publish = true: `tasks/soccer-*`).

## What humanoid soccer tests (owner's decision, 2026-10-12)

Humanoid soccer (2 v 2) tests **embodied execution and the contest for position**: perception from the head cameras,
walking and kicking a real T1 within its limits, timing, being first at the ball and covering. Team play is declared
only as **coordination** (B3: who plays the ball, who covers), not passing or a division of labour; team tactics are
the wheeled 5 v 5's (soccer_ssl, where a passing reference gains about 1.1 goals per match). Why: after the ball and
kick artefacts were fixed (below), a pass could still reach its receiving spot before every opponent at only about 1 %
of on-ball moments (7-12 % even with a kick that takes no time), because the robots walk 0.65-0.8 m/s, the ball rolls
at most about 2.6 m/s on turf and slows down, and every spot has an opponent within 1-3 m; passing itself added
-0.11..+0.04 goals per match in every world and format tested, and all-chase stayed the strongest plan
(`olympiad_review/screening/soccer_physics_pass_zh.md`, `docs/screening/soccer.md` section 12). Humanoid 5 v 5 is
parked; the counter-attacking team plan is a screening candidate, not a level.

## Ball physics and the in-walk kick (soccer-2v2-8 / soccer-5v5-3, 2026-10-12)

Why a team that passes did not beat one that does not (`olympiad_review/screening/soccer_physics_pass_zh.md`, evidence
and sources in `soccer_phys_data/`): every touch cost about 2.7 s of kick preparation plus 2-5 s of lining up, while an
opponent was 1.3 m (about 2 s) away; the real T1 kicks out of its walk in about 1 s. Three simulation artefacts fixed,
no limit loosened (all torques through `bodies/servo.py` as before):

| change | before -> after | evidence | where |
|---|---|---|---|
| the ball bounces (priority-1 contact solref [0.01, 0.15]) | 2 m drop rebounds 0.01 -> 0.81 m; off a post 0.11 -> 0.73 of its speed | FIFA turf 0.60-1.00 m; a FIFA ball on a hard surface about 0.82-0.88 | `soccer_scene.BALL_SOLREF` |
| the ball is a shell (inertia 2/3 m R^2) | a centre-struck ball rolls on at 3/5 of its speed (5/7) | a size 5 ball is a pressurised shell | `soccer_scene.BALL_SHELL` |
| rolling deceleration | 0.15 -> 0.52 m/s^2 (the FIFA roll test from a 1 m ramp: 9.8 m) | FIFA Quality 4-10 m = 0.51-1.28 m/s^2; B-Human's ball model 0.30; grass 0.45-1.5 | `soccer.ROLL_DECEL` |
| **in-walk kick** `kick style step`: the walking policy walks on and keeps the support leg; the first lift-off of the kicking foot with the ball within a step's reach replaces that foot's swing by a kick (wind-up 0.08 s, minimum-jerk swing, 0.08 s back, the policy puts the foot down; the policy's gait clock waits meanwhile) | contact 0.5-1.5 s after the command (swing kick 2.4-2.9 s); ball off at 2.7-3.9 m/s | the real T1's learned kick "integrated into the final step of the approach gait" (about 1.5 s from 1.5 m away), B-Human's T1 in-walk kicks; drill (720 commands, 4 walking speeds, both feet, aims 0 / +-0.3): 300 kicks, 3 falls (all at power 0), 420 called off (no step reached the ball: safe) | `step_kick.py`, `soccer.World` (`_set_prog`, `_cmd_now`, `_physics_tick`, `_step_over`), `plugin.py` schema `style step`, `v`, `/spec robot.kick.step`, `strike_zone.step`; exploit E19 |
| the scripted coach can use it (`lv["step"]`, off for levels 1-2 and ref by default); the power tables are the ball's rolling speed after its skid (`KICK_V`, `KICK_V_STEP`); scripted shots only from where a full kick still arrives (`shoot_m()`) | | the drills | `soccer.py` |

## Passing skills (soccer-2v2-7 / soccer-5v5-2, 2026-10-11)

The skill set lacked passing and receiving (screening of soccer-2v2-5: B3 weak, the two robots rush the ball together;
5 v 5 deferred until passing works). The pass-and-receive study (`olympiad_review/screening/soccer_pass_walkthrough_zh.md`,
evidence in `soccer_pass_data/`) measured what the real T1 can do (upper bound under the real limits, the controller
knowing the true ball) and changed the kick, nothing else:

| change | evidence (step 2, real limits) | where |
|---|---|---|
| **timed kick** `kick at = [t, x, y]` (swing): the player says where the ball's centre will be (team frame) and when; the kick does not wait for a still ball and never reads the ball (it reads only the robot's own state: P2); the foot waits wound up until the swing must start; the same kick sent again before the swing moves t and (x, y); a strike time shorter than the preparation (~2.6 s) strikes at once and is announced (`kick_late`, own robots) | rolling balls 0.5-2 m/s from the front, +-45 deg and +-90 deg, power 0.3-1, oracle timing: 57/60 struck (>= 0.8 m/s), 0 falls in 112 strikes (also with timing errors of +-0.1 s and point errors of +-3-6 cm); the ball leaves at 1.2-3.6 m/s; a ball crossing the swing turns the shot by about 0.3-0.6 rad per m/s of its sideways speed (the player has to allow for it) | `swing_kick.py` (`t_strike`, phase `wait`, `retime`), `soccer.py` `_set_prog`, `_swing_ready`, `_start_swing`, `_timed_target`, `self_state`, event `kick_late`; `plugin.py` schema `at` |
| **wider aim**: up to 0.6 rad away from the support foot's side, 0.3 rad across it (0.2 rad both ways before); the ball's place for each aim in `/spec robot.strike_zone.swing_by_aim` | placement error +-4 / +-3 cm, random heading, 10 kicks per cell: outward 0.4 rad 8-10/10 clean, 0.6 rad 5-8/10, 0.8 rad 2-5/10 (falls at low power); across 0.4 rad 4-7/10 with 3-5 falls (the foot lands on the support foot) | `SWING aim_max`, `aim_in_max`, `SWING_ZONE_AIM`, `swing_zone()` |
| no `receive` skill | a rolling ball that runs into the standing robot's legs stops at its feet (0.5-2 m/s, path within +-0.05 m of the centre: 8/8 stopped within 0.12 m; +-0.10 m: within 0.18 m); a timed foot block (prototype, oracle timing) did worse (the ball glances off the 3 cm thin foot box sideways, 0.4-1.3 m); walking backwards as a cushion did worse (0.3-1.1 m). Receiving = walking into the ball's path (`walk`, `stand`): the player predicts where and when | evidence only |
| the scripted coach can pass (`lv["pass"]`: `block` = the receiver stands in the path, `onetouch` = the receiver arms a timed kick at the goal; `lv["wide"]`: line up with the wider aim) | the reference variants of the screening; levels 1-2 unchanged | `Coach._passing`, `_pass_choice`, `_onetouch_choice`, `_get_open`, `plan_robot` roles `receive` |

## Screening changes (soccer-2v2-5, 2026-10-10; soccer-2v2-6, 2026-10-11)

From the construction and screening process (`docs/screening/soccer.md`, evidence in its data directory). soccer-2v2-5
also had a shot-speed goal rule (D16: a goal only from a shot of >= 2.3 m/s); the user decided against it (2026-10-11:
dribbling the ball in is legitimate soccer), so soccer-2v2-6 removed it and the screening checks instead that the kick
is worth using (the reference against the same decisions without kicks):

| change | why | where |
|---|---|---|
| `kick foot: right \| left` (swing; the strike zone mirrored) | the kick could only be taken from one side of the ball; with an opponent close by the robot rarely got there in time (skill audit: "kicking in a contest"). Drill: left foot 29/30 clean, median 3.19 m/s, no fall (right 30/30, 3.25 m/s) | `soccer.py` `_set_prog`, `_start_swing`; `plugin.py` schema; the coach and the reference choose the foot from their side of the ball |
| 2 x 5 min (D4) | fewer level matches between equal teams | `HALF_S` |
| event `kick_cancelled` (own robots only) with the reason; the kick's control authority in the contract | skill audit P6 (feedback, control authority) | `_swing_ready`, `plugin.PRIVATE_EVENTS`, `[spec] actions` |
| exact self-localization declared (D14); the ball's rolling deceleration declared (D15) | undeclared deviations | `sport.toml` |
| the world takes the team size as a parameter (`SOCCER_TEAM_SIZE`, 2 or 5; 5 = the HSL M-Field, a goalkeeper, kick-ins, goal kicks, corner kicks) from the legacy 5 v 5 (`olympiad-soccer5v5` 7dc17e301), on the real T1 and the swing kick; with it three fairness fixes of that branch also in 2 v 2 (the referee's placement spirals in the team frame, opponent contacts in sorted order, both teams' shoot-out aim on the same side per round) and HSL leaving the field | the 5 v 5 port (no 5 v 5 task yet: plugin, tactical camera, ten-seat team task and its camera reference remain) | `soccer.py`, `soccer_scene.py` |
| built-in player `chase` (every robot a striker) | the simple baseline of the screening | `builtin_players.Chase` |

## Robot limits and the kick (soccer-2v2-4, 2026-10-09)

The robots are held to the real Booster T1 and kick with a real swing of the leg (user decision 2026-10-09, after the
B9 study `olympiad_review/b9_dynamic_sports_zh.md` §4 and the real-limits audit §3.3; research code on the local
branch `b9-dynamic`, `prototypes/b9_soccer/` and `prototypes/soccer_t1_check/`).

**Limits** (`bodies/booster_t1/body.toml`, enforced through the robot library's `bodies/servo.py` `Actuators` and
`WalkCommand`, checked by `ci_checks/check-robot-limits.py`; `/spec` `robot.leg_limits`):

| | soccer-2v2-3 | soccer-2v2-4 | source |
|---|---|---|---|
| leg motor torques (hip pitch / roll / yaw, knee, ankle pitch / roll) | 45 / 45 / 30 / 65 / 24 / 15 N m (sai-mujoco ctrlrange) | 45 / **30** / 30 / **60** / 24 / 15 N m | Booster URDF (ankle pitch: the locomotion URDF) |
| joint speeds | none | 12.5 / 10.9 / 10.9 / 11.7 / 18.8 / 12.4 rad/s as a torque-speed envelope: full torque up to 64 % of the limit, then linearly to 0; braking keeps full torque | URDF; the envelope's shape is an ESTIMATE (no Booster motor curves) |
| motor heat and power (soccer-2v2-10) | none | rated torque 60 % of the URDF torque, an I^2 t foldback to it after about 5 s at the peak; 1 kW bus per robot | ESTIMATES (bodies/booster_t1 [motors.thermal], [motors.bus]) |
| walking command | [-1, 1], any jump | [-1, 1], moving toward the request by at most 2 m/s^2, 2 m/s^2, 4 rad/s^2 (`command.walk_now`) | training range: booster_gym T1.yaml; the rate is an ESTIMATE (the G1 joystick's) |

**Kick** (`kick {power, aim, style}`; `/spec` `robot.kick`, `robot.strike_zone`):

- `style: "swing"` (default): `swing_kick.py`, the G1 pilot's model-based kick ported to the T1 (B9 prototype) with a
  new start and end. The walking controller first stops the robot and the kick waits for the ball to roll slower than
  0.15 m/s (else called off after 1 s: `kicks_cancelled`); stance (0.4 s), weight onto the left foot (1.1 s),
  balance (0.3 s), wind-up (0.3 s), swing through the ball as it lies when the kick starts (minimum jerk; the right
  foot turned to the kick direction `aim`, |aim| <= 0.2 rad), follow-through, retract, lower until the foot is down;
  then, instead of the prototype's one-step return, a 0.4 s joint-space blend to the walking policy's own standing
  targets, a stability check (up to 0.4 s) and 0.6 s of recovery steps in place under the policy. Contact 2.4-2.9 s
  after the command, 4.7-5.6 s in all (soccer-2v2-3's push: about 1.9 s). Strike zone: the ball's centre 0.26-0.38 m
  ahead of the trunk and 0.02-0.10 m to its right. A clean hit sends the ball off at about 0.7 / 1.4 / 2.5 / 3.3 / 3.5
  m/s at power 0 / 0.25 / 0.5 / 0.75 / 1, within about 0.1 rad of the aimed direction. The kick reads the ball's
  position when it starts (the robot's own close-range view; the player never gets these coordinates).
- `style: "push"`: soccer-2v2-3's toe-poke with the B9 study's distance-based run-up (it walks until a foot is past
  the ball's centre, at most twice the nominal 1.32 m, instead of a fixed number of steps, which under the command
  rate limit stopped short of the ball: 1/30 -> 18/30 goals in B9 §4.3). Kept because it is quick (2.6 s), never
  fell in the checks and moves the ball at about 1.3 m/s: a pass or a poke in traffic, never a shot past a keeper
  (B9 (a): 0/30).

**Checks of the kick** (`prototypes/soccer_t1_check/`, real limits, one robot, the B9 study's 30 fixed starts):

| check | B9 prototype (`b9_soccer`, 840 shots) | soccer-2v2-4 skill (840 shots) |
|---|---|---|
| shots past the nominal keeper (0.30 s reaction, 0.74 m/s), upper bound over 28 shots per start (7 aim points x 2 x 2 approach offsets) | 28/30 = 93.3 % | **29/30 = 96.7 %** (>= 90 %: passes) |
| the same with the keeper as a physical box the ball bounces off (best shot per start) | 28/30 | **29/30**, no fall |
| keeper reaction 0.15 s / speed x 1.25 (sensitivity, offline) | 80.0 % / 73.3 % | 90.0 % / 83.3 % |
| falls (in the B9 study: before the ball crossed; here also 2 s after the kick) | 77/840 = 9.2 % before the line | **10/840 = 1.2 %** (6 before the line) |
| ball speed off the foot, median / max | 3.02 / 4.11 m/s | 3.30 / 3.83 m/s |
| contact after the command / whole kick (median) | about 2.9 s / - | 2.58 s / 4.88 s (p90 5.34 s) |

The prototype's end (one 10 ms step back between the feet, then the policy at once) let 42 % of the robots fall in
the 2 s after the kick when that time is watched (`soccer_t1_check` variant C, 89/210); the joint-space return, the
recovery steps and the start conditions (`soccer_t1_check` variants V0-K3) bring the falls down to the 1.2 % above.
The stability check rarely sees the robot settled within its 0.4 s (the CoM still moves ~0.5 m/s after the return),
so the hand-back mostly happens at that limit and the policy's recovery steps do the settling. In 2 v 2 games the
falls are more frequent: scripted team against scripted team (no cameras, 12 matches, 187 swing kicks) 13 = 7.0 %, 7 of
them after contact with another robot (no contact, ball in the zone and still: 4 of 115 = 3.5 %); in the 57 in-process
matches of the checks below (camera reference and scripted teams) 231 kick falls in 2320 kicks of both styles
(10.0 %; the push never falls): more contact and less exact lineups than in the drill (not broken down further).

**Scripted teams and timings.** The coach (`soccer.Coach`, levels 1-2) and the reference player use the same skill and
limits: they line up 0.3 m behind the swing spot and walk straight in slowly, shoot with the swing (the coach at the
goal corner farther from the opponents, y = +-0.8), push when an opponent is within 1.5 m of the ball, and the
shoot-out striker aims at +-0.9. The kick-off robot starts with the ball in its swing zone (-0.32, 0.06). Local game
stuck 10 -> 15 s (a kick now takes about 5 s). Kick tolerances of the coach: level 1 0.10 m / 0.15 rad, level 2
0.07 m / 0.10 rad.

## How the legacy service maps onto the core

| legacy `server.py` / `replay.py` | v0.2 |
|---|---|
| `World.advance()` per control step; `World.command(robot, cmd)` at the step boundary | `step()`; `apply(seat, cmd)` with `cmd["entity"]` = the robot's number |
| one act per robot in flight (429); a penalized / fallen robot: 409 at the request, dropped at the boundary | `entities(seat)` (the core's per-entity lock); `RobotLayer.parse` refuses 409; `apply()` returns `rejected` (the logged command does nothing, also in the replay) |
| a stoppage (`World.stopped`): review by every seat, then `World.resume()` | `phase()` = UNIT_OVER with `review` = every seat; `resume()`; `unit()` = the number of stoppages |
| `World.winner` / `end_reason` | `tiebreaks()`: `goals` (regulation, mercy) -> `shootout` -> `sudden_death` -> `coin_toss` |
| review timeout / start timeout of the duel | core forfeits (`ready_timeout`, `no_show`); `forfeit()` records them outside the world (the hash stays the legacy one) |
| `state_sha()` (qpos, qvel, time, step, score) | `state_digest()` (unchanged, so the legacy records replay to the legacy hashes) |
| per-seat render workers drawing `<robot>:egocentric` from qpos snapshots | `cameras()`: the head camera's world pose from the robot's trunk (our own MjData + `mj_kinematics`/`mj_camlight`), drawn by the core's render workers with the task's renderer from `scene()` and `render_state()` (the same pinhole: the legacy and core images agree to the pixel except the legacy caption) |
| `/say`, `/inbox`, 5 messages per game second + 20 per frozen phase, `TEAM_CHANNEL=off` | the core's team channel (`[team_channel]`: 512 bytes, one message per seat every 0.2 game s, 4000 per match: 3000 would be exactly a full regular time at that rate, leaving nothing for the shoot-out), delivered in every reply's `team_messages` |

## What changed from the legacy tasks

- **Start**: core 0.2.2 rule 5. Duel/team: a side with a seat that never starts loses (`no_show`, valid); legacy:
  "incomplete" (no result). Match: the window opens when the service starts; a player who never starts is a valid
  loss (`no_show`) when it closes or, since core 0.2.7, at the collect if the session ends first; legacy: the
  session's end ("collected") was the valid loss.
- **A single player's session that ends mid-match**: legacy F6 counted a valid loss; so does the core since 0.2.6
  (`abandoned`, a forfeit without a winner side, success 0; cores 0.2.2-0.2.5: `collected`, an unfinished match). Reviews of the match format now have the core's limit
  (`READY_TIMEOUT_S` = 300 s; legacy: none in the single format).
- **Observation**: one tile per own robot (its head camera now; during a review the last frame before the stoppage)
  instead of two frames 0.2 s apart (the core draws one world state per observation: see Core hooks); `events` are
  the referee's events of the current unit (pure) instead of "since your previous call"; new `stoppages` (the number of
  stoppages so far) and `your_robots`; no `pending_acts`. Images come from the task's renderer (no caption).
- **Actions**: `{"type": "walk", "v": [...], "entity": "2"}` (`"robot": 2` still accepted); the core validates the
  schema (numbers rounded to 4 digits as before); one act per robot in flight.
- **Parameters**: the seed and first kick-off are `RCB_SEED` (match, default 1; legacy 101) / `DUEL_SEED` and
  `FIRST_PLAYER` (duel/team; legacy `FIRST_KICKOFF`: `scripts/olympiad/run_duel.sh` exports both, `run_team.sh` only
  `FIRST_KICKOFF` and needs a one-line fix in a legacy-part PR);
  the slowdown is `RCB_SLOWDOWN` with the legacy default x10 (`[timing] slowdown`); `TEAM_CHANNEL=off` (the control condition) is gone.
- **Records**: no `trajectory.json` / `.npz` (the episode holds every command); the replay video is the core's
  spectator video (10 fps of the `[broadcast]` camera).

## Reward keys

The core writes `success`, `valid_trial`, `technical_failure`, `grader_error`, `missing_log`, `deterministic`,
`match_over`, `seed`, `official_config` (overridden by the sport), `video_rendered`, `<seat>_won`, `<seat>_no_show`,
`<seat>_transient_api_errors`, `<seat>_unrecovered_api_errors`, `no_show`, `solo`, `direct_play`,
`lowlevel_actions`. `grade.py` adds the legacy keys: `official_config` (the legacy rule: match seeds per level x
slowdowns 1-10, duel/team seeds 1-3 at x10), `ready_forfeit`, `unfinished`, `slowdown`, `steps`, `A_goals`,
`B_goals`, `A_won`, `B_won` (team: the team's result), `shootout`, `A_shootout_goals`, `B_shootout_goals`,
`end_mercy`, `end_coin_toss`, `n_messages`, `team_channel_on`, `n_reviews`, `sim_lag_late_max_s`,
`sim_lag_late_p99_s`, `sim_effective_slowdown`; match: `won`, `level`, `goals_for`, `goals_against`, unprefixed
seat keys; per seat (prefix `<seat>_` except in the match format): `attempts`, `applied`, `refused`, `refused_robot`,
`dropped`, `observations`, `messages`, `act_s_max`, `metered_requests`, `metered_output_tokens`; per robot
`<robot>_goals`, `_own_goals`, `_shootout_goals`, `_touches`, `_passes_completed`, `_kicks`, `_kick_falls` (falls during
a swing kick, soccer-2v2-4), `_kicks_cancelled` (swing kicks called off: robot or ball not still within 1 s), `_falls`,
`_pushed_over`, `_penalties`, `_distance_m`, `_messages`. Differences:

| legacy key | v0.2 | why |
|---|---|---|
| `refused_busy` | dropped | the core's per-entity lock answers 429 before counting |
| `messages_refused` | dropped | the core's team channel does not count refusals |
| `sim_lag_frac_late_over_1_period`, `sim_step_ms_p99` | dropped | the core's `sim_lag` has `late_max_s`, `late_p99_s`, run steps and wall time only |
| `transient_api_errors`, `unrecovered_api_errors` (duel/team, prefixed) | from the core (same names) | |
| `team_channel_on` | 1 in the team format | no channel-off control condition in the core |
| `unfinished` | `reason` = collected and the match not over | the duel and team formats (a single session that ends mid-match is `abandoned` since core 0.2.6) |

## Golden replays

soccer-2v2-8 (the ball's physics) changed the world again: the two match records were re-recorded with the built-in
camera reference on the new code (`expected.json` keeps the soccer-2v2-7 values under `re_recorded`); the two legacy
no-show records replay unchanged. The table below is of soccer-2v2-5; soccer-2v2-8: `match-L1-s101` 0:1 (the level-1
team wins, 30000 steps), `duel-ref-s1-A` 2:2 (A wins the shoot-out, 31445 steps).

soccer-2v2-5 changed the world again (both feet, D16, 2 x 5 min, the 5 v 5 world merge), so the two match records were
re-recorded from the screening's camera-level batches (`docs/screening/soccer.md`); soccer-2v2-6 (D16 removed) replays
the same commands to the same final step with another score, so their expectations were re-set; `expected.json` keeps
the earlier values under `re_recorded`. `tools/check soccer --stages 3`: 5/5 (2026-10-11).

| case | what | result |
|---|---|---|
| `match-L1-s101` | the reference (seat A) against the level-1 team, seed 101 (screening batch c8) | 2:7, the level-1 team wins, 30115 steps |
| `duel-ref-s1-A` | the reference on both seats, seed 1, A first (screening batch c11) | 4:3, A, 30270 steps |
| `legacy-duel-nop-B`, `legacy-match-nop` | the legacy no-show records (no step) | A wins by no_show; no winner (match) |

Intended rule changes of the core (no_show, abandoned) as before:

| record | legacy | v0.2 |
|---|---|---|
| duel `nop-B`, team `nop-B2` | incomplete, no winner | `no_show`: A wins |
| match `nop` | collected, a valid loss | `no_show`: a valid loss (no winner in a single format) |

## Checks (2026-10-11, soccer-2v2-6)

The construction and screening process (`docs/screening/soccer.md`): stage 1 53/53; the 15 exploit tests 15/15 in
the image; stage 3 (golden) 5/5; B9 (shooting drill) 29/30; World-level gates and camera-level confirmation; model
trials (coach-mode duels Opus 5.5 vs GPT-6 Astra both seat orders, GPT-6.1 SOL vs Astra). Official match levels 2 and 3
(level 3 = the all-forward coach at the reference's speed, a challenge level), level 1 a warm-up. Stage 2 and stages
4-5 at the contract's counts were not run (tens of CPU hours); see the report's §6.5.

## Checks (2026-10-09, soccer-2v2-4, core 0.2; a host with load 25-30 on 32 CPUs)

The match-based checks below (stage 2's matches, the levels, fairness, the B9 numbers above) ran on the branch before
its rebase onto the robot library (#26), where soccer had its own copy of the same limits (a linear torque-speed
envelope and a command rate limiter in bodies/limits.py). The rebased code drives the legs through the library's
`servo.Actuators` / `servo.WalkCommand` with the same numbers: the same model, but not bit-identical in the
envelope's last digits, so the golden cases were re-recorded and the determinism and B9 checks re-run on the final
code: determinism in process and in a fresh-process replay (match, seed 7: 2:1), golden 5/5, the exploit tests 15/15,
the B9 check (840 shots: the same 29/30, 10/840 falls, the same sensitivity table) and the physical keeper (29/30, no
fall). Match outcomes are chaotic in these last digits (the golden match-L1-s101 was 3:2 before the rebase, 0:3
after), so single results, and the levels' 4/9 and 2/5 below, carry a large uncertainty.

- Stage 1: 53/53. Stage 3: 5/5 (the 4 cases above). The 15 exploit tests pass in the sim image (E15 new).
- Stage 2 ran as a sharded queue of the same `core.testkit` code (2-3 worker containers; about 25 minutes per match
  with cameras): observe_is_pure, refuses_bad, fuzz, frame_budget (291 ms of 1000), whitelist declared / enforced /
  no leak, tiebreaks_declared, the 15 exploit tests, solo_quit, prestart_no_show (3 formats), determinism in process
  and in a fresh-process replay (match, seed 7: 0:1), no_draw (all 20 matches of seeds 1-4 of every format and first
  order decided: 18 in regulation, 2 by the shoot-out's coin toss): 50/50 rows pass.
- The reference against the scripted team (match format): level 1 won 4 of 9 (seeds 1-4, 101-141; goals 19:16),
  level 2 won 2 of 5 (seeds 202-242; goals 7:13). soccer-2v2-3's reference won the default configuration; under the
  real limits and the 5 s kick the camera reference is about as strong as the scripted levels (follow-up: retune the
  levels or the reference).
- Fairness (stage 4, duel, the reference on both seats; the same `local.play` as `core.testkit.fairness`): N = 27
  (seeds 1-11, 15, 16 with A first, 1-14 with B first; the contract's 20 per order did not fit the night), first side
  -9.3 pp (95% CI +-18.5), seat A -16.7 pp (+-17.8): no advantage shown, the intervals too wide to rule one out (WARN,
  realtime); 24 decided in regulation, 2 by the shoot-out, 1 by its coin toss; 3.9 goals per match. Seat B's team
  kicked more often (653 against 535 kicks): the world is mirror-symmetric (E12), so a residual seat effect would come
  from the camera images (the sun's direction 0.15, 0.1, -1 is not mirror-symmetric) and the reference's colour
  thresholds; soccer-2v2-3 showed seat A -5.0 pp (+-21.8) with N = 20. Worth a longer run.
- Stage 5 (live through the generated tasks at x10, about an hour per format) was not run for soccer-2v2-4; the match
  format's default configuration (seed 1, level 1) ends 1:1 and goes to the shoot-out's coin toss in process, so the
  oracle's success there is not guaranteed any more.

## Checks of soccer-2v2-3 (2026-10-06, core 0.2.2, a host with load 60-110 on 32 CPUs)

- Stage 1: 48/48. Stage 3: 15/15 (5 cases in `golden/` + the 10 legacy records of the store).
- Stages 2 and 4 ran as a sharded queue of the same code (`core.testkit` functions and `core.local.play`, 2 worker
  containers; the serial check would take about 20 hours, see Core hooks 8): observe_is_pure, refuses_bad,
  frame_budget (277 ms of 1000), whitelist declared / enforced / no leak, tiebreaks_declared, the 14 exploit tests,
  determinism in process and in a fresh-process replay (match, seed 7: 4:2), no_draw (all 20 matches of seeds 1-4 of
  every format and first order decided): all pass.
- Fairness (reference on every seat, team format with its sides A/B, which the core's check does not pass): duel N = 20 (seeds 1-10, both orders), first side -15.0 pp (95% CI +-20.9), seat A -5.0 pp (+-21.8); team N = 20, first side -15.0 pp (+-20.9), seat A1 -5.0 pp (+-21.8): no advantage shown, the intervals too wide to rule one out (WARN); 2 of 40 matches went to the coin toss
  Realtime: an interval wider than 5 pp only warns.
- Stage 5 through Harbor at x10 (the default): match oracle success 1 (won, 3098 s), nop 0; duel oracle success 1
  (B won, 6108 s), nop 0; team oracle success 1 (B won 5:4, valid, deterministic, 15150 steps), nop 0.

## Core hooks soccer needs (not changed here; worked around inside the sport)

1. **Sides in the generated task** (tools/generate.py `sim_env`): the compose file passes no `RO_SIDES`, so the team
   format would be four sides and the team channel would stay off. Workaround: `plugin.py` sets `RO_SIDES` from
   `[formats.team] sides` when core.serve imports it (before it reads the environment). The generator should pass
   `[formats.F] sides`.
2. (done in main a75144ce: the generator passes `RCB_SLOWDOWN`, default `[timing] slowdown` = 10.)
3. **Fairness and sides** (core/testkit.py `fairness`): its MatchConfig has no `sides` and the fair share is
   1 / (number of seats): the team format is judged as four sides (25 % each). It should pass `[formats.F] sides` and
   count sides. `plugin.py` ranks the winning team's seats first when it gets no sides, so the check runs, but its
   team-format verdict is wrong.
4. **Several acts per tick of a built-in player** (core/local.py `play`, core/client.py `play_realtime`): one body per
   call, so a two-robot seat commands one robot per tick (the reference alternates); a list of bodies (one per entity)
   would match what a model's program can do over HTTP.
5. **Team channel for built-in players**: neither the in-process loop nor the HTTP client loop lets a built-in player
   `say`; the reference in the team format plays without its teammate's ball estimates (the legacy reference sent
   one every 0.25 game s). A `say` return (or a `messages()` hook) is needed.
6. **Team channel limits**: the core has one message per `min_interval_s` and a per-match cap; HSL Law 3 / the legacy
   service had a rate per game second plus a separate budget while the world is frozen (20 per review), and a
   channel-off control condition (`TEAM_CHANNEL=off`). Frozen phases now allow one message per seat.
7. **Earlier frames in an observation**: `render_state()` gives one world state per observation, so the legacy pair
   of frames 0.2 s apart cannot be drawn; a per-camera state (or a time offset per camera) would bring it back.
8. **The generic suite's cost** (core/testkit.py): the in-process matches render every seat's cameras every 5
   control steps (about 3000 views per match), so one soccer match takes 12-25 minutes of CPU and `no_draw` plays 20
   of them (seeds 1-4 of every format and first order) plus the fairness matches. A per-sport budget (seeds, a shorter
   test configuration, a render interval) is needed; until then stages 2 and 4 were run as a sharded queue of the
   same testkit / local.play code (below).
9. **Single-format session end**: settled by core 0.2.6 (core rule 6): a session that ends mid-match is a valid loss
   (`abandoned`), the legacy F6 outcome.
10. **Render threads vs pids_limit** (tools/generate.py compose: `pids_limit: 256` for the sim): Mesa's llvmpipe
    starts one thread per host CPU in each render worker, so the team format's four camera workers on a 32-CPU host
    hit the limit ("Cannot fork"). Workaround: `plugin.py` sets `LP_NUM_THREADS=2` (the images are the same). The
    generator (or the core's camera worker) should set it.
11. **Instruction text** (tools/generate.py `instruction`): written for one action per player and turn-based reviews
    ("after each of your actions ... game ready", `wait {"timeout_s"}`, "seat A or B"); a realtime / multi-entity /
    team-channel variant is needed (the rules section, from `[spec]`, is right).

## Overhead formats (2026-10-13)

`match_overhead` and `duel_overhead`: match and duel with fixed overhead cameras instead of the head cameras.
- Control is coach: one seat per team. Perception is overhead: `[formats.<f>] perception = "overhead"`.
- Cameras (`overhead.py`): two, 6.0 m up, straight down, one per half, 600 x 800 px each. Each team gets them in its own frame. The images show the world 40 ms ago; the calibration is in the game spec `camera.overhead`.
- The plugin draws them through `cameras(seat)`. `render_state()` gives the qpos 2 control steps ago.
- The built-in Reference switches to its overhead perception when the spec has `camera.overhead`, and then plays all-chase with the in-walk kick.
- Screening: `docs/screening/soccer.md` section 14.
