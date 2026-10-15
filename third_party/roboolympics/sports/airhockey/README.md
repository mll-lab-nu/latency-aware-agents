# airhockey: robot air hockey (USAA rules)

Robot air hockey on the v0.2 framework (docs/framework_zh.md §4.2, §8), migrated in stage 2 from the legacy tree
(`scripts/olympiad/airhockey/`, `scripts/olympiad/shared/airhockey/sim/`). Real time (`eai-standard-realtime/1.0`): the
world runs at 1/S of real speed and freezes after every rally for the reviews. The simulator `hockey.py` and its robot
files `scene/` are the legacy files (git history follows them) except the goal mouth: 0.35 m since revision v4
(2026-10-08, review item B13, deviation D10; v3: 0.40 m; legacy and v2: 0.25 m). The legacy recordings in which the
game was played belong to revision v2 and no longer replay; `golden/` holds native v4 recordings (below).

**Rules**: USAA Official Rules of Air Hockey with the deviations D1-D10 of `sport.toml`: one game to 7 goals, the
seven-second rule as a foul (the puck goes to the opponent), dead-puck face-offs, a safety cap of 120 s of play or 40
rallies with the stated chain goals -> last goal -> territory -> the player without the first possession (no draws),
a goal mouth 0.35 m wide (D10, below: Why the goal is wider).
Two seven-axis KUKA iiwa arms (AirHockeyChallenge robot model, MIT) hold the mallets; the player commands its mallet's
Cartesian target in its own half (a smooth path of at most 2 m/s and 10 m/s^2, IK, torque-limited motors) and sees the
table only through its fixed camera.

| Format (task) | Seats | Unit | Result |
|---|---|---|---|
| `match` (`airhockey-match-wallclock-program-standard`) | A against the scripted opponent (seat B inside the world, level 1-4, `RCB_LEVEL`) | a rally | success = winning a valid game |
| `duel` (`airhockey-duel-wallclock-program-standard`) | A, B; `FIRST_PLAYER` has the puck first | a rally | first to 7, then the cap chain |

The task names are the Season-1 names (`[[tasks]] id`).

| File | Part of contract ② | What it is |
|---|---|---|
| `sport.toml` | declarations | rulebook and deviations, formats, timing, `game target` / `game none`, observation whitelist, rules, tiebreaks, exploits E1-E10, players, fairness, official set, tasks, capabilities |
| `hockey.py`, `scene/` | the simulator | the legacy simulator with a 0.35 m goal mouth (`airhockey-program-v4`; v3 = 0.40 m, v2 = the legacy 0.25 m): `Hockey` (both arms and the puck in one MuJoCo world, the Cartesian path / IK / torque controller, the scripted opponent `scripted()`, the referee, `winner()`, `digest()`) |
| `plugin.py` | world, actions, referee, observation | `AirHockey` (MatchBase over one `hockey.Hockey`): `step`, `state_digest` = `Hockey.digest()`, `apply` = `set_command`, `phase` / `resume` = `ready()` / `scores` / `tiebreaks` (= `Hockey.winner()` in named steps), `observe` / `cameras` / `render_state`; `TargetLayer` (the schema + the legacy `validate_action`); `AirHockeySport.scene()` (`hockey.scene_xml()`) and `spec()` (hockey.py's workspace, limits, geometry, timing; no seed) |
| `builtin_players.py` | built-in players | `Reference`: camera-only (puck and opponent mallet by colour, velocity fitted over the recent images); predictive defence (crossing point with the rails' restitution, near-post block, deep pucks cleared from the goal side) and hard wind-up strikes at the part of the goal the opponent's mallet needs longest to reach, straight or off a rail (rewritten 2026-10-07, review item B13; the legacy `camera_player.py` scored 0.05 goals per game against itself), `Random` |
| `grade.py` | reward fields | the legacy reward keys that final.json still carries (below) |
| `tests/test_airhockey.py` | exploit list | E1-E10, from the legacy `test_airhockey.py` |
| `golden/` | golden replays | native v3 recordings of the reference player (2 duels, 1 match) and the 3 legacy records without a scored rally, which still replay (`convert.py`; tagged, below) |

Run everything: `PYTHON=python3.11 tools/check airhockey` (stage 3 with `RO_GOLDEN_DATA` also converts the 3 legacy
records that still replay).
Generate the tasks: `python3 tools/generate.py airhockey`.

## How the legacy parts map

- **Seeds and configuration**: `match`: `RCB_SEED` (legacy default 101, now the generator's default 1), `RCB_LEVEL`
  (default 1), `FIRST_PLAYER` (default A); `duel`: `DUEL_SEED`, `FIRST_PLAYER`. `Hockey` keeps its own
  `default_rng(seed)` for the puck spots (moving it to `core.rng` would change every match).
- **Commands**: legacy `POST /act {"x", "y"}` / `{"type": "none"}` -> `{"type": "target", x, y}` / `{"type": "none"}`
  (a body without a type is still a target). The numbers are rounded to 6 digits by the core's schema check.
- **Observation**: the legacy status reply without the session fields (`started`, `running`, `done`, ... are the
  core's) and without the "news since your last call" queue, which is not a pure function of the state: `rallies`
  (every ended rally) and `last_rally` instead. One PNG with one tile (`front`) instead of `image_png_b64` alone; the
  camera is the legacy `front_<seat>` as a camera definition (pos, lookat, up), drawn by the task's renderer from
  `hockey.scene_xml()` and the qpos. Since revision airhockey-program-v4+players-25k (core 0.2.9, renderer rule 6)
  the arms are drawn with the KUKA LBR iiwa 14's visual meshes: `presentation/presentation.toml` declares
  `players = "25k"` (the Menagerie meshes decimated to ~25k triangles per arm, renderers/presentation/lod.json),
  tools/sportlib bundles them into the simulator (`sport/presentation/assets/`, sha256-checked, licence kept) and the
  core's camera worker draws the players' scene (core/observe.players_scene, the same kinematics checked); link 7
  keeps its capsule (the striker's shaft). Table, puck and mallets are unchanged. `[observation] render_threads = 1`.
- **Format match, the opponent**: unchanged (inside the world, from the true puck position a reaction delay ago);
  in `scores` its side is `opponent`.
- **Slowdown**: `RCB_SLOWDOWN` (default x20, `[timing] slowdown`, as both legacy tasks).
- **Reviews**: 300 s in both formats (legacy: 300 s in the duel, no limit in the single-player game).

## Golden replays

`python3 tools/golden.py replay airhockey [--external <golden-data store>]`. Native v7 cases (re-recorded 2026-10-12 with
`tools/golden.py record`: the arms' setpoint generator brakes within the jerk limit; under v6 they ended 1:0, 1:0 and
2:0, now 0:1, 0:0 (A by the tie-break chain: territory or possession) and 0:1; under v5 duel-ref-s1-A ended 0:1): `duel-ref-s1-A` (seed 1, A first),
`duel-ref-s3-B` (seed 3, B first), `match-ref-L1-s1` (the scripted opponent at level 1, seed 1). Legacy records (converted by
`golden/convert.py`): the four in which the game was played (match o-L1; duel oo-s1-A, oo-s1-B, ready-forfeit-B)
belong to revision v2 and are retired (`convert.RETIRED`); the three without a scored rally replayed identically under
v3 and v4 (state hash = the legacy `Hockey.digest()`, goals, winner, steps); under v5 the state hash also covers the
arms' setpoint generators, so all seven legacy records are retired in the converter and golden/ keeps the three
converted no-play cases with their v5 hashes (goals, winner, steps unchanged); v6 adds the mallets' path generators to
the state hash, so the three carry v6 hashes (goals, winner, steps unchanged); under v7 `match-abandon-60s` (whose arms
move) carries a new hash, the two no-show cases (k 0) keep theirs. Tagged:
- duel `nop-B`: the no-show rule of core 0.2.2 (B loses by `no_show`, A wins: the legacy outcome, start_forfeit);
- match `nop`: the no-show rule (a valid loss, success 0; the legacy no_start was a valid loss too);
- match `abandon-60s`: a started single game whose session ends mid-game is a valid loss by forfeit in the core
  since 0.2.6 (`abandoned`, success 0; cores 0.2.2-0.2.5: incomplete), as the legacy `session_ended`; the replay itself
  is the same (tagged as an intended rule change).

## Real arm limits (revision v5)

Since revision v5 (2026-10-08, real-limits audit item 7) both arms, the scripted opponent's too, are held to the real
KUKA LBR iiwa 14's limits (`[arm] body = "kuka_iiwa14"`, `bodies/kuka_iiwa14/body.toml` revision iiwa14-limits-2,
`bodies/limits.py`): the joint reference of every 10 ms step is retimed so that no axis needs more than 0.95 x its
data-sheet velocity (the Air Hockey Challenge's constraint factor); the robot's 1 kHz setpoint generator
(`limits.JerkLimiter`) keeps the setpoint within that velocity, the AHC accelerations (10 x the velocities) and the
estimated jerks (50 x the accelerations); the servo's commanded acceleration is clipped to the same accelerations.
Before (v4): the 3 rad/s clip touched only the velocity feed-forward and the position reference was free (measured
p99 1.9 x the data sheet on axes 2 and 4), the acceleration clip was +-50 rad/s^2. Effect: the mallet reaches about
1.0-1.2 m/s on typical lines (v4: about 2 m/s), shots leave at about 1.4 m/s (v4: 2.6).

The reference player was revised for v5 (B9 review): it strikes only from rest at the wind-up point (within 2.5 cm,
below 0.12 m/s; the old one struck while still moving and, with the slower arm, mostly missed), the near-post block
stands where the 0.35 m mouth actually begins (the old post at |y| = 0.05 left a gap at the mouth's edge), deep pucks
are cleared with 2 cm to spare and never pushed from in front of the mouth, a corner push never drives the puck
towards the own end, the guard never runs through a puck in its own half from in front, and its shot choice assumes
1.4 m/s shots and a 5 m/s^2 defender. Reference-vs-reference duels (camera players, core/local.py as the fairness
check, seeds 1-10 x both first orders, N = 20 unless stated):

| Code | goals per game | own goals (of all goals) | 0:0 games |
|---|---|---|---|
| v4 (main), old reference | 1.45 | 20 of 29 | 2 of 20 |
| v5 limits, old reference (N = 11) | 1.73 | 15 of 19 | 3 of 11 |
| v5 limits, revised reference | 0.45 | 9 of 9 | 14 of 20 |
| v5 limits, revised reference, seeds 1-20 (N = 40, the fairness games) | 0.48 | 17 of 19 | 28 of 40 |
| the same after the rebase onto main `da5500f8` (the library's iiwa14-limits-2 jerks, rounded differently) | 0.47 | 18 of 19 | 25 of 40 |

The revision removes most of the own goals and blunders, which were most of the goals: between two competent
defenders with the real arm, a 0.35 m mouth and 1.4 m/s shots, clean shots almost never score (the B9 finding:
the rule geometry plus slow shots). See pilots/airhockey_modes and the pilot report for a wider goal.

## The mallet on the table (revision v6)

Changhe's physical-realism audit (2026-10-09) found the v5 mallet leaving the table: its centre down to 11.1 cm below
the surface, 4.2 % of the control steps more than 1 cm below it, the rod up to 9 cm into the table. The real robot this
models cannot do that: the Air Hockey Challenge (AHC, TU Darmstadt's KUKA iiwa 14 air-hockey setup that the arm model
comes from) holds the end effector (the universal joint above the mallet) within +-0.02 m of its height on the table
and counts every step outside as a constraint violation. The cause was the controller, not the physics: v5 planned the
mallet's path only against the joint velocities and changed its acceleration at will, and the robot's 1 kHz setpoint
generator (limits.JerkLimiter) then limited each joint's acceleration and jerk on its own, which bends the joint-space
path, so the mallet left the plane (up to 12 cm, up and down).

Revision v6 enforces the band in the motion layer (the IK / servo side), as the real arm's motion planner would:
- the mallet's path is a jerk-limited motion on the table plane (bodies/servo.PlanarBase on the norm, the library's
  shared servo; at most 2 m/s, 10 m/s^2 as before, and 150 m/s^3) whose speed, acceleration and jerk caps are, every
  control step, what the joints allow at that pose: the joints' rad per metre of mallet motion (along the way to the
  command, along the present motion, where that motion leads in 0.1 s and, near the reach, radially) times 0.95 / 0.6
  / 0.3 of their velocity / acceleration / jerk limits; it brakes before the workspace's edge and aims 5 mm inside it;
- the IK keeps the arm near its home posture in the null space (no drift to the joint limits in long rallies) and a
  path step the IK cannot reach on the table is not taken;
- `none` and the level-1 opponent's "stay" hold the path's position (v5: the lagging mallet's, which could oscillate);
- the setpoint generator, the servo and the motors are unchanged (the same limits); physics is unchanged (no contact
  between mallet and table was added: see below).

Measured (2026-10-10, the same 14 games on v5 and v6: the reference against the scripted opponent L1-L4 seeds 1-2,
reference duels seeds 1-2 in both orders, random duels seeds 1-2; both arms, every physics step):

| | v5 (main) | v6 |
|---|---|---|
| mallet centre, lowest | 11.4 cm below the surface | 2.2 cm below (its nominal centre is 1.4 cm above) |
| control steps with the centre more than 1 cm below | 4.09 % | 0.01 % |
| end effector outside +-2 cm of its height | 7.8 % of the steps (from -12.9 to +12.2 cm) | 0.027 % (from -3.6 to +4.0 cm) |
| rod below the table surface, deepest / share of steps | 9.2 cm / 2.24 % | never (lowest 0.0 cm) |

The few v6 steps outside the band are short excursions (5 of the 28 arms, 2-35 control steps each, all within -3.6 /
+4.0 cm) where the setpoint generator still lags the path in a fast reversal. The cost is CPU: about 5 IK solves per
arm and control step instead of 1-3 (a game takes about 1.2-1.8x the CPU of v5 on a loaded host).
The margins were chosen by measurement (the same 14 games and the B9 strike tests): the velocity share 0.95 keeps
shots near v5's speed (B9 straight shots from rest 1.26 m/s, v5 1.45; with 0.9: 1.2 m/s); a velocity share 1.0 or an
acceleration share 0.8 bring v5's speed back but also the excursions (0.14-0.29 % of the steps outside the band, the
rod up to 6.5 cm into the table).
Evaluated and not chosen: making the mallet collide with the table. The table would stop only the downward
excursions (the upward ones, a mallet lifted over the puck, would stay), the model's mallet cylinder already reaches
1.6 cm into the table at its nominal height (it would need a new collision shape and so a physics change), and on the
real robot pressing the mallet into the table is exactly what the AHC constraint and the robot's safety stop prevent.

The two-mode pilot (pilots/airhockey_modes) follows: a low-level plan is also refused when the setpoints the robot's
setpoint generator would make of its rows (and of the hold after them) take the mallet more than 2 cm off the table.

## Fairness

Two copies of the reference player, seeds 1-20, both first orders (N = 40, `tools/check airhockey --stages 4`; one
reference duel takes about 100 s of CPU, so `[check] stage_timeout_s` raises the stage's limit to 3 hours).
Revision v6 (the mallet within +-2 cm of the table, 2026-10-10; the same seeds 1-20 x both orders, N = 40, played by
pilots/airhockey_modes/refgames.py with core/local.py as the stage does, in parallel): first-mover advantage -2.5 pp
(95% CI +-15.5 pp), seat A advantage -2.5 pp (+-15.5 pp). No interval lies beyond +-5 pp (pass); the intervals are
wider than +-5 pp, which a realtime sport only warns about. 23 of 40 games ended 0:0 (decided by territory), 0.53 goals
per game (seat A 10, seat B 11), 18 of the 21 goals touched last by the side scored upon (deflections and clearances).
Straight shots from rest leave a little slower than in v5 (B9: 1.26 m/s median, v5 1.45), because the path keeps the
joints inside their limits with room for the setpoint generator.
Revision v5 (real limits, revised reference, 2026-10-08; the same seeds 1-20 x both orders, N = 40, played by the
pilot's harness pilots/airhockey_modes/refgames.py with core/local.py exactly as the stage does): first-mover
advantage +2.5 pp (95% CI +-15.5 pp), seat A advantage +7.5 pp (+-15.3 pp). No interval lies beyond +-5 pp (pass);
the intervals are wider than +-5 pp, which a realtime sport only warns about. 28 of 40 games ended 0:0 and were decided
by territory; 0.48 goals per game, 17 of the 19 goals own goals. Re-run after the rebase onto main `da5500f8` (2026-10-09): the arm limits now come from the
robot library's entry `bodies/kuka_iiwa14/body.toml` iiwa14-limits-2 (#26; the same accelerations, the jerks rounded
to 2 decimals instead of 3, so play differs slightly and the goldens were re-recorded); the same 40 games: first-mover
advantage +10.0 pp (+-15.2 pp), seat A -10.0 pp (+-15.2 pp): pass with the realtime warning; 25 of 40 games 0:0, 0.47
goals per game, 18 of the 19 goals own goals.
Revision v4 (0.35 m goal, 2026-10-08): first-mover advantage 0.0 pp (95% CI +-15.5 pp), seat A advantage -15.0 pp
(+-14.8 pp). No interval lies beyond +-5 pp (pass); the intervals are wider than +-5 pp, which a realtime sport only
warns about. The same 40 games replayed by a harness (identical winners): 28 decided by goals (26 by the goal count, 2
by the last goal), 12 ended 0:0 and were decided by territory; 1.1 goals per game (seat A 20, seat B 25 goals; seat A
won 14 of 40, within the noise of N = 40: the robots and cameras are point-symmetric, E9).
Revision v3 (0.40 m goal, the new reference, 2026-10-07): first-mover advantage -12.5 pp (95% CI +-15.0 pp), seat A
advantage +12.5 pp (+-15.0 pp). No interval lies beyond +-5 pp (pass); the intervals are wider than +-5 pp, which a
realtime sport only warns about. The same 40 games replayed by a harness (identical winners and advantages): 24
decided by goals (23 by the goal count, 1 by the last goal), 16 ended 0:0 and were decided by territory; 1.0 goals
per game (seat A 27, seat B 13 goals; within the noise of N = 40: the robots and cameras are point-symmetric, E9).
Before (revision v2, 0.25 m goal, the legacy reference): first mover +7.5 pp, seat A -7.5 pp (both +-15.3 pp), 36 of
40 games 0:0 and decided by territory, 4 by a single goal.

## Why the goal is wider (D10)

Measured 2026-10-07 (review item B13) with a harness that plays games of the official format (120 s cap) in the
sport's image; "L-k mirror" = the scripted opponent's logic on both seats; seeds 1-5, both first orders (N = 10 per
row) unless stated; the scripted players and "privileged" reference runs read the true puck position, "camera" runs
are the real players.

| 0.25 m goal (revision v2) | goals per game | decided by goals |
|---|---|---|
| legacy reference mirror, tools/check N = 40 | 0.1 | 4 / 40 (36 by territory) |
| legacy reference mirror, privileged, N = 20 | 0.05 | 1 / 20 |
| L1 / L2 / L3 / L4 mirror | 9.8 / 1.2 / 2.5 / 0.1 | 10 / 8 / 10 / 1 of 10 |
| new reference mirror, camera, N = 20 | 0.15 | 3 / 20 (17 by territory) |
| new reference mirror, privileged | 0.0 | 0 / 10 |

Why so few goals: a mallet on the puck's line blocks puck centres within 0.08 m (mallet + puck radius) of it, the
goal admits them within 0.0934 m of its centre, so a defender standing in front of the goal leaves 1.3 cm per side.
The arms move the mallet at most about 2 m/s (the path's 2 m/s and 10 m/s^2; the path controller stays near
2.1 m/s even with higher caps, the motors near 3.3 m/s), shots leave at 1-1.9 m/s (at most about 2.6 m/s) and
need 0.5-1 s to arrive, while a defender that sees the puck every 50 ms needs about 0.2-0.35 s to move 0.1-0.2 m.
No clean shot on goal (the straight-line path 80 ms after the hit, rail bounces included, entering the mouth)
scored in any mirror above; the few goals came from scrambles at the end rail, mostly own goals. Rallies do not
die: 2-4 rallies per game (1-3 seven-second fouls), every game runs into the 120 s cap.

The USAA rules have no time limit (I.1: first to 7) and nothing that restricts where the mallet may stay (I.12
only keeps it on its side of the centre line; VI.6 goal-tending is palming), so no crease rule exists to adopt.
A longer cap scales a goal rate of about 0.1 per game linearly; a lower goal target changes nothing when nobody
reaches it. The goal width (the rulebook names sanctioned tables, II.A.1, not their dimensions) does change it
(new reference mirror, privileged): 0.30 m 0.2 goals per game (N = 6, an earlier version of the player), 0.35 m 1.2 (9 of 16 games decided by goals),
0.40 m 1.5 (12 of 16), 0.50 m 1.8 (5 of 6). Revision v3 used 0.40 m; revision v4 uses 0.35 m, the medium width
chosen (2026-10-08) after the same study with model players (below: Model players). Faster arms (a physics change, not made:
path 5 m/s, 40 m/s^2, stiffer path gains; the motors then reach 3.2-3.4 m/s) gave 0.3 goals per game at 0.25 m (N = 6, an earlier version of the player),
because the defender gets faster too. The scripted level-4 opponent (true puck position, 30 ms delay, crossing
prediction) stays nearly unbeatable for shots: L4 mirror 0.1 goals per game at 0.40 m and 0.2 at 0.50 m.

Against the scripted opponent at 0.40 m (camera reference, seeds 1-5 and the level's official seeds, both first
orders, N = 16 per level): L1 16/16 won (5.8 : 0.7 goals per game), L2 16/16 (5.0 : 0.3), L3 16/16 (5.3 : 0.3),
L4 10/16 (0.0 : 0.4; the 10 wins by territory, the 6 losses by one goal); on the official seeds alone 6/6, 6/6,
6/6, 3/6. (Legacy reference at 0.25 m, privileged, N = 10: L1 10/10 with 1.4 goals per game, L2 10/10 with 0.4,
L3 10/10 with 3.0, L4 8/10 with 0.1.)

Model players (2026-10-07, `docs/benchmarks/airhockey-goal-width-20261007.md`; Luna and Astra, x20, 14 valid games
with an experiment-only width knob): all 4 games at 0.25 m, 4/4 at 0.35 m, 4/4 at 0.40 m and 1/2 at 0.50 m were
decided by goals (one at 0.35 m by the last goal, after 1:1), but the games against the level-2 opponent mostly
measure how well the model's program plays (10 of the 27 goals at 0.25-0.35 m were own goals), and the three
Luna-Astra duels (0.25 / 0.35 / 0.40 m) had 4, 2 and 1 goals. Revision v4 takes the medium width, 0.35 m: the reference mirror is
decided by goals in 28 of 40 games (Fairness, above), shots still have to beat a defender, and the scripted
opponent's levels stay as they are (L4 unchanged).

## Reward keys

The common keys come from `core/grading.py`, the sport's from `grade.py` (the legacy keys). Differences:
- both formats: new common keys `no_show`, `<seat>_no_show`, `solo`, `direct_play`, `lowlevel_actions`;
  `grader_error` has the core's meaning; `official_config` is recomputed with the legacy per-format rule;
  dropped (the core does not record them): `sim_lag_frac_late_over_1_period`, `sim_step_ms_max`, `refused_busy` /
  `<seat>_refused_busy` (the core refuses a second act in flight with 429 before counting it), and
  `video_rendered` keeps the core's meaning (`replay.mp4` or `video.json`);
- `late_over_100ms` = the core's `sim_lag.late_max_s` > 0.1; `sim_effective_slowdown` = run wall time / run sim time;
  `n_actions` = acts applied (the core's per-seat counters); `revision` = 3 (hockey.py's revision; legacy: 2);
- match: new common keys `A_won`, `A_transient_api_errors`, `A_unrecovered_api_errors` (the legacy unprefixed keys
  stay); `session_ended` = the game ended unfinished because the player never started or the session ended
  (informational; the no-show and, since core 0.2.6, a session that ends after the start, `abandoned`, are valid
  losses);
- duel: unchanged otherwise.

## Experiment knob (not in the official tasks)

`RO_AIRHOCKEY_GOAL_M` (sim environment, 0.2-0.6) replaces the goal width of D10 for experiments only (the goal-width
study, docs/benchmarks/airhockey-goal-width-20261007.md): the revision becomes `airhockey-program-v3+core-0.2+goal<W>m`
(episode header, final.json), final.json carries `goal_width_m`, and `/spec` gives the width in `geometry.goal_width_m`
and in the `table` text. The generated tasks never set it.

## Not in the generated tasks (tools/generate.py hooks)

OpenCV in the seat image (the legacy single-player image had it) and the sport's own prompt sections (the generic
instruction plus `[spec]` replace airhockey_prompt.py). The legacy broadcast adapter
(`scripts/olympiad/broadcast/adapters.py`, `regen_rt.py`) imports `hockey` from a task's sim directory: for a v0.2 task
that is `environment/sim/sport/`.
