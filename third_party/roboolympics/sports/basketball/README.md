# basketball: the robot basketball shootout

Migrated from the legacy task `basketball-shootout-program-standard` (docs/framework_zh.md §10, stage 1; sources were
`scripts/olympiad/basketball/` and `scripts/olympiad/shared/basketball/`). The task name is kept (`[[tasks]] id`);
`tools/generate.py` writes it from `sport.toml`. Not the 3x3 game: a shooting contest. One format: the single
shootout. The legacy duel (`basketball-duel-program-standard`) was retired on 2026-10-07 (review item B10): both
players shot on identical courts without any interaction, so a head-to-head comparison is the two players' single
scores on the same seeds.

**Rules** (FIBA Official Basketball Rules 2024, Art. 16; closest standard contest: the FIBA 3x3 Shoot-out; deviations
D1-D5, D7 of `sport.toml`). A declared shooting machine (D7: a launcher like Toyota's CUE robots) on a 1 m pedestal
shoots from five spots, mid-range to beyond the
three-point line (NBA markings), past a static defender standing on the line to the hoop; every make counts 1. Each
shot is `shoot ANGLE SPEED SPIN YAW` (the machine's release, with a stated scatter); the player sees camera images only (the hoop,
the ball and its own position are not given). Alone, 10 shots (two per spot), success = a complete shootout with at
least 6 makes.

| File | Part of contract ② | What it is |
|---|---|---|
| `sport.toml` | declarations | rule book and deviations, formats, modes, timing, `game shoot`, observation whitelist and renderer, rule constants, tiebreaks, exploit list, players, fairness, tasks, capabilities |
| `bball.py`, `court.py` | simulator | the MuJoCo court and the shooting machine (primitives, no mesh files), the release, the flight with drag and Magnus force, defender, make rule (the legacy simulator with the arm replaced, revision `basketball-5`); the legacy duel module `bb_duel.py` was removed with the duel |
| `plugin.py` | world, actions, referee, observation | `Basketball` (MatchBase) around `Game`, `reply_views` (the legacy picture strip of a shot, drawn by the core); `ShootLayer`; `scene()`: court.py's MJCF with the shooter a mocap body; the four cameras |
| `builtin_players.py` | built-in players | `Reference` (the legacy reference shooter: rim in the zoom tile -> hoop bearing and range -> flight solve), `Random` |
| `grade.py` | reward fields | `evaluate(final)`: the legacy single task's reward keys |
| `tests/test_basketball.py` | exploit list | one test per `[[exploits]]` entry |
| `golden/` | golden replays | shootouts recorded with the shooting machine (revision `basketball-5`); the legacy records of the golden-data store and their converter were retired with the arm |

Run everything: `PYTHON=python3.12 tools/check basketball`.
Last run (core 0.2.12, main a71f92f7 + the release-machine declarations, 2026-10-09): `tools/check basketball --stages
1,2,3` 63/63 (stage 3: `single-s1` reference 10/10, `single-random-s3` random 0/10, both identical). B9 on
basketball-5: a make exists from 150/150 spot situations at the 14 m/s cap (PR #38). The two leftovers of the arm
outside this sport are gone: `tools/arm_limits.py` no longer lists basketball, and `bodies/franka_panda/body.toml` no
longer names it among the Panda sports.

## The shooting machine (revision `basketball-5`, 2026-10-09)

Until `basketball-4` the shooter looked like a Franka Panda that threw the ball: the arm's joints were set
kinematically (velocity p99 6.7x, acceleration 19x the real Panda's limits) and the ball's velocity (3-14 m/s) was
assigned at release, while a Panda's hand reaches about 2 m/s, which makes no shot from any spot (real-limits audit
G3; B9 study HB7). The user decided on 2026-10-09 to declare what the physics always was: a **shooting machine** like
Toyota's CUE basketball robots (deviation D7).

- **Action** (unchanged interface): the machine's release, `shoot ANGLE SPEED SPIN YAW`: elevation 20-80 deg,
  speed 3-14 m/s, backspin -5..5 rev/s, yaw -35..35 deg from the heading.
- **Release envelope**: the muzzle, 0.50 m in front of the pedestal axis and 2.17 m above the floor, turned by the
  yaw (the tube pivots about the muzzle, so the elevation does not move it). Before: 0.5029 / 2.1663 m, the cup of
  the arm's release pose.
- **Speed cap 14 m/s, not lowered** (real machine): CUE6's 24.55 m shot (Guinness World Records "farthest basketball
  shot by a humanoid robot", 2024-09-26) needs at least about 20 m/s with this sport's drag (B9 HB7), so 14 m/s is
  within a real shooting robot's reach. A human-level cap (9 m/s) would make the top-of-the-key three impossible
  (B9: 0/30); the spots need at most about 9.5 m/s.
- **Execution scatter** (seeded per shot, the legacy values kept): 1 sigma 0.5 % speed, 0.4 deg elevation, 0.4 deg
  yaw, 0.15 rev/s spin. Source: CUE3's 2,020 free throws in a row (Guinness World Records, May 2019) put its
  free-throw make rate at >= 99.85 % (one-sided 95 % bound, 0.05^(1/2020)). Aimed at the centre of the make region
  (57 deg, 3 rev/s backspin, 8.12 m/s from a release point on the free-throw line, no defender), this machine makes
  99.87 % (6000 seeded draws; 95 % interval 99.78-99.96 %): the declared scatter is about the largest one consistent
  with CUE3's record (0.8 x these values: 99.98 %).
- **The machine's motion is display only**: the turret and tube slew to the commanded aim in 0.25 s (the loaded ball,
  hidden in the tube, slides up to the muzzle by the release at 0.45 s) and return after the shot; no machine part has a contact pair, the ball is released at the muzzle at exactly the commanded
  yaw with the set velocity (exploit test E10 replays a shot under an arbitrary machine path and gets the same
  flight). The arm was removed rather than kept as a display: any arm path that looks like a throw would show a
  release the arm cannot make, which is what the audit flagged; with it went the Panda meshes (`scene/assets`) and
  the spectator presentation (`presentation/`, Panda visual meshes).
- **Golden**: both cases re-recorded; the legacy records of the golden-data store (and `golden/convert.py`) were
  retired, as badminton's machine-serve records were.

## What changed from the legacy tasks

- **No duel** (review item B10, 2026-10-07): `basketball-duel-program-standard` and everything only it needed are gone
  (`bb_duel.py`, deviation D6, the extra pairs and the clock last resort, the tiebreaks, the fairness check, exploits
  E4, E5 and E7, the duel observation fields, the reference's seeded personal error, the duel golden cases and the
  converted duel records). The exploit ids E1-E3, E6, E8, E9 are kept.
- **Observation**: a pure `observe()` (the shots with their results, `shots`, instead of the legacy
  "since your last call" queue; the own `last_shot`); the legacy single's `last_shot.release_point` is gone (a world
  coordinate that gave away the shooter's position). The images are drawn by the task's renderer from the same scene
  and camera poses (the reference shooter's commands on the legacy records are reproduced to the last digit); the
  tiles carry no name labels (`observation.tiles` names them).
- **Default seed**: `RCB_SEED` default 1 (legacy 7); official seeds 1, 2, 3 and 7, 17, 27 (legacy) in one list.
- **Reward keys**: the legacy single's `metered_requests`, `metered_input_tokens`, `metered_output_tokens` and
  `rate_limit_retries` (from think_usage.json) are not in reward.json (the v0.2 verifier reads final.json only).
- **Picture strip** (core 0.2.8, revision `basketball-4`): the strip in the reply to a shot (2 x 4 tiles around the
  closest approach: rim camera, shooter_zoom) is the same picture, now declared through the core's reply-image hook
  (`reply_views`) and drawn by the core with the task's renderer off the service lock (before, the plugin drew it
  itself inside `reply()`, under the lock, with its own renderer instance). It arrives as `observation.image_png_b64`
  + `observation.tiles` (each tile labelled with its time from the closest approach) instead of `result_image_b64` /
  `result_image_note`.
- **Video**: the core's spectator video (a frame per shot) replaces the legacy real-time `replay.mp4` / `replay_A.mp4`.
