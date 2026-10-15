# soccer_ssl: wheeled robot soccer 5 v 5 (RoboCup SSL Division B)

Ten SSL robots after TIGERs Mannheim's v2020 (`bodies/ssl_robot`) and an orange golf ball on the 9 x 6 m Division B
field, under the SSL rulebook of 2026-09-19 with the deviations S1-S15 of `sport.toml`. Two halves of 300 s of
playing time, a shoot-out when level, an automatic referee. Built in parallel lines (olympiad_review/screening/
`soccer_ssl_interfaces_zh.md`; reports `soccer_ssl_{body,world,camera,plugin}_zh.md`): A the robot, B the field, ball,
world and referee, C the overhead cameras, D the scripted opponents and the reference player, E the plugin, the skill
contract, the observation and the tasks.

| task | format | seats |
|---|---|---|
| `soccer_ssl-match-wallclock-program-standard` | `match` | A (the blue team) against the scripted yellow team (levels 1-3: 1 warm-up, 2 and 3 official) |
| `soccer_ssl-duel-wallclock-program-standard` | `duel` | A (blue) against B (yellow), one program per team as in real SSL |
| `soccer_ssl-team-wallclock-program-standard` | `team` | A0..A4 against B0..B4, one seat per robot, the team channel between teammates (S6, an experimental condition) |

| File | Part of contract ② | What it is |
|---|---|---|
| `sport.toml` | declarations | rulebook and deviations S1-S15, `[body]` ssl_robot, formats, `[team_channel]`, modes (`[modes.high]`: the skill API), timing (x10, reviews of 600 s), observation whitelist, rule constants, tiebreaks, exploits E1-E10, players, fairness, official seeds, tasks, capabilities |
| `skill_api.py` | actions (single source) | the skill contract: `move(x, y, theta, t, v_end)`, `kick(speed, mode, armed_for)`, `dribble(rpm)` as `SkillContract`s, the robot handler's `substitute(to)` and the `batch` envelope; ranges, timing, feedback, control authority, failure handling (P1-P6); the scripted team's interface |
| `plugin.py` | world, actions, referee, observation | `SSLMatch` around `sim/world.py` `World`; `SkillLayer` (schema generated from the contract; `check_act` for players and the scripted team alike); units (goal, half time, full time before a shoot-out, each shoot-out kick); tiebreaks; observation; the four overhead cameras (`sim/camera.py camera_defs()`) drawn from `scene()` and `render_state()` (the world 4 control steps ago) |
| `sim/` | world | field (even venue lighting), ball, world (`World.execute`, vision frames, own-state, truth and referee views, the robot handler), referee, frames, cameras, the stand-in robot |
| `grade.py` | reward fields | goals for and against, goal margin, fouls, cards per seat; the shoot-out; how the match ended |
| `tests/test_plugin.py` | exploit list | one test per `[[exploits]]` entry, the contract against the robot and the field, the B9 probe (`[modes] b9_test`) |
| `tests/test_rules.py`, `test_field_ball.py`, `test_smoke.py` | world | line B's rule tests and smoke matches |
| `builtin_players.py`, `opponents/` | built-in players | line D: the camera-only reference player; the scripted team `ScriptedTeam(level, team, rng)` |

## The interface in short

- **Frames.** Commands and the observation are in your team's frame (you attack +x); the camera images are in the
  world frame; `observation.frame.attack_sign` maps them (team = attack_sign x world for x, y). The ends change at half
  time.
- **Commands** (one per robot; `entity` = the robot's number, "0" is the keeper): `move` (pose; optional arrival game
  time `t` and arrival velocity `v_end`), `kick` (speed, flat or chip, now or armed for a time: it fires when the ball
  blocks the robot's break beam), `dribble` (bar speed), `substitute` (off / on: the robot handler), `batch` (several
  robots at one boundary). A command takes effect at the next 10 ms control step; the reply carries its status
  (`ok`, `late` with the planned `eta`, `clipped`, `kicker_not_ready` with `wait_s` ...).
- **Observation.** Four 800 x 600 overhead tiles (2 x 2, the quadrants as they lie), 40 ms late
  (`vision.t_capture`); your robots' own state (pose, velocity, dribbler, capacitor, armed kick, break beam, target,
  eta); the game controller's information; the unit's events. No ball or opponent coordinates.
- **The scripted team** (match format) is line D's `opponents.ScriptedTeam(level, "yellow", rng)`, called every
  control step with `World.truth_view("yellow")`; its act bodies pass the same `check_act` and `World.execute` as a
  player's. Without `opponents/` the plugin falls back to `Placeholder` (a keeper and a ball chaser; not a level).

## Run

```bash
python3 sports/soccer_ssl/tests/test_plugin.py            # in the sim image (mujoco, numpy)
PYTHON=python3.12 tools/check soccer_ssl --stages 1,2,3
python3 tools/generate.py soccer_ssl && (cd tasks && harbor add --scan .)
```

Known gaps (report `soccer_ssl_plugin_zh.md`): `tools/check` stage 1 looks `[body]` up in `bodies/__init__.py BODIES`
only (ssl_robot is in `PLANAR`: a shared PR); a full
in-process match draws about 14000 observations per seat, so stage 2 runs for hours; the two teams' identical images
are drawn twice (a shared core PR could draw them once per step).
