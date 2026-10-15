# foosball: table football (ITSF)

Foosball on the v0.2 framework (docs/framework_zh.md §4.2, §8), migrated in stage 2 from the legacy tree
(`scripts/olympiad/foosball/`, `scripts/olympiad/shared/foosball/sim/`). The simulator `foos.py` and the scripted
team's policy `foos_policy.py` are the legacy files, unchanged (git history follows them), so every legacy recording
replays bit-exactly (`golden/convert.py`).

**Rules**: ITSF Rules of Table Soccer, Standard Matchplay Rules v2.0 (December 2023), with the deviations D1-D9 of
`sport.toml`: one game to 5 goals, the opening sequence, possession time limits per rod (goalie rods 15 s, 5-rod
10 s, 3-rod 15 s), the spinning rule, dead ball and ball off table restarts, and a safety cap (240 s of play or 40
rallies) with a stated last resort (no draws). A procedural MuJoCo table (1 kHz physics, 10 ms control): 2 x 4 rods,
each with a slide and a rotation joint driven by force- and torque-limited motors that follow a speed- and
acceleration-limited path to the commanded target; a command reaches the motors 20 ms (2 control steps) after the step
that takes it. The player sees the table only through its fixed overhead camera (own goal on the left).

| Format (task) | Seats | Unit | Result |
|---|---|---|---|
| `match` (`foosball-match-wallclock-program-standard`) | A against the scripted red team (seat B inside the world), level 1-4 (`RCB_LEVEL`, default 1) | a rally | success = winning the game |
| `duel` (`foosball-duel-wallclock-program-standard`) | A (blue) against B (red); `FIRST_PLAYER` has the first kick-off | a rally | first to 5 goals, then the tiebreak chain (no draws) |

The task names are the Season-1 names (`[[tasks]] id`).

| File | Part of contract ② | What it is |
|---|---|---|
| `sport.toml` | declarations | rulebook and deviations, formats, timing, observation whitelist, rules (`[spec]`), tiebreaks, exploits E1-E9, players, fairness, official set, tasks, broadcast camera, capabilities |
| `foos.py` | the simulator (legacy, unchanged) | `Foosball`: the table (`scene_xml`), the motor paths, the referee (goals, possession clocks, opening sequence, spinning, dead ball, ball off table, the cap and `winner()`), the scripted team, `digest()`; `validate_action` |
| `foos_policy.py` | the scripted team's policy (legacy, unchanged) | `Policy`: blocking, the opening sequence, aimed kicks and passes, lifting, goal splitting |
| `plugin.py` | world, actions, referee, observation | `FoosMatch` (MatchBase over one `foos.Foosball`): `step`, `state_digest`, `apply` (= `Foosball.schedule`), `phase` / `resume` (= `Foosball.ready`) / `scores` / `tiebreaks`, `observe` / `cameras` / `render_state`; `RodsLayer`; `Foosball.scene()` (= `foos.scene_xml()`) and `spec()` (the legacy /spec) |
| `builtin_players.py` | built-in players | `Reference`: the legacy camera-only player (`solution/camera_player.py`) with a verbatim copy of `foos_policy.py` (the file runs alone in the player's container); `Random` |
| `grade.py` | reward fields | the legacy reward keys that final.json still carries |
| `tests/test_foosball.py` | exploit list | E1-E9 (from the legacy unit tests) and the check that the reference's policy is the scripted team's |
| `golden/` | golden replays | 6 converted legacy recordings + `convert.py` for the golden-data store (7 records; tagged cases below) |

Run everything: `PYTHON=python3.12 tools/check foosball` (stage 3 with `RO_GOLDEN_DATA` also replays the store).
Generate the tasks: `python3 tools/generate.py foosball`.

## What changed from the legacy tasks

- **Action**: the legacy body `{"rods": {ROD: {slide, angle, omega, speed}}}` with `"type": "rods"` (optional in the
  request); the logged low-level command is `{"type": "rods", "rods": ...}` (foos.validate_action's normalised
  targets). There is no `game act ROD=SLIDE,ANGLE` shortcut (core/game.py shortcuts take flat numbers only): `game act
  JSON`. The reply's `took_effect` holds `taken_step` and `effect_step`.
- **Observation**: the legacy state without the core's own fields, pure: `rally`, `t_play`, `t_rally`,
  `play_time_left_s`, `awaiting_review`, `game_over`, `score`, `fouls`, `restart`, `rods`, `last_rally` and `rallies`
  (every ended rally, instead of the legacy "since your previous call" `news` queue). The image is one tile
  (`observation.tiles`) drawn by the task's renderer from the camera definition `overhead` (the legacy MuJoCo camera
  `overhead_<seat>`: same pose, field of view and size).
- **/spec**: the legacy `match`, `action`, `camera` and `geometry` parts; the legacy `timing` part is `game_timing`
  (the core's `timing` holds the slowdown and the control step).
- **Reviews in `match`**: 300 s (`READY_TIMEOUT_S`, the core's rule for every format); the legacy single had no limit.
- **Start and session end** (core rules): a duel seat that never starts loses (`no_show`, valid; the legacy service
  gave the same result as a start forfeit). A single player who never starts has a valid loss (`no_show`, success 0;
  legacy: `no_start`, a valid loss). A single-player session that ends in the middle of the game is a valid loss by
  forfeit since core 0.2.6 (`abandoned`, success 0; cores 0.2.2-0.2.5: incomplete); legacy: `session_ended`, a valid
  loss.
- **Slowdown**: `RCB_SLOWDOWN` (default 20, `[timing] slowdown`), as the legacy `SLOWDOWN`.
- **Default seed of `match`**: `RCB_SEED` default 1 (legacy: 101); `official_config` keeps the legacy official seeds
  and slowdowns (grade.py).
- **Reward keys**: all legacy keys except `sim_lag_frac_late_over_1_period`, `sim_step_ms_max` (the core does not
  record per-step lateness or step times) and `refused_busy` / `<seat>_refused_busy` (the core refuses a second act
  in flight with HTTP 429 without counting it); new from the core: `no_show`, `<seat>_no_show`, `solo`,
  `direct_play`, `lowlevel_actions` (and `A_won`, `A_*_api_errors` in `match`).

## Fairness

The reference on both seats, both first kick-offs, seeds 1-20 (N = 40, `[fairness] matches_per_order = 20`): the
first kick-off's advantage -2.5 pp (95% CI +-15.5), seat A's -7.5 pp (+-15.3): no interval lies beyond +-5 pp (no
FAIL), both are wider (WARN, realtime). Every match was decided by goals (mean 10 600 control steps). A match takes
about 110 s in-process, so the 40 matches do not fit in one stage-4 container of `tools/check` (3600 s timeout):
they were played as the same `core.local` matches as `core.testkit fairness`, in two shards of 20, and merged with
`core.testkit.advantage`.

## Live runs

`tools/check foosball --stages 5 --harbor` (Harbor, x20): reference = success 1 in both formats (match L1 seed 1: won
in 4317 s; duel seed 1: valid, B won 5:1 in 1282 s), nop = 0. The offline live run (`tools/live.py`, no Harbor)
passes the duel and both nop runs, but its reference player is killed after `PLAYER_TIMEOUT_S` = 3000 s: a match
game at the default x20 takes longer (about 21 000 control steps live = 4200 s). At x10 (`--env RCB_SLOWDOWN=10`) the
offline match run wins 5:1 (21 225 steps, valid, deterministic).

## Golden replays

All 7 legacy recordings of the golden-data store replay bit-exactly (state hash, scores, steps; also all 804
intermediate hashes of their `trace.json`, checked once during the migration). Tagged cases (the core's rules, not
this sport's, decide differently): `duel/nop-B` (no-show: A wins, as in the legacy service), `match/nop` (no-show: a
valid loss without a winner side; legacy winner B) and `match/abandon-60s` (a session that ended mid-game: a valid loss by
forfeit, `abandoned`, since core 0.2.6, as legacy; cores 0.2.2-0.2.5: incomplete).
