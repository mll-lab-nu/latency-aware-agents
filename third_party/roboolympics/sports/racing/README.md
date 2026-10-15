# racing: robot racing (RoboRacer)

1:10-scale robot racing on the v0.2 framework (docs/framework_zh.md §4.2, §10 stage 3), migrated from the legacy tree
(`scripts/olympiad/racing/`, `scripts/olympiad/shared/racing/sim/`). The simulator and referee `race.py` is the legacy
file (git history follows it; only its fingerprint helper, which read the deleted `replay.py`, and one docstring
changed), so every legacy recording replays bit-exactly (`golden/convert.py`).

**Rules**: RoboRacer Rules v3.2026-draft (the 31st International RoboRacer Autonomous Racing Competition), with the
deviations D1-D10 of `sport.toml`; the multi-car race takes its points and count-back from the FIA 2026 Formula 1
Sporting Regulations. Identical MuJoCo cars (rigid chassis, four rolling wheels, wheel-speed and steering servos with
finite force) race one-lap heats on a seeded walled circuit with 2-4 fixed obstacles; the player commands the servo
targets (`drive`: speed, steering) and sees only its car's cameras, its own car and the referee's calls.

| Format (task) | Seats | Unit | Result |
|---|---|---|---|
| `match` (`racing-match-wallclock-program-standard`) | A against the scripted car (side `opponent`), level 1-4 (`RCB_LEVEL`) | a heat | first to two heat wins (or the third warning loses); success = A wins |
| `duel` (`racing-duel-wallclock-program-standard`) | A, B | a heat | as the match; a valid duel always has a winner (`<seat>_won`) |
| `race` (`racing-race-wallclock-program-standard`) | A, B, C, D | a heat (one per car, rotating grid) | F1 points 25-18-15-12, count-back, the seed's draw; full classification (`<seat>_place`, `Match.ranking`); a car whose player never starts, misses a review or is disqualified is retired (`Match.retire`) and the others race on |

`[[tasks]] id` keeps the Season-1 task names.

| File | Part of contract ② | What it is |
|---|---|---|
| `sport.toml` | declarations | rulebook and deviations D1-D10, formats, timing, `game drive` shortcut, observation whitelist, rules, tiebreaks, exploits E1-E9, players, fairness, official set, tasks, spectator camera, capabilities |
| `race.py` | world + referee (legacy) | the seeded circuit and obstacles, the MJCF scene with 2-6 cars, `Race`: control step, gates, crash and obstacle rules, heats, the scripted car, the multi-car classification and retirements, `digest()` |
| `plugin.py` | world, actions, referee, observation | `RacingMatch` (MatchBase over one `race.Race`): `step`, `state_digest`, `apply`, `phase` / `resume` / `scores` / `tiebreaks` / `ranking` / `retire`, `observe` / `cameras` / `render_state`; `DriveLayer`; `Racing.scene()` (race.scene_xml) and `spec()` (the legacy /spec: circuit map, action limits, camera mounts, referee constants) |
| `builtin_players.py` | built-in players | `Reference`: the legacy camera-only lane follower (`solution/camera_player.py`), unchanged except that it integrates its travel over game time; `Random` |
| `grade.py` | reward fields | the legacy reward keys that final.json still carries (differences below) |
| `tests/test_racing.py` | exploit list | E1-E9 (from the legacy unit tests) |
| `golden/` | golden replays | the 11 legacy recordings of the golden-data store, converted (`convert.py`; 4 tagged, see below) |

Run everything: `PYTHON=python3.12 tools/check racing` (stage 3 with `RO_GOLDEN_DATA=<store>` also converts the store
again). Generate the tasks: `python3 tools/generate.py racing`.

## What changed from the legacy tasks

- **Service**: the core's realtime protocol replaces `server.py` (its pid-before-SIGUSR1 race is gone with it): `/<seat>/<call>`
  instead of `/<call>` with an `X-Seat` header; reviews, start window and retirements are the core's.
- **Action**: `{"type": "drive", "speed", "steering"}` (a body without `type` is still a drive, so the legacy
  `{"speed", "steering"}` works) or `{"type": "none"}`, checked by the core's schema (numbers rounded to 6 digits);
  `game drive S D` replaces `game act S D`.
- **Observation**: the camera is an observation parameter: `POST /observe {"query": {"camera": "chase"}}` (legacy
  `{"camera": "chase"}`), default front; one PNG tile drawn by the task's renderer from the camera's world pose
  (`Match.cameras`) instead of the scene's named camera. The state fields are the legacy `status` fields under the
  same names (`own`, `race`, `crashes_this_heat`, `results`, `classification`), except the heat's game time
  (`heat_time_s`; the core's `t_sim` is the whole match's) and `race_over` / `race_winner` (the core has `done`).
- **Match format**: the scripted car is the side `opponent` in the core's scores and outcome (car `B` in race.py, the
  observation and the reward keys); the default seed is 1 (legacy: 101); a review now has the core's limit
  (`READY_TIMEOUT_S`, 300 s; legacy: none) and a missed review is a valid loss.
- **Start and collection** (core 0.2.2): a duel seat that never starts loses (`no_show`, as the legacy
  `start_forfeit`); a race car that never starts is retired (as legacy); a single player who never starts loses when
  the start window closes (counted from the service start) or at the collect, whichever comes first (core 0.2.7;
  legacy: whenever the session was collected; cores 0.2.2-0.2.6: only once the window had closed); a
  single session collected mid-match is a valid loss by forfeit since core 0.2.6 (`abandoned`, success 0; cores
  0.2.2-0.2.5: `collected`, an incomplete match), as the legacy `session_ended`. With four cars and only one starter the core retires the absent cars one by one and the last
  absent one forfeits (legacy: the race ended at once): same winner, different state.
- **Slowdown**: `RCB_SLOWDOWN`, default x20 for the match and the duel (`[timing] slowdown`) and x10 for the race
  (`[[tasks]] slowdown`), as the legacy tasks.
- **Fairness rules** (found by the fairness check with identical reference drivers): the inner side of heat 3 of a
  head-to-head race goes to the seed's draw (legacy: the faster winning lap of heats 1-2, which between cars of equal
  pace depended on the control timing and always went to the car that started inside in heat 1; the draw is A for
  seed 1, as in both legacy duel records); the race classification after count-back is the seed's draw (legacy: the
  better place in the last heat, which with cars of equal pace was the last heat's pole car, a slot fixed by
  FIRST_PLAYER). Only the classification of `oooo-s1-A` changes among the legacy records (tagged).
- **Renderer threads**: plugin.py caps Mesa llvmpipe at 4 threads per GL context (`LP_NUM_THREADS`, unless set):
  with 2 x nproc threads per render worker the race's four workers exceeded the sim container's pids_limit (256)
  under Harbor and the fresh-process replay could not fork. The images are byte-identical.
- **Race size**: the race format declares 4 cars (seats A-D, the legacy default and official size); race.py still
  supports 3-6 (a format per size could be added); the legacy task declared six seats and left the unused ones idle.

### Reward keys

Kept: `replay_error`, `meter_error`, `proxy_error`, `timing_invalid`, `official_config` (legacy sets, plus seeds 1-3
for the match), `level`, `seed`, `slowdown`, `first_player_A`, `n_actions`, `steps`, `sim_steps`, `heats`, `crashes`,
`cars`, `obstacle_crashes`, `won`, `<car>_won`, `race_over`, `disqualification`, `ready_forfeit`, `start_forfeit`,
`session_ended` (= reason `collected` or `abandoned`), `video_rendered`, `car_contact_steps`, `transient_api_errors`,
`unrecovered_api_errors`, `sim_lag_late_max_s`, `sim_lag_late_p99_s`, `sim_effective_slowdown`, and per car
`<car>_{heats_won,warnings,checkpoints,laps,obstacle_crashes}` (race: also `points`, `position`, `retired`),
`<car>_wall_contact_steps`, `<car>_dropped_acts`, `<car>_*_api_errors`, `<car>_proxy_error`, `<car>_meter_error`,
`<car>_metered_{requests,input_tokens,output_tokens}`, `<car>_n_observe`, `<car>_n_act`; race: `winner_seat_index`,
`heat<h>_<car>_position`.

Dropped (final.json no longer has the value): `source_mismatch` (the core's replay check covers the source and is in
`deterministic` / `replay_error`), `sim_lag_late_mean_s` and `sim_lag_frac_late_over_1_period` (the core reports max
and p99 only), `<car>_refused_busy` (the core answers a second act in flight with 429 without counting it),
`<car>_n_wait` (not counted). The legacy verifier's `classification.json` is gone: the classification and every heat's
order are in final.json (`classification`, `results`), which the verifier copies to its logs.

Added by the core: `match_over`, `no_show`, `<seat>_no_show`, `solo`, `direct_play`, `lowlevel_actions`, and in the
race `<seat>_place` (= `<seat>_position` of a valid race).

## Golden records

`golden/` holds the 11 legacy recordings (commit d4664a8b of the golden-data store), converted 1:1 (`act` -> `cmd`,
`ready` -> `resume`, `retire` -> `retire` with the core's reason; legacy forfeits from final.json). All 11 replay to
the legacy state hash, scores and step count. Four carry a `tag` in expected.json because the core decides their
outcome by its own rules: duel `nop-B` (no-show: A wins, as the legacy start forfeit), match `nop` (no-show of a
single player, collected inside the start window: no winner, a valid loss at the collect since core 0.2.7), race
`nop-D` (no-show with four cars: D is
retired, as legacy) and match `abandon-60s` (collected mid-race: a valid loss by forfeit, `abandoned`, since core 0.2.6, as legacy;
cores 0.2.2-0.2.5: incomplete).

## Fairness

Built-in reference cars on every seat, both first orders of the duel and all four pole seats of the race, 20 seeds
each (`[fairness] matches_per_order = 20`; core.testkit's jobs, run in two shards and combined with its
`advantage()`, 2026-10-06):

| Format | N | first side (FIRST_PLAYER) | seat A | wins |
|---|---|---|---|---|
| duel | 40 | 0.0 pp, 95% CI ±15.5 | +10.0 pp, ±15.2 | A 24, B 16 |
| race (fair rate 1/4) | 80 | -2.5 pp, ±9.2 | +1.3 pp, ±9.6 | A 21, B 12, C 24, D 23 |

No interval lies beyond ±5 pp (no failure); both are wider than ±5 pp (a realtime WARN: a reference duel costs about
65 s of CPU, a race about 200 s). Before the two tie rules above the duel's first side was at -50 pp (the heat-1
inner car always got the heat-3 inner side) and the race's heat-1 pole car never won.
