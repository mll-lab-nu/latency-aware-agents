# bowling: ten-pin bowling

Ten-pin bowling on a regulation MuJoCo lane (oiled front end, dry back end, 10 rigid pins, a 7 kg ball), migrated
from the Season-1 task `bowling-program-standard` to the v0.2 framework (docs/framework_zh.md §10, stage 1). Rules:
USBC Playing Rules 2025-2026; the deviations, the rule constants and the exploit list are in `sport.toml`. One format:
the single game. The legacy duel (`bowling-duel-program-standard`) was retired on 2026-10-08 (user decision, the
criterion of review item B10 for archery and basketball): both players bowled identical lanes without any interaction,
so a head-to-head comparison is the two players' single scores on the same seeds.

- **Single**: one 10-frame game; success = a complete game of at least 200 (the old task's line).
- **Action** (one level, `low`): `{"type": "roll", board, angle_deg, speed, rpm, axis_deg}` with seeded Gaussian
  execution scatter. **Observation**: the decision point (frame, ball, pins standing, the last ball's pin count),
  the scorecard, and one PNG of four views (approach, pin deck now, pin deck after the last ball, the lane from
  above with the last ball's path). No ball, pin or oil numbers, no seed.

| File | Part of contract ② | What it is |
|---|---|---|
| `sport.toml` | declarations | rule book and deviations, formats, modes, timing, `game roll`, observation whitelist, rule constants, exploits, players, official seeds, tasks, capabilities |
| `sim/bowling.py`, `sim/sport.py` | world, referee, cameras | the Season-1 simulator and rules module, **byte-identical** to `scripts/olympiad/shared/bowling/sim/` at 39bea8a1 except one line: `sim/sport.py` `LIMITS` (and its `/spec` text) caps the release speed to 4.47-10.73 m/s since bowling-4 (the rules module's duel code is no longer used) |
| `turnshot.py` | world, actions, referee, observation | the generic adapter of the shared turn-based server (below) |
| `plugin.py` | the sport | `Bowling(TurnShotMatch)`: the action schema and a renderer cache; `BowlingSport`: build, the action layer, `/spec`, players |
| `builtin_players.py` | built-in players | `Reference` (camera-only: the path dots and the standing pins in the lane view), `Random` |
| `grade.py` | reward fields | the Season-1 single task's reward keys (`total`, `strikes`, `pins_f1..10`, ...) |
| `tests/test_bowling.py` | exploit list | one test per `[[exploits]]` entry |
| `golden/` | golden replays | recorded single games (`single-s1`, `single-random-s3`); `convert.py`: the Season-1 single records (golden-data store) -> core episodes |

Check: `PYTHON=python3.12 RO_GOLDEN_DATA=<store> tools/check bowling`. Generate: `python3 tools/generate.py bowling`.

**Golden replays.** The 4 Season-1 single records (nop, clock stall, oracle on 20260930 and on seed 7) replay through
the v0.2 core and this plugin to the legacy scores, winner, number of balls and state hash (the 4 legacy duel records
were retired with the duel). The hash is `sha_json({seat: legacy game hash})`, so it
matches only if every seat's lane state after every ball (the hash chain of `sim/bowling.py`) is bit-identical.

**Changes from Season 1** (the single game's play is unchanged): **no duel** (retired on 2026-10-08, see above: the
format, deviation D4, the roll-off and clock tiebreaks, the fairness check, exploits E6 and E7, the reference's role
jitter, the duel golden cases and the converter's duel records are gone; the exploit ids E1-E5 are kept; revision
bowling-3); the core replaces the legacy service; the seed is no longer in `/spec` (rule O3); the observation
is stateless (both scorecards and the last ball instead of "shots since your last call"); the default seed is 1 (a
drawn lane; 20260930 is the standard lane and stays an official seed); the single task is generated as
`bowling-single-program-standard` (the generator's naming).

**Release machine (bowling-4, deviation D5).** The ball is delivered by a declared release machine like USBC's
[E.A.R.L.](https://bowl.com/equipment-specifications2024/e-a-r-l-the-robot) (Enhanced Automated Robotic Launcher):
"ball speeds anywhere from 10-24 miles per hour and rev rates anywhere from 50-900 rpm". The release speed is capped
to that range, 4.47-10.73 m/s (until bowling-3: 3-11 m/s); the rev rate stays 0-700 rpm (below E.A.R.L.'s 50 rpm is
kept as an allowance, declared in D5). No recorded ball used a speed outside the new range (the reference rolls
8.0-9.0 m/s, the random player 5-10 m/s, the Season-1 records 8.0-9.0 m/s), so the golden cases and the converted
Season-1 records replay unchanged. B9 (real-machine feasibility) on bowling-4: see the pull request.

**The reference player** sees only the images and `/spec`: it reads the yellow dots of its previous ball's path in
the lane view (pixels -> lane with the panel cameras of `/spec`), estimates the launch angle that was executed and
where the ball reached the head pin, and corrects its strike line's launch angle towards the 1-3 pocket; for a spare
it finds the standing pins (white silhouettes in the last lane panel) and rolls a straight ball at the front pin,
cross-lane for the outside pins. About 200 on average over seeds 1-8 (212 on the default seed 1).

## Pattern for the shared turn-based server

Season 1 ran bowling, mini-golf putting and disc golf on one ~650-line turn service,
`scripts/olympiad/shared/<sport>/sim/server.py` (byte-identical for bowling and golf; disc golf hides the seed in
`spec`), plus a per-sport **rules module** `sport.py` with one fixed interface: `NAME`, `LOWER_WINS`, `ACTION_KEYS`,
`TILES`, `IMAGE_SIZE`, `IMAGE_LAYOUT`, `TIEBREAK_MAX`, `validate(body)`, `decide(scores)`, `spec(duel)`, and
`Match(seats, order)` with `turn`, `shoot`, `done`, `winner`, `start_tiebreak`, `n_tiebreaks`, `decision`, `views`,
`summary`, `score`, `state_sha`. On v0.2 the core replaces `server.py`, and **`turnshot.py` maps any such rules
module onto contract ②** once. `turnshot.py` still supports a two-seat format (each seat on its own copy of the
world, the tiebreaks below), but bowling, golf putting and disc golf now have the single format only (their duels had
no interaction between the players and were retired on 2026-10-08); the duel parts below apply to a new sport whose
players do interact:

| legacy `server.py` | v0.2 |
|---|---|
| the seed read from `TURN_SEED` at import | `load_rules(sim_dir, seed)`: a fresh copy of the `sim/` modules per seed (environment and `sys.modules` restored), cached |
| `M.turn(ORDER)`; a review after each shot | `to_move()`; the core's review after each action |
| `validate(body)`; both chess clocks with every shot | `ShotLayer`: the plugin's schema (strict), then `validate()` (its rounding); both clocks (ms) go into the low-level command, so the replay sees them |
| `M.shoot` + start the tie-break when level | `apply()` |
| every game done -> `match_over` | `phase()` |
| `decision()`: forfeit -> `decide` -> clock | the core's forfeit rule, then `tiebreaks()`: `total` (by `LOWER_WINS`), `rulebook_tiebreak` (`M.winner()`), `clock_left` (the clocks at the last shot; equal -> the second player) |
| `render_views` (tiles -> one PNG) | `camera_job()` draws the PNG (`views(seat)`, overridable), `Sport.worker_call` passes it through |
| `final.json` fields | `final_fields()` with the legacy names (`scores` per seat stays under `sport_fields`, which live_board reads first) |
| `final_sha` per seat | `state_digest()` = `sha_json({seat: M.state_sha(seat)})` |

**To migrate golf putting or disc golf**, copy this directory's shape:
1. `sports/<sport>/sim/`: `git mv` the simulator and `sport.py` from `scripts/olympiad/shared/<sport>/sim/`
   **unchanged** (disc golf: `discgolf.py`, `flight.py`, `sport.py`); not `server.py`.
2. `turnshot.py`: copy byte-identical (a sport never imports another sport; keep the copies equal, like the legacy
   `server.py` copies).
3. `plugin.py` (~60 lines, copy bowling's): `class <Sport>(TurnShotMatch)` with `schema(seat)` (the legacy
   `validate()` limits, `"digits": 4` if it rounds to 4 decimals) and, if useful, a `views()` override (bowling caches
   one renderer per process and the lane panels that cannot change); `Sport.build = <Sport>(cfg, load_rules(SIM,
   cfg.seed))`, `action_layer = ShotLayer(lambda m: m.S)`, `worker_call` returns the job, `spec()` = `S.spec(duel)`
   plus the image tiles (and in a duel the first player), **without the seed** (rule O3); a single-only sport calls
   `S.spec(False)`.
4. `sport.toml`: a duel needs `[tiebreaks] order = ["total", "rulebook_tiebreak", "clock_left"]` (turnshot's names);
   `[observation] fields = ["decision", "game_over", "score", "last_shot"]` (+ `"first_player"` in a duel); `[modes] action_levels =
   ["low"]`; the single format is solo (`success = { score_at_most = <par line> }` for golf: lower wins);
   `[capabilities] physics` from the current vocabulary: `ballistics_wind` is now split into `ballistics` and `wind`
   (disc golf: `["ballistics", "wind", "drag"]`). A duel's turn-based fairness is strict (a 95% CI wider than +-5 pp
   fails): `[fairness] matches_per_order = 400`.
5. `golden/convert.py`: copy bowling's; change `TASKS` and `KEYS` (= the rules module's `ACTION_KEYS`). The expected
   state hash is `sha_json(manifest final_sha)`. Forfeits: `clock_out` -> `clock`, `start_timeout` -> `no_show` (core 0.2.2).
6. `grade.py`: the Season-1 grader's keys from `final.json` (`sport_fields.scores`, `seats`, `clock_used_s`) and the
   metering files.
7. A camera-only reference in `builtin_players.py`. Do not offer a duel in which the players do not interact (two
   identical worlds side by side: the head-to-head is the single scores on the same seeds); a duel must also give
   each player its own execution-scatter stream (an n-th shot that scatters alike for both players lets the one who
   shoots later pre-compensate).
