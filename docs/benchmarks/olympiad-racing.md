# Three-dimensional robot racing

Branch `olympiad-racing-v3` (on `olympiad-racing-fix`). Revision `racing-program-v3`. Three generated, parameterized
standard tasks:

- `racing-match-wallclock-program-standard`: a camera-driven agent against a scripted car, levels 1-4.
- `racing-duel-wallclock-program-standard`: two isolated agents in one physical world.
- `racing-race-wallclock-program-standard`: a multi-car race, one isolated agent per car (3-6 cars, default 4).

v3 replaces the stadium oval by seeded circuits, adds fixed obstacles and the multi-car race. Single player and duel
keep the v2 format.

## Rules

Rulebook: RoboRacer Rules v3.2026-draft (general rules of the 31st International RoboRacer, formerly F1TENTH,
Autonomous Racing Competition, <https://iros2026-race.roboracer.ai/rules.html>), Head-to-Head race, adapted. The
RoboRacer rules race at most two cars; the multi-car race takes its classification from the FIA 2026 Formula 1
Sporting Regulations (points per place, count-back). The deviations, each with the rule it changes, are `DEVIATIONS`
in `scripts/olympiad/racing/spec.py` and the Rules row of the READMEs. All three tasks use the same referee
(`race.py`); only the opponents differ (R3).

- **Format (single, duel).** Up to three heats of one lap; the first car to win two heats wins the race. Heat time
  limit: the lap length at 1.25 m/s, rounded up to 5 s (55-60 game s on the official circuits). Side-by-side start
  0.42 m either side of the centre line. Heat 1: `FIRST_PLAYER` outside; heat 2: sides switch; heat 3: the faster
  winning lap of heats 1-2 starts inside.
- **Format (multi-car race).** N one-lap heats. Staggered grid behind the start line (two columns, slots 0.9 m apart,
  pole on the inner side); heat h rotates the grid by one slot (heat 1: `FIRST_PLAYER` on pole, then the seats in
  order), so every car starts once from every slot. A heat ends when all running cars but one have finished, 10 game s
  after the first finisher (`FINISH_WINDOW_S`), or at the time limit. Heat order: finishers by interpolated finish
  time, the others by checkpoint-bounded progress, then who reached it first, then a draw fixed by the seed. Points
  per heat place 25-18-15-12-10-8 (F1's first six). Classification: points, then count-back (more first places, then
  more second places, ...), then the better place in the last heat; no ties (every heat order is strict). Retired cars
  (third warning, not started when the start window closes, missed review) are parked off the track, score no more
  points and are classified after the running cars (the later retirement first). The race ends after N heats or when
  fewer than two cars run.
- **Heat winner (single, duel).** The first car across the finish line after the 16 ordered checkpoints (crossing time
  interpolated inside the 10 ms step); the heat ends there. Without a finisher: more checkpoint-bounded progress, then
  who reached it first, then a coin flip fixed by the seed. No draws.
- **Car-car crashes.** Every touching pair is judged on its own. A contact episode is a crash when its contact impulse
  reaches car mass x 0.30 m/s, or after 2 s of contact; lighter touching is free. At fault is the car with the larger
  closing speed towards the other (margin 0.1 m/s), else the car behind on the track, else the seed's draw. The
  at-fault car is placed 2 m behind the other car on that car's line, at rest (further back in 0.5 m steps if the spot
  touches another car or an obstacle); the other car continues. Every at-fault crash is a warning; the third
  disqualifies (head-to-head: the other car wins the race; multi-car: retired).
- **Obstacles.** RoboRacer "Race penalties": upon crashing into an obstacle the car is stopped and moved to the side of
  the track next to its position before the crash. Here: an obstacle contact episode that passes the same impulse /
  2 s test places the car at rest 0.75 m before the obstacle, in its free gap, and holds it for 1 s; no warning.
  Touching below the test is free. Obstacles are static world geoms: they cannot move or be pushed into anyone's path.
- **Walls.** Continuous rigid walls; contact is free; the car must free itself (the lost time is the penalty).

## Circuit generator (`race.Circuit`, a pure function of the seed)

- Blocks: one left hairpin (170-190 degrees, radius 2.5-3.0 m), an S-bend (two opposite 50-80 degree turns), a
  chicane (+a, -2a, +a with a = 25-40 degrees), a right-hand sweeper (70 % of seeds) and two or three left turns that
  make the total turning 360 degrees. Each turn is a curvature plateau with cosine ramps (<= 1.2 m), so the centre line
  is C2; the smallest centre-line radius is 2.5 m (the car turns down to 0.46 / tan 0.48 = 0.89 m).
- The blocks are put in a seeded order with a straight before each (the first is the main straight, at least 9 m). The
  straight lengths are solved by non-negative least squares so that the loop closes exactly (heading and position);
  orders are tried until a solution with straights of at most 15 m (main) / 9 m (others) exists.
- Checks: length 45-70 m; bounding box at most 26 m; parts of the loop more than 6.5 m apart along it stay a corridor
  width + 1.2 m apart (no self-intersection, room for both walls). Width 2.6-2.8 m with a smooth variation of up to
  +-0.15 m. The dense centre line (0.02 m) starts at the start/finish line, 6 m into the main straight, and is centred
  on the origin. `project(p)` is the nearest dense sample refined by the orthogonal projection onto its segments.
- Walls: boxes along the two offset polylines (every 0.24 m), road tiles and the yellow stripe along the centre line,
  a white start/finish line.
- Obstacles: 2-4 magenta barrels (r 0.17 m) and blocks (0.5 x 0.4 m) on sections with radius >= 4.5 m (3.2 m if fewer
  than two fit), at least 7 m apart and 9 m from the start, beside a wall (gap 0.05-0.55 m), each leaving a free gap
  of at least 1.5 m. `Circuit.line(s, lane)` is a collision-free driving line through the free gaps (trusted drivers
  only: the scripted car and the tests); the tests check its clearance (>= car half width + 0.25 m from every
  obstacle, 0.4 m from the walls) and radius (> 1.2 m) on every official seed and 40 more.
- The map (centre line every ~0.5 m with the width, length, obstacles) is published in `/spec` as the rulebook's map.

## Observation and action

Front and chase RGB (240 x 160, 70 degrees; front 0.22 m ahead, 0.395 m high, 20 degrees down; chase 1.5 m behind,
1.505 m high, 35.4 degrees down; both in `/spec`), own forward speed, command, grid slot and side, obstacle hold.
Referee information: heats won or points, warnings, retirements, car-car crash calls; an obstacle crash call only to
its own car during the heat (it would locate another car), every call in the heat result; after each heat its result
(order, how each place was decided, checkpoints, finish times, obstacle crashes). No pose, track progress or running
order of any car is returned during a heat, and `/spec` has no seed. Action: target speed [-1.5, 4] m/s and steering
[-0.48, 0.48] rad, held until replaced, applied at the next 10 ms control step; out of range is HTTP 400, a second act
in flight 429, an act while frozen 409, an act of a retired car 409.

## Scripted opponent

Trusted (it knows the poses and the map; never part of the reference). It cruises at 1.75 / 2.2 / 2.7 / 3.2 m/s
(L1-L4) on its grid line, detours around the obstacles (`Circuit.line`), moves to the freer side to pass a car ahead in
its line and slows to keep its distance. L1 laps in 35.5-39.8 s on the official seeds, L4 in 19.7-22.0 s, never
touching a wall or an obstacle. Tests drive blocking, crawling, weaving, brake-checking, reversing and idle agents
against it at L1 and L4: it never gets a warning and always wins.

## Service, isolation, evidence

- Physics 2 ms, control 10 ms; the step loop runs in the service (0.5 ms per step with 4 cars), images in one render
  process per seat.
- Each seat: its own internal network, setup-only egress, metering proxy (ports 18090 + seat index) and render process;
  identity by source address. The race task declares seats A-F; `RACE_CARS` of them race, the others get Harbor's
  `nop` agent from `run_race.sh` and no car, proxy or renderer. The metering files (`think_proxy.py`, `think_clock.py`)
  are table tennis's, copied by the generator: the single-seat copy in the single task, the per-seat copy in the duel
  and the race.
- `trajectory.json` records the applied commands with their steps, the review transitions and (multi-car) the
  service's retirements. The sim replays it in a fresh process; the separate verifier replays it twice more with its
  own physics/referee copy and requires the final state hash, heat results, crash calls, scores, classification and
  winner to match. `source_sha` covers every `.py` of `environment/sim` and must equal the grader's `SIM_SOURCE_SHA`
  (single / duel / race; asserted by `spec.py`).
- `reward.json`: `success`, `valid_trial`, `technical_failure`, `deterministic`, `official_config`, `level`, `seed`,
  `slowdown`, `cars`, `timing_invalid`, `<seat>_won` (every seat), forfeits, heats won, warnings, laps, checkpoints,
  crashes, obstacle crashes, per-seat `transient_api_errors`/`unrecovered_api_errors`/`proxy_error`/`refused_busy`/
  metered tokens, `sim_lag_*`; multi-car also `<seat>_position`, `<seat>_points`, `heat<h>_<seat>_position` and
  `winner_seat_index`. `classification.json` (multi-car, valid races) holds the full classification and every heat's
  finishing order and grid: what a Plackett-Luce rating needs. A single-player session that ends mid-race is a valid
  loss; a duel or race cut off is a technical failure.

## Broadcast

`broadcast=True`: `/data/broadcast.jsonl` (never read by the grader).

Header (v2 fields kept): sport `racing`, `mode` (`single`, `duel` or **`race`**), slowdown, dt 0.01, `nq`,
`qpos_layout` (qpos index of every car's free joint: A, B and, in a race, C...), `players`, seed, level,
`first_player`, and new: **`cars`**, **`heat_seconds`**, **`max_heats`**. `track` keeps `straight_half_m`,
`radius_m` (now oval-equivalent extents: `straight_half_m + radius_m` = half width of the bounding box, `radius_m` =
half height, the circuit centred on the origin), `width_m` (now the mean width) and `gates`, and adds `kind:
"circuit"`, `length_m`, `width_range_m` ([min, max]), `extent_m` (half width / half height), `min_radius_m`,
`features` (kind, s0, s1 of each block), `centreline` ([x, y, width] every ~0.5 m from the start line in driving
order) and `obstacles` (id, kind, x, y, yaw).

Frames every 4 steps (and at every event): full `qpos` plus `heat`. Events (v2): `start`, `run` (with the grid: side
per seat, or slot per seat in a race), `checkpoint` (seat, gate, t), `contact` (now with `cars`: the pair), `crash`
(at fault, `other`, rule, impulse, warning), `disqualified`, `heat_end` (public result; a race adds order, positions,
heat points, points, decided_by_car, retired), `pause`, `review` (why/plan, truncated), `end` (a race adds the
classification). New: `obstacle_contact` (seat, obstacle), `obstacle_crash` (seat, obstacle, kind, impulse),
`retired` (seat, why: disqualified / did_not_start / review_forfeit) and `positions` (the live running order, every
game second; spectators only). Score fields on every event: `heats_won` (single, duel) or `points` (race).

What the broadcast adapter (`scripts/olympiad/broadcast/adapters.py`, `Racing`; and `regen_rt.py`) must change:

1. Build the scene with `race.scene_xml(seed, cars)` (`cars` from the header; `scene_xml(seed)` is still the 2-car
   scene). The scene is larger (up to about 26 x 13 m, centred on the origin); `race.track(seed)[:2]` still frames the
   `high` camera (oval-equivalent extents), but the header's `track.extent_m` is the exact half size.
2. Read every car's free joint from `qpos_layout` (2 to 6 entries) instead of a fixed A/B pair; colours A blue, B
   orange, C cyan, D purple, E brown, F grey (`race.CAR_COLORS`); obstacles are magenta.
3. Map `mode == "race"` to the Racing adapter (the table has `("racing", "single")` and `("racing", "duel")` only) and
   draw points / places instead of heats won when `points` is present; a race has `max_heats` = N heats.
4. `regen_rt.py`: `race.Race(**cfg)` takes the new config key `cars` (always present in v3 logs); its header should
   add the new fields above; the v2 `track` fields still exist.
5. Optional overlays: obstacle crashes, retirements and the `positions` running order.

## Sources

| Component | Source |
| --- | --- |
| Physics, circuit, obstacles, referee, scripted car | `scripts/olympiad/shared/racing/sim/race.py` |
| Service, clock, identity, broadcast | `scripts/olympiad/shared/racing/sim/server.py` |
| Replay / fingerprint / renderer / launcher | `replay.py`, `server_fingerprint.py`, `render.py`, `launch.py` (same directory) |
| Prompt sections, registration, rules | `scripts/olympiad/racing/racing_prompt.py`, `spec.py` |
| Task sources | `scripts/olympiad/racing/match/`, `duel/`, `race/` (no copies of sim, referee or metering files) |
| Tests | `scripts/olympiad/racing/test_racing.py` |
| Run scripts | `run_single.sh`, `run_duel.sh`, `run_race.sh`; `run_round.py` kind `race` (`--race-players`) |

```bash
python3 scripts/olympiad/racing/test_racing.py        # MuJoCo 3.3.7
python3 scripts/olympiad/generate.py check
python3 scripts/olympiad/test_program_play.py tasks/racing-race-wallclock-program-standard
scripts/olympiad/run_race.sh --players luna,luna,luna,luna --slowdown 10 --seed 1 --tag racev3
```

When the sim code changes, `generate.py` stops with the new `SIM_SOURCE_SHA` to put into the three graders.

## Open points

- The shared duel prompt says "After one player runs `game start`, the other has ... to start" and "you forfeit the
  race" (`program_prompt.py`); in the multi-car race "the others" and "your car is retired" would be exact. A neutral
  wording needs a shared change.
- The shared prompt text says "predicts its flight"; racing replies carry all finished heats' results.
- `realtime_standard.readme` hard-codes "Simulator / robot: mujoco / franka-panda"; `normalize` fixes the header from
  task.toml, but the shared default should change.
- Blocking that stays below the crash threshold is not judged; the rulebook leaves "abnormal change of direction" to
  referees.
- Multi-car ratings: the grader records the full orders; a Plackett-Luce fit over seasons is still to be written.
