# drone: drone racing (FAI F9U)

Drone racing on the v0.2 framework (docs/framework_zh.md §4.2, §8), migrated in stage 2 from the legacy tree
(`scripts/olympiad/drone/`, `scripts/olympiad/shared/drone/sim/`). The simulator `race.py` is the legacy file
(git history follows it; only one docstring reference changed), so every legacy recording replays bit-exactly
(`golden/convert.py`).

**Rules**: FAI Sporting Code Section 4, Volume F9 Drone Sports, 2026, class F9U, with the deviations D1-D10 of
`sport.toml`. An original 0.35 kg quadrotor (CPU MuJoCo, 500 Hz physics, 10 ms control) flies through a seeded course
of numbered gates; the player commands a world-frame velocity and yaw rate (`fly`) through the onboard attitude
controller and sees only its FPV camera and its own telemetry. Level 1: 6 gates; level 2: 8 gates, wind, two moving
shutters; level 3: 10 gates, stronger wind, four shutters, two floor-to-ceiling panels.

| Format (task) | Seats | Unit | Result |
|---|---|---|---|
| `racing` (`drone-racing-wallclock-program-standard`) | A, alone, level 1-3 (`RCB_LEVEL`) | a qualifying round (3) | rounds finished (success: >= 1 in a complete event); ranked by `best_time_s`, then 2nd, 3rd, `best_gates` |
| `duel` (`drone-duel-wallclock-program-standard`) | A, B, level 2 | a heat (both fly at once, each in its own world) | first to 2 heat wins; at most 5 heats, then the tiebreak chain (no draws) |

`[[tasks]] id` keeps the legacy task names.

| File | Part of contract ② | What it is |
|---|---|---|
| `sport.toml` | declarations | rulebook and deviations, formats, timing, `game fly` shortcut, observation whitelist, rules, tiebreaks, exploits, players, fairness, official set, tasks, capabilities |
| `race.py` | the simulator | unchanged legacy `drone-race/4.0`: `Race` (one drone's world and judge), `Heat` (one world per seat on one clock), `Event` (rounds / heats and the duel's tiebreak chain) |
| `plugin.py` | world, actions, referee, observation | `Drone` (MatchBase over one `race.Event`): `step`, `state_digest`, `apply`, `phase` / `resume` / `scores` / `tiebreaks`, `observe` / `cameras` / `render_state`; `FlyLayer`; `DroneSport.scene()` (the course without any drone) and `spec()` (race.py's public geometry) |
| `builtin_players.py` | built-in players | `Reference`: the legacy camera-only vision pilot, ported from OpenCV / SciPy to numpy; `Random` |
| `grade.py` | reward fields | the legacy reward keys that final.json still carries |
| `tests/test_drone.py` | exploit list | E1-E8 (from the legacy unit tests) |
| `golden/` | golden replays | the 7 legacy recordings of the golden-data store, converted (`convert.py`; nop-B tagged: the no-show rule) |

The players' images: the FPV camera's world pose at the observation (`Match.cameras`) drawn by the task's renderer
from the course scene (race.py's, with its shadows and floor reflection), which holds no drone, so neither seat
ever sees the other's drone. The spectators' broadcast log keeps the legacy frames (`qA`, `qB`,
`m`, `h`); the replay video shows the course and the shutters only (the scene has no drone).

Run everything: `PYTHON=python3.11 tools/check drone` (stage 3 with `RO_GOLDEN_DATA` also replays the store). Generate
the tasks: `python3 tools/generate.py drone`.
