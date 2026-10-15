# hello: the example sport

The smallest complete sport of the v0.2 framework (docs/framework_zh.md §4.2, §8) and the core's test fixture.
Copy this directory to start a new sport (CONTRIBUTING.md, "Add a sport").

**Rules** (this file is hello's rule book; the constants are in `sport.toml [spec]`). A target is painted on the ground
4-6 m ahead of a fixed release point (the seed places it). Each ball is a throw (`yaw`, `pitch`, `speed`, with seeded
execution scatter); it flies, lands and rolls to rest. Points by the ring the ball rests in: 10 / 8 / 6 / 4 / 2 for
radii 0.10 / 0.25 / 0.45 / 0.70 / 1.00 m, else 0. Single: 5 balls, success = at least 30 points. Duel: 5 balls each,
alternating (the first side throws yellow, the other blue); the higher total wins; if level, sudden-death pairs (up to
3), then the side whose best ball is closest to the centre, then (last resort) the side that threw second.

| File | Part of contract ② | What it is |
|---|---|---|
| `sport.toml` | declarations | rule book and deviations, formats, modes, timing, `game` shortcuts, observation whitelist and renderer, rule constants, tiebreaks, exploit list, players, fairness, golden, tasks |
| `plugin.py` (about 215 lines) | world, actions, referee, observation | `Hello` (MatchBase): `state_digest`, `schema`/`apply`, `phase`/`scores`/`tiebreaks`/`to_move`, `observe`/`cameras`/`render_state`, the optional `reply_views` (the reply to a throw shows the ball at rest); `AimLayer` (the high level: `lower` an aim point to a throw); `HelloSport.scene()` for the renderers |
| `builtin_players.py` | built-in players | `Reference` (camera-only: finds the target disc and its own balls in the overhead image, corrects by its last errors), `Random` (check policy) |
| `grade.py` | reward fields | `evaluate(final)` -> numbers (the common fields come from core/grading.py) |
| `skill_api.py` | the two modes | the skill API `aim(x, y)`: its contract (skilleval/contract.py); the high level's act schema is built from it |
| `seat/toolkit.toml`, `seat/lib/flight.py` | the two modes | the low level's toolkit manifest and its one library (the launcher's flight model, forward only) |
| `lowlevel/` | the two modes | `evaluator.py` (the post-match evaluator's hooks; `make-cases`), `reference/aim.py` and `boundary/aim.py` (rules 4, 5), `cases/` (hidden + public requests), `reference_results.json` |
| `tests/test_hello.py` | exploit list | one test per `[[exploits]]` entry |
| `golden/` | golden replays | recorded matches + expected results (`tools/golden.py record`) |

Run everything: `PYTHON=python3.11 tools/check hello`. Generate its tasks: `python3 tools/generate.py hello`
(into `build/tasks/`, since `publish = false`: the example is not part of the benchmark).

## The two modes (the minimal example: docs/framework_zh.md §6.3, CONTRIBUTING.md "如何给一项运动加两档")

| | high: Agent as Athlete | low: Agent as Robotics Engineer |
|---|---|---|
| the skill `aim(x, y)` | the platform's (`plugin.AimLayer`: the pitch-35 throw that lands at the point) | the player's own, from the toolkit: `/app/output/skill/aim.py` with `aim(x, y, *, state, spec) -> {type: throw, yaw, pitch, speed}` |
| what the player gets | `{type: aim, x, y}` acts; `skill_contract` in /spec | `{type: throw, ...}` acts, `/opt/ro/toolkit.toml` (the throw interface, `/opt/ro/lib/flight.py`), 60 public practice requests at `/opt/ro/practice.jsonl` |
| evaluation | the match | the match, then (`tools/skill_eval.py hello <skill dir>`) the API test on 140 hidden requests (landing within 3 cm) and the competition test (our reference player, its aims lowered by the player's aim, 8 single matches) |

Reference rows (`lowlevel/reference_results.json`, 2026-10-08): our implementation 140/140 and 8/8; the reference
skill from the toolkit only 140/140 and 8/8; the boundary baseline (yaw at the point, the nominal 7 m/s) 1/140 and
1/8. Run them: `python3 tools/skill_eval.py hello platform|reference|boundary` (or `tools/check hello --stages 6`).
