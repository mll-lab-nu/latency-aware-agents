# curling

Curling on the v0.2 framework (docs/framework_zh.md §4.2, §10 stage 1), migrated from the legacy tree
(`scripts/olympiad/curling/`, `scripts/olympiad/shared/curling/`). Two tasks, same names as in Season 1:
`curling-match-program-standard` (format `match`: seat A against the scripted red team, levels 1-2) and
`curling-duel-program-standard` (format `duel`: seat A yellow against seat B red). Rules: World Curling, The Rules of
Curling (July 2025) with the deviations of `sport.toml` (4 ends of 5 stones, at most 3 extra ends, no sweeping, ...).

**The thrower is a declared delivery machine** (deviation D7, 2026-10-09; no change of play, no new revision): like
Sone and Kawamura's pneumatic stone delivery robot ([J. Robot. Mechatron. 36(4) 2024](https://doi.org/10.20965/jrm.2024.p0864),
2.0-4.0 m/s) and Curly ([Science Robotics 2020](https://doi.org/10.1126/scirobotics.abb9764)). The release speed
0.5-4.0 m/s tops out at the Sone machine's 4.0 m/s; below about 2.2 m/s a stone stops short of the far hog line and
is removed from play, so every throw that stays in play is within the machine's range. The spin and the execution
scatter have no robot source (estimates). B9 (real-machine feasibility, research branch b9-release): draws into the
house 200/200 and takeouts of guarded stones 200/200; every successful throw used 2.38-3.61 m/s.

| File | Part of contract ② | What it is |
|---|---|---|
| `sport.toml` | declarations | rulebook and deviations D1-D7, formats, modes, timing, observation whitelist, rule constants, tiebreaks, exploits E1-E6, players, fairness, official configurations, tasks, capabilities |
| `plugin.py` | world, actions, referee, observation | `CurlingMatch` (MatchBase) around the legacy `curling.Match` / `duel_core.Duel`; `ThrowLayer` (the one action level); `Curling.scene()` (render.py's MuJoCo sheet) |
| `curling.py` | world (legacy; the noise streams since curling-2) | FCV1 ice model, impulse collisions, rules (in play, free guard zone, scoring, hammer), the scripted team, `Match` |
| `duel_core.py` | world (legacy; the noise streams since curling-2) | `Duel(curling.Match)`: two seats, each player's own execution-noise stream (the first player's 1, the other's 2) |
| `render.py` | scene (legacy, unchanged) | the MJCF sheet and stones (mocap bodies) and the two camera poses; its own `Renderer` / video code is no longer used (the core's renderer and spectator video draw from `scene()`) |
| `builtin_players.py` | built-in players | `Reference` (the Season-1 scripted player; perception rewritten without SciPy, which the sim image lacks), `Random` |
| `grade.py` | reward fields | the legacy reward keys of both tasks |
| `tests/test_curling.py` | exploit list | one test per `[[exploits]]` entry |
| `golden/` | golden replays | recorded v0.2 matches + `convert.py` (the legacy records of the golden-data store) |

Run: `PYTHON=python3.12 tools/check curling` (all stages); `RO_GOLDEN_DATA=<store> tools/check curling --stages 3`
for the legacy records. Generate: `python3 tools/generate.py curling` (publish = true: `tasks/curling-*`).

## What changed from the legacy tasks

- **Independent noise streams** (revision curling-2, user decision of 2026-10-08): the n-th stone of a team in an
  end draws its execution noise from the team's own stream, `default_rng([seed, end, stream, n])` (match: the player
  1, the scripted team 2; duel: the player who throws the first stone of end 1 1, the other 2). Legacy (and
  curling-1): both teams' n-th stones drew the same numbers (`default_rng([seed, end, n])`), so the team throwing
  later could read its coming error off the other team's stone and aim against it (the loophole the models used in
  darts). The duel is a pair of matches on the same seed with the first player swapped: the coin toss's result swaps
  with it, so the two players swap the streams and meet the same draws over the pair. The legacy records with throws
  no longer replay and are retired; the golden cases were re-recorded. Exploit probe and fairness:
  `docs/benchmarks/rng-duels-20261008.md` (a scripted team that aims against the other team's n-th error won 69 of
  100 matches under curling-1, 46 under curling-2). Fairness under curling-2 (the reference on both seats, seeds
  1-400 x both first orders, N = 800, the stage-4 computation in shards): first player +0.5 pp (95% CI +-3.46), seat
  A 0.0 pp (+-3.46): pass; the two orders of every seed are exact mirror images (the streams follow the role).

- **Duel, hammer in end 1**: a seeded coin toss (WCF R5(a); `core.rng` stream `coin_toss`) decides who throws the
  first stone of end 1: FIRST_PLAYER if it falls 0, else the other seat. Legacy: FIRST_PLAYER always threw first. The
  hammer is worth about 14 pp of win rate between the reference players, which failed the first-order fairness
  check; the toss falls 0 for the official seeds 1-3, so those configurations (and the golden records) are unchanged.
- **Match, the scripted team's reply**: it throws inside the player's action (the reply to `act` already holds its
  stones); legacy: after the review. A match that ended during a review (clock) would differ; none of the records does.
- **Observation**: the throws since your last one (`throws_since_your_last`, pure) instead of the legacy
  "since your last call" news queue; one PNG of two tiles (`house`, `hack`) instead of two PNGs.
- **Action**: `{"type": "throw", "speed", "angle", "spin"}` checked by the core's schema (numbers rounded to 6
  digits; `spin` exactly `ccw` / `cw`); the chess clocks ride along in the logged command (`clock_left_s`) for the tie
  rule. There is no `game throw` shortcut (core/game.py shortcuts take numbers only): `game act JSON`.
- **Start timeout**: in core 0.2.1 an incomplete match (no winner); legacy: a start forfeit, the present seat won.
  Core 0.2.2 (a duel seat that never starts loses) brings the legacy outcome back: the converted `nop-B` record
  then expects winner A (an intended rule change of the core, not of this sport).
- **Default seed of `match`**: `RCB_SEED` default 1 (legacy: the level's first official seed, 7101 / 7202).

## Pattern for duel_core sports

Curling and billiards share the legacy duel design: a pure simulator module (`curling.py`, `eightball.py`), a
`duel_core.py` with a `Duel` class (the match as a pure function of the seed, the first player and the commands, with
`play(seat, dec)`, `turn`, `done`, `winner`, `state_sha()`), and the generic `server_duel.py` + `sport.py` adapter
(validate, observation, spec, shot texts). To migrate such a sport:

1. `git mv` the simulator files and `duel_core.py` (and the single-player `Match` if the sport has one) into
   `sports/<sport>/`, **unchanged**. `server_duel.py`, `sport.py`, `replay.py`, `launch.py`, the metering copies, the
   prompts and the task sources are replaced by the core and are deleted (each sport had its own copies).
2. `plugin.py`: a `MatchBase` subclass that holds the `Duel` (or the single-player `Match`): `state_digest()` =
   `state_sha()` (this is what makes the legacy records replay bit-exactly), `apply()` = `play()`, `to_move()` =
   `turn`, `phase()` from `done`, `scores()` by side, `tiebreaks()` = the legacy winner rule split into named steps.
   Port `sport.py`'s validate into a `schema()` + a `SchemaLayer` (copy the clocks with `ctx_keys=("clock_left_s",)`
   if a legacy rule reads them, and apply that rule inside `apply()` so the state hash holds it as before), its
   observation into a pure `observe()` (no news queue: compute "since your last action" from the log) and its
   cameras into `cameras()` + `render_state()` + `Sport.scene()` (the legacy render.py's scene XML and camera poses).
   Clock and start forfeits are core decisions (`forfeit` rows): leave `forfeit()` at the default when the legacy
   server did not change the world on a forfeit, so the hash stays the legacy one.
3. Keep the legacy RNG draws where they are fair (moving them to `core.rng` changes every match); new randomness (here
   the coin toss) comes from `core.rng.stream`. But an execution draw shared by both players (the n-th stone of each
   team, the n-th shot of each player) is not fair: a player who acts later reads its own coming error off the
   opponent's. Give each player its own stream, keyed by its role in the pair of matches (curling-2, billiards keeps
   one draw per shot of the game, which only one player ever uses).
4. `golden/convert.py`: map each legacy command to the new command format, add the forfeit row from the raw
   episode's end row (the commands file has no forfeits), expected = manifest `final_sha`, scores (renamed to the new
   sides), the live winner, the number of commands. Check per-command hashes against `trace.json` too.
5. The reference player runs in the sim image for the checks (no SciPy there): numpy + Pillow only, and it reads its
   camera from the observation's tiles.
6. Fairness: a structural first-mover advantage (the hammer here, the break in 8-ball) fails the check; find what the
   rulebook says about who starts (a toss, a lag) and draw it from the seed, keeping the official seeds' order.
