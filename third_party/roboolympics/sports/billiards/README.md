# billiards

Billiards (one rack of American 8-ball) on the v0.2 framework (docs/framework_zh.md §4.2, §10 stage 1), migrated from
the legacy tree (`scripts/olympiad/billiards/`, `scripts/olympiad/shared/billiards/`). Two tasks, same names as in
Season 1: `billiards-eightball-program-standard` (format `eightball`: seat A against the scripted opponent, levels
1-2) and `billiards-duel-program-standard` (format `duel`: seat A against seat B). Rules: WPA Rules of Play
(effective 2025-09-15), 8-Ball, with the deviations of `sport.toml` (D1-D11: the 8 on the break always spotted, every
shot called, no jump shots, a 60-shot limit, a coin toss instead of the lag for the break, ...). Revision
`billiards-2` (2026-10-07, review item B7): in the duel a seeded coin toss decides who breaks (billiards-1: FIRST_PLAYER). Physics: pooltool 0.6.0 (its own image,
`ghcr.io/mll-lab-nu/eai-olympiad-billiards`), not MuJoCo.

| File | Part of contract ② | What it is |
|---|---|---|
| `sport.toml` | declarations | rulebook and deviations D1-D11, formats, modes, timing, observation whitelist, rule constants, tiebreaks, exploits E1-E7, players, fairness, official configurations, tasks, capabilities |
| `plugin.py` | world, actions, referee, observation | `BilliardsMatch` (MatchBase) around the legacy `eightball.Match` / `duel_core.Duel`; `ShootLayer` (the one action level); `Billiards.worker_call` draws the cameras with `render.py` |
| `eightball.py` | world (legacy, unchanged) | the rack, the shot (pooltool), the WPA rules, the scripted opponent, `Match` |
| `duel_core.py` | world (legacy, unchanged) | `Duel(seed, first)`: `eightball.Game` with two seats; `first` breaks (plugin.py passes the coin toss's breaker) |
| `render.py` | cameras (legacy, unchanged) | the PIL pinhole renderer of the two cameras (top, sight); its video code is not used |
| `builtin_players.py` | built-in players | `Reference` (the Season-1 scripted player, eight_player.py; reads the `top` tile), `Random` |
| `grade.py` | reward fields | the legacy reward keys of both tasks (+ `breaker`, `physics_retries`) |
| `tests/test_billiards.py` | exploit list | one test per `[[exploits]]` entry |
| `golden/` | golden replays | recorded v0.2 matches (duel-s1-A (re-recorded under billiards-2: seed 1's toss lets B break), duel-s3-B, duel-random-s2-A, eightball-L1-s1) + `convert.py` (the legacy records of the golden-data store; the superseded duel records are retired, see below) |

Run: `PYTHON=python3.12 tools/check billiards` (all stages); `RO_GOLDEN_DATA=<store> tools/check billiards --stages 3`
for the legacy records (billiards-1: 7/7 bit-exact, and all 60 per-command hashes equal the legacy traces; billiards-2:
the 3 eightball records, the 4 duel records are retired). Generate:
`python3 tools/generate.py billiards` (publish = true: `tasks/billiards-*`).

Rechecked on core 0.2.12 (main a71f92f7, 2026-10-09): `RO_GOLDEN_DATA=<store> tools/check billiards --stages 1,2,3`
75/75 (stage 2: determinism, mirror symmetry on seeds 1-4, malformed and fuzzed actions, E1-E7, quit and no-show;
stage 3: the 4 golden cases and the 3 eightball records of the store, all identical).

**The cue is a declared cue machine** (deviation D11, 2026-10-09; no change of play, no new revision), like Deep
Green, the pool-playing gantry robot of Queen's University ([IEEE Computer 41(1) 2008](https://doi.org/10.1109/MC.2008.33)).
The cue speed 0.3-7 m/s lies within the speeds reported for its cue (about 3 m/s for the normal stroke, up to
12 m/s for the break stroke). B9 (research branch b9-release): a legal break without a foul exists for 100/100 racks
and a pot after the break for 99/100 tables, and capping every shot but the break at 3 m/s changes none of the 200
results.

## What changed from the legacy tasks

- **The images**: one PNG of two tiles (`top` 1120 x 640, `sight` 960 x 540; `observation.tiles`) instead of two
  PNGs; the camera poses are in the observation's `cameras` as before. The sighting direction is the observation query
  `{"aim": DEG}` (`game observe '{"aim": 87.5}'`; legacy: `game observe AIM_DEG`). The sport draws its own images
  (`Match.camera_job`, final.json `renderer = "sport"`): its image has no MuJoCo, and the scene is pooltool's.
- **Action**: `{"type": "shoot", "aim_deg", "speed", "spin_x", "spin_y", ["cue_x", "cue_y"], ["call"]}` checked by
  the core's schema (numbers rounded to 6 digits, `call` one of the 91 `BALL:POCKET` values or `safety`, matched
  case-insensitively), then by the rules of the moment (`eightball.Game.check`) before the shot is logged; the chess
  clocks ride along (`clock_left_s`) for the tie rule. There is no `game shoot` shortcut (core/game.py shortcuts take
  numbers only): `game act JSON`.
- **eightball, the opponent's reply**: it plays inside the player's action (the reply to `act` already holds its
  shots); legacy: after the review. A rack that ended during a review (clock) would differ; none of the records does.
  The tie rule at the shot limit reads the clock at the player's last action (legacy: also after the review).
- **Observation**: the shots since your last one (`shots_since_your_last`, pure) instead of the legacy "since your
  last call" news queue; the state also says who broke (`breaker`) and the opponent's shots taken.
- **Engine failures** (D10): pooltool 0.6.0 sometimes fails an internal assertion on a shot (e.g. `v_n_0 < 0` in its
  cushion model; seen once in 300 reference racks (seed 16, the second shot; resolved by the first nudge)). Legacy: a simulator error, an invalid trial that the same
  configuration repeats. Now the shot is simulated again from the same state with the aim nudged by 0.001 degree steps
  (`physics_retry` in the shot record, `physics_retries` in reward.json); deterministic, so the replay agrees, and a
  shot the engine resolves (every legacy record) is untouched.
- **Start timeout**: core 0.2.2 (a duel seat that never starts loses, valid) keeps the legacy duel outcome; the converted
  `nop-B` record expects winner A (tagged `core-0.2.2-no-show`: the legacy manifest has no winner there). A single
  player who never starts loses when its start window closes or, since core 0.2.7, at the collect if the session ends
  first (`no_show`, valid, success 0); the legacy `nop` record (a session collected inside the window; legacy: a
  valid loss) converts to that no_show forfeit without a winner (tagged `rule-change:core-0.2.7-prestart-noshow`;
  cores 0.2.2-0.2.6: an incomplete match).
- **Default seed of `eightball`**: `RCB_SEED` default 1 (legacy: the level's first official seed, 1 / 2).
- **No replay video**: the core's spectator video draws `Sport.scene()` with a renderer of `renderers/`, which this
  sport cannot provide (no MuJoCo in its image); the legacy pooltool video (render.py `render_video`) is not wired.
- **Reward keys**: the legacy keys of both tasks with the same meaning, `official_config` with the legacy per-format
  official configurations; added by the core: `direct_play`, `lowlevel_actions`, `solo`, `no_show`, `<seat>_no_show`;
  added here: `breaker` (duel), `physics_retries`.

- **Who breaks in the duel** (billiards-2, user decision 2026-10-07, review item B7; deviation D9): a coin toss drawn
  from the seed (`core.rng` stream `coin_toss`, as curling's hammer toss) decides it: FIRST_PLAYER breaks if it falls
  0, else the other player (`game spec` match.breaker / match.you_break / match.coin_toss; reward `breaker`, 0 = A,
  1 = B, may now differ from `first_player`). Legacy and billiards-1: FIRST_PLAYER always broke. The rulebook (WPA
  Rules of Play 1.2 and 4.1, 2025-09-15) has the players lag and the winner of the lag choose who breaks; neither the
  Rules of Play nor the Playing Regulations (same date) offer a coin toss as an alternative (Regulation 1 leaves only
  who breaks after the first rack at 9-Ball to the event), so the toss is a deviation: it replaces a skill shot by a
  draw, and the toss winner always breaks (no choice). The toss falls 1 for seeds 1, 5, 7-10 and 0 for 2, 3, 4, 6
  (official duel seeds 1, 3, 4: seed 1 now lets the other player break). The eightball format is unchanged (level 1
  you break, level 2 the scripted opponent).
- **Retired legacy records**: the four legacy duel records of the golden-data store (`clock-1s`, `nop-B`, `oo-s1-A`,
  `oo-s1-B`, all seed 1, first A) were played with A breaking; under billiards-2 seed 1's toss lets B break, so they
  no longer replay (their first shot is out of turn; `nop-B` has no shot but the state hash holds the breaker).
  `golden/convert.py` RETIRED skips them (kept in the store as they are); the replacement is `golden/duel-s1-A`,
  re-recorded with the reference players under billiards-2 (B breaks and wins 1-0, the mirror of the old record).
  The earlier matches were demos and all future matches are re-run (user, 2026-10-07).

## Fairness

tools/check stage 4 (reference players on both seats, both first orders, 250 seeds each, N = 500), billiards-2
(2026-10-07): first side -2.8 pp, 95% CI +-4.38 pp; seat A 0.0 pp, +-4.38 pp (passes). The first side is no longer
the breaker: the seed's coin toss lets it break in 114 of the 250 seeds (toss 0) and the other player in 136, so the
breaker's edge now averages out across seeds instead of landing on the side named first. The duel is mirror-exact
(the n-th shot of the rack has the same execution draw whoever shoots it, and the toss depends on the seed only), so
the two orders of a seed are the same rack mirrored: the seat advantage is exactly 0, and both numbers rest on 250
independent racks.

Before (billiards-1, FIRST_PLAYER always broke; same N = 500): first side (= the breaker) +6.0 pp, 95% CI +-4.35 pp;
seat A 0.0 pp, +-4.38 pp (a separate run of 300 racks: the breaker won 161, 53.7 %). The break itself still carries
that edge; the official evaluation plays both first orders of every seed, which gives each player the break once.

