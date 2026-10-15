# darts: 501 double-out

Migrated from the legacy tree in stage 1 (docs/framework_zh.md §10). The rules are the WDF Playing and Tournament
Rules (2018) with the deviations D1-D5 of `sport.toml`: best of 3 legs of 501 double-out, visits of 3 darts, the first
player throws first in legs 1 and 3; a leg not checked out after 30 visits each goes to the lower remaining score,
then to the player with more chess-clock time left, then to the player who did not start the leg.

| Format | Task | Seats | Result |
|---|---|---|---|
| `501` | `darts-501-program-standard` | A against the scripted thrower inside the world (`darts.Match`; level 1-4 = `RCB_LEVEL`, default 1; it throws first in legs 1 and 3 and aims with the optimal-aim table for its own scatter) | success = A wins a valid match |
| `duel` | `darts-duel-program-standard` | A and B (`darts.Duel`; `FIRST_PLAYER` throws first in legs 1 and 3; both orders are played) | success = a valid match; the winner is `A_won` / `B_won` |

| File | Part of contract ② | What it is |
|---|---|---|
| `sport.toml` | declarations | rule book and deviations, formats, modes, timing, `game throw`, observation whitelist and renderer, rule constants, tiebreaks, exploit list, players, fairness, official configurations, tasks |
| `darts.py` | world + referee | the legacy simulator: board scoring, RK4 flight under gravity and quadratic drag (k by the seed), seeded execution scatter (in the duel one independent stream per player since revision `darts-5`), the scripted opponent (`policy_l1..4.json`, built by `python3 darts.py policy L OUT`), `Match` (501) and `Duel` |
| `plugin.py` | world, actions, referee, observation | `Darts` (MatchBase) over `Match` / `Duel`: `schema`/`apply` (a throw, 4 decimals; the chess clocks at the throw go into the command), `phase`/`scores`/`tiebreaks`/`to_move`/`forfeit`, `observe` (score board and the caller's calls, no coordinates), `cameras`/`render_state`, `reply_views` (the board close-up in each dart's reply); `DartsSport.spec()` adds the first player and the opponent |
| `board.py` | observation | the MuJoCo scene for the renderers (board texture, oche, release mechanism, the latest visit's darts as mocap bodies) and the camera definitions (oche, board, side) |
| `builtin_players.py` | built-in players | `Reference` (camera-only: optimal-aim table built from /spec, drag learnt from its darts in the board close-up), `Rival` (the reference with its own aim errors: the fairness policy), `Random` |
| `grade.py` | reward fields | the legacy tasks' reward keys from final.json |
| `tests/test_darts.py` | exploit list | one test per `[[exploits]]` entry |
| `golden/` | golden replays | `convert.py` (the legacy records of both tasks in the golden-data store -> core episodes; the 501 records and the duel `nop-B` replay bit-exactly, the duel records with throws are retired since `darts-5`), duel cases re-recorded under `darts-5`, one 501 case recorded with the reference |

Changes against the legacy tasks (revision `darts-4`): the players' images come from the task renderer
(`renderers/`, default MuJoCo) instead of the sport's own drawing; the side view no longer draws the last dart's path;
the reply to a throw carries the board close-up with the darts of the thrower's current visit, the one just thrown
included, as the legacy `landed_image_b64` did (since core 0.2.8 through the core's reply-image hook `reply_views`,
drawn by the task renderer off the service lock, as `observation.image_png_b64` + `tiles`; revision `darts-3` had
dropped it, so after a player's third dart, once the opponent threw, that visit could no longer be seen); the
observation names the caller's calls since your last dart (`darts_since_your_last`) and your last
dart (`your_last_dart`) and is a pure function of the match; the official configurations are seeds 1-3 (format 501:
with every level 1-4; duel: both first orders).

**Independent scatter streams** (revision `darts-5`, user decision of 2026-10-08). Until `darts-4` the duel gave the
n-th dart of either player in a leg the same execution scatter (common random numbers). The player who threw later
could read its own coming deviation off the opponent's dart and aim against it: in the duel matches of 2026-10-07 all
four model players (SOL, Astra, Fable, Opus) did so, about +16 points per dart. Since `darts-5` the n-th dart of a
player draws `default_rng([seed, 100 + leg, stream, n])` with stream 1 for the first player and 2 for the other. The
duel is played as a pair of matches on the same seed with the first player swapped, so the two players swap the streams
and over the pair both meet the same draws. The 501 format already drew one stream per thrower and is unchanged.
Exploit probe (`docs/benchmarks/rng-duels-20261008.md`): a scripted thrower that reads the opponent's n-th deviation
and aims against it, compared with a non-compensating twin, wins 100 % of 400 matches under `darts-4` (the twin 50 %)
and 27.5 % under `darts-5` (the compensation only adds noise). Fairness under `darts-5` (`rival` on both seats, seeds
1-400 x both first orders, N = 800, the stage-4 computation in four shards): first player +2.5 pp (95% CI +-3.46),
seat A +1.75 pp (+-3.46): pass.

**Release speed 5-7 m/s** (revision `darts-6`, deviation D5; review HB11 / audit D1, 2026-10-09). The release
mechanism throws at a human elite level, an estimate: no dart-throwing robot publishes its release speed, and a skilled
thrower releases at about 5.1-5.5 m/s. Until `darts-5` the range was 4-9 m/s: 9 m/s is far beyond a human throw, and
below about 5 m/s no dart reaches the top of the board (at 4 m/s 62 of the 82 beds are out of reach even at 40° pitch).
The built-in players are unchanged (the reference and `rival` throw at 7.0 m/s, the scripted opponent at 6.5 m/s; the
random player now draws 5.5-7.0 m/s instead of 5.5-8.5), so the golden cases replay unchanged and the fairness numbers
above (all `rival` darts at 7.0 m/s) still hold. B9 (real-machine feasibility) on `darts-6`: see the pull request.

Run everything: `PYTHON=python3.11 RO_GOLDEN_DATA=<store> tools/check darts`. Generate the tasks:
`python3 tools/generate.py darts`.
