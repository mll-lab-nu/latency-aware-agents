# discgolf: disc golf

A 6-hole disc golf round (par 3, 3, 4, 3, 3, 4: par 20) through woods, around a left and a right dogleg, over OB
water and past a mandatory, in a seeded wind, migrated from the Season-1 task `discgolf-course-program-standard` to
the v0.2 framework (docs/framework_zh.md §10, stage 1). Rules: PDGA Official Rules of Disc Golf (rev. 2026-01-01); the
deviations, the rule constants and the exploit list are in `sport.toml`. One format: the single round. The legacy duel
(`discgolf-duel-program-standard`) was retired on 2026-10-08 (user decision, the criterion of review item B10 for
archery and basketball): both players threw their own discs on identical courses without any interaction, so a
head-to-head comparison is the two players' single scores on the same seeds.

- **Single**: one round; success = the round complete in 21 throws or fewer (par + 1, the old task's line; lower is
  better: `success = { score_at_most = 21 }`).
- **Action** (one level, `low`): `{"type": "throw", disc, speed, launch_deg, nose_deg, bank_deg, spin_rps, aim_deg}`
  with seeded Gaussian execution scatter. **Observation**: the decision point (hole, par, throw number, the lie, the
  basket's position, distance and heading, the mandatory, the last throw's result), the scorecard, and one PNG of
  three views (overview straight down on the hole with the previous throw's flight as yellow dots, the tee view behind
  the launcher, the windsock from above). No tree coordinates, wind numbers, flight track or seed.

| File | Part of contract ② | What it is |
|---|---|---|
| `sport.toml` | declarations | rule book and deviations, formats, modes, timing, `game throw`, observation whitelist, rule constants, exploits, players, official seeds, tasks, capabilities |
| `sim/flight.py`, `sim/discgolf.py`, `sim/sport.py` | world, referee, cameras | the Season-1 disc-flight model, course and rules module, **byte-identical** to `scripts/olympiad/shared/discgolf/sim/` at bfec6311 (moved with `git mv`; the rules module's duel code is no longer used) |
| `turnshot.py` | world, actions, referee, observation | the generic adapter of the shared turn-based server, a byte-identical copy of `sports/bowling/turnshot.py` |
| `plugin.py` | the sport | `DiscGolf(TurnShotMatch)`: the action schema; `DiscGolfSport`: build, the action layer, `/spec` (the legacy `spec(False)` plus the image tiles, no seed), players |
| `builtin_players.py` | built-in players | `Reference` (the Season-1 camera-only planner, unchanged), `Random` |
| `grade.py` | reward fields | the Season-1 single task's reward keys (`total`, `to_par`, `score_h1..6`, `penalty_throws`, ...) |
| `tests/test_discgolf.py` | exploit list | one test per `[[exploits]]` entry (ported from the Season-1 unit tests) |
| `golden/` | golden replays | recorded single rounds (`single-s1`, `single-random-s3`); `convert.py`: the Season-1 single records (golden-data store) -> core episodes |

Check: `PYTHON=python3.12 RO_GOLDEN_DATA=<store> tools/check discgolf`. Generate: `python3 tools/generate.py discgolf`.

**Golden replays.** The 4 Season-1 single records (nop, clock stall, oracle on 20261005 and on seed 7) replay through
the v0.2 core and this plugin to the legacy scores, winner, number of throws and state hash. The hash is `sha_json({seat: legacy round hash})`, so it
matches only if every seat's round (every throw's exact rest point, status and result, the hash chain of
`sim/discgolf.py`) is bit-identical (the 4 legacy duel records were retired with the duel). Local cases recorded with
the built-in players: `single-s1` (reference, seed 1) and `single-random-s3` (random, seed 3; the duel cases were
retired).

**Changes from Season 1** (the single round's play is unchanged): **no duel** (retired on 2026-10-08, see above: the
format, deviations D6 and D7, the order of play, the play-off and clock tiebreaks, the fairness check, exploits E7 and
E8, the reference's role-seeded refinements, the duel golden cases and the converter's duel records are gone; the
exploit ids E1-E6 are kept; revision discgolf-3); the core replaces the legacy service; the observation is stateless
(the scorecard and the last throw instead of "news since your last call"); the default seed is 1 (a drawn course;
20261005, the standard course, stays an official seed); the single task keeps its Season-1 name
`discgolf-course-program-standard` (`[[tasks]] id`). The Season-1 prompt's physics section is now the `flight`,
`wind` and `trees_basket_ground` constants of `[spec]`; the full coefficients are in `/spec` as before.

**The reference player** is the Season-1 reference disc golfer: it sees only the images, the decision point and
`/spec`; it reads the wind from the windsock tile, maps the trees from the overview tile, simulates candidate throws
with its own vectorized copy of the documented flight model and picks the lowest expected number of throws under the
published scatter. Single: seed 1 -> 18 throws, 20261005 -> 19, 7 -> 21 (the Season-1 rounds, replayed exactly);
Season-1 diagnostic over 20261005 and 2-9: 17-22 (mean 19.3; 8 of 9 within the line), against 19-29 for the same
planner ignoring the wind and 27-35 for straight aiming. Its random refinements are seeded by the throw number, as in
Season 1.
