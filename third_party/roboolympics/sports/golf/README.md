# golf: mini-golf putting

A round of 4 mini-golf putting holes (par 2, 2, 2, 3; par 9) on sloped greens enclosed by walls, one hole with an
obstacle block, migrated from the Season-1 task `golf-putting-program-standard` to the v0.2 framework
(docs/framework_zh.md §10, stage 1). Rules: WMF Rulebook 2.3 (secondary: R&A/USGA Rules of Golf 2023); the deviations,
the rule constants and the exploit list are in `sport.toml`. One format: the single round. The legacy duel
(`golf-putting-duel-program-standard`) was retired on 2026-10-08 (user decision, the criterion of review item B10 for
archery and basketball): both players putted their own balls on identical holes without any interaction, so a
head-to-head comparison is the two players' single scores on the same seeds.

- **Single**: one round; success = a complete round in 9 strokes or fewer (par, the old task's line).
- **The putter is a declared putting machine** (deviation D5, 2026-10-09; no change of play, no new revision), like
  the swing robots of [Golf Laboratories](https://www.golflabs.com/robotics-information): their club-head speeds
  reach from 0.1 mph for putters ([media report, 2016](https://robohub.org/robot-golfer-hits-hole-in-one/)) to
  120-130 mph, so the putt's 0.05-6.0 m/s is within them (the slow end rests on the media report only). B9: a
  one-putt exists from 59 of 60 seeded starts and par is reachable from all 60; the 6.0 m/s cap is used in full (two
  starts on the hole with the block need a bank shot of at least 5.87 m/s), so it must not be lowered.
- **Action** (one level, `low`): `{"type": "putt", aim_deg, speed}` with seeded Gaussian execution scatter (0.3 deg,
  1.5 %). **Observation**: the decision point (hole, par, stroke, the last stroke's result), the scorecard, and one
  PNG of three views (behind the ball, the side of the green, the green from above with the previous stroke's path).
  No ball, cup or slope numbers, no seed.

| File | Part of contract ② | What it is |
|---|---|---|
| `sport.toml` | declarations | rule book and deviations, formats, modes, timing, `game putt`, observation whitelist, rule constants, exploits, players, official seeds, tasks, capabilities |
| `sim/golf.py`, `sim/sport.py` | world, referee, cameras | the Season-1 simulator and rules module, **byte-identical** to `scripts/olympiad/shared/golf/sim/` at bfec6311 (the rules module's duel code is no longer used) |
| `turnshot.py` | world, actions, referee, observation | the generic adapter of the shared turn-based server, byte-identical to `sports/bowling/turnshot.py` |
| `plugin.py` | the sport | `Golf(TurnShotMatch)`: the action schema; `GolfSport`: build, the action layer, `/spec`, players |
| `builtin_players.py` | built-in players | `Reference` (camera-only: reads the green's heights from the grid lines and plans with its own rolling model), `Random` |
| `grade.py` | reward fields | the Season-1 single task's reward keys (`total`, `to_par`, `score_h1..4`, ...) |
| `tests/test_golf.py` | exploit list | one test per `[[exploits]]` entry |
| `golden/` | golden replays | recorded single rounds (`single-s1`, `single-random-s3`); `convert.py`: the Season-1 single records (golden-data store) -> core episodes |

Check: `PYTHON=python3.12 RO_GOLDEN_DATA=<store> tools/check golf`. Generate: `python3 tools/generate.py golf`.

**Golden replays.** The 4 Season-1 single records (nop, clock stall, oracle on 20260928 and on seed 7) replay through
the v0.2 core and this plugin to the legacy scores, winner, number of strokes and state hash. The hash is `sha_json({seat: legacy round hash})`, so it
matches only if every seat's ball after every stroke (the `Course.state_sha` chain of `sim/golf.py`, play-off holes
included) is bit-identical. The 4 legacy duel records were retired with the duel.

**Changes from Season 1** (the single round's play is unchanged): **no duel** (retired on 2026-10-08, see above: the
format, deviations D3 and D4, the play-off and clock tiebreaks, the fairness check, exploits E6 and E7, the
reference's role jitter, the duel golden cases and the converter's duel records are gone; the exploit ids E1-E5 and E8
are kept; revision golf-3); the core replaces the legacy service; the seed is no longer in `/spec` (rule O3); the
observation is stateless (the scorecard and the last stroke instead of "shots since your last call"); the default
seed is 1 (a drawn course; 20260928 is the original course and stays an official seed); the task name stays the
Season-1 name (`[[tasks]] id`).

**The reference player** sees only the images and `/spec`. In the overhead view (a pinhole straight down; its height
and focal length are in `/spec`) a white grid line at a known x is drawn at u = cx + f (x - L/2) / (H - z), so its
sub-pixel position gives the height z of the surface under it (and likewise the lines at a known y): about 10,000
height samples per green, good to about a millimetre except near the image centre. It fits a cubic B-spline with
knots on the grid lines to them, finds the ball (white; after a stroke, the end of the yellow path drawn over it), the
cup (black disc) and the block (red), and plans each putt with its own model of the published physics ((5/7) g slope
plus the constant rolling resistance of the stimpmeter, the wall rebound, the cup's capture speed of about 1.7 m/s):
a coarse fan of aims and speeds (bank shots included, a small cost per wall hit), then fine grids around the best,
choosing the putt with the fewest expected strokes still to play under the published scatter.

How well it plays (the rounds of the former duel's fairness check, 1600 rounds on seeds 1-400 with a small duel jitter): 6.8
strokes per round on average, 98 % of the rounds at par (9) or better; 6 strokes on the default seed 1 (golden case
`single-s1`). Its main weaknesses: a ball that comes to rest against a wall after a slow rebound is sometimes located
at the wrong end of its path (the last dots overlap there), and the height field is least certain near the image
centre, where the grid lines move least with height.

**Reward keys**: every Season-1 key is kept (`total`, `complete`, `clock_out`, `n_shots`, `n_reviews`, `refused_*`,
`clock_used_s`, `metered_*`, `to_par`, `picked_up_holes`, `score_h1..4`, `transient_api_errors`, ...; the duel's
keys went with the duel). The core adds `no_show`, `<seat>_no_show`, `direct_play`, `lowlevel_actions`, `solo`,
`A_won`, `A_transient_api_errors`, `A_unrecovered_api_errors`.
