# baseball: robot baseball in real time

Migrated from the Season-1 tasks `baseball-batting-wallclock-program-standard` and
`baseball-duel-wallclock-program-standard` (docs/framework_zh.md §10, stage 4). The simulator is the legacy one
(`bb.py`, `bbrt.py`, `bbduel.py`, moved with `git mv`), unchanged except for the duel's scatter streams (revision
duel-2, below), so every legacy batting recording replays bit-exactly through the v0.2 core, the state hash after
every pitch included (`golden/convert.py`); the legacy duel recordings with pitches are retired. The task names are kept
(`[[tasks]] id`); the formats are `batting` and `duel`.

**Rules** (MLB Official Baseball Rules 2026 with the deviations D1-D9 of `sport.toml`; the constants are `[spec]`).
A kinematic swing rig with a 0.84 m bat (4 parameters: plane, height, speed, reach, plus the time T to square) bats
against pitches flown with drag and Magnus lift; the bat-ball impact, the batted ball's flight and nine fixed fielders
decide the result; a plate appearance scores the bases the batter reaches (walk or single 1, double 2, home run 4).
`batting`: seat A bats 5 plate appearances against the scripted pitcher of level 1-4 (`RCB_LEVEL`) and needs 4
bases. `duel`: plate appearances alternate (`FIRST_PLAYER` bats first), one seat bats and the other pitches (type,
speed, target; a seeded scatter); 5 each, then up to 5 extra pairs, then the winner of the last pitch. Each batter's
plate appearances draw their own seeded stream (windups, pitch and swing scatter); the two games of a duel pair (the
same seed, the first batter swapped) swap the streams.

**Independent scatter streams** (revision `duel-2`, user decision of 2026-10-08). Revision `duel-1` gave the n-th pitch
to either batter the same windup and the same pitch and swing scatter (common random numbers): a player who had seen
the other half's n-th pitch (as pitcher or batter) knew its own coming deviation and could aim against it. Since
`duel-2` the draws are keyed by (seed, kind, slot, j): j = the batter's pitch number, slot 1 when the first batter
bats, 2 when the other one does, so within a game the two streams are independent, and the pair of games with the
first batter swapped swaps them (over the pair both players meet the same draws). The batting format is unchanged.
Exploit probe and fairness: `docs/benchmarks/rng-duels-20261008.md` (a scripted player that aims against the other
half's n-th deviation hits its pitch target to 0.005 cm under duel-1 and misses by 5.3 cm instead of 3.8 cm under
duel-2). Fairness under duel-2 (`rival` on both seats, seeds 1-20 x both first batters, N = 40): first batter
-5.0 pp (95% CI +-15.4), seat A 0.0 pp (+-15.5): pass (WARN: wider than +-5 pp, as every realtime sport at N = 40).

| File | Part of contract ② | What it is |
|---|---|---|
| `sport.toml` | declarations | rule book and deviations, formats, modes, timing, `game swing`, observation whitelist, rule constants, tiebreaks, exploits, players, fairness, official configurations, tasks, capabilities |
| `plugin.py` | world, actions, referee, observation | `Baseball` (MatchBase) around `bbrt.RT` / `bbduel.Duel`; `RigLayer` (the one action level: `swing`, `pitch`, `none`); `Views` (the legacy 2 x 4 tiles, MuJoCo, drawn in the seat's render worker); `scene()` and `render_state()` for the spectator video |
| `bb.py`, `bbrt.py`, `bbduel.py` | world (legacy; the duel's scatter streams since duel-2) | ball, rig, impact, fielders, the scripted pitcher and the count; the 100 Hz pitch clock; the duel |
| `builtin_players.py` | built-in players | `Reference` (the legacy camera-only batter `bat_player.py` as `act(view)`, and the legacy reference pitcher), `Rival` (the reference with seeded aim errors: the fairness policy), `Random` |
| `grade.py` | reward fields | `evaluate(final)` -> the legacy reward keys |
| `tests/test_baseball.py` | exploit list | one test per `[[exploits]]` entry |
| `golden/convert.py` | golden replays | legacy records -> core episodes; `python3 golden/convert.py STORE` (in the sim image) also checks every pitch's hash against the legacy trace; the duel records with pitches (`oo-s1-A`, `oo-s1-B`, `quit-B`) are retired since duel-2 (`RETIRED`) |
| `golden/duel-rival-s2-B`, `golden/batting-s1` | golden replays | native cases (tools/golden.py record); the duel case re-recorded under duel-2 |

Run everything: `PYTHON=python3.12 RO_GOLDEN_DATA=<store> tools/check baseball`. Generate the tasks:
`python3 tools/generate.py baseball`.

## How the legacy services map onto the core

- **World and clock**: one core control step = one legacy pitch-clock step (`advance(1)`, 10 ms). The pitch is judged
  inside the step that ends it (`finish()`: the batted ball, the fielders, the count) and `phase()` turns `unit_over`
  (both seats review in the duel); `resume()` is the legacy `next_pitch()`. A unit is a pitch (`unit()` = judged
  pitches), so a command still pending at a pitch's end is dropped, as in the legacy services.
- **State digest**: the legacy `state_sha()` (batting: the judged pitches only; duel: the judged pitches and the step
  count). It changes once per pitch, so hash checkpoints are per pitch.
- **Commands**: the legacy commands unchanged (`{"type": "swing", "t", "plane", "height", "speed", "reach"}`,
  `{"type": "pitch", "pitch", "speed", "x", "z"}`, `{"type": "none"}`), checked by the core's schema when they arrive
  (numbers rounded to 6 digits; the other role's command and a pitch that cannot be thrown to its target are refused
  with HTTP 400) and applied at the next control step. The duel's pitch flight (`bbduel.make_pitch`, pure) is computed
  in-process and cached (the legacy service computed it in a separate process).
- **Observation**: the legacy state minus the core's own fields: `pitch_phase` (legacy `phase`, a core field now),
  the count, bases or score, `t_since_release`, `heard`, `command`, `last_pitch` (pure: from the judged pitches; the
  legacy "news" queue is gone); no `step_phase`, `wall_s` (core timing fields instead). The images are the legacy
  ones (same scene, cameras, labels; a tile's scene is drawn once and reused, the PNG encoder is faster).

## Differences from the legacy tasks

- **No-show** (core 0.2.2): a duel side that never starts loses (legacy: an invalid match); a batter who never
  starts has a valid loss after the start window (1800 s), or at the collect if the session ends first (core 0.2.7;
  legacy: no start window, the game waited until collected).
  The converted records `nop-B` (winner A, legacy none) and `nop` (same result) carry the tag in their header.
- **Batting reviews**: the core gives every review 300 s (`ready_timeout_s`); a batter who does not review loses
  (legacy batting: no review timeout).
- **Batting success needs a complete game** (decision B2 of 2026-10-07): the legacy service counted a game that was
  collected before its end (the player stopped) as valid, and success = 4 bases or more in a valid game, so a batter
  who quit after a home run succeeded. Now the core's solo rule applies (`grade.py` no longer recomputes success,
  valid_trial or technical_failure): success = a valid, complete game (reason match_over) with at least 4 bases. A
  batter whose session ends after the start and before the end of the game has a valid loss (core 0.2.6, reason
  `abandoned`, success 0, whatever the bases; `passed` still reports bases >= 4). The converted record `quit-60s`
  (4 bases, legacy success 1) carries the tag in its header; its replayed result is the same.
- **Default seeds**: the generator's `RCB_SEED` / `DUEL_SEED` default 1 (legacy batting: the level's first official
  seed, 301 at level 1); any seed can be set. The duel's first batter is `FIRST_PLAYER` (legacy `FIRST_RECEIVER`).
- **Reward keys**. Common ones come from `core/grading.py` (added: `no_show`, `<seat>_no_show`, `solo`,
  `direct_play`, `lowlevel_actions` = 1 (the rig's command is the sport's only level), `match_over`, `seed`).
  Batting: kept every legacy key except `act_delay_steps_max`, `act_rule_wait_max_s`, `act_rule_wait_mean_s`,
  `n_calls`, `n_wait`, `refused_busy`, `sim_lag_frac_late_over_1_period`, `sim_step_ms_max`, `sim_commit_ms_max`,
  `sim_judge_ms_max`, `sim_next_pitch_ms_max` (timings and counters of the legacy service's own scheduler that the
  core does not record; the core's HTTP layer refuses a second act in flight before the match sees it);
  `revision` = 4; `n_decisions` = applied acts, `refused_not_model` = the core's `refused`, `grader_error` is the
  core's (0 unless the verifier failed). Duel: kept every legacy key except `<seat>_refused_busy` (as above) and
  `<seat>_prepare_s_max` (the pitch is prepared in-process now); added `<seat>_rejected` (commands the rig refused
  when they took effect: a swing before the release, a change after the swing started).
- **Spec**: the legacy `timing` block is `pitch_clock` (the core owns `timing`: slowdown, period); the legacy
  `calls`, `revision` and `player_mode` entries come from the core.
- The legacy `game` shortcuts are kept for the swing (`game swing T PLANE HEIGHT SPEED REACH`, `game none`); a pitch
  is `game act JSON`.

## Porting PR #56 (MLB innings, branch `baseball-innings`)

PR #56 changes `bb.py` (runners in the scene, seven pitch types), `bbrt.py` (render_snaps draws the runners),
`bbduel.py` (innings, outs, runners, runs, the run limit, pitcher-chosen spin) and the legacy duel service. To port it:
1. apply its diffs of `scripts/olympiad/shared/baseball/sim/bb.py`, `bbrt.py` and
   `scripts/olympiad/baseball/duel/environment/sim/bbduel.py` to `sports/baseball/bb.py`, `bbrt.py`, `bbduel.py`
   (`bb.py` and `bbrt.py` are unchanged since the migration, so their patches apply as they are, with the paths
   changed; `bbduel.py` has the per-batter scatter streams of revision duel-2: keep `slot()` and the
   `(seed, kind, slot, j)` keys of `windup_s`, `swing_scatter` and `make_pitch` when applying the PR's diff, and give a
   pitcher-chosen spin's scatter the same key);
2. `plugin.py`: `PITCH` (the schema) gains `spin_rpm` and `break_deg` with the type's default (`bbduel.default_spin`)
   and the limits of `bbduel.spin_limits` (check them in `RigLayer.parse`, as `validate_pitch` does), the arsenal
   comes from `bbduel.ARSENAL` already; `observe()` adds `inning`, `half`, `outs`, `on_base` (and the whitelist in
   `sport.toml`); `tiebreaks()` becomes runs, then the last pitch (the batter if it put the batter on base);
   `scores()` = runs; `final_fields()` the new `format` block; `BaseballSport.spec()` the PR's `match` texts; `Views`
   passes the runners (`rd.set(..., runners=sn.get("runners", ()))`);
3. `sport.toml`: bump `revision`, replace D1/D9 with the innings rules (5.09(e), 7.01, the run limit, the WBSC
   extra-inning runners), update `[spec] duel`, `duel_pitch`, `[tiebreaks]`; `tests/`: the run limit and the pitcher
   who walks every batter (the PR's exploit);
4. record new golden cases (`tools/golden.py record baseball ...`) or re-record the legacy duel records with the PR's
   service; the old duel records then no longer replay (a new revision), the batting ones still do.

## Open decisions

- The fairness check plays `Rival` on both seats: in-process (no command delay) two references hit a home run in
  nearly every plate appearance, every duel goes to the last-resort rule and the second batter wins it (seed 1:
  40:40, B by the last pitch). With strong batters the format leans on that rule (it favours the second batter); the
  live legacy reference (with its command delay) made outs (golden oo-s1: 32:30). With N = 40 (20 per order, an in-process duel draws about
  1800 views of 8 tiles) the 95% interval is wider than +-5 pp, which only warns for a realtime sport.
  Result (tools/check, 2026-10-06): N = 40, first batter +5.0 pp (95% CI +-15.4), seat A -10.0 pp (+-15.2): no
  advantage shown, the interval too wide to rule one out (WARN).
