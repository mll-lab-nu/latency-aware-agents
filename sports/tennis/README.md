# tennis: robot singles tennis in real time

Migrated from the Season-1 tasks `tennis-match-wallclock-program-standard` and `tennis-duel-wallclock-program-standard`
(docs/framework_zh.md §10, stage 3). The physics is the legacy one, unchanged (ball drag / Magnus lift, bounce, net,
racket law, robots: `sim/tn.py`, `sim/duel.py`, `sim/scene/`); the rules are not: since 2026-10-07 (revision
`tennis-wallclock-7+duel-5`; `-8+duel-6` on 2026-10-08: no cap) **the players serve** by the ITF service rules instead of the scripted opponent / a serving
machine, the rail base covers the whole half, and (since revision `tennis-wallclock-9+duel-7`, 2026-10-09, rule A6)
**each robot draws its own execution-scatter stream** (below; revisions 7+5 to 8+6 shared one sequence). The legacy
recordings no longer replay (`golden/convert.py` and the legacy cases are retired; the golden cases are re-recorded).
The task names are kept (`[[tasks]] id`); the formats are `match` and `duel`.

**Independent scatter streams** (revision wallclock-9 / duel-7, rule A6 of docs/benchmarks/roboolympics-rules.md,
docs/conventions/racket_serve_zh.md §6). The scatter of a player's n-th act is `tn.scatter_rng(seed, kind, stream, n)` =
`default_rng([seed, 1000, kind, stream, n])`: in the duel stream 1 for the first server's robot and 2 for the other's
(`duel.Duel.plan_snapshot`); the duel is a pair of matches on the same seed with the first server swapped, so the two
players swap the streams and meet the same draws over the pair. In the match format the player's stream (1) starts
with the tag 1000, which none of the scripted opponent's draws does (its serve choice `[seed, 11, n]` and return choice
`[seed, point, leg]`; the old player key `[seed, 13, n]` coincided with the opponent's return draws of point 13).
Revisions duel-5 / duel-6 drew one sequence for both robots (the n-th act of each got the same draw), which a player
acting later could pre-compensate. Exploit E11.

**Rules** (the constants are `sport.toml [spec]`): a Franka Panda arm with a racket on an x-y rail base plays singles
on an ITF court; the ball has drag, Magnus lift, a hard-court bounce and a racket impact law; first to 5 points (from 4:4 by 2, no cap); the match also ends after 400 hits or 40 points played. `match`: seat A against a scripted
opponent (levels 1-4, `RCB_LEVEL`); `duel`: two seats in mirrored frames (`FIRST_PLAYER` serves first). Deviations D1-D7
in `sport.toml`.

**The serve** (ITF Rules 16-22, 5 b, 24 e; D2, D4, D6, D7; the cross-sport serve convention of the racket sports):
- *Who serves.* The tie-break order (Rule 5 b): the first server serves point 1, then each player serves two points in
  turn; the serving half alternates every point, point 1 from the right half (Rule 17). Duel: the first server is
  `match.first_server` (the runner and the fairness check play both orders); match: you serve first, the scripted
  opponent serves its points with its own seeded serve (from 2.75 m; a serve missing the box is its fault).
- *Where.* At the release the base must be at rest (< 0.1 m/s) with its centre behind the base line (y <= -11.885)
  between the centre mark and the singles sideline of the serving half (Rule 16; judged on the base, D4); a base that
  moves more than 0.10 m between the release and the strike is a foot fault (Rule 18).
- *The toss (the missing free hand).* Rule 16 lets a player who can use only one arm release the ball with the racket.
  From 1.0 s after the point starts (the receiver is ready) the server sends `{"type": "toss", "pos": [x, y, z],
  "height": H}`: when it takes effect (the act lead; low level: the next step) the ball is at that point (0.3-2.4 m
  high, within 1.3 m horizontally of the base) and rises straight up by H (0-1.0 m), without spin (D2); then the
  server hits it with an `act` (which may be sent first). The receiver sees the toss point and height.
- *Faults* (Rules 19-20; a second serve from the same half, a second fault loses the point): the serve lands outside
  the box diagonally opposite or in the net, a foot fault, the toss missed while a stroke is under way (19 b; let drop
  without a stroke: tossed again, 19 Case 1), no toss within the window (6 s; a second serve 4 s): undue delay (D6,
  every time; no disqualification). A serve that touches the net and lands in is a let (22 a). The receiver may not
  hit the serve before it bounces (24 e). Point records: `serve_fault:<reason> (double fault: ...)`,
  `receive_fault:volley`.
- *Observation.* `you_serve`, `you_receive_serve`, `serve` {phase waiting / open / tossed / done, court,
  serve_number, opens_in_s, deadline_in_s, tossed_at_step, toss}.
- *The rails* reach the net (-15.4 <= by <= -0.4); a racket contact beyond the net (y > 0) loses the point (24 i).

| File | Part of contract ② | What it is |
|---|---|---|
| `sport.toml` | declarations | rule book and deviations, formats, modes, timing, observation whitelist, rule constants, tiebreaks, exploits, players, fairness, tasks, capabilities |
| `plugin.py` | world, actions, referee, observation | `Tennis` (MatchBase) around `tn.Match` / `duel.Duel`; `LowLayer` (joint targets `traj`, `move`, `none`) and `RacketLayer` (the legacy `act`, planned in the plan worker by `tn.PlanCtx` and lowered to `traj`); `Views` (the legacy 2 x 4 tiles) |
| `sim/` | world (unchanged) | `tn.py` (ball, robot, stroke controller, scripted opponent, referee), `duel.py` (two robots), `scene/` |
| `builtin_players.py` | built-in players | `Reference` (the legacy camera-only `tn_player.py` as `act(view)`, plus the serve and the receiving spot), `Random` |
| `grade.py` | reward fields | `evaluate(final)` -> the legacy reward keys the core records |
| `tests/test_tennis.py` | exploit list | one test per `[[exploits]]` entry (incl. the legacy net-cord test) |
| `golden/` | golden replays | cases recorded under revision wallclock-9 + duel-7 (`tools/golden.py record`; duel-s1-A and match-high-s1 re-recorded for the scatter streams, match-low-s1 unchanged) |

**Actions.** The episode logs low-level commands only: `traj` (joint targets per control step, then the ready pose),
`move` (the base's rest position) and `none`. The high level (the tasks' level) is the legacy racket command `act`:
accepted at the next step, its stroke is planned in the seat's plan worker from a snapshot of that step and takes
effect `lead` steps later (lead = ceil(0.5 s / wall time per step), 3 at x20, as in the legacy services), lowered to a
`traj` with that effect step (`k0`). A command sent before the previous act took effect is refused (409); the legacy
service queued it instead. If the plan worker misses the effect step (a loaded host), the act is planned again when
it is lowered, from the robot's state then, for an effect at that step and the same contact time (the world waits
meanwhile; the legacy service stalled the world until the plan came and kept the effect step). A traj whose planned
robot state differs from the robot's at its effect step is refused (counted in `rejected_acts`).

**Differences from the legacy tasks.**
- Reward keys: the common ones come from `core/grading.py`. Duel: kept `match_limit`, `ready_forfeit`, `timing_invalid`,
  `steps`, `hits`, `slowdown`, `sim_lag_max_s`, `<seat>_points`, `<seat>_attempts|applied|refused|dropped|observations`;
  dropped `<seat>_refused_busy` (the core's HTTP layer refuses a second request in flight before the match sees it) and
  `<seat>_act_delay_steps_max` (not recorded by the core); added `<seat>_rejected_acts`. Single: dropped
  `act_delay_steps_max`, `act_rule_wait_max_s`, `act_rule_wait_mean_s`, `n_calls`, `n_wait`, `sim_step_ms_max`,
  `plan_slack_ms_min`, `plan_wait_ms_max`, `opp_cpu_ms_max`, `opp_slack_ms_min`, `opp_wait_ms_max`, `render_ms_max`,
  `sim_lag_frac_late_over_1_period`, `opp_cache_miss` (timings of the legacy service's own scheduler and workers);
  added `rejected_acts`; `revision` = 4; `match_over` is the core's (1 also at the match limit).
- The opponent's return search runs in-process at its hit step (the legacy service precomputed it in a worker; same
  result, a short stall at x20).
- The default seed of `match` is the generator's 1 (legacy: the level's seed, 101 at level 1); `RCB_SEED`
  sets any seed. `RCB_SLOWDOWN` (default 20, `[timing] slowdown`) sets the slowdown as before.
- The `game` shortcuts `game act T PX ...` and `game move BX BY` are gone (vector fields); use `game act JSON`.
- A duel side that never starts loses (core 0.2.2); the legacy service voided that match. A single player who never
  starts has a valid loss (`no_show`) when the start window closes or at the collect (core 0.2.7; legacy: a valid
  loss; cores 0.2.2-0.2.6: an incomplete match if the session ended inside the window).

Run everything: `PYTHON=python3.12 tools/check tennis`. Generate its tasks: `python3 tools/generate.py tennis`.

**Legacy code.** The legacy tennis tree is deleted; its last files (`scripts/olympiad/tennis/{match,duel}/environment/sim/`
`think_clock.py`, `think_proxy.py`, `duel/.../launch.py`, kept for the legacy drone tasks) were removed on 2026-10-10
after drone migrated.

**Fairness.** The contract asks for N = 200 reference duels (`[fairness] matches_per_order = 100`, review 2026-10-07).
`tools/check` stage 4 plays them with the 8-tile images in one process (20-40 min per duel on the shared host: about
100 hours), so it was not run at N = 200; `prototypes/tennis/fairness_run.py` (branch `archive/prototypes`, moved off
main on 2026-10-10, like the data files named below) plays the same duels (the same Reference,
both first-server orders, seeds 1-100) with perfect perception (the ball's true position projected into the two
documented cameras, no rendering), in two processes: N = 200, first server
-12.0 pp ± 13.8 (95% CI), seat A -4.0 pp ± 13.8: no interval lies entirely beyond ±5 pp (the usual realtime WARN: too
wide). The server wins 21 % of the points (the reference returns 88 % of the serves and the serving robot rarely
reaches the return); a match averages 9.9 points, at most 32. With the earlier 7-point cap the same duels gave
-17.0 ± 13.7 pp (the exact DP: a cap favours one serving order whenever the server's point rate is far from 1/2; "win by
2, no cap" has no first-server bias, so revision 8 + 6 dropped the cap, user decision 2026-10-08). Data:
`prototypes/tennis/results/fairness_v4_*.jsonl`, `fairness_v4_summary.json`.
**After the rebase onto #33 (revision wallclock-9 + duel-7, one scatter stream per robot; 2026-10-09):** the same
duels, perfect perception, seeds 1-50 x both first-server orders, N = 100 (three processes): first server -8.0 pp ±
19.5, seat A -8.0 pp ± 19.5: no interval lies entirely beyond ±5 pp (pass; WARN: too wide). The server wins 20.7 % of
the points (ace 7.0 %, first serve in 88.7 %, returns in play 87.6 %); 9.9 points per match. Data:
`prototypes/tennis/results/fairness_r35_*of3.jsonl`, `fairness_r35_summary.json` (an extension to seeds 51-100,
N = 200, `fairness_r35b_*`, runs after the push).
History (docs/benchmarks/tennis-player-serve-20261007.md, section V.7): with the first serve at a head speed of 3.0 m/s the
first server won +20-21 pp (N = 200, FAIL): the reference's forehand serve reaches 24-31 m/s from the right half but only
16-21 m/s from the left half (the arm's joint speeds along that swing), so the server won 88 % of the points served from
the right half and 43 % from the left, and the tie-break order hands the first server the strong half first. The first
serve is now 2.0 m/s (about 20 m/s from both halves).
