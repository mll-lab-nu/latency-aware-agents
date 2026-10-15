# badminton: robot singles badminton in real time

Migrated from the Season-1 tasks `badminton-match-wallclock-program-standard` and
`badminton-duel-wallclock-program-standard` (docs/framework_zh.md §10, stage 3). Since 2026-10-07 (revision
badminton-wallclock-4+duel-6) **the players' robots serve** by the BWF service laws instead of a serving machine / the
scripted opponent serving every point. Since revision wallclock-6 / duel-8 (2026-10-08) every undue delay of the
service is a service fault, never a disqualification (*Delay* below); since revision wallclock-7 / duel-9 (2026-10-08)
**each robot draws its own execution-scatter stream** (below). Since 2026-10-08 (revision **wallclock-8 / duel-10**,
user decision "option B": keep real physics, shrink the court) **every body obeys real-robot limits and the court is
small**:

- the racket's contact-law gain is **4** (was 20): a real racket sends the shuttle at about 1.5 x the head speed; 4 lets
  the Panda's best swing stand for that of a real fast badminton arm (ANYmal + DynaArm, 2025), D9;
- the arm is held to the **Franka Panda's FCI limits** (velocity, acceleration, jerk, torque, torque rate:
  `bodies/franka_panda`, `bodies/limits.py`, as fencing-3) through a 1 kHz setpoint generator and torque stage, with the
  base's commanded acceleration fed forward into the arm torques; the base is a **Festo EGC-185-TB-KF gantry** (5 m/s,
  50 m/s^2, 2500 N; the go-to generator also limits the jerk, 1250 m/s^3, and plans at 40 m/s^2); a low-level traj is
  refused beyond the gantry's speed and acceleration (D4, D9, E12);
- the **court is the BWF court x 0.4** (singles 2.072 x 5.36 m, short service lines 0.792 m from the net), the net keeps
  its BWF height 1.524 m (D8); the shuttle keeps the real drag (terminal speed 6.8 m/s): with gain 4 and the Panda's
  real head speeds (about 1.5-2.5 m/s along a forward face) it leaves the racket at 5-12 m/s and flies 2.5-5 m;
- the stroke controller is the legacy one, adapted to the limits: a lower ready pose (racket head 1.15 m), the IK
  solution nearest in rest-to-rest time under the acceleration limits, the swing within 90 % of the acceleration
  limits (a slower head when the time left does not allow the commanded one), the base counted arrived within 1.5 cm,
  the arm back to its ready pose at 90 % of its speed limits, a rehearsal score that weighs the head velocity;
- the reference player and the scripted opponent are re-tuned for the slow game (below).

Why, the numbers and the measurements: `docs/benchmarks/badminton-minicourt-20261008.md` (and the receivability
study before it, branch `badminton-receive-study`). The task names are kept: the format names are `match-wallclock`
and `duel-wallclock`.

**Self-collision** (revision **wallclock-9 / duel-11**, 2026-10-10; real limits B0, PR #54 on the library's
self-collision, #44). The arm, its racket and its stand collide: the scenes carry the pairs that
`bodies/selfcollision.py` generates (links two or more joints apart, the racket with the arm's links, the arm and the
racket with the 1.0 m pedestal and the platform), MuJoCo's contacts are on (they were switched off globally before),
and every physics step feeds `bodies/selfcollision.Monitor`: the final fields carry `self_collision` per seat
(events, warnings = events whose 10 ms mean force exceeds 20 N, time in contact, peak force, deepest penetration,
worst pairs). **Recorded only, no penalty.** With the contacts on, the reference player's low forehands drove the
forearm and the racket into the pedestal (up to 15 cm deep on the planned poses; in perception-free reference duels
about 9 events per match, 0.1-0.9 s in contact, peaks of 1-6 kN). The cause was the stroke controller's planning,
not the stance: for contacts below about 0.75 m the IK solution nearest in move time often folds the forearm down
beside the pedestal, and the joint-space swing, its follow-through and the return to the ready pose (the arm lagging
its targets under the Panda's acceleration and jerk limits) cut through it. What changed (`sim/bm.py`; no physics,
limit or stand changed):
- `Phys.clearance`: the least distance of the arm's self-collision pairs at a pose, from MuJoCo's own collision
  detection on a copy of the model whose pairs have a 5 cm margin (`mj_geomDistance` reads 0.0 for some mesh pairs
  centimetres apart in MuJoCo 3.3.7); pairs structurally closer than 4 cm in the ready pose (the wrist's link5 x link7,
  2.2 cm) are left out of the check (they still collide and are recorded);
- `ik_multi`: the hit poses at least 4 cm clear of the stand and links first (nearest in move time among them), the
  others after them by clearance; when no start gives a clear one, five more starts (forehand poses with the forearm
  held out from the pedestal);
- `plan_racket`: of the candidate swings, one whose rehearsals on the robot-only twin have no self-collision contact
  first (the twin carries the same pairs, so the stand blocks the rehearsed arm physically): the swing's correction
  passes (`_swing` keeps a contact-free pass first), then the rest of the motion from the kept pass, the
  follow-through and the return to the ready pose (`_rollout_contact`); when every candidate touches, the one with
  the fewest contact steps, and the stroke notes say `self-collision: ...`;
- the reference player (`builtin_players.Z_MIN`): contacts from 0.65 m (was 0.6): below that the swing rarely gets
  clear of the pedestal in the time left (and was rarely delivered).

**Independent scatter streams** (revision wallclock-7 / duel-9, user decision of 2026-10-08). Revisions duel-5 to
duel-8 drew ONE scatter sequence for both robots (the n-th act of each got the same draw), so a player acting later
could read its own coming deviation off the opponent's stroke and pre-compensate (the loophole the models used in
darts). Now the scatter of a player's n-th act is `bm.scatter_rng(seed, kind, stream, n)` =
`default_rng([seed, 1000, kind, stream, n])`: in the duel stream 1 for the first server's robot and 2 for the other's;
the duel is a pair of matches on the same seed with the first server swapped, so the two players swap the streams and
meet the same draws over the pair. In the match format the player's stream (1) starts with the tag 1000, which none of
the scripted opponent's draws does (its serve choice `[seed, 11, n]` and return choice `[seed, point, leg]`; the old
player key `[seed, 13, n]` coincided with the opponent's choices of point 13). The opponent itself has no execution
scatter (it sets the shuttle's velocity). Exploit probe and fairness: `docs/benchmarks/rng-duels-20261008.md`.

**Rules** (the constants are `sport.toml [spec]`): a Franka Panda arm (real limits) with a racket on an x-y gantry plays
singles on a small court (the BWF court x 0.4, the net at 1.524 m); the shuttle is a point mass with the real
shuttle's high quadratic drag (terminal speed 6.8 m/s), no spin and no bounce; a racket contact law of gain 4 sends it
along the face normal; rally scoring, one game: match-wallclock first to 11 (from 10-10 by 2, at most 15; since
revision wallclock-10, the screening's step 5), duel-wallclock first to 5 (from 4-4 by 2, at most 7); the match also
ends after 400 hits
or 40 points played. `match-wallclock`: seat A against a scripted opponent (levels 1-3, `RCB_LEVEL`);
`duel-wallclock`: two seats in mirrored frames. Deviations D1-D9 in `sport.toml`.

**The serve** (BWF Laws 9, 10, 14, 16; D2, D4, D6, D7):
- *Who serves.* Law 10.3: the winner of a rally serves the next point; a let keeps the server (14.3). The first
  server: duel = `FIRST_PLAYER` (`match.first_server`; the runner and the fairness check play both orders, there is no
  toss, Law 6); match = you (seat A) serve the first point.
- *Where.* Before every serve the referee puts both robots at rest in the ready pose at a fixed spot in their service
  courts, x = +0.12 (right court) or -0.12 (left court), y = -1.6 in each robot's own frame: both in their right courts
  when the server's score is even, else in their left ones (Laws 10.1, 9.1.3: diagonally opposite). The spot is the
  robot's home for the point. Standing still until the serve is delivered (9.1.4) is judged on the base: more than
  5 cm off the spot is that player's fault (both at once: a let, 14.2.2).
- *The toss (the missing free hand).* The arm holds the racket and has no hand for the shuttle. From 1 s after
  play resumes (the receiver is ready, Law 9.4) the server sends `{"type": "toss", "pos": [x, y, z], "height": H}`, the
  toss of all racket sports (docs/conventions/racket_serve_zh.md): when it takes effect (the act lead) the shuttle
  appears at that point and is released from rest (H = 0, the default) or rises H straight up (at most 0.5 m: the
  Laws ask only for no spin, 9.1.5, and do not forbid a small toss); the point must be 0.5-1.8 m high, within 1.0 m
  (horizontally) of the server's base and on its half (the reach of a hand; these bound the mechanism, not the
  serve). Until then the shuttle is held beside the server (drawn there). State: `you_serve`, `you_receive_serve`,
  `serve` = {`phase` waiting/open/tossed/done, `court` right/left, `opens_in_s`, `deadline_in_s`, `tossed_at_step`};
  points lost by a service fault say `serve_fault:<reason>: ...` (undue_delay, missed, too_high, not_upwards, net, out,
  moved), the receiver's early move `receive_fault:moved: ...`.
- *The hit.* The server hits the falling shuttle with an ordinary `act` (it may be sent before the toss, so the
  swing starts first). Judged: a miss is a fault (9.1.8); the shuttle must be below 1.15 m when hit (9.1.6; judged at
  the point mass's position) and leave the racket upwards (9.1: "the flight of the shuttle shall be upwards"); the
  serve must pass over the net (13.3.2; a serve into the net is now a fault, not a let) and land in the receiver's
  diagonal service court unless the receiver returns it (9.1, 13.3.1). Not judged (D2): hitting the base of the shuttle
  first and a continuous forward racket movement (9.1.5, 9.1.7).
- *Delay.* The server has 3 s (from the receiver being ready) to toss; after that it is a service fault (undue
  delay, 9.1.1): the receiver wins the rally (its point and the next serve). Every undue delay is such a fault, any
  number of times, and never a disqualification (user decision 2026-10-08, as in tennis: each delay is one fault; the
  umpire's warning and the referee's discretion of Laws 16.6-16.7 are replaced by this fixed rule, D6). A player who
  never serves gives back every rally it wins as receiver at its next serve, so it never leads by two points; it can
  still win only by winning the deciding rally at 6-6 or the last rally before the match limit.
- *The scripted opponent* (match-wallclock) serves when it won the last rally, as the legacy opponent served: a seeded
  high-deep or short serve hit at 1.05 m, upwards, into your diagonal service court, always within your reach (level 1:
  70 % high serves towards the middle of the court; levels 2-3: 50 % high, wider and tighter to the short service line);
  while you serve it stands still in its service court. It has no execution scatter (it sets the shuttle's velocity);
  its seeded choices are independent of the player's scatter stream (revision wallclock-7). Since wallclock-8 its
  depths and widths are the legacy ones x 0.4,
  its shots leave its racket at most at 18 m/s (`SHUTTLE_VMAX`), they leave your robot a contact at least 1.05 s after
  its hit, 0.7-1.7 m high, with 0.95 s for the swing after your base arrives (`REACH_*`; your arm needs that time under
  the Panda's limits). Since revision wallclock-10 (the screening, `docs/screening/badminton.md`) it is a robot
  whose placement matters: `reaction_s` after your hit it predicts the flight, its base moves at `chase` (2.0 / 2.2 /
  2.2 m/s) until the shuttle is within `reach` (0.5 / 0.5 / 0.75 m), and its arm then needs `swing` (0.55 s); the slack
  left at the best interception point decides: none is a winner for you, less than `window` (0.3 s) presses it (a weak
  lift when hard pressed, a normal return, an attack when it has time); after its own stroke its base stays put for
  `follow` (0.35 s) and then walks at `recover` (0.6 m/s) to where it waits: `anticipate` (70 %) of the way from home
  towards the mean landing point of your last 4 shots. Until wallclock-9 the position term never fired on the small
  court and every point was won by `min_t` (1.15 s at level 1): a player that always hit to the middle won 10 of 10.
  Levels: 1 (reaction 0.30 s, chase 2.0 m/s, mild attacks), 2 (reaction 0.25 s, chase 2.2 m/s, level 2's attacks:
  smashes up to 17 m/s, full width), 3 (as 2, reach 0.75 m); no `min_t`.
- *The reference player* serves too (`builtin_players.serve_plan`): contact at its forehand point at 1.05 m (where the
  stroke controller's stance is the base itself, so the base stays on its spot), a toss point (height 0) straight above it
  from which the shuttle falls to the contact in 0.35 s, a long (launch about 55 deg, to the back of the service court)
  or short (about 55 deg, just beyond the short service line) serve into the diagonal service court (60 % long; a seeded
  choice per point, the same for either seat), the stroke sent 1.3 s ahead (from rest the arm needs about 1 s to
  deliver a serving pose reliably under the Panda's limits), then the toss timed to meet it. On the small court the
  serve's contact is about 1.07 m behind the net, so a serve must leave at 40-75 deg to rise over it.
- *The reference player's rally* (wallclock-8): from the camera fit it scores a dozen contacts on the predicted flight
  (0.65-1.75 m high since wallclock-9, the base there in time) by the time left for the swing (a stroke planned at
  least 0.9 s ahead is delivered; under 0.85 s rarely), the head-speed margin (what the arm delivers in that time:
  about 2.2 t - 0.2 m/s) and the placement: a lofted shot (55 deg) deep to the corner away from the opponent first,
  against an opponent standing back a downward shot or a drop to the open side, then safer deep and mid shots;
  clearing the net by 0.4 m. It reads the stroke controller's rehearsal (`state.command.vel_rehearsed`,
  `pos_error_rehearsed_m`): a pose not delivered moves the contact later (once), a head speed off by more than 8 %
  scales the command (once).

**Actions.** The episode logs low-level commands only: `traj` (joint targets per control step, then the ready pose and
the base to its spot), `none` and `toss` (the serve's toss). The high level (the tasks' level) is the legacy racket command `act` with the legacy act
lead: an act (or `none`) takes effect `lead` steps after the step at which it arrives (lead = ceil(0.4 s / wall time per
step), 2 at x20). The plan worker restores the exact copy of the match (single) or of the seat's robot (duel) taken at
that step, advances it to the effect step with the command in force and plans the act there with the legacy stroke
controller (`wc.plan_on` / `bduel.plan_on`), as the legacy planner process did; the result is lowered to a `traj` with
its effect step `k0`, the planner's state key and the stroke's hit configuration, and takes effect at `k0` with
exactly the attribute changes of the legacy install. A traj whose state key differs from the live one at `k0` is
refused (counted in `rejected_acts` and `plan_mismatch`; never seen). The core's `delay_steps` hook is not used: it
counts the delay from the control-step boundary at which the act is lowered, which depends on how long the planner
took, while the legacy lead (and so the meaning of T) counts from the step at which the act arrived. A command sent
before the previous act took effect is refused (409); the legacy service replied only once an act had taken effect,
so it never had two in flight either. If the plan worker misses the effect step (a loaded host), the act is planned
again when it is lowered, from the state then, for an effect at that step and the same contact time (the world waits
meanwhile; the legacy service stalled the world until the plan came and kept the effect step).

**Observation.** The legacy state of each format, without the wall-clock fields (`wall_s`, `step_phase`) and with the
core's (`step`, `t_sim`, `world`, ...). The legacy single task's one-shot news (`shots_since_your_last_call`,
`points_finished`) is replaced by `last_points` / `last_shots` (the last three; an observation must not change
anything); the duel's `points` (last two) is `last_points` (last three). The images are the legacy 2 cameras x 4
frames (480 x 360 tiles, 60 ms apart, stamped `k=.. t=..`), drawn by the sport in the seat's render worker (cached
tiles, a faster PNG encoder). The legacy single task's privileged `POST /oracle` endpoint (true
shuttle state and a reference decision, oracle runs only) is gone: the reference player uses the cameras only.
Since revision wallclock-5 / duel-7 (core 0.2.9, renderer rule 6) the arms are drawn with the Franka Panda's visual
meshes: `presentation/presentation.toml` declares `players = "25k"` (the Menagerie meshes decimated to ~25k triangles
per arm, renderers/presentation/lod.json), tools/sportlib bundles them into the simulator (`sport/presentation/assets/`,
sha256-checked, licence kept) and `Views` draws the compiled players' scene (`renderers.players_model`, the same
kinematics checked) in both formats. Shuttle, court, net, pedestal and racket are unchanged (the reference player
sees the shuttle as before). `[observation] render_threads = 2`, `sim_cpus = 6`.

**Fairness** (revision duel-11, self-collision; 2026-10-10; the reference on both seats, seeds 1-20 x both first
servers, N = 40, perception-free as below): the first server won 19 of 40, **-2.5 +- 15.5 pp**; seat A won 25 of
40, **+12.5 +- 15.0 pp**; 2.65 hits per point (median 2, longest 10; 45 % of points with >= 3 hits, 15 % with >= 4).
**Pass** (no interval lies beyond +-5 pp; WARN: wider than +-5 pp at N = 40). Self-collision in these 40 matches: 8 events (all warnings) in 5 of the 80 robots,
0.81 s in contact in all, deepest 1.5 mm, all low forehands (0.65-0.68 m) for which every candidate swing touched the
pedestal in its rehearsal (the stroke notes say so); with the duel-10 planner and the contacts on: about 9 events per
match (54 in 6 matches, 4.5 s in contact, peaks of 1.3-2.8 kN).

**Fairness** (revision duel-10, the small court with real physics and one scatter stream per robot; 2026-10-08): the
reference on both seats, seeds 1-20 x both first servers, N = 40, perception-free (the true shuttle position at the
frames' times + 3 mm noise, the reference's own detection otherwise unchanged;
`docs/benchmarks/badminton-minicourt-20261008/fair_sum.py`, results `fair-r34`, run after the rebase onto revision
wallclock-7 / duel-9): the first server won 17 of 40, **-7.5 +- 15.3 pp**; seat A won 21 of 40, **+2.5 +- 15.5 pp**;
the server won 142 of 345 points (41 %); 2.8 hits per point. **Pass** (no interval lies beyond +-5 pp; WARN: wider
than +-5 pp at N = 40, as for every realtime sport). Before the rebase (one scatter sequence for both robots): first
server -10.0 +- 15.2 pp, seat 0.0 +- 15.5 pp. The earlier revisions' numbers (duel-4 FAIL by the first receiver,
duel-5, duel-6, duel-9 pass) are in the git history.

**B9 feasibility and rallies** (2026-10-08, `docs/benchmarks/badminton-minicourt-20261008.md`; scripts not part of the
sport). Re-run on revision wallclock-9 / duel-11 (self-collision; 2026-10-10, the same grid and upper-bound receiver,
reaction 0, each tried stroke played on through its follow-through with the receiver's self-collision counted):
**87 of 88 legal serves returned (98.9 %)**, all 87 by a stroke without a self-collision warning; the forced server's
serves touched the stand in 0 of 105 (with the contacts on but the duel-10 planner: 86 of 87, 98.9 %, the server's
serve touching its stand in 10 of 105; before the contacts, duel-10: 88 of 89).
- *The server's serves* (the reference's serve forced to the receivability study's grid scaled x 0.4, 105 targets in
  three launch-angle families of the small court): 100 legal: 41 short, 29 mid, 13 long-flat, 17 long-high
  (5.5-12.6 m/s). Both short and long serves exist; 5 flat serves to the front corners hit the net.
- *The upper-bound receiver* (true state, a search of up to 40 strokes through the official planner): 97 of 100 legal
  serves returned with an instant reaction, 88 of 100 with a 0.2 s reaction, 92 of 100 when it searches up to 100
  strokes (the misses: flat serves landing 1.7-2.3 m deep after 1.1 s, two short serves to the front corner; the BWF
  court with real physics: 71 %).
- *Basic shots exist* (upper bound): clear, drop, drive and a downward shot from contacts above 2.2 m.
- *Rallies:* informed players (true state, strokes searched) 4.7 hits per point; the reference (perception-free,
  N = 40, before the rebase onto the independent scatter streams) **2.7 hits per point** (median 2, longest 9; 46 % of
  points with >= 3 hits, 17 % with >= 4; after the rebase, the fairness run above: 2.8, median 2, longest 11, 47 % /
  19 %). The reference's limit is the arm's time: a stroke planned >= 0.9 s ahead is delivered, under 0.85 s rarely,
  and most shots leave the receiver 0.5-0.9 s after its reaction; points end by unreturned shots (40 %), the hitter's
  own side (29 %), the net (20 %), unreturned serves (8 %).
- *The match format* (perception-free): on revision wallclock-8/9 level 1 seeds 101, 111, 121: the reference won 3 of 3;
  level 2 seeds 202, 212, 222: lost 3 of 3. Since wallclock-10 (game to 11, the new levels; `docs/screening/badminton.md`,
  seeds 101-191): level 1 10 of 10 (point share 0.64), level 2 3 of 10 (0.47), level 3 see the report; every simple
  baseline loses every level. With the images only, the reference wins 1 of 10 at level 1 (51 % legal returns: it commits too late for the arm;
  a follow-up, the report); a model won level 1 with its own camera pipeline (14:12).
- *Re-run on 3473d9e7* (the screening, with self-collision): 87 of 88 legal serves returned (98.9 %), identical.

**Golden.** `golden/` holds matches recorded under the current rules with the built-in reference (tools/golden.py
record; each replays in a fresh process to its state hash, scores, winner and steps): `duel-s1-A`, `duel-s3-B` and
`match-L1-s1`, to be re-recorded on revision wallclock-11 / duel-13 (the screening: game to 11, the EGC-185-BS base,
no automatic return, the new opponent levels; on top of main's JerkLimiter servo). Before them: `duel-s1-A` (6-4,
k 5242), `duel-s3-B` (5-1, k 2539) and `match-L1-s1` (5-0, k 2472) on wallclock-10 / duel-12 (2026-10-12: the arm's
setpoint generator, bodies/limits.JerkLimiter, brakes within the jerk in force); `duel-s1-A` (5-0) and `duel-s3-B`
(5-1) on wallclock-9 / duel-11 (2026-10-10, self-collision) and `match-L1-s1` (5-0, recorded on wallclock-8). Before
those: `duel-s1-A` (4-6)
and `duel-s3-B` (2-5) on wallclock-8 / duel-10 (2026-10-08, after the rebase onto revision wallclock-7 / duel-9, the
independent scatter streams; the earlier records were played on the BWF court with gain 20 or with one scatter
sequence and cannot replay). Retired: the legacy recordings of the golden-data store (the Season-1 tasks' machine-serve matches,
converted by the former `golden/convert.py`) and the earlier `duel-*` / `match-L1-s1` cases: they were played under
other rules (a serving machine, the opponent serving every point, per-role scatter), so they cannot replay under the
current rules and were not kept; every future match is re-run (compatibility policy of 2026-10-07).

**Differences from the legacy tasks.**
- Reward keys: the common ones come from `core/grading.py`.
  Single: kept `won`, `sim_error`, `replay_error`, `meter_error`, `proxy_error`, `timing_invalid`, `points_for`,
  `points_against`, `replay_points_for`, `replay_points_against`, `replay_joint_targets_equal`, `finalized`,
  `revision` (now 8), `timing_lag_enforced`, `metered_*`, `rate_limit_retries`, `transient_api_errors`,
  `unrecovered_api_errors`, `level`, `slowdown`, `our_shots`, `good_shots`, `n_decisions`, `refused_not_model`,
  `dropped_acts`, `n_actions`, `sim_steps`, `steps`, `n_reviews`, `n_observe`, `n_act`, `elapsed_s`, `max_lag_s`,
  `sim_lag_late_max_s`, `sim_lag_late_p99_s`, `sim_effective_slowdown`, `sim_run_wall_s`, `plan_ms_max`,
  `plan_lead_steps`, `plan_mismatch` (now: acts re-planned late or refused for a state-key mismatch), plus
  `official_config`, `match_over`, `video_rendered`, `deterministic`, `missing_log`, `grader_error` from the core;
  dropped `refused_busy` (the core's HTTP layer refuses a second request in flight before the match sees it),
  `act_delay_steps_max` (the effect delay is the fixed lead, not recorded per act by the core), `act_rule_wait_max_s`,
  `act_rule_wait_mean_s`, `n_calls`, `n_wait` (not counted by the core), `sim_step_ms_max`, `plan_wait_ms_max`,
  `sim_lag_frac_late_over_1_period` (timings of the legacy service's own scheduler and planner); added
  `rejected_acts`, `match_limit`, `our_serves`, `our_service_faults`.
  Duel: kept `ready_forfeit`, `timing_invalid`, `steps`, `hits`, `slowdown`, `lead_steps`,
  `plan_mismatch`, `sim_lag_max_s`, `<seat>_points`, `<seat>_attempts|applied|refused|dropped|observations`,
  `<seat>_plan_s_max`, `<seat>_act_s_max`, `<seat>_obs_age_s_max` and the core's common keys; dropped
  `<seat>_refused_busy` (as above) and `plan_wait_max_s` (the world never waits for a plan: a late act is re-planned);
  added `<seat>_rejected_acts`, `match_limit`, `<seat>_serves`, `<seat>_service_faults`; `first_receiver_A` is now
  `first_server_A`.
- The default seed of `match-wallclock` is the generator's 1 (legacy: 101); `RCB_SEED` sets any seed (the official
  ones are 101/111/121 at level 1, 202/212/222 at level 2). The slowdown is `RCB_SLOWDOWN` (default 20, `[timing] slowdown`, as in the legacy tasks).
- The `game` shortcuts `game act T PX ...` and `game act none` are `game act JSON` and `game none`.
- A duel side that never starts loses (core 0.2.2, `no_show`); the legacy service voided that match. A single player
  who never starts loses (no_show), as in the legacy task.

**Base study** (revision wallclock-11 / duel-13, user decisions of 2026-10-11): the base is now a real **ball-screw
gantry, Festo EGC-185-BS-KF-40P** (spindle axes EGC-BS-KF technical data: 2.0 m/s, 15 m/s^2, feed force 3000 N, working
stroke up to 3000 mm, so |x| <= 1.5 m; jerk 25 x the acceleration, estimated), chosen over the EGC toothed-belt axis (5
m/s, 50 m/s^2) and mobile bases (RB-Y1 1.5 m/s returned only 40 of 88 legal serves in B9) in a study (`docs/screening/
badminton.md`). There is **no automatic return**: after a stroke the base stays where it struck; `move {base}` sets
where it waits and `act` may carry `base` (where it stands at contact). The scripted opponent obeys the same base
limits, plays to the side away from your base (`open_side`) and guarantees reachability only from the middle of your
court with a 0.45 s margin (was 0.95 s, which made it lift to the middle almost always). Both formats are a game to 11.

**Screening** (revision wallclock-10, 2026-10-10/11): `docs/screening/badminton.md`: the scripted opponent rebuilt
(placement-sensitive, reads your habits; levels 1-3), match-wallclock a game to 11, the skill contract `skill_api.py`
(`[modes.high]`), the aim-based legacy stroke removed.

**Opponent and reference, revision wallclock-11 / duel-13** (2026-10-12, user decisions BMOPP and BMRAND): the
scripted opponent reads a habit of your placement (4 of your last 5 changes of side alike, or the same spot three
times) and moves to the predicted spot at full speed, with its stroke prepared when it read right; it blocks a shuttle
it meets too late for a full stroke (0.45 s of arm time); it lifts over a base standing within 1.2 m of the net (its
clears pass above 2.8 m over your forward court); it attacks a high shuttle overhead; some of its attacks and normal
returns are kills (`kill`): the hardest of a few shots for your robot from where its base stands, without the reach
guarantee. Levels: L1 reaction 0.30 s, reach 0.85 m, kill 0.2, 0.45 s margin, mild attacks; L2 as L1 with level-2
attacks, kill 0.5, 0.25 s margin; L3 reaction 0.20 s, reach 0.9 m, kill 0.9, 0.25 s margin, recovery at 2 m/s (a
ceiling level). The reference plays net shots against a back opponent, mixes its sides (60 % to the open side) and
kills from contacts 1.9-2.2 m. A bug of the interception choice (it could take a point it would miss while a later
one was makeable, since wallclock-10) is fixed. The match ends after 400 hits (was 300). Exploit E14: a flat short
serve is returned at every level.

Run everything: `PYTHON=python3.12 tools/check badminton`. Generate its tasks: `python3 tools/generate.py badminton`.
