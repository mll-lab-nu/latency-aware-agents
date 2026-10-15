# sports/tabletennis

Robot table tennis on the v0.2 framework (docs/framework_zh.md), migrated from the legacy tree of Season 1
(`scripts/olympiad/tabletennis`, `scripts/olympiad/shared/tabletennis`). Real time (`eai-standard-realtime/1.0`): the
world runs at 1/S of real speed and freezes after every point for the reviews.

| | |
|---|---|
| Formats | `match`: one seat against the scripted opponent (level 1-4); `duel`: two seats. The players serve (ITTF 2.6, see "The service"): 2 each, from 4:4 one each; match: you serve first; duel: `first` = the first server |
| Action levels | `high`: the paddle state at contact (+ optional base position), the service plans the stroke and moves the base, `move` sets the base's rest position; `low`: targets of the 9 joints [base_x, base_y, q1..q7] at 100 Hz (traj / slew / hold); `toss` at both levels |
| Plays | `program`, `direct` (no code: one act per model response, the players' no-code audit) |
| Tasks | `tools/generate.py tabletennis` writes the six `tasks/tabletennis-{match,duel}-wallclock-{program,program-lowlevel,direct}-standard` (the legacy names, `[[tasks]] id`) |
| Robot | Franka Panda + paddle on an x-y rail base behind the end line (revision wallclock-7 / duel-wallclock-5, see "The rail base"), both held to the real robot's limits since revision wallclock-10 / duel-wallclock-8 (see "The real limits in the sport") |
| Physics | MuJoCo; ball, table, net and paddle contact unchanged from Season 1; the arm and the rail run on the real-limits servo (`rcb_tt.TTServo`) |
| Capabilities | a1-a4 = 2, b1 = 1, b2 = 2, b3 = 0; physics `spin_bounce` |
| Slowdown | **S = 10** (`[timing] slowdown`, the generated tasks' `RCB_SLOWDOWN` default; user decision 2026-10-14: table tennis plays official matches at x10, was 20; badminton stays at x20 this round). The graded official slowdowns are unchanged (`grade.py`: match program 1, 2, 5, 10, 20; duel program 10, 20) |

## Files

| File | What |
|---|---|
| `sport.toml` | the declarations: rules and deviations, formats, modes, timing, whitelist, rule constants, tiebreaks, exploits, players, fairness, official configurations, tasks |
| `plugin.py` | contract ②: `TTMatch` (arm_commands.Core), `TTDuel` (duel.Duel), `StrokeLayer` (high), `JointLayer` (low), the cameras (`MatchView`, `DuelView`: the legacy cameras and tiles, with a tile cache; the arms drawn with the players' presentation scene) |
| `sim/` | the simulator: rcb_tt (physics, the real-limits servo `TTServo` (on the library's JerkLimiter and TorqueRateLimiter), the rail's limiter and go-to generator, the row checks), rally (ball model, swing helpers, the limits-aware stroke rows), match, arm_commands, action_high, action_low, levels, ladder, duel, workers, ipc, the scenes and data (`serve_library.json`: the official serve library, see "The real limits in the sport") |
| `../../bodies/limits.py`, `../../bodies/trajopt.py` | the real limits and their enforcement (`JerkLimiter`, `TorqueRateLimiter`), the limits-aware stroke planner; sport.toml `[arm]` copies them into the simulator (`<sim>/bodies/`) |
| `builtin_players.py` | `Reference` (the legacy reference player: wc_player + arm + the oracle loops, camera-only, both levels and formats), `SkillReference` (the low tasks' oracle: the Reference's decisions sent as `hit` calls, the skill in the loop) and `Idle` (never acts; the check policy) |
| `grade.py` | the sport's reward keys (the legacy keys, see below) |
| `robot_model.py` | the robot's own model (`robot_only.xml`) cut out of the scene, given at the low level in `/spec` |
| `tests/` | one test per exploit |
| `golden/` | four cases recorded under the current rules (revision 10/8: real limits, player serves), see "Golden replays" |
| `sim/serve.py` | the service: the order of serving (2.13.3), the toss limits, `ServeJudge` (2.6.2-2.6.4, 2.9.1.1, 2.10.1.6) |
| `prototypes/tabletennis_serve/` (branch `archive/prototypes`) | the serve study (not part of the sport; moved off main 2026-10-10): `servelab.py`, `validate.py`, `receivegrid.py`, `render_serve.py`, `build_library.py` (makes `sim/serve_library.json`), `opp_levels.py`, `fairness.py`, results |

## The rail base (embodiment change, 2026-10-07, deviation D9)

The robot used to stand on a fixed base at (0, -1.75). Since revision wallclock-7 (duel-wallclock-5) its pedestal
rides on an x-y rail base, like tennis (x-y rails, 4 m/s, 8 m/s^2) and badminton (x-y sliding base, 6 m/s, 20 m/s^2):

| | value | why |
|---|---|---|
| joints | `base_x`, `base_y` (prismatic, joint value = the base's position in the robot's frame), then q1..q7 | the low level's rows are `[base_x, base_y, q1..q7]`, as in tennis |
| travel | \|x\| <= 1.0 m; y from -1.50 to -2.40 m (0.13-1.03 m behind the end line), home (0, -1.75) | covers the whole 1.525 m table width with the 0.85 m arm reach; the column (radius 0.12 m) stops 1 cm behind the end line, so the robot cannot lean over the table (a human can) |
| speed | 2 m/s per axis | table-tennis footwork is short (0.3-1 m within a 0.6-0.9 s ball flight); the prototype measured the same return rate at 2 and 4 m/s with 8 m/s^2 (94 % at 4.4 m/s); a belt-driven linear gantry reaches this easily |
| acceleration | 8 m/s^2 (the drives' 620 N force limit for the 76.8 kg moving mass) | the binding limit for short moves: 0.1 / 0.3 / 0.5 / 1.0 m in 0.26 / 0.45 / 0.57 / 0.83 s rest to rest (the go-to generator plans at 85 % of the limits); real wheeled humanoids (1.5 m/s, ~2.5 m/s^2) returned only 67 % of 6 m/s balls in the prototype |
| control | the same PD controller as the arm (kp 6e4, kd 4e3), the speed checked on the targets like a joint, the acceleration bounded physically | tennis' scheme |
| high level | `act` with an optional `base` (the base position at contact; default: move only as much as needed, `rcb_tt.stance`), `move` (the rest position, home at each point) | tennis' interface |

Arm, paddle, ball, table and contact physics are unchanged. The column and the carriage are drawn only (no collision),
as the old pedestal was. The prototype study is on branch `tt-proto` (docs/benchmarks/tabletennis-prototypes-20261007.md).

## The service (players serve, 2026-10-07, review item C2)

The players' robots serve; there is no serve machine any more. The only constraints are the ITTF Laws (2.6 the service,
2.9.1.1 let, 2.10.1.1 a failed service, 2.10.1.6 obstruction, 2.13.3 the order of serving), scaled to our game, and the
stated deviations. No landing-zone or "must bounce out" limits: every serve that is legal under the Laws is allowed.

| Law | here |
|---|---|
| 2.6.1 the ball rests on the open palm, stationary | no free hand (D2): `toss {pos, height}` (the racket sports' common form) puts the ball at rest at `pos` on a toss device for 0.1 s; until then it waits out of sight in the server's column |
| 2.6.2 projected near vertically upwards, without spin, rising at least 16 cm; it falls without touching anything before being struck | projected straight up, no spin, to rise `height` m above `pos` (0.16-1.0; the speed is solved on the simulator's own ball model, `serve.vz_for_height`: 16 cm = 1.79 m/s); any contact before the strike (the arm, the paddle while it rests or rises) is a fault |
| 2.6.3 struck as it falls; first the server's court, then directly the receiver's | `ServeJudge`: the strike needs a falling ball; the next contacts must be the server's court (top surface) and then the receiver's |
| 2.6.4 above the playing surface and behind the end line until struck | checked at every 1 ms physics step on the whole ball (centre z >= 0.02, y <= -1.39 in the server's frame); a dropped toss is a fault. The toss point must satisfy it too (else refused, 400), and lie within 0.9 m of the base, at most 0.9 m above the table (the device's reach). Not judged: hidden from the receiver |
| 2.6.5 the free arm out of the way | no free arm |
| 2.9.1.1 a service touching the net assembly, otherwise correct (or obstructed by the receiver) | let, at most 2 in a row (D12, since revision 11/9): the server's third service let since the last point scored is a failed service (`serve_fault:repeated_let`); see "The let limit" |
| 2.10.1.6 obstruction | the receiver's robot touching the service before it bounces on its court, while it is over or travelling towards that court, loses the point |
| 2.13.3 change after every 2 points; from 10:10 after every point | after every 2 points, from 4:4 after every point (lets do not count) |
| 2.13.1 lot for the first service | D11: match - you serve first; duel - the runner plays both first servers |
| undue delay (chapter 3, warnings and penalty points) | D10: toss within 5 s (game time) of the serve opening, else a failed service; the third time loses the match |
| state (common with tennis and badminton) | `you_serve`, `you_receive_serve`, `serve` = {`phase` waiting / open / tossed / done, `court` "any" (singles), `opens_in_s`, `deadline_in_s`, `tossed_at_step`, extras: `server`, `detail`, the server's `delay_faults`, `lets_in_row` / `let_limit` and own `toss`}; service faults are recorded as `serve_fault:<reason>` |

Who serves when: match - points 1-2 you, 3-4 the opponent, ..., from 4:4 alternately; duel - the same with `first` as
the first server. The scripted opponent (format match) serves its turns: a toss of 19.8 cm from its end, then a
recorded real serve (DeepMind, the 92 of 109 whose recorded start is behind its end line) struck as the toss falls back
through its start point; level 1 draws from the easiest 40 % (by speed and spin), level 2 60 %, level 3 80 %, level 4
all. It receives your service like any of your returns (its pressure rule reads only how wide and fast the ball is).

Information: the observation never holds ball coordinates; the server sees its own toss (`serve.toss`), the receiver
only the serve phase (and the cameras). The expedite rule's 13-return limit stays on from the first point, but the
service still changes every 2 points (D6).

## Table contact, table avoidance, the opponent's receive (2026-10-07, user decisions after the first serve study)

- **D4 relaxed to the Laws**: the robot or its paddle touching the playing surface (or the floor) is no longer a fault.
  Only touching the net assembly loses the point (2.10.1.10). 2.10.1.9 (moving the playing surface) cannot happen
  (the table is fixed in the simulation); 2.10.1.11 (the free hand) does not apply to a one-arm robot. Physics
  unchanged: the table still stops a robot or paddle that presses on it.
- **Table avoidance in the high-level planner** (match `plan_paddle`, duel `workers.prepare/trajectory`): the planned
  swing is checked kinematically against the table, net and floor (`rally.touches_table`, every 2nd row up to the
  contact); if the direct cubic swing would pass through them, the planner goes first to a pre-contact pose 0.10 m
  (else 0.18 m) above and 0.10 m behind the contact and comes down onto it over the last 0.12 s (reply field
  `approach`). This changes some strokes and serves (low contacts over or near the table); the reference serve
  repertoire was re-validated.
- **The scripted opponent's receive of your service, recalibrated** (levels monotone): it moves at `chase_serve`
  0.75 / 0.90 / 1.05 / 1.20 m/s (L1-L4; 0.47-0.55 for rally balls), so a wide service is no automatic winner, and a
  short service adds a pressure term from 0 where it meets the ball 1.25 m from the net to 1 at `short_y1` 0.80 /
  0.70 / 0.60 / 0.50 m (beyond it: a miss, "too short"; above 0.75: a weak return). Rally balls are judged as before.
  Measured (`opp_levels.py`, the reference, seeds 1-3 per level, `results/opp_levels_v2.jsonl`): the opponent returned
  3/12, 7/9, 9/9, 12/12 of the reference's services at L1-L4 (missed as too short: 9, 2, 0, 0); the reference won all
  12 matches, the opponent took 3, 0, 0, 4 points.

## Serve study (2026-10-07, `prototypes/tabletennis_serve/`)

The study's code and results are on branch `archive/prototypes` (moved off main on 2026-10-10); the paths below are
paths in that branch.

All runs: the real referee (`serve.ServeJudge`), the high level unless noted, execution scatter on.

- **What the arm can serve.** Grid of 936 serve recipes (`servelab.py`: contact beside / in front of the base, strike
  height 0.15-0.35 m, second bounce 0.35-1.1 m into the receiver's court, blade speed cap 2.6 / 3.2 m/s, brush for
  spin), 594 with a flight solution: 135 good, 24 lets, 435 faults (246 bounced twice on the server's own court - the
  arm cannot hit hard enough; 150 the paddle touched the table striking a low ball; 36 the swing touched the toss
  first). Legal serves: 3.1-4.7 m/s (median 3.4), spin 0.4-33 rad/s (median 10, mostly light topspin); 131 of 135 are
  short (they would bounce twice on the receiver's court), 4 long. Tosses to the right of the base: 0 of 36 legal (the
  arm meets the toss). The blade speed the arm delivers at a serving contact is 1.0-2.6 m/s (best 0.4 m to the side
  and 0.35-0.45 m in front of the base, swinging forward-up). Speed caps are not needed: the bounce order of Law 2.6.3
  and the arm limit the serve.
- **The reference's repertoire** (`builtin_players.SERVES`, six short serves with 3.6-10.6 rad/s of topspin, aimed
  wide, central and short), with the toss's height interface (revision 9/7, `validate.py`, 12 seeds each, the
  reference player's own code): 72 of 72 good at the low level (`results/validate_low_height.json`); 69 of 72 at the
  high level (`results/validate_high.json`: 1 let, 2 faults - the service came down on the server's own court
  again, both from the wide topspin serves).
- **Receiving player serves** (`receivegrid.py`: one duel point per serve, 36 recipes from the grid, the camera-only
  reference receiving): with the mobile base 7 of 26 legal serves returned (27 %; failures: 18 the arm or paddle
  touched the table reaching over it, 1 a second bounce); the same receiver with its base held at home returned 0 of 25
  (19 bounced twice, 5 table contacts). The prototype's numbers (fixed arm, other serves, perfect perception): 4 of
  47 returned (fixed), 12 of 47 (base up to y = -1.55), 30 of 47 (base under the table). Short player serves remain
  nearly unreturnable for an arm whose base stops behind the end line and that might not touch the table (the old D4, D9);
  nothing restricts them (the user's decision). After D4 was relaxed and with table avoidance (revision 9/7,
  `results/receivegrid_mobile_v2.json`): 19 of 28 legal serves returned (68 %); the 9 unreturned: 7 returns into the
  receiver's own half, 1 second bounce, 1 out. Unreturnable serves are reported, not forbidden.
- **Matches** (stage 4's matches, `fairness.py`, `results/fairness_A.jsonl`, `results/fairness_B.jsonl`: format duel,
  the reference on both seats, high level, seeds 1-20 per first server, N = 40, revision 9/7): first-server advantage
  +5.0 pp (22/40), seat A advantage 0.0 pp (20/40), both with a 95% CI of +-15.5 pp (no significant bias). The server
  won 184 of 321 points (57.3 %, 95% CI 51.9-62.6 %); 320 of 321 services legal (1 fault: the second bounce missed the
  receiver's court), no lets. Games: 8.0 points on average, 10 of 40 reached 4:4, 2 were decided at 6:6 (one won by
  the first server). The 6:6 cap, exact (both players alike, the server wins a point with p): first-server bias
  +0.5 pp at the measured p = 0.573 (+2.0 pp at 0.7, +5.0 at 0.8, +13.9 at 0.9); "lead by 2, no cap" has none, and
  with the 40-point last resort it stays below 0.1 pp up to p = 0.8 (the limit is reached with probability 6e-6 at
  p = 0.573, 8e-4 at 0.8; 2 % and -1.1 pp at 0.9), costing 0.2 points per game on average at the measured p.
  The cap is unchanged (the user's decision).

## Real arm limits: a measurement (2026-10-07, before they were applied; superseded by revision wallclock-10)

The question: with the real Franka Panda's limits (`bodies/limits.py`, `docs/benchmarks/roboolympics-arm-limits-2026-10-07.md`)
can serves become unreturnable? `prototypes/tabletennis_serve/reallimits/sitecustomize.py` (RCB_TT_REAL_LIMITS=1, every
process incl. the duel's plan workers and the planner's rehearsals) swaps the arms' servo for fencing-3's: JerkLimiter
setpoint (velocity, acceleration, jerk at 100 %, jerk capped by the torque-rate share), inertia feed-forward + PD +
bias, TorqueRateLimiter (87 / 12 N m, 1000 N m/s), re-sync. The rail base and everything else are unchanged; the
planners are NOT acceleration- or jerk-aware (phase 2), so this measures today's reference and recipes on a limited
arm. Setpoints: acceleration reaches the limit (binds), velocity <= 0.999 of it; the actual joint acceleration still
exceeds the limit in 25 % of physics steps (servo and contacts, as in fencing-3).

| | unlimited (today) | real limits, both arms |
|---|---|---|
| serve grid, 1728 recipes (`servelab.py`, high level; `results/servelab_grid_v3_*.json`) | 280 good (43 lets, 805 faults) | 170 good (32 lets, 926 faults) |
| good serves: ball speed min / median / max (m/s) | 2.99 / 3.41 / 4.91 (21 at >= 4 m/s) | 2.92 / 3.30 / 3.92 (none at 4) |
| good serves: spin median / p90 / max (rad/s) | 10.8 / 25.2 / 37.8 | 9.7 / 31.7 / 47.3 |
| long serves (second bounce aimed >= 1.0 m deep) good | 18 | 7 |
| the reference's repertoire, 12 seeds (`validate.py`) | 69/72 good (1 let, 2 faults) | 67/72 good (5 faults) |
| receive grid, 36 recipes, mobile base (`receivegrid.py`) | 28 legal, 19 returned (68 %) | 18 legal, 2 returned (11 %): 10 second bounces (late), 3 into its own half |
| the 17 recipes legal in both | 9 returned | 2 returned |

Reading: the limited arm serves slower (max 3.9 m/s against 4.9) and fewer of its recipes stay legal, but the receiver
loses far more: its swing, planned for the velocity limit only, arrives late. With real limits and today's planner
most legal serves are unreturnable; how much an acceleration-aware planner recovers is phase 2 (servo, planner, the
low level's acceleration check, opponent recalibration, golden matches and fairness; not a trivial change).

## Limits-aware stroke planner (branch tt-planner, 2026-10-08: measured before the limits were applied; the sport runs it since revision wallclock-10)

The real-limits measurement above showed the reference collapsing (11 % of legal serves returned) because its swing
was planned for the velocity limit only. tt-planner replaces it (high level; `RCB_TT_PLANNER=hermite` restores it):

- **bodies/trajopt.py** (shared, numpy only, tests `bodies/tests/test_trajopt.py`): fixed-time jerk-limited joint
  trajectories: per joint a least-squared-jerk QP over piecewise-constant jerk at 10 ms with hard velocity /
  acceleration / jerk boxes (95 % of the Panda's, the jerk also capped by the torque rate at the start, middle and
  contact poses, `limits.jerk_caps`) and the joint ranges, soft waypoints (ADMM, adaptive rho, cached
  factorisation; 10-60 ms per plan). `stroke()`: an anticipation move from the current state to a wind-up pose at
  rest, the stroke started as late as the slowest joint's ramp time allows, the contact (q, qd) at the contact step
  (through the table-avoidance pose if needed), a follow-through to rest; with too little time the best reachable
  contact (position first for returns, velocity first for services). Infeasible requests are reported, not hidden.
- **TT** (`rally.stroke_rows`, match `plan_paddle`, duel `workers.prepare / trajectory`): the contact state from the
  IK with continuity (a re-plan of the same ball keeps the last contact pose's branch; joint travel judged under the
  acceleration limit); a re-plan the arm can no longer follow is not taken if the trajectory in force predicts a
  better paddle state at the new contact step (`workers.keep_better`: position + 0.3 s x velocity error). Fixed on
  the way: the arm inertia for the jerk caps needs `mj_makeM` (MuJoCo 3.3: `mj_crb` no longer fills qM).
- **Reference player**: anticipation of a service from its first frames (before its first bounce), so the base and
  arm start ~0.4 s earlier; the service stroke is sent once before the toss (the wind-up starts early); a ball's
  stroke choice is kept across re-plans (no flips between aims and speeds).

Measured (`prototypes/tabletennis_serve/planner_report.py`, results in `results/planner/`; old = tt-serve's planner
and player, new = tt-planner's; ON = both arms on the real-limits servo of `reallimits/`, OFF = the sport's PD):

| | old OFF | old ON | new OFF | new ON |
|---|---|---|---|---|
| receive grid, 36 recipes: legal / returned | 28 / 19 (68 %) | 18 / 2 (11 %) | 29 / 18 (62 %) | 29 / 15 (52 %) |
| reference repertoire, 72 services | 69 good | 67 good | 72 good | 72 good |
| serve grid, 1728 recipes: good; max speed | 280; 4.91 m/s | 170; 3.92 | 240; 4.94 | 194; 4.45 |
| reference v scripted L1 / L2 / L3 / L4 (3 seeds each) | 3 / 3 / 3 / 3 won | 0 / 0 / 0 / 0 | 3 / 3 / 3 / 1 | 2 / 3 / 0 / 0 |
| duels (8): hits per point (serve = 1) | 3.76 | 1.38 | 3.49 | 2.25 |
| duels: server won / services legal | 37/63, 63/63 | 80/84, 84/84 | 48/78, 76/84 | 41/69, 66/70 |

Real-limits servo, actual joints: the setpoints stay within every limit (acceleration and velocity at most 1.000),
but the joints' actual acceleration exceeds the limit in 30-50 % of the physics steps of matches (10 % in isolated
services; velocity 1.6 %): PD vibration under the torque-rate-limited servo and contacts (fencing-3 saw the same
at a lower rate); a phase-2 servo needs its gains tuned against it.

Not yet done (phase 2, needs a decision): the servo in the sport (`sport.toml [arm]`), the low level (its rows and
the low-level reference's own swing), recalibrating the scripted opponent (the limited reference loses L3/L4), the
remaining gap in returns (the shoulder and elbow's torque-rate jerk, about 230-400 rad/s^3, cannot add upward
paddle speed late: most failed returns hit the receiver's own half), golden matches, fairness, tasks.

## The real limits in the sport (revision wallclock-10 / duel-wallclock-8, 2026-10-09)

Principle B0 (user, 2026-10-09): every body obeys the real robot's limits - velocity, acceleration, jerk, torque,
torque rate, the base too - never unlimited; B9: an upper-bound receiver must return >= 90 % of the legal serves. The
measurements above (tt-serve, tt-planner) are now the sport:

| | what | where |
|---|---|---|
| arm servo | fencing-3's servo on both robots, every 1 ms: a JerkLimiter setpoint within the Panda's joint velocity, acceleration and jerk limits (the jerk also capped by the torque-rate share at the pose, `limits.jerk_caps`), torque = M qdd_sp + PD + gravity/Coriolis, then the TorqueRateLimiter (87 / 12 N m, 1000 N m/s); a joint pushed > 0.02 rad from its setpoint re-syncs it; a teleported robot (reset, the duel's ready pose, a rehearsal) starts from its actual state | `rcb_tt.TTServo` (match, duel, the planners' rehearsal twins; the twin copies the live servo state) |
| rail | per axis 2 m/s (DeepMind's Festo gantry), 8 m/s^2 (the drives' 620 N on 76.8 kg), 200 m/s^3 (ESTIMATED: no manufacturer value; 25 x the acceleration, as the receive study): a jerk-limited setpoint (`RailLimiter`, its position loop brakes on the jerk-limited stopping distance; the arm's cascade oscillates at 200 m/s^3) + PD + bias, clipped at 620 N; the go-to generator (`base_goto`) is jerk-limited too (0.1 / 0.23 / 0.5 / 1.0 m rest to rest in 0.33 / 0.47 / 0.67 / 0.94 s) | `rcb_tt.RailLimiter`, `rcb_tt.base_goto` |
| low-level checks | a `traj` is refused (400) if consecutive rows break the speed limit or three rows the acceleration limit (second difference <= acc x (10 ms)^2: 8 m/s^2 per base axis, 15 / 7.5 / 10 / 12.5 / 15 / 20 / 20 rad/s^2); its first row is rejected if it does so with the target in force and its velocity; jerk is left to the servo; `/spec` limits gives every limit and the servo's law | `action_low.validate`, `rcb_tt.rows_problem`, `arm_commands.submit`, `duel.install` |
| slews, rest | every slew (the high level's return to the ready pose, the low level's `slew`) is acceleration-limited per joint and brakes before the joint ranges | `rcb_tt.seek` |
| high level | the limits-aware stroke planner (bodies/trajopt via `rally.stroke_rows`), with a last guard that makes every lowered row pass the low level's checks (E5) | `rally.stroke_rows`, `rcb_tt.limit_rows` |
| scripted opponent | obeys the same limits: its paddle moves along its end line on a jerk-limited setpoint within the rail's limits; its shots stay within the limited arm's envelope (<= 5.5 m/s off the paddle, <= 34 rad/s, blade speed <= 2 m/s by the contact law); it serves from the official serve library (below) instead of recorded human serves (3.9-6.4 m/s, up to 97 rad/s); levels recalibrated (below) | `match.Match` (`_opp_step`, `_within_limits`, `OPP_*`), `rcb_tt.OPP_LIM` |
| serve library | `sim/serve_library.json`: the 279 legal services (of the receive study's 284; 5 dropped whose reconstructed strike point lies in front of the end line) that a Franka on the rail made under the real limits (the serve grid + the receive study's extension), stored as the strike state, ordered by difficulty for the receiver (how low the ball's apex after its bounce is, then spin) | `prototypes/tabletennis_serve/build_library.py` |
| reference player | high level: waits for the ready pose and the serve stance before serving; receives services from a stance 0.20 m forward (0, -1.55); low level: a limits-aware stroke of its own (a bang-bang move to a wind-up pose at rest, a constant-acceleration stroke at 90 % of each joint's limit, the same deceleration after the contact), every row within the speed and acceleration checks, the command in force read from the observation (a refused or rejected command never took effect), the repertoire's three serves that are legal at the low level | `builtin_players.Reference`, `Arm.stroke` |

Opponent levels (recalibrated; every value monotone in the level; revision wallclock-13 changed the short limit at
every level and L1 / L2's reaction, chase and reach, see "Screening 2026-10-11"; the old values in brackets):

| level | reaction s | chase / recover m/s | reach m | serve m/s | short limit m | width m | pace m/s | spin rad/s | sidespin | serve library share |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.34 (0.30) | 0.36 (0.42) / 0.30 | 0.20 (0.24) | 0.65 | 0.62 (0.85) | 0.12 | 3.9 | 8 | 0 | easiest 25 % |
| 2 | 0.31 (0.28) | 0.40 (0.45) / 0.32 | 0.23 (0.26) | 0.72 | 0.58 (0.80) | 0.15 | 4.0 | 10 | 0 | 40 % |
| 3 | 0.27 | 0.46 / 0.33 | 0.27 | 0.78 | 0.54 (0.76) | 0.18 | 4.1 | 12 | 0.1 | 55 % |
| 4 | 0.24 | 0.52 / 0.36 | 0.31 | 1.00 | 0.50 (0.62) | 0.26 | 4.6 | 25 | 0.5 | all |

Measured (revision wallclock-10 / duel-wallclock-8, the sport's image, in-process; scripts and results in
`prototypes/tabletennis_serve/reallimits/`, `results/reallimits/`):

- **The reference against the scripted opponent** (format match, high level, seeds 1-5 per level,
  `reallimits/levels.py`, `results/reallimits/levels_v10.jsonl`):

  | level | won | games (you : opponent) | points | services of the opponent returned | your rally returns | your services legal |
  |---|---|---|---|---|---|---|
  | 1 | **5/5** | 5:2, 7:6, 5:0, 7:6, 5:0 | 29 : 14 | 8/19 | 10/11 | 22/24 |
  | 2 | **5/5** | 5:0, 5:1, 6:4, 5:3, 5:1 | 26 : 9 | 8/15 | 13/14 | 19/20 |
  | 3 | **2/5** | 5:2, 4:6, 5:3, 3:5, 6:7 | 23 : 23 | 8/22 | 17/21 | 19/25 |
  | 4 | **1/5** | 2:5, 4:6, 5:1, 4:6, 1:5 | 16 : 23 | 10/17 | 13/28 | 21/23 |

  Under tt-planner's measurement (the limits on, the old levels) the reference won L1 2/3, L2 3/3, L3 0/3, L4 0/3.
  The reference's weak spot is the service receive (8-10 of 15-22 per level returned): the library's services are
  slow, short and low; a stance 0.20 m forward raised its receive from 15/32 to 23/37 in the receive test
  (`reallimits/receive_test.py`, 40 library serves; `results/reallimits/receive_*.jsonl`), the other variants tried
  (contacts higher, the base at the front limit, no late contact, a lower blade-speed cap) did not help.
- **The envelope the opponent is held to** (the same 20 matches, every shot of the limited reference): ball speed off
  the paddle median 3.6, p95 5.3, max 6.3 m/s (227 shots); blade speed at the contact median 1.2, p95 2.1, max 3.1 m/s;
  spin after the contact median 20, p95 52 rad/s. The opponent's caps (5.5 m/s, 2.0 m/s of blade speed, 34 rad/s)
  lie at or below these; its 74 shots in these matches: pace <= 4.7 m/s, spin <= 23 rad/s.
- **Your services** (the reference's repertoire, in matches): 81 of 92 legal (2 lets). It serves only once its base
  stands at the serve stance and its arm rests in the ready pose (from a moving arm the same recipes faulted in about
  1 of 5 serves: `reallimits/serve_test.py`, every point served, 19/21 legal after the change).
- **Fairness** (stage 4's matches, format duel, the reference on both seats, high level, seeds 1-20 per first server,
  N = 40; `fairness.py`, `results/reallimits/fairness_*.jsonl`): first-server advantage 0.0 pp (20/40), seat A 0.0 pp (20/40), both with
  a 95 % CI of +-15.5 pp (no bias; realtime sports only warn on the wide CI). The server won 147 of 324 points
  (45.4 %); 309 of 337 services legal (13 lets, 15 faults); the 6:6 cap unchanged.
- **B9 recheck** (an upper-bound receiver must return >= 90 % of the legal serves): the receive study's full-simulation check (`fullsim.py` of branch
  tt-receive-study 329ff155, unchanged, run on this revision: the sport's own real-limits servo and rail, the duel's
  high-level path; the 284 legal serves of the library; `results/reallimits/b9_fullsim_*.json`). Returned from the
  best fixed stance **94.7 %** (near_bh (-0.25, -1.55): 269/284; near_fh 93.3 %), from any stance 97.5 % - **passes**
  (before, with the unlimited rail and the measurement servo: 94.0 % / 97.2 %). From home (0, -1.75) only 77.8 %
  (93.0 % before): the jerk-limited rail reaches the short serves later, so a receiver should wait 0.20 m forward (the
  reference does).
- **The arm against its limits** (`tools/arm_limits.py measure tabletennis` on the golden cases match-high-s101 and
  duel-high-s1-A, 51990 physics steps; `results/reallimits/arm_limits_golden.json`): the servo's commanded setpoints
  never exceed the limits (velocity max/limit 0.999, acceleration 1.000, jerk 0.733); the actual joints: velocity
  p99/limit <= 1.00, torque max/limit <= 1.00, torque rate p99/limit 1.00; the actual acceleration exceeds the limit in
  0.5-12.7 % of the 1 ms samples (p99/limit up to 1.43: PD vibration under the torque-rate-limited servo and contacts,
  as in fencing-3; tt-planner's measurement servo saw 30-50 %). The extreme maxima are the duel's ready-pose resets
  and ball contacts, not commands.
- **The low level** (the reference's own limits-aware stroke, no rehearsal): every row passes the checks; against level 1 (seeds 1-3,
  `results/reallimits/levels_v10_low.jsonl`) it lost all three (3:5, 2:5, 2:5): without the high level's rehearsal it
  delivers the blade state less precisely under the real servo, its returns of the library's low services mostly land
  on its own half (3 of 14 balls returned), and 5 of its 13 services faulted. A known gap of the reference (the low level's task is being
  redesigned in PR #31), not of the body: the high-level reference on the same robot wins level 1 5/5.

## The let limit (revision wallclock-11 / duel-wallclock-9, 2026-10-09, deviation D12)

**The hole ("let stalling").** A service that touches the net and is otherwise good is a let (Law 2.9.1.1), a let does
not change the server (2.13.3), and the Laws set no limit on lets. The only bound was the 40-point guard (D8: at 40
points, lets included, the player ahead wins; at a level score the player who won the last point). A server that can
make its service clip the net at will can therefore replay it until the guard ends the game. At the low level this
is exact: before every point both robots are reset to the same rest pose and the physics is deterministic, so the
same toss and joint trajectory give the same let every time. Reproduced on revision 10/8 (the sport's image,
in-process; duel, low level, seed 1, B idle): A served point 1 normally (1:0), then the reference's low-level
`mid_wide_bh_top` service re-aimed 2 cm below the net-clearing height on the reference's own flight model
(`tests/test_tabletennis.py::_let_service`) and got a let 39 times in a row; the game ended 1:0 for A at point 40
(k 10034). Served from 0:0, all 40 points are lets and the guard has no winner (its "last point won" does not exist).
At the high level the execution scatter (4 mm, 0.6 deg, 2 % per shot) makes a let only sometimes (2 of 4 tries with
the same aim), so the hole was a low-level one.

**The fix (D12).** At most 2 service lets in a row: the server's third service let since the last point scored is a
failed service, the receiver's point (`serve_fault:repeated_let`), in both formats and for the scripted opponent
too. The rulebook leaves this to chapter 3 (Regulations for International Competitions, ITTF Statutes 2025): play
shall be continuous (3.4.4.1), and behaviour that may unfairly affect an opponent (3.5.2.1) earns a yellow-card
warning, then a penalty point to the opponent (3.5.2.2-3.5.2.3). As for undue delay (D10) the umpire's judgement is
replaced by a fixed rule: two lets in a row are free (the warning), the third costs the point. Honest play is not
affected: the reference's services were lets in 13 of 337 (fairness run, revision 10/8), so three in a row would come
about once in 17 000 service turns ((337/13)^3). With the limit a game has at most 13 scoring points and 2 service lets before each,
so lets from the service alone can no longer reach the 40-point guard. `state.serve` gives the server `lets_in_row`
and `let_limit`; `/spec` `serve.lets_in_a_row_max = 2`. Exploit test E11 (`test_exploit_let_stalling`): the same let
service at the low level in both formats gets at most 2 lets in a row and the third is the receiver's point.

## How the legacy parts map

- **Seeds**: `match`: `RCB_SEED` -> `MATCH_SEED` (the opponent's serves and returns, execution scatter; the legacy
  default was the level's seed 101/202/...); `duel`: `DUEL_SEED`, first server = `FIRST_PLAYER` (`FIRST_SERVER`; until
  revision duel-wallclock-4 it named the first receiver). duel.py reads both as module constants; the plugin sets them.
- **High level**: the paddle act is lowered at the effect boundary to a low-level traj command (match: `plan_paddle` ->
  9-joint traj then the rest slew; duel: base goal and IK in the plan worker, base trajectory and swing at the boundary);
  `move` and `toss` pass through. The episode holds only low-level commands; `test_exploit_high_level_lowers_to_valid_low_level`
  checks every lowered command against the low level's validator and replays the episode bit for bit.
- **State hash**: duel = `Duel.state_sha()`; match = sha256(the simulator's `state_sha()` + the points), so the score is
  covered (the legacy match hash was physics only).
- **Cameras**: the sport draws its own images (`camera_job`): the 4-frame history is not one renderer state, so the
  renderer plugins cannot draw it. The cameras and tiles are the legacy ones; `scene()` and `render_state()` serve the
  spectator video. Since revision wallclock-6 / duel-wallclock-4 (core 0.2.9, renderer rule 6) the arms are drawn
  with the Franka Panda's visual meshes: `presentation/presentation.toml` declares `players = "25k"` (the Menagerie
  meshes decimated to ~25k triangles per arm, renderers/presentation/lod.json), tools/sportlib bundles them into the
  simulator (`sport/presentation/assets/`, sha256-checked, licence kept) and `MatchView` / `DuelView` draw the
  compiled players' scene (`renderers.players_model`, the same kinematics checked). Ball, table, net and
  paddle are unchanged (the reference player sees the ball as before). `[observation] render_threads = 2`,
  `sim_cpus = 6`.

## Checks (2026-10-09, revision 11/9: the let limit)

`tools/check tabletennis` stages 1-3: 82/82 in 6802 s (stage 1 contract 47/47; stage 2 30/30 with the new exploit test
E11; stage 3 golden 4/4 identical: no golden match had 3 lets in a row, so the records of revision 10/8 stand).
`ci_checks/run_all.sh` passes. Stages 4-5 were not re-run (D12 changes nothing for players that do not serve 3 lets
in a row; the reference's lets were 13 of 337 services).

### Revision 10/8 (2026-10-09)

`tools/check tabletennis` stages 1-3: 79/79 (stage 1 contract 45/45; stage 2 29/29: determinism in process and in a
fresh-process replay, no draw, whitelist, purity, refusals at both levels, the exploit tests E1-E10, quit and no-show;
stage 3 golden 4/4; 7200 s); stage 5 live 5/5 (the match and duel program tasks offline: oracle = 1, nop = 0); stage 4
fairness split over processes (`fairness.py`, N = 40: first server 0.0 pp, seat A 0.0 pp, 95% CI +-15.5 pp; see "The
real limits in the sport"). After the rebase onto main (robot library #26) the golden cases replay 4/4 identical and
`ci_checks/run_all.sh` passes (check-robot-limits: the sport's servo class is `TTServo`, composed of the library's
JerkLimiter and TorqueRateLimiter).

### Earlier (2026-10-07, revision 9/7)

`tools/check tabletennis`: stage 1 contract, stage 2 24/24 (determinism in process and in a fresh-process replay, no
draw over 48 matches, whitelist, purity, refusals at both levels, the exploit tests E1-E9, quit and no-show; 5281 s),
stage 3 golden 4/4, stage 5 live (the two program tasks offline: oracle = 1, nop = 0); stages 1, 3, 5: 53/53. Stage 4
fairness was run split over two processes (`prototypes/tabletennis_serve/fairness.py`, the same matches: N = 40,
first server +5.0 pp, seat A 0.0 pp, 95% CI +-15.5 pp; see "Serve study" above; realtime sports only warn on a wide
CI). Earlier (2026-10-06, machine serves, fixed base): stage 2 17/17, golden 23/23, fairness N = 12 (first receiver
-8.3 pp, seat A -8.3 pp, +-27.9 pp).

## The arm's setpoint ringing (revision wallclock-12 / duel-wallclock-10, 2026-10-11/12)

The arm shook visibly in the videos (about +-5 deg at 2-3 Hz on joints 1-4, also while the reference held still). Cause:
the shared setpoint generator (`bodies/limits.JerkLimiter`) braked its position loop on a sqrt law at 0.6 x the
acceleration limit, which assumes the deceleration is there at once; the jerk a step may use is capped by the torque
rate at the pose (`limits.jerk_caps`: 70 % of 1000 N m/s over the inertia row, about 230-430 rad/s^3 on joints 1-4
against the 7500 of the datasheet), so the setpoint overshot, reversed and settled into a limit cycle (a 2.9 deg step at
231 rad/s^3: out to 6.4 deg, then +-3.5 deg at 2.5 Hz). The generator now brakes within the jerk in force
(`servo.SetpointGenerator`'s law: the error left once the acceleration has ramped to zero, `brake_speed`), with that
coast distance taken relative to the reference's velocity (qd - rd: with qd alone the setpoint braked early toward a
moving reference, the strokes delivered too little blade speed and B9 fell to 248/284; with qd - rd B9 is 263/284 =
92.6 %, best fixed stance, on the screening branch's revision), and its velocity bound keeps one step of jerk and one of
acceleration below the limit. The four golden matches, reference on every seat, per arm second, joints 1-4: setpoint
velocity reversals while the reference holds still 0.74 / 0.24 / 1.25 / 1.75 per s before, 0 / 0 / 0 / 0.02 after;
swings of the joint beyond +-1 deg of the reference 2.35 / 0.58 / 0.92 / 0.68 per s before, 0.60 / 0.46 / 0.33 / 0.59
after (the rest are strokes); RMS of joint minus reference 3.99 / 2.41 / 2.26 / 2.14 deg before, 1.48 / 2.23 / 1.58 /
2.34 after. The reference against the level-1 opponent (match, high, seeds 1-20): 18/20 won, points 112:66 before;
19/20, 103:36 after. Setpoint peaks stay within the velocity, acceleration and jerk limits
(`bodies/tests/test_limits.py`, in CI, also the tracking of moving references). The player's toolkit model of the servo
(`seat/lib/tt_servo.py`, on the screening branch) must take the same law when it is merged.

## The skill in the loop (revision +low-inloop-1, 2026-10-13, shared PR shared/skill-in-loop)

The low tasks play the match with the player's own `hit` (`sport.toml [modes.low.inloop]`, `skilleval/inloop.py`):
the generated low tasks run each seat's `/app/output/skill/hit.py` in a skill container of its own (no network,
read-only, no simulator files; the toolkit at the seat's paths); the player's program sends `hit` calls and the
simulator lowers each with the player's skill; its compute counts in match time (the world runs on, the command takes
effect when the call returns; #90 decision 6); while a ball is in play `traj` / `slew` / `hold` are refused (409),
`toss` is free; `step()` every 20 ms; every call is logged with the skill's sha256 (`inloop_<seat>.jsonl`), and the
post-match evaluation (`tools/skill_eval.py tabletennis <skill> --inloop-log <log>`) says whether it tested the version
that played. The reference skill plans around its own compute: with `spec.timing` its rows start with the target in
force over the steps it expects to take, its rehearsals stop after 0.5 s of wall time and a call that ran over gives no
command (the API test, without `timing`, is unchanged). The oracle of the low tasks is `SkillReference` (the serve
stance by a held `traj`: a `slew`'s base go-to keeps moving by 1-2 mm around its goal, which a stroke planned on a
held target does not join) with `solution/skill` = `lowlevel/reference`. `lowlevel/make_inloop_task.py` is gone (the
generator does it). 

Measured 2026-10-13 (L1, seed 101): in process (the world waits) 1:5, 20 hits of which 12 failed (the reference hit's
p95 is 5.0 s, at the 5 s limit: a time-out stops the skill for the rest of the point); live (`tools/live.py`, x20, the
skill host over its socket, compute counted) valid and deterministic, 17 hits all taking effect (0 rejected), 0.57 s
wall p50 = 3 control steps, 912 step() calls without error, one skill version, 0:5 (3 service faults, 2 volleys); the
reference row re-recorded with that match's log: API test 88/118, competition test 0/16 (as #110 recorded), the
evaluated skill = the one that played (929 calls, 1 version).

## Golden replays

`python3 tools/golden.py replay tabletennis`: the four cases of `golden/`, recorded with `tools/golden.py record` (the
reference on every seat, in-process) under revision wallclock-15 / duel-wallclock-12 (re-recorded 2026-10-12, the
re-screen: realistic restitution, 1 cm scatter, D13, the hit's blade-speed fix, the 0.9 velocity margin and the
velocity and collision reflexes, the recalibrated opponent; earlier records cannot replay):

| case | format, level, seed, first server | result | steps k |
|---|---|---|---|
| `duel-high-s1-A` | duel, high, 1, A | B 7:5 | 4258 |
| `duel-low-s2-B` | duel, low, 2, B | B 7:6 | 3119 |
| `match-high-s101` | match (L1), high, 101 | you 5:2 | 2711 |
| `match-low-s101` | match (L1), low, 101 | opponent 5:0 | 1428 |

**Retired (2026-10-07, the rule change C2):** the machine-serve records can no longer replay - the robot gained two
joints and the players serve - and every future match is re-run (no result backward compatibility). Retired from the
golden set, not converted or faked: the two stage-0 cases `golden/duel-high-s1-A-x20` (7:5, k 3598) and
`golden/duel-low-s2-B-x20`, `golden/convert.py` with the 21 legacy records of the golden-data store it converted, and
`tests/data/hl_oracle_L1_s101_x50.json` (the original service's high-level match). They remain in git history (last at
commit c479b71e); `sim/rally.json` (the Season-1 rally episode's balls) went with the rally episode.

## The legacy tree

`scripts/olympiad/tabletennis` and `scripts/olympiad/shared/tabletennis` are removed. The last metering reference
copies (`scripts/olympiad/tabletennis/{match,duel}/environment/sim/think_{proxy,clock}.py`) were removed on 2026-10-10:
every sport is migrated, `core/metering/think_clock.py` and `players/metering/think_proxy.py` are the sources and
`tools/generate.py check` keeps every task's copy identical.

## Reward keys

The common keys come from `core/grading.py`, the sport's from `grade.py` (the legacy keys). Differences:
- both formats: `official_config` is recomputed with the legacy per-format rule (match: the level's own seeds and the
  program/direct slowdowns; duel: the slowdown only), since `[official]` cannot express it;
- match: new common keys `A_won`, `A_transient_api_errors`, `A_unrecovered_api_errors`, `solo` (and `A_nocode_*` in
  direct play); `grader_error` has the core's meaning (the grader crashed), not the legacy "sim, replay or meter
  error"; `replay_joint_targets_equal` = `deterministic` (the core's replay compares the commands, state and steps);
  dropped (the core does not record them): `act_delay_steps_max`, `act_rule_wait_max_s`, `act_rule_wait_mean_s`,
  `n_wait`, `sim_lag_frac_late_over_1_period`; `n_calls` = acts + observations (waits are not counted);
- duel: unchanged key set (the core adds `solo`, `direct_play`, `lowlevel_actions` as before).

## Seat images (tools/generate.py)

- Low level (`[seat.low]`): `/opt/robot/robot_only.xml` (+ the Menagerie meshes in `/opt/robot/assets`) and MuJoCo
  3.3.7 with its closure, the sim image's version (tests check the pin and that `seat/robot_only.xml` is what
  `robot_model.py` makes from the scene); every seat also has OpenCV headless (`[seat]`), as in Season 1.
- Direct (no-code) play, its two tasks (`tabletennis-{match,duel}-wallclock-direct-standard`), `[actions.direct]`,
  the direct slowdowns and exploit E7 (`test_exploit_nocode_audit_forfeits`) were removed from main on 2026-10-10 and
  are kept on branch `archive/direct-play`. `grade.py` still writes the legacy keys `nocode_*` and `gate_blocks`
  (always 0 in program play), so the program tasks' reward.json is unchanged.

## Screening 2026-10-11 (docs/screening/tabletennis.md)

The sport went through the construction and screening process (docs/construction_zh.md); the report is
`docs/screening/tabletennis.md`. Changes it made here:

- **Revision 12/10: self-collision on the rail version** (`[self_collision]`): the arm and paddle against their own
  links and against the rail base's column and carriage (which were drawn only); recorded per seat in the final fields,
  not penalised; `seat/robot_only.xml` carries the pairs. B9 re-run on this revision: 269/284 = 94.7 % from the best fixed
  ready position.
- **Revision 13 (match only): the scripted opponent recalibrated.** The model trial runs found that short services were
  near-automatic points (GPT-6.1 SOL took all 6 of its points at L2 that way without one good return; a program that
  only served short won 4/10 at L1). The opponent now receives services down to 0.62 / 0.58 / 0.54 / 0.50 m from the net
  and L1 / L2 are easier in the rally (table above). The match golden cases were re-recorded.
- **The hit contract** (`[spec] action_high`): control authority and failure handling written out.
- **Low level** (toolkit 2.0.0, `skill_api.py`, `lowlevel/`): the #31 hit evaluation redone on the rail version; the
  API-test cases replay against the opponent they were recorded with (`lowlevel/evaluator.py` CASE_LEVELS).
- **Screening hooks** (`screen/hooks.py`, `screen/analyze.py`, not part of the simulator): the reference and its
  variants for `tools/screen` (simple baselines, skill necessity, capability-sensitivity impairments, truth-perception
  variants), and the points-by-source analysis.
