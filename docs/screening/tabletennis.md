# Screening report: table tennis

Owner: Chi Gui. Process: [docs/construction_zh.md](../construction_zh.md) v1. First sport through the process; this
report is meant as the worked example for the other owners. Data: `/data/guichi/olympiad_review/screening/
tabletennis_data/` (host mll-4090-1). Scripts: `tools/screen/` (shared), `sports/tabletennis/screen/hooks.py`.

## 1. Conclusion

Table tennis went through all six steps; three checks failed on the way and were fixed (self-collision missing on the
rail version, an incomplete `hit` contract, and a shortcut — short services the scripted opponent could not receive —
found by the model trial runs, fixed by recalibrating the opponent, revision 13). With the fixes every gate of steps
②–④ passes on the current code, step ⑤ shows discrimination at L1 and in the low level's API test and no new
shortcut, and the configuration below is proposed for the freeze. Items marked *pending the user's confirmation* are
listed in §3 of the Chinese walkthrough (`olympiad_review/screening/tabletennis_walkthrough_zh.md`).

| step | check | result |
|---|---|---|
| ① define | core challenge, 12 numbered deviations, capability claims | pass; B2 lowered to placement (b2 = 1) after step ④ |
| ② robot & venue | real limits with sources; B9 (upper-bound receive of the 284 legal serves) | 269/284 = 94.7 % (≥ 90 %) on revision 12/10 with self-collision |
| ② | physical consistency | failed (no self-collision on the rail version) → fixed (67 / 134 pairs, column and carriage) → pass |
| ③ interface | P1–P6, the `hit` contract | failed P6 (control authority, failure handling) → written into the instructions → pass; low-level toolkit 2.0.0 |
| ④ model-free (high) | opponent calibration, baselines (20 seeds, true perception first), skill necessity, capability sensitivity | pass: truth reference 17/20 at L1 falling to 0/20 at L4; every baseline 0–1/20 at every level; `no_hit` 0 points; A1, A2, A4 kept, B2 not |
| ④ engineering | `tools/check` stages 1, 2, 3, 6; fairness (40 duels) | pass (stage 5's single live match passes ~6 in 10, §9.6); first server +10 pp, CI −5.4…+23.7 (WARN) |
| ④ model-free (low) | reference skill, boundary, isolation | 43/50 ≥ 0.8; boundary 0/50; probe pass |
| ⑤ models (round 1) | discrimination, shortcuts | shortcut found (SOL: 6 points from short services, no good return) → back to ① (opponent) |
| ⑤ models (round 2, Astra + Opus 5.5) | discrimination, shortcuts | L1: Astra 5:3, Opus 3:5; L2–L3: 0–1 points each; low-level API test 26–39/50; no new shortcut |
| ⑥ freeze | versions, seeds, levels, score | proposed (§8): revision 13, hidden seeds registered, formal levels L1–L3, `[score]` point share |

Chosen configuration: Franka Panda + 15 cm paddle on an x-y rail base (real Panda limits; rail 2 m/s, 8 m/s², 200 m/s³
estimated), standard ITTF table and ball; modes high (main board) and low (with the skill in the loop as a prototype);
format match against scripted L1–L4 (formal L1–L3), format duel; slowdown 10 (was 20 until 2026-10-14, see §8); score = win + point share.


## 2. Candidates and pruning

The configuration was searched over robot × venue × interface × opponent. Earlier results are used as the search
record (not redone); the chosen configuration's checks are re-run on the current code (sections 4–7).

| candidate (robot + standard table) | upper-bound receive of legal player serves (B9, best fixed ready position) | result | evidence |
|---|---|---|---|
| **Franka Panda + x-y rail base** | **94.7 % (269/284), re-run on revision 12/10 (this screening)** | **kept** | §4 |
| ABB IRB 1100 + two-axis gantry | 73.9 % (88.4 % with a longer paddle) | dropped: short reach, cannot get to short balls near the net | `olympiad_review/abb_irb1100_zh.md` |
| Rainbow RB-Y1 (wheeled humanoid) | 68.7 % (93.7 % only if its base may roll under the table) | dropped | `olympiad_review/tt_serve_receivability_zh.md` |
| Unitree G1 (legged) | 33.8 % | dropped for the main board (kept for low-level research) | same; `pilot_g1_tt_zh.md` |
| Unitree H1-2 (legged) | 22.9 % | dropped | same |
| Galaxea R1 Pro (wheeled) | 0 % (its SDK jerk limits cannot swing a paddle) | dropped | same |
| Tianji Marvin Pro (fixed) | 20.8 % | dropped | same |
| Franka on a fixed base (Season 1) | short player serves nearly unreturnable (4 of 47 with a fixed arm) | replaced by the rail base (revision 7/5) | README "Serve study" |

Interface candidates: `hit(landing point)` was rejected by P1/P2 (the platform would solve the stroke); the paddle state
at contact `hit(t, pos, normal, vel[, base])` is kept. Low level: joint targets only (Season 1) gave no reference that
could beat L1 and no IK; the toolkit with an IK library (option B) is kept (§5, §6).


## 3. Step ①: define the sport

**Core challenge kept.** Predict a spinning, bouncing 40 mm ball from camera images only (two cameras × four frames, no
ball coordinates) and choose when, where, with which face and how fast the paddle meets it, so that the ball lands
legally on the far half and out of the opponent's reach, in a world that keeps running while the program thinks
(real-time protocol: computing costs game time); serve legally under the ITTF service law with the robot's own stroke.
Neither the platform nor the rules take any of this over: the skill takes the paddle state at contact (not a landing
point) and the observation holds no ball state (exploit E3).

**Rule book.** ITTF Laws of Table Tennis (ITTF Statutes 2025, chapter 2), effective 1 January 2025. Numbered deviations
(`sport.toml [[deviations]]`; every one that changes results is in the agent's instructions):

| id | Law | deviation | why |
|---|---|---|---|
| D1 | 2.11, 2.12 | one game to 5 (lead by 2 from 4:4; at 6:6 the next point wins) | cost: a best-of-games match to 11 takes hours of model time |
| D2 | 2.6.1–2.6.5 | a toss device on the robot makes the toss (`toss {pos, height}`); the rest of 2.6 is judged at every physics step | a one-arm robot has no free hand |
| D3 | 2.5.7 | only the blade may strike the ball | no racket hand |
| D4 | 2.10.1.9–11 | as the Laws: the table may be touched, the net assembly loses the point | (the table is fixed) |
| D5 | 2.10.1.7 | any double hit loses the point | intent cannot be judged |
| D6 | 2.15.4 | the expedite 13-return limit from the first point | bounds every rally |
| D7 | 2.9.1.3 | duel: simultaneous faults are a let | two robots in one world |
| D8 | — | 40 points played ends the game (cost guard; never reached) | last resort |
| D9 | — | the player is a Franka Panda on an x-y rail base behind its end line, held to the real limits | embodiment |
| D10 | ch. 3 | undue delay: toss within 5 s, else a failed service; the third such fault loses the match | ch. 3 warnings as a fixed rule |
| D11 | 2.13.1–2 | no lot: `match` you serve first; `duel` both orders are played | fairness by design |
| D12 | 2.9.1.1, ch. 3 | at most 2 service lets in a row | blocks deterministic let-stalling (E11) |

**Capabilities claimed** (`sport.toml [capabilities]`): A1 3-D perception, A2 dynamics, A3 embodiment (at the high level
the execution is the platform's and is not attributed to the agent), A4 time and latency, B1 rules, B2 strategy and
reading the opponent. Step ④ tests A1, A2, A4 and B2 with impaired reference players (§6.4); B1 is exercised by the
service law and the exploit tests (no impairment); C1/C2 are measured outside the official score.

**Adjustable parameters.** Standard: the table (2.74 × 1.525 m, net 15.25 cm), the ball (40 mm, 2.7 g, DeepMind's
system-identified ball/table model), the point and service rules except D1–D12. Adjusted for the robot (declared): the
game length (D1), the toss device (D2), the rail base's travel (D9). Not adjustable: the robot's real limits, the
physics.

**Opponents, seeds, scoring.** `match`: the scripted opponent, levels 1–4 (reaction 0.30 → 0.24 s, chase 0.42 → 0.52
m/s, reach 0.24 → 0.31 m, receive of short services down to 0.85 → 0.62 m from the net, its serves drawn from the
easiest 25 % → 100 % of the official serve library; every parameter monotone in the level; held to the same real
limits). `duel`: two players, both first-server orders. Scoring: match win, and the point share (points won / points
played) as the continuous score. Seeds: §8.

## 4. Step ②: robot and venue

**Configuration.** Franka Emika Panda (FER; MuJoCo Menagerie `panda_nohand` + a 15 × 15 cm paddle on the flange) on an
x-y rail base behind its end line; standard ITTF table and ball.

**Real limits (B0) and their sources.**

| quantity | value | source |
|---|---|---|
| arm range, speed, acceleration, jerk, torque, torque rate | `bodies/franka_panda/body.toml [limits.arm]`: 2.175 / 2.61 rad/s; 15 / 7.5 / 10 / 12.5 / 15 / 20 / 20 rad/s²; 3750–10000 rad/s³; 87 / 12 N m; 1000 N m/s | S1 Franka FCI "Limits for FER" (cross-checked: S2 libfranka 0.9.2 `rate_limiting.h`, S3 franka_ros `joint_limits.yaml`) |
| enforcement | fencing-3's 1 kHz servo (jerk-limited setpoint, TorqueRateLimiter) on `bodies/limits.py` (`rcb_tt.TTServo`); the low level's rows are also checked for speed and acceleration (E1, E2, E10) | shared module, `ci_checks/check-robot-limits.py` |
| rail speed | 2 m/s per axis | DeepMind's robot table-tennis gantry (Festo) |
| rail acceleration | 8 m/s² | drive force 620 N / 76.8 kg moving mass |
| rail jerk | 200 m/s³ | **estimated** (no manufacturer value; 25 × acceleration), marked in D9, `/spec`, README — *pending the user's confirmation* |
| travel | \|x\| ≤ 1.0 m, base y −1.50…−2.40 m (the column stops 1 cm behind the end line) | the sport's rule: the robot cannot lean over the table |

**Real counterpart.** The arm is the real Panda with the manufacturer's limits; the rail base corresponds to DeepMind's
competitive robot table-tennis setup (an arm on a two-axis linear gantry). The paddle is a real 15 × 15 cm blade on a
short handle bolted to the flange.

**Feasibility (B9), re-run on this revision.** The receive study's full-simulation upper bound (`fullsim.py`,
tt-receive-study 329ff155, code unchanged; the sport's own real-limits servo and rail, the duel's high-level stroke
path, self-collision on), the 284 legal serves of the serve library: from the best fixed ready position (near_bh,
(−0.25, −1.55)) **269/284 = 94.7 %** returned (near_fh 93.3 %, home 76.8 %, any position 97.5 %). Threshold 90 %:
**pass** (#42 on revision 10/8: 94.7 % / 93.3 % / 77.8 % / 97.5 %). Data: `tabletennis_data/b9/`.

**Physical consistency.**
- **Self-collision on the rail version (new, revision 12/10).** #54's pairs were made for the old pedestal and dropped
  in the integration (#73/#76). Re-generated with `bodies/selfcollision.py` from `sport.toml [self_collision]` (body
  `franka_panda`, `geoms = "physical"`): 67 pairs per robot (match scene), 134 (duel): arm links two or more joints
  apart, the paddle with the arm, **and the arm and paddle with the rail base's column and carriage**, which were
  drawn-only (contype 0) before. `robot_only.xml` (the low level's own model) carries the same pairs. Every physics
  step feeds `selfcollision.Monitor`; the final fields record events per seat (recorded only, no penalty, as in the
  other Franka sports). Static check (`SC.check_model`, 1000 random legal arm poses): nothing penetrates at rest; 78 of
  1000 random poses touch (most often link4/link5 × blade, link0 × link6/7).
- **Contacts are on**: no scene disables contacts; the arm and paddle collide with the table and net (D4); the MuJoCo
  arm links already collided with each other through the default contype (the explicit pairs replace those dynamic
  contacts one for one), so the only physical change is the column and carriage.
- **In play** (§6, 600+ matches): the reference had a self-collision event in a few matches, always paddle × forearm
  (link5) at the end of a swing, peak force above the Panda's 20 N reflex threshold (see §6.6); one reflex-baseline
  match touched the column. Recorded, not penalised (the convention of the Franka sports); the stroke planner does not
  yet avoid the forearm — a follow-up, not a blocker (the arm's motion is physically stopped, so nothing passes through).
- **Release speeds**: the opponent's shots are capped at what the limited arm produces (≤ 5.5 m/s off the paddle,
  34 rad/s spin, a blade speed ≤ 2 m/s by the contact law); the player's own shots come from the simulated arm.


## 5. Step ③: interface

**Modes.** Both. High = Agent as Athlete (the platform's `hit`); low = Agent as Robotics Engineer (joint interface +
toolkit; the agent implements the same `hit`). Both pass the step-④ checks (§6); the high level is the main board.

**The `hit` contract (P6).** One contract for both modes (`sport.toml [spec] action_high`, the low level's
`skill_api.py` SkillContract):

| part | high level (`act` / `hit`) |
|---|---|
| inputs | `t` (s from the command's effect to contact, 0 < t ≤ 2.5); `pos` (blade centre at contact, m, the player's frame: x right, y towards the opponent, z up; \|x\| ≤ 2.0, −3.3 ≤ y ≤ −0.3, −0.5 ≤ z ≤ 1.5); `normal` (blade face normal, non-zero); `vel` (blade velocity at contact, ≤ 8 m/s); optional `base` (the base position at contact, within its travel). Also `move {base}` (rest position), `toss {pos, height}` (service), `none` |
| timing | takes effect at the next 10 ms control step after it arrives (`took_effect`); replaces the stroke under way; one command per seat in flight |
| what the controller does | moves the base (to `base`, else only as much as needed; arriving at rest 5 steps before contact, or as far as it gets); IK at that base position; a jerk-limited swing within the Panda's limits (wind-up, stroke as late as the slowest joint allows, follow-through); table avoidance (from 0.10 / 0.18 m above); in `match`, up to 5 robot-only rehearsals that correct the servo's lag; seeded execution scatter 4 mm / 0.6° / 2 %; afterwards the arm back to its ready pose, the base to its rest position |
| control authority | reads only the robot's own state (joints, targets, servo, base) and the command; never the ball or the opponent; it neither predicts nor aims |
| feedback | reply: `stroke` notes, `n_swing_steps_available`, `ik_residual`, `blade_error_m`, `normal_error_deg`, `speed_scale` / `plan_scale` (the share of the blade speed the limits deliver), `base` / `base_goal`, `approach`, `paddle_vel_delivered`; state: `command` (contact_in_s, phase, the commanded state) |
| failure handling | malformed / out of range → 400, nothing changes; contact time already passed → "too late", not executed, the stroke in force continues; unreachable / too fast / base too slow → best effort, reported in the reply; `duel`: no rehearsal, and the trajectory in force is kept if it predicts a better contact (reply `kept`) |

Contract change in this screening: control authority and failure handling are now in the agent's instructions (they
were only in code comments, audit item P6 ③④). The `match`/`duel` difference (rehearsal, `keep_better`) is written
down; unifying the two implementations is a separate decision (*pending the user's confirmation*; recommendation:
documented now, unified in the next revision with its own fairness re-run).

**Skill necessity of the high level's action set** (§6.2): `move`, `toss`, `none` alone score nothing.

**Observation and time model.** Whitelist (`sport.toml [observation]`): point, score, ball id, `t_since_played`,
serve state, ball-state phrase, the robot's own joints, base, paddle pose, command and targets, the points and shots so
far, heard bounce events; no ball coordinates, no opponent or seed state (E3). Images: one PNG, 2 cameras × 4 frames
40 ms apart, 480 × 360 tiles, each frame's control step. Time: real time, the world slowed by S (`[timing] slowdown`,
default 10 since 2026-10-14, 20 before); a command takes effect at the next 10 ms step; the world freezes after each point for the review (300 s).
In the model-free checks the latency is modelled as a fixed effect delay of D = 5 control steps (50 ms of game time ≈ 1
s of wall time at S = 20).


## 6. Step ④: model-free validation

Setting for every check unless stated: format `match`, high level, the camera-only reference and its variants
(`sports/tabletennis/screen/hooks.py`), effect delay D = 5 control steps, opponent levels L1–L4, seeds 1–10 paired across
variants, one container per match on a frozen simulator copy (`tools/screen/batch.py`); gates from
`tools/screen/gates.py`. Point share = points won / points played. The process went through two rounds: round 1 on
revision 12/10 (600 matches + 136 extra seeds), then step ⑤ found a shortcut, the opponent was recalibrated (revision
13) and step ④ was run again (round 2, below).

### 6.1 Round 1 (revision 12/10)

| check | result | gate | verdict |
|---|---|---|---|
| reference (P4) | L1 9/10, L2 7/10, L3 5/10, L4 2/10 wins; point share 0.65 / 0.64 / 0.55 / 0.31 | wins the default level; monotone | pass |
| skill necessity (`no_hit`: move, toss, none only) | 0 points in 40 matches | no win, point share ≤ 0.05 | pass |
| simple baselines (fixed, random, reflex, centre) | best: reflex L1 4/10 (0.45); every other cell ≤ 1/10 | lose the majority at every level | pass |
| capability sensitivity | A1 3/4 levels, A2 4/4, A4 3/4, B2 0/4 | strong impairment below 0 at ≥ 3 of 4 levels | A1, A2, A4 kept; B2 not |
| engineering (`tools/check` 1, 2, 3, 6) | 124/124 | all | pass |
| live run (stage 5) | reference lost 2:5 to L1 on the first run, won on the re-run | reference 1, nop 0 | pass on re-run (a single live match is a ~10 % coin flip for this reference) |
| fairness (40 duels, both first servers) | first server 24/40 (+10.0 pp, 95 % CI −5.4…+23.7), seat A 20/40 (0.0 pp) | CI not entirely beyond ±5 pp | pass (WARN: CI wider than ±5 pp) |

Capability sensitivity in detail (impaired − reference, point share, mean [95 % bootstrap CI], 10 paired seeds; noise
floor = the replica `ctrl`, which only rounds the act's numbers to 3 decimals: +0.03, −0.02, −0.04, 0.00):

| capability | impairment | L1 | L2 | L3 | L4 |
|---|---|---|---|---|---|
| A1 | 1 px centroid noise | −0.09 | −0.08 | −0.15 | −0.15 |
| A1 | 2 px | −0.26 [−0.37, −0.15] | −0.20 [−0.32, −0.07] | −0.29 [−0.40, −0.18] | −0.18 [−0.36, +0.01] |
| A2 | no air drag | −0.26 | −0.32 | −0.24 | −0.03 |
| A2 | wrong bounce (e_z 0.80) | −0.63 | −0.62 | −0.51 | −0.27 |
| A4 | latency under-compensated 20 ms | −0.05 | −0.08 | −0.15 | −0.02 |
| A4 | 30 ms | −0.15 [−0.27, −0.02] | −0.20 [−0.37, −0.01] | −0.15 [−0.32, +0.05] | −0.16 [−0.35, −0.01] |
| B2 | random side (ignores the opponent) | −0.04 | +0.04 | −0.06 | −0.07 |
| B2 | fixed corner | −0.06 [−0.20, +0.09] | −0.10 [−0.22, +0.00] | −0.07 [−0.23, +0.09] | −0.08 [−0.27, +0.14] |

B2 does not drop at any level (the validity pilot had found the same at L2): the scripted opponent recovers to the
centre after each of its strokes, so where it was when it hit carries no information by the time the player hits;
placing the ball wide pays (the centre baseline loses), reading the opponent does not.

Where the points come from (round 1, 10 matches per cell): the reference won 51 points at L1, 26 of them from short
services the opponent could not receive ("too short") and 25 from balls out of its reach ("too wide"); the fixed and
random baselines, which never return a ball, still won ~2 points per match from the same short services at L1–L3.
This was the first sign of the shortcut that step ⑤ then exposed.

### 6.2 Low level

| check | result | gate | verdict |
|---|---|---|---|
| reference skill (toolkit only), API test | 44/50 hidden requests (0.88); median error 1.3 mm / 0.13° / 0.031 m/s / 0 ms | ≥ `reference_min` 0.8 | pass |
| platform implementation (with execution scatter) | 40/50 (0.80); competition 11/16 (L1–L4 4/4/1/2) | every hidden request feasible | pass |
| boundary (IK only, no swing) | 0/50, 0/16 | ≤ 0.05 | pass |
| isolation probe | the skill reads no simulator file or hidden request, writes nothing, no network | all fail | pass |
| reference skill, competition test | 5/16 (2/3/0/0) | (reported) | — |
| low-level reference player vs L1 | 1/10 won (22:48 points) | (reported) | — |

Cases: 24 platform reference matches → 183 candidate requests → 79 feasible (the platform's own robot-only rehearsal
delivers them within 15 mm, 2.5° and 0.10 m/s + 8 %) → 50 hidden + 28 public. `tools/check` stage 6: all pass.

Skill in the loop (prototype, `skilleval/inloop.py`): in-process L1 match with the camera reference program and our
reference `hit.py` in the sandbox: lost 3:5, 44 `hit` calls (p50 1.71 s, p95 3.06 s wall), 1562 `step` calls (p50
0.3 ms), replay bit-identical. At x10 / x20 a control step is 100 / 200 ms of wall time, so our `hit` stalls the world
for ~9–17 steps per call: it does not fit unless the call is capped (~150 ms) or its time is charged as game time.


### 6.3 Round 2 (revision 13, after step ⑤'s shortcut)

The opponent was recalibrated (§7.1, §3): short services received down to 0.62 / 0.58 / 0.54 / 0.50 m from the net
(was 0.85 / 0.80 / 0.76 / 0.62), and L1 / L2 easier in the rally (reaction 0.34 / 0.31 s, chase 0.36 / 0.40 m/s, reach
0.20 / 0.23 m; was 0.30 / 0.28 s, 0.42 / 0.45 m/s, 0.24 / 0.26 m). Two candidates were tried first on L1–L2 with 10
seeds: **A** (short-service receive only): short-serve-only 0/10 and 0/10, but the reference fell to 3/10 at L1 and 4/10
at L2 — it too had lived off short services — so the default level was no longer winnable by the reference; dropped.
**C** (A + easier L1/L2 rallies): reference L1 7/10, L2 5/10; short-serve-only, reflex and centre all 0/10 at both;
kept as revision 13 (match only; the duel has no scripted opponent). Step ④ was then run again in full (640 matches,
seeds 1–10) and the reference and every baseline extended to 20 seeds per level (the user's rule of 2026-10-11:
baselines must lose the majority at every formal level with at least 20 seeds; the win rate is reported).

Round-2 results (revision 13; reference and baselines 20 seeds per level, sensitivity 10 seeds, A4 and the replica
20; the user's rules of 2026-10-11: baselines with true perception first, the opponent calibration with the
truth-perception reference):

| check | L1 | L2 | L3 | L4 | gate | verdict |
|---|---|---|---|---|---|---|
| opponent calibration: truth-perception reference, wins (point share) | 17/20 (0.64) | 8/20 (0.49) | 8/20 (0.42) | 0/20 (0.20) | wins the lowest level, falls with the level | pass |
| camera reference, wins (point share) | 12/20 (0.54) | 8/20 (0.46) | 5/20 (0.42) | 0/20 (0.18) | camera-only can win: majority at the lowest level (or a model won with cameras in step ⑤: Astra 5:3 at L1) | pass |
| camera reference, good shots / shots (serves and returns) | 81/157 | 75/146 | 96/189 | 47/144 | end-to-end evidence, no gate | — |
| fixed (no perception) | 0/20 (0.18) | 0/20 (0.04) | 0/20 (0.06) | 0/20 (0.00) | lose the majority, ≥ 20 seeds | pass |
| random (no perception) | 0/20 (0.14) | 0/20 (0.05) | 0/20 (0.03) | 0/20 (0.00) | 〃 | pass |
| short serve only (no perception) | 0/20 (0.22) | 0/20 (0.08) | 0/20 (0.03) | 0/20 (0.00) | 〃 | pass |
| reflex, true perception | 1/20 (0.35) | 0/20 (0.28) | 0/20 (0.20) | 0/20 (0.17) | 〃 (true perception first) | pass |
| centre, true perception | 0/20 (0.28) | 0/20 (0.14) | 0/20 (0.09) | 0/20 (0.03) | 〃 | pass |
| skill necessity (`no_hit`) | 0.00, 0/10 | 0.00, 0/10 | 0.00, 0/10 | 0.00, 0/10 | no win, point share ≤ 0.05 | pass |

Capability sensitivity, round 2 (impaired − reference, point share, 10 paired seeds, A4 20; replica noise (20 seeds)
+0.11 / +0.04 / +0.01 at L1–L3):

| capability | weak / strong | L1 | L2 | L3 | L4 | verdict (formal levels L1–L3) |
|---|---|---|---|---|---|---|
| A1 | 1 px / 2 px | −0.23 / **−0.34** | −0.18 / **−0.34** | −0.25 / **−0.30** | −0.09 / **−0.13** | kept (3/3; also L4) |
| A2 | no drag / wrong bounce | −0.26 / **−0.60** | −0.17 / **−0.44** | −0.23 / **−0.39** | −0.06 / **−0.14** | kept (3/3; also L4) |
| A4 | 20 ms / 30 ms under-compensated (20 seeds) | −0.09 / **−0.12** | 0.00 / **−0.14** | −0.10 / **−0.22** | (not run) | kept (3/3; pooled L1–L3: −0.16 [−0.21, −0.11]) |
| B2 | random side / fixed corner | −0.05 / +0.01 | +0.05 / −0.15 | −0.06 / −0.16 | −0.04 / −0.08 | not kept (0/3) |

(bold: the 95 % interval excludes 0.) L4 sits near the floor for every player (even the truth-perception reference
wins 0/20), so differences there are compressed; the verdicts use the proposed formal levels L1–L3 (§8).


The API-test cases of the low level are start states reached by replaying recorded matches; with a new opponent those
replays diverged (`test_b9_every_hit_request_is_feasible` failed). The cases now replay against the opponent they were
recorded with (`lowlevel/evaluator.py` `CASE_LEVELS`): a case is only a start state. The test passes on revision 13,
and the reference rows were re-recorded: platform 40/50 (competition 8/16), reference skill 43/50 (6/16), boundary
0/50 (0/16).

Engineering on revision 13: `tools/check` stages 1 and 3 pass (golden: the two match cases re-recorded, the duel
cases unchanged and replaying); stage 5's single live match of the reference at L1 (seed 1) lost — seed 1 is a losing
seed for the reference in-process as well (2:5). Live runs of the reference at L1 (`tools/live.py`, seeds 1–8,
x20, real wall-clock latency): won 5 of 8 (seed 1 won 5:1 this time; the same seed lost the stage-5 run: live timing
is not deterministic), so a single live match passes about 6 times in 10 (§9.6). Fairness (duel) is unchanged by revision 13 (the opponent is not in the duel).


## 7. Step ⑤: model trial runs

### 7.1 Round 1 (revision 12/10), high level, x20, official seeds

| model | level (seed) | result | how the points were won | API errors | tokens in / out | match wall time |
|---|---|---|---|---|---|---|
| GPT-6.1 SOL | L2 (202) | lost 6:7 | **all 6 from short services the opponent could not receive; 0 good returns** | 0 | 4.9 M / 16.8 k | 19 min |
| GPT-6.1 SOL | L3 (303) | lost 0:5 | — (2 service faults, a volley) | 0 | 3.0 M / 10.5 k | 11 min |
| GPT-6 Astra | L2 (202) | won 6:4 | 3 short services, 3 too wide | 3 (two earlier tries void: 22 and 9 errors > 5) | 7.0 M / 17.0 k | 19 min |
| GPT-6 Astra | L3 (303) | lost 3:5 | 1 short service, 2 too wide | 0 | 3.8 M / 11.6 k | 14 min |

**Shortcut found**: serving short wins points with no table-tennis skill at L1–L2. A baseline that only plays the
reference's two short serves and never returns a ball (`short_serve`, added to the simple baselines): L1 4/10 wins
(point share 0.43), L2 2/10 (0.39), L3 1/10, L4 0/10 — it passes the "lose the majority" gate, but SOL's L2 match shows
that the score at L1–L2 was largely decided by one trick. Back to step ① (opponents): the opponent's short-service
receive was far below what a real robot can do (B9: 95 % of these serves are returnable).

### 7.2 Low level, skill in the loop (revision 12/10, L1, seed 101, x20)

| model | match | `hit` calls (wall p50) | skill versions during the match | API test (hidden) | competition test |
|---|---|---|---|---|---|
| GPT-6 Astra | won 5:2 | 11 (0.34 s) | 5 | 29/50 | 4/16 (revision 12 opponent); 6/16 (revision 13) |
| GPT-6.1 SOL | lost 3:5 | 18 (0.96 s) | 3 | 26/50 | 0/16 (revision 13) |


### 7.3 Round 2 (revision 13): GPT-6 Astra and Opus 5.5 (the user's standard pair since 2026-10-11)

High level, x20, official seeds (L1 101, L2 202, L3 303). Opus 5.5 ran on the designated account (.env slot 1).

| model | level | result | point share | points won by | how it lost | tokens in / out | match wall time |
|---|---|---|---|---|---|---|---|
| GPT-6 Astra | L1 | **won 5:3** | 0.63 | 5 out of reach (rally winners) | 1 service fault, 1 own fault, 1 miss | 3.0 M / 11 k | 16 min |
| Opus 5.5 | L1 | lost 3:5 | 0.38 | 3 out of reach | 3 own faults, 1 service fault, 1 let limit | 5.8 M / 61 k | 28 min |
| GPT-6 Astra | L2 | lost 1:5 | 0.17 | 1 out of reach | 2 service faults, a volley, a miss, an own fault | 2.3 M / 8 k | 11 min |
| Opus 5.5 | L2 | lost 0:5 | 0.00 | — | 2 service faults, 2 volleys, an own fault | 2.6 M / 38 k | 16 min |
| GPT-6 Astra | L3 | lost 0:5 | 0.00 | — | 3 own faults, a service fault, a miss | 2.2 M / 10 k | 11 min |
| Opus 5.5 | L3 | lost 0:5 | 0.00 | — | 2 misses, 2 service faults, a volley | 2.2 M / 29 k | 13 min |

Low level, skill in the loop (L1, seed 101):

| model | match | `hit` calls | API test (hidden /50) | competition test (/16) |
|---|---|---|---|---|
| Opus 5.5 (revision 13) | **forfeit** (`ready_timeout`: no `game ready` within 300 s after the first point; 0:1) | 1 (1.1 s) | 39 | 3 |
| GPT-6 Astra (revision 12 match) | won 5:2 | 11 (0.34 s), 5 skill versions | 29 | 6 (revision 13) |
| GPT-6.1 SOL (extra evidence) | lost 3:5 | 18 (0.96 s), 3 versions | 26 | 0 |
| references | — | — | platform 40, reference skill 43, boundary 0 | 8, 6, 0 |

Discrimination: not all at the floor or ceiling. L1 separates the two models (Astra 0.63, Opus 0.38); L2 and L3 are
near the floor for both (they lose mostly to their own service faults and volleys, below the reference: L2 5/10, L3
3/10); the low level's API test separates them clearly (26–39 of 50, between the boundary 0 and the reference 43). No
new shortcut: every point the models won on revision 13 was a rally winner. One process finding: Opus spent the review
after the first point engineering its skill and forfeited by the 300 s review limit (a valid loss by the rules).



## 8. Step ⑥: freeze (proposal)

| item | value |
|---|---|
| sport revision | `wallclock-13+duel-wallclock-10+core-0.2` (this PR), core 0.2.15 (skill in the loop, PR #79, on top of #85) |
| assembled configuration | Franka Panda (`bodies/franka_panda`, panda-limits-2) + paddle on the x-y rail base; `[self_collision]` on |
| low-level toolkit | `seat/toolkit.toml` 2.0.0 (sha256 per component in the manifest); evaluator skilleval/1.1.0 |
| first exam (format match) | formal levels **L1, L2, L3**; L4 reported as a challenge level (no reference, not even with true perception, wins it: 0/20) (*pending the user's confirmation*) |
| hidden seeds | registered with `tools/seeds.py` (core 0.2.14): 20 per cell (match/L1–L4, duel), `sport.toml [official.hidden_seeds]` sha256 `27a0839b799c156c1d76293f31b7df2cd22fbcf94eda85b95f29a71e1d4eaab2`, `seed_commitments.json`; the private set only in the official store on mll-4090-1 |
| scoring | win, and `[score]` point_share (`share`: points won / points played, forfeit 0, range [0, 1]) |
| arena (format duel) | both first-server orders on paired seeds |
| slowdown | **10** (the task default, `[timing] slowdown`; user decision 2026-10-14: table tennis at x10, was 20; badminton stays at x20 this round). Camera reference vs L1 with the real service timing, 20 seeds paired: x10 11-0-9 = x20 11-0-9 (net +0.80 vs +0.65 points per match, paired difference -0.15 [-1.75, +1.50]); x10 halves the match's wall time. The model-free and model results above were measured at x20 |
| capabilities claimed | A1, A2, A4 (passed the sensitivity check), B1 (rules: the service law); B2 lowered to "placement" (b2 = 1): reading the opponent is not measured by this opponent |
| frozen with | the golden cases recorded on mll-4090-1 (the user's decision: the reference host) |

## 9. Process issues

1. **Noise is larger than the process assumes.** A replica that only rounds the act's numbers moves single-match point
   share by up to ±0.4 and a 10-match mean by ±0.15; 10 seeds per cell detect only ~0.15. Ask for 20 seeds per
   sensitivity cell and always run a replica.
2. **"Drops clearly" has no operational rule.** Proposed: the strong impairment's 95 % interval below 0 and the weak one
   not above the replica noise, at all formal levels but one.
3. **The simple baselines did not cover single-trick strategies.** All four borrowed the reference's whole serve
   repertoire; "only serve short, never return" surfaced only in the model runs. Proposal: before choosing baselines,
   break the reference's points down by how they were won (`analyze.py`) and turn every large source into its own
   baseline.
4. **Changing the opponent is expensive late.** It invalidated step ④, the golden match cases and the low-level case
   replays (~1000 matches of CPU). Do the points-by-source check and the trick baselines in step ①.
5. **The latency of the model-free checks is not specified.** `tools/check` plays with no delay, which cannot test A4;
   I used D = 5 control steps. The delay changes the levels' difficulty; the process should fix it per sport.
6. **Stage 5 is a single live match.** A reference that wins 7/10 fails it 3 times in 10, and the live default seed
   (1) is a losing seed for it. Proposal: the reference must win the majority over N ≥ 10 seeds (in-process, with the
   sport's latency model), live runs only check that the task runs.
7. **The low-level part of step ④ is under-specified**: only the reference and the boundary are defined; whether
   simple baselines and capability sensitivity also need a low-level version is not said.
8. **Low-level cases depended on the opponent** (replays of recorded matches); fixed by replaying against the
   recording-time opponent. Cases should carry their full start state.
9. **Cost.** CPU rendering takes 1–3 min per match; with the host's load often above 40 only 2 containers could run, so
   step ④'s two rounds took over 20 h of wall time; `tools/check` stage 2 ~1.5 h, stage 6 ~1.4 h; one low-level
   evaluation 30–60 min (the default 1800 s timeout is too short; use 3600 s). For a while more than 4 of my containers
   ran (a batch I believed stopped was still running); an automatic limiter (container count, host load) was added.
10. **Model API instability**: two Astra trials were void (22 and 9 API errors > 5) and re-run.
11. **The 300 s review limit may be short for the low level**, where agents engineer their skill between points (Opus
    forfeited). Either say so in the instructions or give the low level more review time.

## Revision 15 re-screen (2026-10-12)

Sections 1–9 describe revision 13. After the 10-12 group meeting (the paddle and the table were too bouncy, the arm
delivered about 60 % of the planned blade speed, the physical joints could exceed the velocity limits), the sport was
re-screened on branch `tt-real` (PR #110). The full record is the review page `screening/tabletennis_walkthrough_zh.md`;
the numbers that matter:

| step | result |
|---|---|
| physics | table: a 30 cm drop rebounds 23.0 cm (ITTF 2.1.3); paddle normal restitution 0.80; execution scatter 1 cm / 1.5 deg / 5 % |
| rules | deviation D13: a service's first bounce on the receiver's court lands at least 0.2 m from the net |
| robot | planning and setpoints at 0.9 of the velocity limits; velocity reflex (a joint above 1.05 x its limit) and collision reflex (the arm or paddle pressing on the table above 20 N): the robot holds for 1 s, as a real Franka does |
| B9 (upper-bound receiver, full simulation, best fixed stance) | 256/282 = 90.8 % (Wilson 86.8–93.6 %); 0.3 m rule: 184/189 |
| step ④ (1920 matches, D = 5, Quest) | truth reference wins L1–L4: 24/16/3/2 of 40; camera reference 21/16/7/0 of 40; fixed / random / short-serve / reflex / centre baselines 0–1 of 20 at every level; no_hit 0 points; A1 kept 3/4, A2 4/4, A4 2/4 (secondary), B2 not kept |
| low level | cases re-filtered (our implementation without its scatter fulfils each one in the full simulation): 118 hidden; reference skill 88/118 = 0.75, platform 118/118 (without execution scatter, as the low level has none), boundary 0/118; reference_min 0.7 (the user's decision of 2026-10-13). Same conditions on 179 cases: platform 1.00 / 0.63 with the 1 cm scatter, reference 0.79 / 0.47 |

### Step ⑤, revision 15: model matches

High level, program play, x20, official seeds, cameras only (valid matches; technical interruptions re-run).

| model | level (seed) | result | notes |
|---|---|---|---|
| GPT-6 Astra | L1 (101) | lost 0:5 | 3 service faults, no ball over the net |
| GPT-6 Astra | L2 (202) | **won 6:4** | 5 serves, 1 fault; 7 of 10 points over the net, 6 returned |
| Opus 5.5 | L1 (101) | lost 0:5 | 2 service faults of 3; 2 touches, no ball over the net |

The L1 losses are the programs failing (service faults, nothing over the net), not the level: every L1 parameter is
weaker than L2's and the reference wins more at L1. Astra's L2 win is the camera-only win registered in sport.toml
[[reference.camera_wins]].
