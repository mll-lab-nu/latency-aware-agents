# Screening report: wheeled robot soccer 5 v 5 (RoboCup SSL Division B)

Owners: Chi Gui with the screening lines A (robot), B (world and referee), C (overhead cameras), D (scripted opponents,
reference player, step-④ hooks; integrator of this branch from 10-12), E (plugin and tasks), and SSLONB / SSLONBREF
(the second perception mode, onboard cameras). Process: [docs/construction_zh.md](../construction_zh.md).

Chinese walkthroughs for review (host mll-4090-1, `/data/guichi/olympiad_review/screening/`):
`soccer_ssl_proposal_zh.md` (①), `soccer_ssl_interfaces_zh.md` (decisions), `soccer_ssl_body_zh.md` (A),
`soccer_ssl_world_zh.md` (B), `soccer_ssl_camera_zh.md` (C), `soccer_ssl_players_zh.md` (D), `soccer_ssl_plugin_zh.md`
(E), `soccer_ssl_onboard_zh.md` and `soccer_ssl_onboard_ref_zh.md` (onboard). Data: `soccer_ssl_data/` next to them.
Scripts: `tools/screen/` (shared), `sports/soccer_ssl/screen/` (world-level batches, paired comparisons).

This report covers the **main mode (overhead cameras: formats `match`, `duel`, `team`)**, the official formats of this
freeze. The onboard formats (`match_onboard`, `duel_onboard`) stay in the plugin as a **preview, not official**
(coordinator, 10-13): no generated task, no hidden seeds, because there is no onboard camera reference yet; the onboard
work continues later (per-robot onboard perception, first on the humanoid 2 v 2).

## 1. Conclusion

The overhead mode passes steps ①–⑥. On 2026-10-13 the user chose to strengthen L2's defence against shots (camera
shoot-only had won 52.5 % of L2 under the measured service latency) and to recalibrate at the declared reaction delay
D = 2 control steps (core 0.2.18, `sport.toml [timing] reaction_delay_steps = 2`). All step-④ numbers below are at
D = 2 (§6.6). Step ⑤ was re-run on the new L2 (§7); the earlier trials (Opus 4:3, Astra 3:10 against the old L2) are
kept as the old revision's record. The configuration below is proposed for the freeze. Decisions taken for the user
are marked *decided for the user*.

| step | check | result |
|---|---|---|
| ① define | core challenge, 17 numbered deviations (S1–S17), capability claims | pass; after step ④: A2 down to secondary, A1 up to primary (*decided for the user*) |
| ② robot & venue | real limits with sources; feasibility; physical consistency | pass: TIGERs v2020 limits (3 m/s, 3 m/s², kick ≤ 6.5 m/s legal); upper-bound probe 20/20 (pass, receive, turn with the ball, shoot); dribbler = contact + capped grip (S16, user decision) |
| ③ interface | P1–P6 for `move` / `kick` / `dribble` (+ `batch`, `substitute`) | pass (contracts in `skill_api.py`, the schema is generated from them) |
| ④ reference (D = 2) | truth reference ≥ 20 seeds per official level; camera-only evidence | L1 20/20; L2 70-12-18 (100 seeds, +2.05 goals/match); L3 68-31-101 (200 seeds); camera reference vs L2 26-6-8 in 40 (65 %, Wilson 50–78 %; +1.68 goals/match) |
| ④ baselines (D = 2) | every simple baseline loses at the official levels | L2 with truth: shoot-only 34-7-59 (−0.93 [−1.52, −0.33] goals/match), all-chase 21/100, fixed 2/40, random and idle 0; camera shoot-only 24-9-47 in 80 (30 %, Wilson 21–41 %; −1.01 [−1.64, −0.39] goals/match). L3: shoot-only 45/200, all-chase 3/200, fixed 1/40, random and idle 0 |
| ④ skill necessity (D = 2) | passing (value skill): reference vs the same program that never passes | per match L2 +2.34 [1.39, 3.27], L3 +0.85 [0.18, 1.52] (paired, 100 / 200 seeds) |
| ④ capability sensitivity (D = 2) | A1, A2, A4, B2, B3 weak/strong ablations, 40 paired seeds, noise floor from replicas | A1, A4, B3 clearly sensitive; B2 at the strong level; A2 within the noise |
| ④ engineering | `tools/check` stages 1, 2, 3, 5, 6 | §6.5 |
| ⑤ models | discrimination, shortcuts | new L2: valid trials, Opus 5.5 0:10 and Astra 2:7 (both lost), no shortcut; old L2 (old revision): Opus 4:3, Astra 3:10 (valid) |
| ⑥ freeze | versions, seeds, levels, score | proposed (§8): revision soccer_ssl-1, D = 2, hidden seeds registered (20 per cell: match L2/L3, duel, team; onboard formats are a preview), official levels L2 and L3, score = goal margin |

Chosen configuration: ten SSL robots after TIGERs Mannheim v2020 (`bodies/ssl_robot`, planar base at the real limits,
impulse kicker and chip kicker, contact dribbler with capped grip), Division B field 9 × 6 m, orange golf ball with
TIGERs' fitted two-phase model; two halves of 300 s game time, shoot-out when level; overhead vision of four cameras
40 ms late; declared reaction delay D = 2 control steps; skills `move`, `kick`, `dribble`; formats match (scripted opponent L1 warm-up, L2, L3 official), duel,
team; slowdown 10; score = goals for minus goals against.

## 2. Candidates and pruning

| decision | candidates | kept | evidence |
|---|---|---|---|
| sport | humanoid 2 v 2 only; add wheeled SSL 5 v 5 | add SSL: it tests strategy and team play, while humanoid 2 v 2 tests embodiment. The SSL rules (1 m dribbling limit, crash fouls, defense areas) already make "everyone chases the ball" illegal or costly | proposal §0 |
| perception | detection results (SSL-Vision-like); overhead images; onboard cameras | overhead images as the main mode (keeps A1); onboard as a second, separately ranked mode | proposal §3, onboard report |
| dribbler model | pure contact; contact + capped grip | contact + capped grip (S16): pure contact loses the ball at 0.55 m/s sideways and about 3 rad/s turning; the grip's release limits match ZJUNlict v2023 measurements | body report §2 |
| match length | 2 × 300 s at slowdown 10 (≤ 4 h); 5 × slowdown; shorter halves | 2 × 300 s at slowdown 10 (user, 10-11) | plugin report §3 (1) |
| L3 reaction delay | 0.1, 0.05, 0 s | 0 (user, 10-12): all-chase wins 27.5 % at 0, 40 % at 0.05 | players report §3.4 |
| opponent shooting | reference decisions without / with the aggressive shooting rule | with (option A, user 10-12): without it, shoot-only beat L3 (140/200) | players report §2.4 |
| counter-attacking opponent | official level or optional variant | optional variant (`scheme="counter"`): it punishes swarming strongly, but passing stops paying (gap −0.03) | players report §2.4 |

## 3. Step ①: define the sport

**Core challenge kept.** Five robots must work as a team under rules that punish solo play: read the game from overhead
images that arrive 40 ms late, predict a fast sliding and rolling ball, and decide who presses, who receives, where to
run and when to pass or shoot, in a world that keeps running while the program thinks. Neither the platform nor the
rules do this for the agent: there is no `pass(teammate)` or `receive()` skill, the kick goes along the robot's heading,
and the observation holds no ball or opponent coordinates (exploit E3).

**Rule book.** RoboCup SSL rules, 2026-09-19 edition, Division B. Deviations S1–S17 are in `sport.toml
[[deviations]]`: automatic referee (S1), overhead vision (S2), referee ball placement (S3), no halt or timeouts (S4), no
overtime (S5), team communication (S6), vision pattern (S7), fitted ball model (S8), 5 v 5 and cards (S9), no tipping
(S10), substitution after cards (S11), wall height (S12), robot model (S13), own state from own sensors (S14), slowdown
and reviews (S15), dribbler (S16), onboard mode (S17).

**Capabilities** (`sport.toml [capabilities]`, after step ④): primary A1 (reading the overhead images), A4 (fast ball,
latency, real time), B2 (pass / shoot / fall back, reading the defence), B3 (passing, off-ball running, defensive
roles); secondary A2 (ball dynamics), B1 (rules); none A3 (the platform executes the skills). Step ④ moved A2 from
primary to secondary and A1 from secondary to primary (§6.4; *decided for the user*; the framework requires at least
one of A1–A3 to be primary).

**Opponents, seeds, scoring.** Scripted levels L1 (warm-up), L2 and L3 (official); public seeds 1–3; hidden seeds 20 per
cell; score `goal_margin` (regulation goals for minus against, forfeit −10, range −10..10).

## 4. Step ②: robot and venue

- **Real limits** (body report): TIGERs Mannheim v2020 dimensions (diameter 178 mm, height 148 mm, 2.62 kg),
  omnidirectional base 3 m/s and 3 m/s² (symmetric deceleration, stricter than the real robot), kicker flat up to
  8.5 m/s (legal ≤ 6.5 m/s, direction σ 1°, speed σ 3 %), chip up to 5.5 m/s at 45°, capacitor recharge 0.5 s,
  break-beam sensor, team colours on the sides (rule 3.2.3).
- **Venue** (world report): Division B field, walls, goals, defense areas; golf ball with TIGERs' two-phase model
  (sliding 3.4 m/s² down to 5/7 of the initial speed, then rolling 0.45 m/s²; measured in the world, players report §2.1).
- **Feasibility (upper-bound player)**: the plugin's probe with truth, using only the platform's skills: 20/20 for a
  2.5 m pass at 3 m/s received and held for 0.5 s, a turn with the ball, and a goal from 3 m at 5 m/s. Drills with the
  reference: unopposed passing 10/10; one-touch redirect 24/24; keeper vs 5.5 m/s shots from 2.5–5 m 27–30/30.
- **Physical consistency**: all contacts on (robot-robot, robot-ball, walls, goals); kick speeds capped at the legal
  limit by the referee; the dribbler's grip releases at the published limits.
- **Speed**: about 3.0–3.5 × real time for ten robots without rendering; one observation (4 cameras) 68 ms in the
  match image.

## 5. Step ③: interface

Modes: high only (`move`, `kick`, `dribble`, plus the robot-handler commands `batch` and `substitute`); a `low` layer is
declared as the servo interface for the framework rule but not generated as a task. Contracts (`skill_api.py`,
the plugin schema is generated from it): inputs and ranges, timing (an act takes effect at max(the next control step,
the seat's latest world state + D), D = 2 steps (core 0.2.18); vision 40 ms late), feedback (each command's reply and the unit's events), control authority (the robot's own servo; `kick` may be
armed to fire when the ball reaches the break beam), failure handling (refusals, illegal kicks). Observation: four
800 × 600 overhead images 40 ms late, own robots' state from their own sensors (S14), referee signals; no ball or
opponent state, no seed (E3). Time: realtime protocol, slowdown 10, reviews at the half and after goals (S15).

## 6. Step ④: model-free validation

All batches play at world level with 2 × 150 s halves unless noted (decision of 10-12: short halves for screening, full
matches for confirmation). Since 10-13 every batch plays at the declared reaction delay D = 2 (`screen/worldrun.py`'s
default): a player's act takes effect 2 control steps after its observation and it observes again 6 steps after the
effect (the reference program's loop in the official service: act, wait 0.05 s, observe); a truth variant sees the
ball and the opponents as the cameras show them (40 ms late) and its own robots now. Before 10-13 the harness observed
every 4 steps with effect 4 steps later and the truth of the moment (`SSL_LAT_OLD=1`; review LATENCY). Paired
comparisons use the same seeds; intervals are 95 % bootstrap. Cluster batches ran on Quest (native, bit-identical to
the local host; 20 matches of the merged code without any setting reproduce the Quest results exactly).

### 6.1 Scripted opponents (`sports/soccer_ssl/opponents/`)

They read the truth, use the same skills and limits as the agent, and react with a delay only to sudden changes (a
rolling ball is extrapolated, the ball at their own robot is felt at once; user decision 10-12).

| level | delay | speed cap | defence | attack | exploitable habit |
|---|---|---|---|---|---|
| L1 warm-up | 0.4 s | 1.5 m/s | keeper + four chase the ball | shoot past midfield, no passing | everyone chases, the back is empty |
| L2 official | 0.2 s | 2.5 m/s | man marking with a double press, loose marks (0.6 m), one forward outlet; shot defence (10-13): an angle-keeping keeper that reads the shooter's heading, and a shot screen within 5 m of the goal | the reference's decisions incl. aggressive shooting, no chips | outside the shooting zone always two at the ball: one attacker is free; the keeper follows the shooter's heading 0.2 s late |
| L3 official | 0 | 3.0 m/s | man marking, a shot blocker, tight marks (0.4 m), an extra screen when the opponent swarms | the reference's decisions incl. aggressive shooting, chips only to clear | marks follow runners: off-ball runs open space |

### 6.2 Reference check (D = 2)

| player | L1 | L2 | L3 |
|---|---|---|---|
| truth reference (`builtin_players.Reference`) | 20-0-0, +9.05 | 70-12-18, +2.05 goals/match (100 seeds) | 68-31-101, −0.57 (200 seeds) |
| camera-only reference (overhead images only) | — | 26-6-8 in 40 (65 %, Wilson 50–78 %; +1.68 goals/match) (seeds 71–110) | — |

Camera detection (colour thresholds, `builtin_players.py`): ball within 0.3 m of the truth in 96–99 % of frames, ball
error median 0.4–1.4 cm, opponents' median error about 2 cm. A camera match of 2 × 150 s costs about 50 min of CPU.

### 6.3 Simple baselines and skill necessity (D = 2)

| program (truth perception) | L1 (warm-up) | L2 | L3 |
|---|---|---|---|
| reference | 20-0-0 | 70-12-18 | 68-31-101 |
| never passes (same decisions otherwise) | gap −3.75 | gap −2.34 [−3.27, −1.39] | gap −0.85 [−1.52, −0.18] |
| shoot-only | 20-0-0 | 34-7-59 (34 %, Wilson 25–44 %; −0.93 [−1.52, −0.33]) | 45-22-133 |
| all-chase (with the reference's movement layer) | 17-2-1 | 21-13-66 | 3-11-186 |
| fixed positions | 16-3-1 | 2-3-35 | 1-1-38 |
| random, idle | 0/20 | 0/40 | 0/40 |

Camera perception at L2 (seeds 71–150): shoot-only 24-9-47 in 80 (30 %, Wilson 21–41 %; −1.01 [−1.64, −0.39] goals/match); on the seeds both played the camera reference scores
+2.95 [1.75, 4.12] goals a match more. Every simple baseline loses L2 significantly, with truth and with camera perception. L1 is
the warm-up: simple baselines win it by design.

### 6.4 Capability sensitivity (D = 2, new L2; 40 paired seeds 101–140, variant minus reference, goals per match)

| ablation | L2 weak / strong | L3 weak / strong | verdict |
|---|---|---|---|
| A1: ball-position noise 1 cm / 5 cm, velocity from 0.1 s differences | −3.00 / −8.68 | −1.43 / −6.45 | sensitive → primary |
| A2: rolling deceleration wrong 50 % / no sliding phase and 100 % wrong | −0.88 / −1.68 | −0.20 / −0.78 | within the noise → secondary |
| A4: act on the world 0.1 s / 0.2 s old, no prediction | −6.80 / −8.97 | −4.42 / −6.97 | sensitive |
| B2: ignore opponents for the receiving point / and the pass lane | −0.38 / −1.32 | −0.20 / −3.40 | sensitive at the strong level (L3) |
| B3: random striker every 2 s / everybody presses | −5.50 / −3.60 | −4.50 / −4.38 | sensitive |
| replicas (0.1 / 0.2 mm target nudges, chaos only) | −1.45 / −2.15 | −0.53 / +0.53 | noise floor about ±2 at 40 seeds |

The verdicts are those of the old harness (40 ms, old L2), re-run on 10-13. A2 is insensitive because the receiver
corrects only sideways in the last 1.2 m and the dribbler catches anything below about 4 m/s: a wrong ball model costs
little. This is a property of the sport's physics, not of the reference.

### 6.5 Engineering checks (code of 10-13: new L2, D = 2, merged with main / core 0.2.18)

- `tools/check soccer_ssl --stages 1`: 59/59 (incl. #101's `[body]`, the `reaction_delay_steps` declaration = hooks
  DELAY).
- Stage 2 on a reduced configuration (run on mll-4090-2; a full stage 2 renders about 20 000 frames per seat per match,
  over 100 h): 48/48. Every quick test in full (observe_is_pure, refuses_bad, fuzz_actions 10 sets, frame_budget
  47 ms per observation, exploits E1–E11, solo_quit, prestart_no_show, continuous_score, seed_private); determinism
  (twice in process + a fresh-process replay) and no_draw / whitelist (16 matches, all decided, no leak) without
  rendering, seed 1.
- Stage 3: 3/3 on the goldens re-recorded at D = 2 (match 10:1, k 51864; duel 6:4, k 96834).
- Stage 5 (the stock checks, run with one container per seed / task in parallel on mll-4090-2): 11/11. Opponent
  calibration with the truth reference L2 19/20, L3 17/20 (full matches through the plugin); camera reference L2 4/5;
  the live tasks: match (camera reference, valid to its end), duel and team (oracle on every seat, success 1), nop on
  every seat 0. Not covered by the stock tool: `tools/live.task_entry` cannot parse a format name with "_"
  (`match_onboard`); moot now that the onboard formats have no tasks.
- Stage 6: not applicable (no `[modes.low]`).
- `ci_checks/run_all.sh`, `scripts/task_conventions.py --check`, `tools/test_check.py`, `test_program_play` on the
  soccer_ssl tasks, `test_run_plan`, `tools/generate.py check`: pass; GitHub CI green on #101 and #107.

### 6.6 L2's shot defence (10-13)

**Problem.** Under the measured service latency (review LATENCY), camera shoot-only won 42-8-30 against L2 and the
camera reference was only +0.88 [−1.31, +3.06] goals a match better. At D = 2 with truth, shoot-only won 29-0-11 of 40.

**Diagnosis** (8 traced matches of shoot-only against the old L2, D = 2): its goals came from 1.5–3.5 m (conversion
75 % under 2 m, 66 % at 2–3 m, 26 % at 3–4 m, 13 % at 4–5 m, 0 of 76 from 5 m or more). With a 0.2 s reaction delay
the keeper has about 0.25 s to move after it sees a 6 m/s shot from 2.8 m and cannot cover a far corner; standing
further out does not help (covering the whole mouth from 2.8 m needs 1.9 m, outside the defense area). In 43 of 45 goals
no defender stood on the shot's line: the double press rarely gets goal-side in time.

**Alternatives tried** (D = 2, truth, 40 seeds; finalists 100): keeper angle keeping alone (shoot-only 23/40); a shot
screen alone (22/40); both (20/40); plus no risky build-up passes (24/40) or no forward outlet in the zone (25/40),
both dropped; angle keeping + reading the shooter (35/100, goal difference −0.58 [−1.22, +0.05]); a keeper with its own
0.1 s delay (42/100; it changes the level's delay, not adopted); a 0 s keeper that reads the shooter (the reference
also loses: 13/40, too strong). **Adopted: angle keeping + reading the shooter + the shot screen** (100 seeds:
reference 70/100, shoot-only 34/100 −0.93 [−1.52, −0.33], all-chase 21/100, passing +2.34).

**What L2 does now** (`opponents/__init__.py`, `LEVELS[2]`: `keeper="angle", keeper_read=True, screen_r=5.0,
screen_prio=True`):
- *angle keeping*: the keeper stands on the bisector of the angle the goal mouth subtends from the ball, 0.15 m out
  plus 0.2 m per metre the ball comes closer, at most 0.5 m, inside the defense area;
- *reading the shooter*: when an opponent controls the ball within 6 m and its heading line enters the mouth, the
  keeper stands on that line (a flat kick leaves along the robot's heading, σ 1°); it sees the heading 0.2 s late,
  carried forward by the angular velocity for at most 0.15 s;
- *shot screen*: with the ball within 5 m of the goal, the second presser stands on that bisector 0.9 m from the ball
  (its body covers the corners a shooter aims for at 2–4 m), filled right after the presser even when short-handed,
  and holds its place; in possession there, the safety stands on the same line 1.2 m from the ball.
L1 and L3 are unchanged (bit-identical on the same seeds); the keeper's default placement moved into
`TeamBase.keeper_spot()` without any change. Exploitable: a feint (aim at one corner, turn late) beats the reading
keeper, and so does a one-touch finish after a pass.

## 7. Step ⑤: model trial runs

Format `match` against L2, seed 1, full match at slowdown 10, through Harbor (`scripts/olympiad/run_single.sh`),
cameras only: GPT-6 Astra (Codex, Azure) and Opus 5.5 (Claude Code).

**New L2 (10-13, D = 2, commit 48d2a1d3):**

| model | score | steps | agent time | notes |
|---|---|---|---|---|
| Opus 5.5 | 0:10 (lost; 10-goal limit) | 68 460 | 9 155 s | valid, deterministic, 0 API errors; 17 fouls, 3 yellow + 2 red; 3 of the 10 goals were its own keeper's |
| GPT-6 Astra | 2:7 (lost; full time) | 90 120 | 10 782 s | valid, deterministic, 1 transient (0 unrecovered) API error; 9 fouls, 3 yellow; 2 of the 7 goals were its own keeper's |

Both models read the brief, wrote camera-only controllers, played the match to its end and were graded; 0 refused
commands, goals in open play, no shortcut. Neither won the new L2 in one try (Opus had won the old L2 4:3); the
camera reference wins it 26-6-8, so "camera-only can win" rests on the reference (§6.2). Step ⑤ passes; whether L2
should be easier for models is listed for the user.

**Old L2 (10-12; old revision, kept as a record; Codex 0.158.0, Claude Code 2.1.284):**

| round | model | score | steps | model calls | notes |
|---|---|---|---|---|---|
| 1 | Astra | 7:8 (lost) | 87 014 | 320 | 9 fouls, 2 yellow + 1 red; grading failed (harness), replay exact |
| 1 | Opus 5.5 | 1:7 (lost) | 95 563 | 189 | 14 fouls, 2 yellow + 2 red: STOP-speed fouls (1.8–2.7 m/s against the 1.5 m/s limit) cost two robots by 163 s |
| 2 | Astra | 6:8 (lost) | 97 925 | 342 (17 failed) | 9 fouls, 3 yellow; graded, but a technical failure (17 failed model calls under Azure rate limits) → re-run |
| 2 | Opus 5.5 | **4:3 (won)** | 93 515 | 148 (0 failed) | 11 fouls, 3 yellow; **valid trial, success 1** |
| 2b | Astra (re-run) | 3:10 (lost; 10-goal limit) | 79 018 | 308 (0 failed) | 9 fouls, 3 yellow; **valid trial** |

In round 2, Astra's model calls were throttled when several agents ran Astra seats at once. The first 110 minutes had
5 rate-limit failovers to another Azure deployment, absorbed in 1–4 s each; the agent saw no error. Later there were
94 failovers and 17 failed calls, which is a technical interruption, so the match is re-run. The re-run and Opus play
from the freeze commit with the agent time limit raised locally to 6 h: Astra's match took 11 552 s, close to the
4 h limit. Since the freeze, the default seed is hidden, so the trials name public seed 1.

Opus's round-2 controller (about 850 lines) reads the overhead images with colour thresholds and assigns roles
(attacker, support, defensive spots, keeper). Its four goals came in open play (no penalties); two of L2's three goals
were own goals by Opus's robots. Neither its code nor the match record shows a rule or platform loophole. The old L2 could thus be won by a model from the images alone, and it was
not a free win (Opus 1:7 in round 1; Astra 7:8, 6:8 and 3:10). The two models are separated and no shortcut was found:
step ⑤ passed on the old L2.

Round 1's grading failed with "cwd outside of container mount namespace" in the simulator's collect hook: the task
directory was regenerated during the runs. The match content is valid: `core.episode` replays of both episodes give
the live state hash, score and step count exactly. Both models wrote full controllers and played to the end; neither
found a shortcut. Opus's own notes name the STOP speed limit and wall-aware set pieces as its main problems.

## 8. Freeze (proposed)

- Versions: sport revision `soccer_ssl-1` (L2's defence changed before any official result, PR not merged: not bumped,
  *decided for the user*), world revision `soccer_ssl-world-1`, core 0.2.18, image
  `ghcr.io/mll-lab-nu/eai-olympiad:0.1.0`, MuJoCo 3.3.7.
- Declared reaction delay: `[timing] reaction_delay_steps = 2` (= `screen/hooks.py` DELAY; the official service, local
  matches, goldens and screening all use it).
- Hidden seeds: 20 per cell (duel, match/L2, match/L3, team), registered in `sport.toml [official.hidden_seeds]`
  (sha256 0199230b…) with public commitments in `seed_commitments.json`. The onboard cells (duel_onboard,
  match_onboard/L2, match_onboard/L3) were dropped from the set of 10-12 (the other cells keep their seeds); the private
  set is kept off the repository and backed up.
- Official levels: L2 and L3 (L1 warm-up). Score: `goal_margin`.
- Goldens re-recorded on this code (new L2, D = 2, core 0.2.18): `match-L2-s1-reference` camera reference vs L2 10:1 (10-goal rule), k 51864; `duel-s1-A-reference` 6:4, k 96834. Stage 3: 3/3.

## 9. Process issues

- **The screening harness's command latency disagreed with the official service** (40 ms against a measured median of
  20 ms; review LATENCY). Fixed by the declared reaction delay (core 0.2.18): the service, local play and screening now
  share D = 2 (all of §6, the ablations included, re-run at D = 2).
- **Close-range shooting is strong in this physics** (6 m/s flat kicks, 40 ms vision): a program that shoots on sight
  beats a slow keeper. A level needs a real goal defence, else it only tests perception (§6.6).
- **Selection and validation shared seeds.** The adopted L2 was chosen on seeds 101–200; its numbers come from a fresh
  run of the same seeds (bit-identical). L1, L3 and the camera matches were run only after the choice.
- **World-level samples are expensive.** A full match takes 7–8 min of CPU, and a camera match of 2 × 150 s takes
  about 40 min. Screening therefore used short halves on the Quest cluster (thousands of matches) with full-match
  confirmation. Stage 2 at full size needs about 100 h with rendering.
- **The process does not say how strong a simple baseline's movement layer should be.** We gave all-chase the
  reference's movement layer (collision avoidance, legal tackling, set-piece positioning), which is its strongest
  version (user, 10-12).
- **How to implement the opponent's reaction delay was undefined.** Taken literally ("sees the world 0.2 s ago"), the
  opponent could not even receive passes. The delay applies to sudden changes only (user, 10-12).
- **Small samples mislead.** The same program moved from 2-3-1 to 4-0-2 between 6-seed batches. Every conclusion here
  uses ≥ 40 paired seeds; the replicas showed a ±1.5-goal noise floor at 40 seeds.
- **Never regenerate tasks while a model run uses them.** It voided round 1's grading (§7).
