# Screening report: soccer (2 v 2; 5 v 5 as far as it goes)

Owner: Chi Gui. Process: [../construction_zh.md](../construction_zh.md) v1 (2026-10-10). Code: `sports/soccer`, revision
**soccer-2v2-6** (branch `screen-soccer`, PR #82). Data: `/data/guichi/olympiad_review/screening/soccer_data/` on the lab host
(mll-4090-1): raw JSONL per batch, `summary_*.md`, `gates_final.md`, `walkthrough` (Chinese process record: `olympiad_review/screening/soccer_walkthrough_zh.md`), and the scripts (`scripts/screen.py` World-level
matches of scripted teams, `scripts/cammatch.py` camera-level matches in process, `scripts/kickdrill.py`,
`scripts/physcheck.py`, `scripts/t1check.py` (the B9 check of PR #43), `scripts/rescore.py`, `scripts/gates.py`).
Steps 1-4 ran without any model; step 5 used GPT-6 Astra and Opus 5.5 (the step-5 standard; Opus on Kangrui's account)
and, before that standard was set, GPT-6.1 SOL. No match used the coding session's Claude account.

## 1. Conclusion

**2 v 2 passes steps 1-5 in revision soccer-2v2-6 (no shot-speed rule; official levels L2 and L3, L1 a warm-up;
confirmed by the user on 2026-10-11). 5 v 5 is ported onto the real robot but stops at step 4.**

What the user should know first:

- **With dribbled goals counting, "both robots attack" is the strongest strategy in 2 v 2 with the current skills.** The
  true-state all-forward team beats level 2 (68 %) and beats the reference's division of labour coach (58 %); dividing
  the work does not pay. So the humanoid 2 v 2 mainly tests embodiment and control (seeing the ball with a head camera,
  lining up and timing a 5 s kick, the walking policy's limits); team play is left to the pass / receive skill work and
  to the wheeled SSL 5 v 5. The capability B3 is reduced to "who plays the ball" (B3 = 1).
- **Official levels after the change: L2 (the default and the lowest official level) and L3 (new, a challenge level:
  the all-forward coach at the reference's speed and precision, the strongest scripted team found). L1 is a warm-up.**
  The reference does not beat L3 (expected for a challenge level, as badminton's L3).
- **Step 4 at L2 under the BASEPERC rule**: the true-state all-forward wins L2 (27 / 9 / 4 in 40, 68 %), so it was re-run with camera
  perception at L2: the camera all-forward loses (1 win in 5, 11:28): L2 passes, flagged "this level mainly tests
  perception". At L3 the true-state all-forward loses the majority ((11 / 16 / 13 in 40, 28 %)). **Camera-only-can-win evidence at L2:
  GPT-6 Astra and Opus 5.5 each won a single match against L2 10:0 (mercy rule) with cameras only (§7, registered in
  `sport.toml` `[[reference.camera_wins]]`)**; the camera reference itself wins 2 of 5 against L2 (7/10 against L1).
  Both models crushed L2, so **L2 saturates for strong models; L3 is the level that separates them**.
- **Follow-up**: revisit the scripted opponents (a tackling / body-blocking defence) and the B3 claim once the pass /
  receive skills are done.

| step | check | result |
|---|---|---|
| 1 define | rules, deviations, capabilities, opponents, score | pass after changes: kick with either foot; 2 x 5 min; D14/D15 declared; the shot-speed rule D16 of soccer-2v2-5 removed (user decision); levels L2 + L3 official; continuous score = goal difference per match in regulation |
| 2 robot and venue | real limits, counterpart, B9, physical consistency | pass: Booster T1 limits from Booster's URDF (envelope and command rate ESTIMATES); B9 (shooting drill) re-run 29/30, 0 falls; robot-robot and leg-leg contacts on, no leg penetration > 5 mm; the ball rolls farther than on real turf (D15) |
| 3 interface | P1-P6, modes | pass: high level only; the kick's control authority and its called-off event in the contract |
| 4 model-free | opponent calibration, value of the kick, simple baselines, sensitivity, engineering | pass with the flags above: reference 32/6/2 vs L2, 10 / 15 / 15 vs L3; the kick is worth using at both levels (walk-only minus reference -1.20 (95 % interval -1.68 .. -0.70) at L2, -0.65 (-1.10 .. -0.23) at L3); baselines by BASEPERC; A1, A4 and B3-coordination impairments clearly below the reference at L2, B3 division not; 15/15 exploit tests, golden 5/5 |
| 5 model trials | discrimination; camera-only evidence | pass: Opus 5.5 vs Astra in both seat orders, Astra won 7:1 and 5:3 (re-scored without D16); SOL vs Astra 2:3; all decided in regulation; single matches against L2: Astra 10:0, Opus 10:0 (camera only) |
| 6 freeze | | proposal in §8 |

**Chosen configuration**: Booster T1 lower bodies (bodies/booster_t1 t1-1) on the released walking policy and the
platform swing kick (right or left foot), the boarded 10 x 7 m pitch, HSL 2026 with D1-D15, formats match (levels 2
and 3 official, default 2; level 1 a warm-up), duel and team, x10, 2 x 5 min, review timeout 600 s; score win,
continuous score goal difference; public seeds as now, the hidden official set through the core's common mechanism.

## 2. Candidates and pruning

The configuration was searched over robot x venue x interface x opponent x rules. Earlier results were the search record;
every check of the chosen configuration was re-run on the current code (sections 4-7).

| candidate | evidence | result |
|---|---|---|
| 2 v 2, soccer-2v2-3 (push kick, no real limits) | B9 push shot past a keeper 0/30 (`b9_dynamic_sports_zh.md`); limits not held (G9) | dropped (PR #43) |
| 2 v 2, soccer-2v2-4 (main: real T1 limits, swing kick, right foot only, 2 x 2.5 min) | re-run here (round 0, 180 World-level matches): walking the ball in (no kick) scores 0.5 goals per match and beats level 1 4/10 (by shoot-out); "both strikers" beats L1 10/10 and L2 8/10; competent teams draw (ref vs ref 6/10 coin tosses) | changed |
| + passing under pressure, an open support, a realistic turf (0.6 m/s^2) | round 1 (120 matches against "both strikers"): none made the division of labour pay (GD -0.6 .. +0.15, intervals around 0); passing opened the game (1.5 : 2.1 goals) but the open support left the goal empty | dropped (code removed) |
| a quicker swing (weight shift 1.1 -> 0.8 / 0.6 s) | kick drill: 30/30 falls before the touch at power 1 and 0.5 | not possible with this controller |
| 2 v 2, soccer-2v2-5: left or right foot, 2 x 5 min, a goal only from a shot of >= 2.3 m/s (D16) | passed steps 1-5 (screening of 2026-10-10/11, the first version of this report) | **D16 rejected by the user** (2026-10-11): dribbling the ball in is legitimate soccer |
| **2 v 2, soccer-2v2-6: soccer-2v2-5 without D16; official levels L2 and L3 (new: the all-forward coach at reference speed), L1 a warm-up** | sections 3-7 | **chosen** |
| 5 v 5 (legacy `olympiad-soccer5v5`, push kick, no real limits) | skill audit: P3 fails | ported onto the real T1 + swing kick (world level, §10) |
| 5 v 5 on the real T1 + swing kick | 66 World-level matches: 0.3 goals per match between the reference and level 2, 11/12 draws for "chase" vs positional | stops at step 4 (§10) |
| low level for soccer | G1 two-mode pilot: walking-only 0/30 with the shot rules, the toolkit kick 25/30 (G1, not T1) | not opened (§5) |
| other robots (G1, H1-2) for the high level | no soccer B9 on them with real limits and a kick skill; T1 is HSL's robot and has the B9 pass | not searched further |

## 3. Step 1: define the sport

**Core challenge kept.** A walking policy is given (the real T1 ships with one, and without it every robot would only
be learning to stand). What it leaves to the agent:

- **see the ball and the other robots** with the head cameras only (no ball or opponent coordinates; the camera sees
  the floor from 0.55 m ahead of the trunk, so the ball at the robot's feet is out of view and must be remembered);
- **put the ball into the kick's strike zone** (0.26-0.38 m ahead, 0.02-0.10 m to the kicking foot's side): the
  walking policy reaches only about 0.75 of a command, turns slowly and the command is rate-limited, so lining up is a
  closed-loop control task;
- **choose when, which foot, how hard and where** (power, aim, left or right foot), under a 2.5 s window in which an
  opponent can take the ball or knock the robot over;
- **decide who goes for the ball** and where the other robot stands (2 robots per team), within HSL's fouls (pushing,
  ball holding, illegal defence, game stuck).

**Rules.** RoboCup Humanoid Soccer League Rule Book 2026 (commit 06c44d4, 2026-06-29), Laws 1-17. Deviations D1-D15
(`sport.toml`); new or changed in this screening:

| id | law | deviation | why |
|---|---|---|---|
| D4 | 7 | 2 x **5** min (was 2 x 2.5 min; HSL 2 x 10 min) | decided 2026-10 for 2 v 2; fewer level matches (§6) |
| D14 | 4 | exact self-localization from the simulator (real T1: camera + odometry) | was undeclared (skill audit P6) |
| D15 | 1 | the ball rolls with 0.15 m/s^2 (real turf about 0.6-1.5 m/s^2, ESTIMATE) | physical-consistency finding of step 2, declared, kept (§4) |
| (D16, removed) | 10 | soccer-2v2-5 only: a goal needed the scoring team's last touch to send the ball off at >= 2.3 m/s | added for skill necessity, removed by the user's decision (dribbling in is legitimate); skill necessity is now the "value" check of §6.2 |
| (timing) | - | the review timeout 300 -> 600 s | step 5: the first model duel ended incomplete at a goal review |

Deviations now D1-D15.

**Capabilities claimed** (`[capabilities]`, after the sensitivity check of §6.4): A1 = 2, A2 = 1, A3 = 2, A4 = 1,
B1 = 2, B2 = 2, B3 = 1 (only the coordination of who plays the ball; the division of labour does not pay, §6.4).

**Adjustable parameters**: half length, the ball's rolling deceleration (D15), the scripted levels. Fixed: the robot, its limits, the walking policy, the kick
skill, the pitch.

**Opponents, seeds, scoring.** Scripted team `soccer.Coach`: level 1 (warm-up, not official: reacts 0.6 s late,
0.45 m/s, loose kick tolerances), **level 2** (official, the default: 0.3 s, 0.65 m/s), **level 3** (official, a
challenge level, new in soccer-2v2-6: both robots go for the ball, no delay, 0.8 m/s, the reference's precision: the
strongest scripted team found, §6.3); all kick with either foot. Official seeds: match 202-222 (L2), 303-323 (L3),
duel / team 1-3; the hidden official set comes from the core (decision 2026-10-11: one mechanism for all sports, only
its sha256 published). Score: win (match format: success = winning). The continuous score of soccer (for the core's
common field, decision 2026-10-11) is the **goal difference per match in regulation time** (the shoot-out not
counted); the screening reads it because two decent teams often draw and the shoot-out of two scripted teams ends in
a coin toss.

## 4. Step 2: robot and venue

**Robot.** Booster T1 lower body (sai-mujoco 0.1.30 model = Booster's URDF kinematics), walking policy
`SaiResearch/booster_soccer_models` (MIT). Real limits from `bodies/booster_t1/body.toml` (PR #43): leg torques and
joint speeds from Booster's URDF (S1), the walking command's range from Booster's training config (S2); the
torque-speed envelope and the command rate limit are ESTIMATES (Booster publishes neither). Executed by the shared
`bodies/servo.py` (Actuators, WalkCommand) in every mode, the kick included (exploit test E15). Real counterpart:
Booster T1 (listed in HSL's Large Division; used by several 2026 HSL teams). Arms not modelled (D13).

**Upper-bound feasibility (B9)**, re-run on the current code: the shooting check of PR #43 (30 fixed starts 2.5-4 m
from the goal, the robot behind a 3 m line, a T1-class keeper with 0.30 s reaction and 0.74 m/s), best searched shot
per start against the physical keeper: **29/30, 0 falls** (PR #43: 29/30, 0 falls; threshold 27/30 = 90 %). Kick drill (standing start, ball in the
zone, 30 random headings each): right foot 30/30 clean, median 3.25 m/s, no fall; **left foot 29/30, median 3.19
m/s, no fall** (new); push median 1.40 m/s; walking into the ball at full command: median 1.89, max 2.06 m/s.

**Physical consistency** (`physcheck.py`, two full matches of scripted teams, sampled every 10 ms):

| item | result | grade |
|---|---|---|
| contacts globally on | yes (`disableflags` without contact), no excluded pairs | ok |
| robot x robot contacts (Changhe's audit: the robot library's humanoids never collide) | enabled for all 49 geom pairs of two robots; 505 / 198 contact samples per match, deepest 6.9 mm (soft contact) | ok: soccer's scene has them, unlike the library's G1 test scene |
| left leg x right leg of one robot | enabled for all 9 pairs; closest 0.0 m (touching), never > 5 mm penetration in 6000 samples | ok |
| robot x ball | enabled (7 geoms) | ok |
| fall cost | lies 3 s, then stood up where it lies; real T1 gets up in 2.7 ± 0.3 s (supine) / 2.9 ± 0.5 s (prone) (HiFAR, arXiv 2502.20061, Table VIII) | ok (slightly lenient: no get-up motion) |
| ball rolling | 0.15 m/s^2; FIFA Quality turf ball-roll 4-10 m from a 1 m ramp: about 0.6-1.5 m/s^2 (ESTIMATE) | **too loose**, declared D15; a 0.6 m/s^2 world did not change the conclusions (§6) |
| release speed | ball off the swing kick <= 3.64 m/s (drill max); no published T1 kick speed found | unknown (no real counterpart number) |

## 5. Step 3: interface

**Mode: high level only** (Agent as Athlete). Low level is not offered: the core motor skill left after the walking
policy is the kick, and a low-level kick (G1 pilot: the walking-push boundary baseline scores 0/30 with the shot line
and the 2.2 m/s rule, the agent's own kick 25/30) is measurable as a *separate drill* (`pilots/g1_modes`), but inside a
match the low level would mostly re-test the same kick; kept as the four-criteria candidate below, not opened.

| criterion | low-level soccer (C: walking policy + joint arbitration) | verdict |
|---|---|---|
| feasible | G1 pilot: reference kick from the toolkit 25/30 | yes (on G1; not ported to T1) |
| non-trivial | walking-only boundary 0/30 with the shot rule | yes |
| extensible | arbitration by joint group exists for G1 only | not for T1 yet |
| measurable | API test of the kick (shooting drill) exists for G1 | yes on G1 |

Low level for soccer = port the G1 two-mode pilot to T1 (arbiter, IK, toolkit manifest): about 3-5 days, later.

**Skill contract** (high level, `sport.toml [spec] actions`, `/spec`): per robot

| | walk | stand | kick |
|---|---|---|---|
| inputs | v = [vx, vy, wz], body frame, each [-1, 1] | - | power [0, 1]; aim [-0.2, 0.2] rad; **foot right / left (new)**; style swing / push |
| execution | next 20 ms step; replaces the running command; rate-limited 2 / 2 / 4 per s | same | swing: waits up to 1 s for the robot to stop and the ball < 0.15 m/s, touch 2.4-2.9 s, end 4.7-5.6 s; push 2.6 s |
| feedback | state: command, walk_now | same | state: phase, ends_in_s; **event kick_cancelled with the reason (new)** |
| control authority | the walking policy reads only proprioception | - | **reads the ball's position once at the start; moves the foot's path by <= 0.08 m sideways, strike point clipped 0.05-0.32 m ahead and 0.03-0.12 m high; never follows a rolling ball, predicts or aims (now in the contract)** |
| failure | 409 when fallen or penalized | same | called off (event) when the robot or ball does not settle within 1 s; a fall ends the kick |

P1 (action targets): yes. P2 (no decisions): yes; the one-time ball read is execution, bounded and declared. P3 (real
limits): yes (E15). P4 (complete): §6.1. P5: §6. P6: the contract above, in `/spec`.

Rejected skill changes: a quicker swing (shorter weight shift 1.1 -> 0.8 s or 0.6 s: 30/30 falls before the touch at
every power, `drill/q*.jsonl`): not possible with this controller; kicking a rolling ball / a contact time (audit
item 2a): the controller would have to predict the ball (P2) or the agent would give a foot target in the world, a new
skill API, not done in this round.

**Observation**: head-camera tiles (480 x 360) per own robot, own state (exact pose, D14), referee information, own
events. No tactical camera in 2 v 2 (5 v 5 has one). **Time**: real time at x10 (official), x1/x2/x5 possible.
**Formats**: `match` (coach mode against the scripted team), `duel` (coach mode, model against model), `team` (one seat
per robot, team channel 512 B / 0.2 s). Model trials use coach mode (§7).

## 6. Step 4: model-free validation

**Two levels of matches.** A camera-level match (`cammatch.py`: `core.local.play`, the players get exactly the
whitelisted observation and head-camera images a model's program gets) takes about 25 minutes of CPU at 2 x 5 min; a
World-level match (`screen.py`: scripted teams on `soccer.World`, true state, no rendering) about 1.5 minutes. The
World level carries the 20-seed checks; the camera level confirms. Score: the goal difference per match (GD) in
regulation; wins are regulation wins (between equal teams many matches end level and the scripted shoot-out ends in a
coin toss).

**Rounds** (data directory; World-level r0-r7, camera-level c2-c12):

| round | code | what | finding |
|---|---|---|---|
| r0 | soccer-2v2-4 as on main | 180 World-level matches | walking in scores 0.5 goals per match against L1; "both strikers" beats L1 9/10, L2 6/10 |
| r1 | + passing, open support, 0.6 m/s^2 turf | 120 matches against "both strikers" | none makes the division of labour pay; dropped |
| r2 | + both feet, 2 x 5 min | order-A half of the matrix | walking in beats L1 10/10 (23:2) -> the shot-speed rule D16 (soccer-2v2-5) |
| c2-c12, r3, r4 | soccer-2v2-5 (D16) | the full step 4 under D16 and the six fix rounds of the camera reference (first version of this report) | all gates passed under D16; the camera reference needed a press rule to win L1 |
| rescore | soccer-2v2-5 records | 35 camera-level and model episodes replayed with the D16 code; the goals D16 had disallowed added back | below |
| r5 | soccer-2v2-6 (no D16) | L2: reference, walk-only, all-forward (20 seeds x both kick-offs), sensitivity (10 seeds x both) | all-forward wins L2 68 %; division of labour does not pay |
| r6 | soccer-2v2-6 | the reference's coach as a candidate L3 | all-forward beats it 14/6/4: not usable as the top level (stopped at 24 matches) |
| r7 | soccer-2v2-6 + L3 = the all-forward coach | reference, walk-only, all-forward vs L3 (20 seeds x both) | below |

**Why re-scoring is valid, and where it was not possible.** D16 never changed the play: a disallowed goal restarted
exactly like a goal (the other team's kick-off, a review), so with and without D16 the world runs bit-identically
until the score itself ends a match (the 10-goal mercy rule) or decides whether regulation ends level. Checks: the two
golden episodes replayed under soccer-2v2-6 reach the same final step as under soccer-2v2-5 (30115 and 30270), only the
score and state hash change (2:6 -> 2:7, 3:2 -> 4:3); and in round r2 the 51 matches played both with and without the
rule had identical goal times in 50 (the other ended early by the mercy rule). Camera-level and model matches were
therefore re-scored from their saved episodes (`epevents.py`: each episode replayed with the D16 code, the goals D16
had disallowed in regulation added back). The World-level batches r3 and r4 had not logged the disallowed goals, so
the World-level cells were played again (r5, r7) instead of re-scored.

### 6.1 Reference: opponent calibration and end-to-end playability

Opponent calibration (the true-state reference wins the lowest official level, its results fall with the level):

| opponent | seeds / matches | W / D / L | win rate | GD per match |
|---|---|---|---|---|
| L1 (warm-up) | 20 / 40 (soccer-2v2-5, D16) | 36 / 4 / 0 | 90 % | +3.48 |
| **L2** | 20 / 40 | 32 / 6 / 2 | **80 %** | +1.45 |
| **L3** (challenge) | 20 / 40 | 10 / 15 / 15 | 25 % | -0.23 | |

Camera-only can win (the camera reference, re-scored without D16; `summary_camrescore.md`):

| camera reference | matches | W / D / L | goals for : against |
|---|---|---|---|
| vs L1 (warm-up) | 10 | 7 / 2 / 1 | 32 : 19 |
| vs L2 | 5 | 2 / 0 / 3 | 11 : 15 |

At L2 the camera reference wins 2 of 5, not a majority; **the camera-only evidence at L2 comes from the models**: Astra
and Opus 5.5 each won a single match against L2 10:0 with cameras only (§7, "Model matches against L2"; registered in
`sport.toml` `[[reference.camera_wins]]`). At L1 the camera reference wins 7 of 10.

`tools/check` stage 5 runs this check through the sport's `[reference]` match hook
(`sports/soccer/reference_check.py:reference_match(format, level, seed, perception)`): `perception="truth"` plays the
scripted reference coach (team A, true state) against the level's coach in the World, `perception="camera"` plays the
built-in camera Reference through `core.local.play`; both return `{won, scores, report}` with the playability counters
"kicks not called off" and "kicks without a fall". Declared seeds: truth 1-20 per official level (L2, L3), camera
202, 212, 222, 232, 242 at L2.

End-to-end playability of the camera reference (from the existing camera records, per match, no gate):

| opponent | matches | swing / push kicks commanded | kicks called off (robot or ball not settled) | falls during a kick | all falls | pushing offences | goals |
|---|---|---|---|---|---|---|---|
| L1 | 10 | 35.9 | 9.7 | 3.8 | 4.2 | 0.9 | 3.2 |
| L2 | 5 | 28.8 | 7.8 | 3.6 | 3.6 | 0.2 | 2.2 |

About a quarter of the commanded kicks are called off (the ball still rolling or the robot not yet still) and about one
in ten ends in a fall; the camera reference scores 2-3 goals per match.

The D16-specific adaptations of the camera reference (no walk-in tap in front of the goal; the push only far from the
goal) were kept: re-scored without D16 the reference wins L1 7/10 (before the adaptations, without D16: 4/5, c2), and
testing a revert needs new camera matches (about an hour per five); the other adaptations (never walking into an
opponent, pressing the ball, guarding the goal when the ball is lost) do not depend on D16.

### 6.2 Value of the kick (skill necessity, "value" type)

User decision 2026-10-11: dribbling the ball in is legitimate; the kick need not be strictly necessary, but the
kicking reference must be clearly better than the same decisions without kicks (walk-only: every kick replaced by
walking through the ball), on the same seeds, at every official level: walk-only minus reference, paired GD
difference, 95 % bootstrap interval below 0, 20 seeds x both kick-offs.

| level | pairs | walk-only W / D / L | walk-only GD | reference GD | walk-only minus reference | 95 % interval | gate |
|---|---|---|---|---|---|---|---|
| L2 | 40 | 13 / 21 / 6 | +0.25 | +1.45 | -1.20 | -1.68 .. -0.70 | pass | |
| L3 | 40 | 0 / 13 / 27 | -0.88 | -0.23 | -0.65 | -1.10 .. -0.23 | pass | |

### 6.3 Simple baselines (BASEPERC rule)

Rule (user decision 2026-10-11): baselines with the true state first; a baseline that wins a level is re-run with
camera perception at that level only, and losing with the camera passes that level, flagged "this level mainly tests
perception"; at the highest official level even the true-state baselines must lose the majority. At least 20 seeds per
level (World level), camera runs confirm.

| baseline | level | true state: seeds / matches, W / D / L, win rate | camera re-run | verdict |
|---|---|---|---|---|
| all-forward (every robot a striker, reference execution) | L2 | 20 / 40, 27 / 9 / 4, **68 %** | camera all-forward (`chase`, re-scored without D16): 1 / 0 / 4 in 5, 11 : 28 | **pass, flagged: L2 mainly tests perception** |
| all-forward | **L3 (top)** | 20 / 40, 11 / 16 / 13, 28 % | - | **pass** (the true-state baseline loses the majority at the top level) |
| stand, random commands | L2 | 20 / 40 each: 0 / 0 / 40 and 0 / 0 / 40 (soccer-2v2-5; they score nothing, D16 only removed their opponents' slow goals) | - | pass |
| all-forward | L1 (warm-up, not official) | 36 / 2 / 2 (soccer-2v2-5) | camera: 8 / 0 / 2 in 10 without D16 | fails: one reason L1 is no longer official |

The all-forward team also beat the reference's own decisions at full speed (the first L3 candidate, r6: 14 / 6 / 4 in 24
matches); that is why L3 is the all-forward coach itself.

### 6.4 Capability sensitivity (L2, paired with the reference, 95 % bootstrap)

| capability | weak impairment | GD - reference (pairs, 95 % interval) | strong impairment | GD - reference | gate |
|---|---|---|---|---|---|
| A1 3D perception | ball seen with N(0, 0.15 m) error | -0.20 (10, -0.90 .. +0.60) | 0.35 m | -5.10 (10, -7.60 .. -3.00) | pass |
| A4 time and latency | acting on the world 0.3 s ago | +0.20 (10, -1.10 .. +1.40) | 0.6 s | -1.70 (10, -2.80 .. -0.50) | pass |
| B3 division of labour | both strikers half the time | +0.30 (10, -0.70 .. +1.30) | both strikers always | L2 -0.20 (40, -0.70 .. +0.38); L3 +0.20 (40, -0.28 .. +0.68) | **fail** |
| B3 coordination (who plays the ball) | - | - | the striker drawn at random every 2 s | -3.50 (10, -5.00 .. -1.90) | pass |

Gate: the strong impairment below the reference at all official levels but one (here: at L2). A1 (ball position
error), A4 (latency) and B3 coordination (who presses drawn at random) pass; **B3 division of labour fails** (both
strikers is not worse), so the B3 claim is reduced to coordination (B3 = 1). A2, B1, B2 not measured (no clean
impairment that keeps the skill); C1/C2 need models.

### 6.5 Engineering and isolation checks

| check | result |
|---|---|
| `tools/check soccer --stages 1` (contract) | 53/53 |
| the sport's 15 exploit tests (E16 removed with D16) in the soccer image | 15/15 |
| B9 (the shooting drill, physical only, user decision 2026-10-11) | 29/30, 0 falls (soccer-2v2-5; the shot rule never applied in the drill) |
| golden `match-L1-s101`, `duel-ref-s1-A` re-expected on soccer-2v2-6 (same commands, same final step); `tools/check soccer --stages 3` | 5/5 |
| determinism | the re-scoring replays reproduced every recorded camera and model match |
| fairness (first kick-off, seat) | World-level ref vs ref, soccer-2v2-5, 20 matches: first kick-off 9/20, seat A 11/20 (WARN: interval +-22 pp) |
| stage 2 (generic tests in the image) | not run: about 20 camera matches (8-10 h) |
| stage 5 (reference hook + live tasks) | hook declared and smoke-tested (truth and camera, short halves); the full run was not started in this session (the run was refused by the session's permission check; about 2-3 h for 40 truth + 5 camera matches with `--jobs 2`, plus about an hour per format for the live tasks). Command: `PYTHON=<python 3.11+> tools/check soccer --stages 5 --jobs 2`, at host load < 40 |

## 7. Step 5: model trial runs

Coach mode (`duel`: each model commands its whole team), Codex 0.158.0 for the GPT models, Claude Code for Opus,
reasoning effort medium, no web search, x10, official duel seed 1, both seat orders. The step-5 standard is GPT-6 Astra
+ Opus 5.5 (user decision 2026-10-11; Opus on Kangrui's account, the run tree's `.env` slot 1, never the coding
session's); the SOL match was played before that standard was set and is kept as an extra. The matches were played
under soccer-2v2-5 (with D16) and are **re-scored without D16** from their saved episodes (§6: the play is identical).

| match | seats (A / B) | with D16 (as played) | **without D16 (re-scored)** | D16 had disallowed (A : B) | reviews | model requests / output tokens (A; B) |
|---|---|---|---|---|---|---|
| 1 | SOL / Astra | incomplete at 3:08 (neither seat answered a goal review within 300 s) | - | - | 8 | 90 / 13 k; 98 / 22 k |
| 1 (re-run, review timeout 600 s) | SOL / Astra | Astra 3:1 | **Astra 3:2** | 1 : 0 | 16 | 232 / 25 k; 236 / 26 k |
| 2 | Opus 5.5 / Astra | Astra 4:1 | **Astra 7:1** | 0 : 3 | 20 | 137 / 64 k; 229 / 31 k |
| 3 | Astra / Opus 5.5 | Astra 5:3 | **Astra 5:3** | 0 : 0 | 18 | 242 / 25 k; 118 / 64 k |

**Discrimination: pass.** All valid matches were decided in regulation time (3-8 goals each, no shoot-out); Astra beat
Opus in both seat orders and SOL once. Three matches are not a ranking.

**Shortcuts and loopholes: none found.** Replayed referee events: the models also walked or poked balls in (the goals
D16 had disallowed, now legal), committed pushing (1-2 per team per match) and leaving-the-field offences, had 0-5 swing
kicks called off per team, and all three used the left foot in some match (SOL 4, Astra 12, Opus 7 left-foot kicks);
Opus used the push 10 times in one match. No rule was exploited.

### Model matches against L2

One single match per model against the default level L2 (format `match`, official seed 202, team A kicks off, x10,
soccer-2v2-6 + core 0.2.14; the same harness and settings as the duels: Codex 0.158.0 for Astra, Claude Code 2.1.284
for Opus 5.5 on Kangrui's account, reasoning medium, no web search). The models see only the robots' camera images.

| model | result | ended | steps (of 30 000) | reviews | requests / output tokens | kicks (A1 ; A2) | goals |
|---|---|---|---|---|---|---|---|
| GPT-6 Astra | **won 10:0** | mercy rule | 23 448 | 10 | 190 / 19 k | 50 ; 3, none called off, no falls | A1 all 10 |
| Opus 5.5 | **won 10:0** | mercy rule | 17 633 | 10 | 46 / 26 k | 3 ; 3, no falls | A1 4, A2 4, two own goals by B2 |

Both are valid, deterministic, official-configuration trials (job directories
`jobs/scr5-soccer-astra-L2-s202-20261011T201928Z` and `jobs/scr5-soccer-opus-L2-s202-20261011T215450Z` on the screening
host). They are the camera-only-can-win evidence at L2 and are registered in `sport.toml` `[[reference.camera_wins]]`.
The two models won differently: Astra kicked (50 swing kicks by one robot, none called off), Opus mostly dribbled the
ball in. **Reading**: L2 is far below the models (a one-sided 10:0 for both), so L2 does not separate strong models; L3
(the all-forward coach, which the reference does not beat) is the discriminating level. One match per model, not a
ranking; L3 has not been played by a model.

## 8. Freeze (proposal)

| item | value |
|---|---|
| sport revision | soccer-2v2-6+core-0.2 (world soccer-2v2-6) |
| robot | booster_t1 (body revision t1-1), walking policy SaiResearch booster_soccer_models (`models/robot/lower_t1_control.pt` sha256 697cac61dbed2e76…), kick `swing_kick.py` |
| rules | HSL 2026 (06c44d4) with D1-D15 (no shot-speed rule) |
| formats | match (levels 2 and 3 official, default 2; level 1 a warm-up), duel, team (channel 512 B / 0.2 s) |
| timing | x10 official; 2 x 5 min; review timeout 600 s; agent budget 10 800 s |
| seeds | public: match 202-222 (L2), 303-323 (L3), duel / team 1-3; the hidden official set through the core's common mechanism (only its sha256 published; decision 2026-10-11), 20 seeds per level; **registered 2026-10-13** (`tools/seeds.py`): 20 each in match/L2, match/L3 and duel; **team: preview** (no cell, never official; one agent per robot is still in screening, owner 2026-10-13) |
| score | win (match: success = winning a valid match); continuous score `[score]` goal_difference (`from = "margin"`, regulation goals; forfeit -10, the mercy-rule bound; range -10..10; core 0.2.14) |
| reference check | `[reference] match = "reference_check:reference_match"` (truth seeds 1-20 per level, camera seeds 202-242 at L2); camera-only evidence `[[reference.camera_wins]]`: Astra and Opus 5.5 won L2 10:0 |
| before freezing | run `tools/check soccer --stages 5 --jobs 2` (not yet run); register the hidden seeds |
| capabilities | A1 2, A2 1, A3 2, A4 1, B1 2, B2 2, B3 1 (coordination only) |
| not frozen | 5 v 5 (§10), the low level (§5) |

Confirmed by the user (2026-10-11): L3 = the all-forward coach as the challenge level; B3 reduced to coordination;
the humanoid 2 v 2 mainly tests embodiment and control. The camera-only evidence at L2 now exists (both models won L2
10:0); L2 saturates for strong models, L3 is the discriminating level. Follow-up: revisit the opponents (tackling / body-blocking defence) and the B3 claim once the pass /
receive skills are done.

## 9. Process issues

| issue | what happened | what we did | suggestion for `construction_zh.md` |
|---|---|---|---|
| a rule for skill necessity can be the wrong fix | D16 made "walk-only" score nothing, but it also outlawed legitimate dribbled goals; it cost the camera reference two thirds of its goals and six fix rounds, and the user removed it | replaced by the "value" check (the kick must be clearly better than walk-only at every level) | name the "value" type of skill necessity for sports where the core skill is one option among several |
| simple baselines: which perception | the true-state all-forward beats L1 (90 %) and L2 (60 %) through perfect perception | BASEPERC rule (user decision): true state first, camera re-run where a baseline wins, true state must lose at the top level | (now decided) |
| the strongest simple strategy can beat every scripted coach | without D16 the all-forward team beat L2 and the reference's own coach | the top level became the all-forward coach itself (challenge level) | say what to do when a "simple" strategy is the best known one: the top level may play it, and the capability it removes is not claimed |
| capability sensitivity: against whom | against an equal team nearly every match is level | measured against the official levels | say "against every official level" |
| P4 "a meaningful match" | equal scripted teams draw often | read the goal difference; 2 x 5 min | quantify P4 |
| camera-only-can-win evidence | at the new lowest level L2 the camera reference wins 2/5; step 5 had only played duels | one single match per model against L2 (both won 10:0), registered in `[[reference.camera_wins]]` | plan one single match per model against the lowest official level in step 5 by default |
| cost | a camera match 25 min; the host's load (40-60) allowed one container for hours; this follow-up needed about 600 World-level matches | World-level for the 20-seed gates; camera only to confirm; re-scoring instead of replaying where the episodes existed | log every referee event of every screening match, so a rule change can always be re-scored |
| `tools/screen` | one container per match, camera players only | own World-level harness, same gate rules | a World-level hook in `tools/screen` |
| model trials | the first duel ended incomplete at a 300 s goal review | review timeout 600 s | check the review cadence before step 5 |
| time | ① ② ③ about 3 h; ④ about 14 h wall (first version) + about 8 h (this follow-up); ⑤ about 7 h | | |

## 10. 5 v 5: how far it got

**Port (step 2 prerequisite).** The legacy 5 v 5 (`/data/guichi/rcb-oly-soccer5`, branch `olympiad-soccer5v5`,
7dc17e301, rules v4 / revision soccer-4: HSL 2026 Large Division "Advanced" on the M-Field 14 x 9 m, a goalkeeper,
kick-ins, goal kicks and corner kicks, leaving the field, a tactical camera, a ten-seat team task) ran on the old body
(no real limits, the push kick: P3 fails in the skill audit). Its world changes were a clean three-way merge onto
main's world (base: the legacy file main was migrated from, identical; 5 conflicts, all resolved by hand), so the
world now takes the team size as a parameter (`SOCCER_TEAM_SIZE` = 2 or 5) and 5 v 5 runs on the **real T1 limits and
the swing kick** with both feet. Revision `soccer-5v5-1` (world only). Not ported yet: the plugin's
tactical camera observation, the 5 v 5 `sport.toml` (rules and deviations of the legacy `DEVIATIONS_5`), the ten-seat
team format and its generated task, the camera reference for 5 v 5 (the legacy `cam_player.py`, 580 lines, as an
`act(view)` player), golden records, the 5 v 5 exploit tests (33 legacy tests) and the fairness run.

**Model-free checks at the World level** (true state, scripted coaches, 2 x 5 min, 6 seeds x both kick-offs;
`f1.jsonl`, `summary_f1.md`; played under soccer-2v2-5's shot rule, not re-run: so few goals that the rule hardly
matters):

| team A (plan) | team B | n | GD per match (95 % CI) | goals per match A : B | W / D / L |
|---|---|---|---|---|---|
| positional, reference speed | positional L1 | 12 | +1.58 (+0.57 .. +2.59) | 1.58 : 0.00 | 8 / 4 / 0 |
| positional, reference speed | positional L2 | 12 | +0.17 (-0.27 .. +0.60) | 0.25 : 0.08 | 2 / 9 / 1 |
| chase (every field player goes for the ball) | positional reference | 12 | +0.08 (-0.09 .. +0.26) | 0.08 : 0.00 | 1 / 11 / 0 |
| chase | positional L2 | 12 | +0.08 (-0.09 .. +0.26) | 0.08 : 0.00 | 1 / 11 / 0 |
| walking only (no kick) | positional L2 | 12 | -0.08 | 0.00 : 0.08 | 0 / 11 / 1 |
| positional reference (mirror) | positional reference | 6 | -0.17 | 0.00 : 0.17 | 0 / 5 / 1 |

Matches take about 3 minutes of CPU each (10 robots).

**Verdict: 5 v 5 stops at step 4.** With a real 5 s kick, a goalkeeper and four field players per side, competent
teams almost never score (0.1-0.3 goals per match except against level 1), so no check that compares programs can
pass: the division of labour (chase vs positional) and everything else end level. The legacy 5 v 5 scored 0.85 goals
per match only because its push kick was quick. What 5 v 5 needs before step 4 can pass is the same thing 2 v 2 lacks
between equal teams (§6.6): a kick that can be taken in a contest (a shorter weight shift needs a new kick controller;
a kick at a rolling ball needs a contact-time skill API). Remaining work after that: the 5 v 5 plugin and tasks (the
list above), about 3-4 days, then steps 4-5 again.

## 11. Passing and receiving (soccer-2v2-7 / soccer-5v5-2, 2026-10-11)

The skill set had no passing or receiving (B3 weak, the robots rush the ball together; 5 v 5 deferred until passing
works). A separate study (walkthrough `olympiad_review/screening/soccer_pass_walkthrough_zh.md`, data
`olympiad_review/screening/soccer_pass_data/`) went through steps 2-4 again for passing; 5 v 5 was the main target.

**Step 2 (upper bound under the real T1 limits, the controller knowing the true ball).**

| question | result |
|---|---|
| receive a rolling ball (0.5-2 m/s) | standing in its path stops it at the feet (path within +-0.05 m of the centre: 8/8 within 0.12 m; +-0.10 m: 4/4 within 0.18 m); a timed foot block (prototype) did worse (0.4-1.3 m, falls when low); walking backwards as a cushion did worse. With 2.5 s warning a robot 1 m off the path gets there 10/12 times. No `receive` skill: receiving is prediction and walking |
| a wider kick direction | outward (away from the support foot) 0.4 rad as reliable as 0 rad, 0.6 rad 5-8/10 clean, 0.8 rad falls at low power; across the support foot falls from 0.2-0.4 rad at high power; a sideways (no-turn) kick fell 24/24 |
| kick a rolling ball | a swing timed to a given point and time: 57/60 rolling balls struck (front, +-45 deg, +-90 deg, 0.5-2 m/s), 0 falls in 112; needs the command about 2.6 s ahead; a crossing ball turns the shot by 0.3-0.6 rad per m/s of its sideways speed |
| physical consistency (new finding) | the ball hardly bounces anywhere (dropped from 2 m it rebounds 0.01 m; FIFA turf: 0.60-1.00 m); a rolling ball against a leg stops almost the same with a realistic bounce (its topspin eats the rebound), but the same swing sends the ball 22-30 % faster (power 1: 3.6 -> 4.7 m/s): a world-wide decision, left to the owner |

**Step 3.** `kick` gets an optional `at = [t, x, y]` (the timed kick: the ball's centre at (x, y) of the team frame t s
from now; no wait for a still ball, the ball never read; re-sending before the swing moves t and the point; late
strikes announced with `kick_late`) and a wider aim (0.6 rad outward, 0.3 across; the ball's place per aim in
`/spec robot.strike_zone.swing_by_aim`). No `receive`, no `pass(teammate)`. Exploits E17, E18.

**Step 4 (World level, no D16).** The reference coach learned to pass (open-play passes at the moment it kicks,
cut-back passes to a teammate armed with a timed kick, 5 v 5 free-kick set plays). Paired against the same reference
without passing (goal difference per match, 95 % bootstrap interval):

| format | level | matches | passing reference minus reference |
|---|---|---|---|
| 2 v 2 | L1 | 10 | -0.70 (-3.10 .. +1.40) |
| 2 v 2 | L2 | 10 | -2.00 (-2.90 .. -1.10) |
| 5 v 5 | pos:L1 | 20 | +0.40 (-0.65 .. +1.50) |
| 5 v 5 | pos:L2 | 20 | +0.20 (-0.35 .. +0.80) |

Passing does not pay at the scripted levels: a pass costs the passer's line-up (2-5 s) plus the 2.6 s kick
preparation and the ball's roll, while possession windows of 6 s or more are rare (5 v 5: 1-3 per 10-minute match);
one-touch passes almost never complete (5 of 27 planned passes kicked, 1 touched). Camera level: a 30-line camera-only
rule strikes rolling balls with the timed kick 9/20 times (the same rule on the true ball: 17/20).

Division of labour (B3) does not pay with passing either: the passing reference against all-forward (2 v 2 L3) 6 / 6 / 8;
5 v 5 against pos:L2 the all-chase team scores 0.95 goals per match more than the positional reference (+0.50 .. +1.40)
and 0.75 more than the passing one; breaking the pass coordination (the receiver ignores the pass) costs nothing. An
opponent that leaves a robot forward to counter-attack makes division pay in 2 v 2 at level-1/2 speeds (+1.8, +1.5
goals per match over all-forward) but is weaker, and the advantage vanishes at reference speed (+0.1); in 5 v 5 the
all-chase team stays ahead (+1.2). 5 v 5 simple baseline: the true-state all-chase team wins 30/30 against pos:L1 and
25/5/0 against pos:L2, so the 5 v 5 levels must be redesigned before 5 v 5 can enter the leaderboard.

## 12. Ball physics, the in-walk kick, and why passing does not pay (soccer-2v2-8, 2026-10-12)

Report: `olympiad_review/screening/soccer_physics_pass_zh.md`. Evidence and sources:
`olympiad_review/screening/soccer_phys_data/`; `sources.md` has every real-robot and turf number with its citation.

**Timeline (before).** A 5 v 5 pass took the following, as medians:
- decision -> command 4.8 s;
- command -> contact 3.1 s;
- 2.2 m of roll in 1.7 s.

8 of 9 received passes were lost before the receiver kicked again. The nearest opponent was 1.2 m from the ball when a
swing kick was commanded, about 2 s of walking away, against the 2.7 s preparation. As a result, 41-46 % of swing
kicks were touched by an opponent before contact.

**Artefacts vs. the real robot.** Three things were simulation artefacts:
- The swing kick was twice as slow as the real T1's. The real T1's learned kick is part of the approach step, about 1.5 s from 1.5 m away.
- The ball hardly bounced: a 2 m drop rebounded 0.01 m, where FIFA turf gives 0.60-1.00 m.
- The rolling deceleration was too low: 0.15 m/s^2, where the FIFA roll test gives 0.51-1.28 m/s^2.

Walking speed (0.65-0.8 m/s) and ball speed (at most about 4 m/s) are the real robot's.

**Fixes.** These are listed in the soccer README:
- the ball's restitution, calibrated to FIFA;
- a thin-shell ball;
- `ROLL_DECEL` 0.52 m/s^2;
- a new `kick style step`, the in-walk kick: 0.8 s command -> contact, 3 falls in 290 drill kicks.

No limit was loosened.

**Re-test.** World level, Delta portable mode, 80 paired matches per cell, all without D16. Three worlds: before, physics
only, physics plus step. Results:
- Passing reference minus the same off-ball moves without passes: -0.11..+0.04 goals per match. No cell is significant.
- 5 v 5: more passing is worse. With 8 passes per match the team loses 0.80 goals per match (-1.23..-0.33).
- All-chase is the strongest plan in every world: +1.2 / +2.5 / +2.8.
- Reachable receptions: a spot that the ball reaches before every opponent exists at about 1 % of on-ball moments after the fixes, and at 7-12 % even with a zero-time kick.

**Decision (owner, 2026-10-12):**
- Humanoid soccer tests embodied execution and position contest. B3 declares coordination only.
- Team tactics belong to the wheeled 5 v 5.
- The physics fix goes into the 2 v 2 line.
- Humanoid 5 v 5 is parked.
- The counter-attack opponent is not a level.


## 13. Step 4 again on the new physics; the levels kick with the in-walk kick (soccer-2v2-9, 2026-10-12 night)

World level (true state), 20 seeds x both first orders = 40 matches per cell, Quest native; data
`olympiad_review/screening/soccer_v2_data/` (q4, q4c). Decisions taken for the owner overnight are marked *(for the owner)*.

**The swing-kick levels are too weak on soccer-2v2-8.** Any team using the in-walk kick (`style step`) beats them:
the reference's decisions with the in-walk kick win L1 40/0/0 (+7.45), L2 40/0/0 (+6.12), L3 31/6/3; with swing kicks only,
the same decisions win L2 20/15/5 and lose L3 1/17/22.

**The strongest play is all-chase with the in-walk kick.** Head to head (40 matches each), all-chase + step beats
striker + cover + step 33/3/4 and a coordinated chase (the second robot presses 0.9 m behind the ball, never
double-committing) + step 29/4/7. No coordinated plan beats it, so coordination is not measured either: **B3 is no
longer declared (b3 = 0)** *(for the owner, rule "not measurable, not declared")*.

**soccer-2v2-9** *(for the owner)*: the 2 v 2 levels keep their speed, reaction and precision and kick with the in-walk
kick; `Coach("reference")` (every robot goes for the ball at the reference's speed, in-walk kick) is the stage-5 truth
reference. Calibration (40 per cell):

| team A | L1 | L2 | L3 |
|---|---|---|---|
| reference (all-chase + step) | 40/0/0, +8.55 | 40/0/0, +6.67 | 15/7/18, +0.12 (-0.50 .. +0.80) |
| striker + cover + step | 39/1/0 | 35/1/4 | 4/3/33 |
| coordinated chase + step | 40/0/0 | 36/4/0 | 7/4/29 |
| walk only (no kicks) | 5/18/17 | 0/11/29 | 0/1/39 |
| stand / random | 0/0/40 | 0/0/40 | 0/0/40 |

Value of the kick: walk-only minus the reference at L2 -7.83 (-8.53 .. -7.08). Capability sensitivity at L2 (paired with
the reference): A1 ball noise 0.35 m -11.70 (-12.70 .. -10.70), A4 0.6 s late -10.82 (-11.78 .. -9.88), random choice of
who plays the ball -9.47. Camera level: the built-in camera reference and camera chase (swing kicks only) lose even the
soccer-2v2-8 swing-kick L2 (0/1/9, 1/0/9) and L3 (0/10 each); the soccer-2v2-6 `[[reference.camera_wins]]` were
removed, and the camera-only evidence comes from the step-5 model matches against the new L2.

**Open decision for the owner.** The truth reference is now the simplest soccer strategy. (a) Accept humanoid 2 v 2 as an
embodied-execution sport (discrimination from cameras and execution, L3 headroom 38 %), recommended; or (b) treat
all-chase as a simple baseline with a coordinated reference: then the baseline wins L2 40/40 and does not lose L3
significantly (15/7/18), so L3 fails the BASEPERC gate, and no level that punishes all-chase was found (the
counter-attack opponent of 2026-10-12 did not).

### Step 5 on soccer-2v2-9

Match format, x10, cameras only (the players see the head-camera images, their own state and the referee), 2026-10-13.

| model | level | seed | score | valid | note |
|---|---|---|---|---|---|
| GPT-6 Astra (Codex 0.158.0, medium) | L2 | 202 | **12:2 won** (mercy rule) | yes, 0 API errors | 256 kicks, 3 falls; registered in `[[reference.camera_wins]]` |
| Opus 5.5 (Claude Code) | L3 | 303 | 4:6 lost (full time) | yes, 0 API errors | 13 kicks, 2 falls |
| GPT-6 Astra (Codex 0.158.0, medium) | L3 | 303 | **11:5 won** (full time) | yes (1 transient API error; an earlier run lost 5:8 but was invalid: 17 transient errors) | registered in `[[reference.camera_wins]]` |

Camera-only play beats the default level L2 decisively and can beat the challenge level L3; the two models differ at L3 (Astra 11:5, Opus 4:6). tools/check stage 5 on soccer-2v2-9: 10/11 when run, the one failure (camera-only can win) predates the registered wins and passes with them (the truth calibration L2 20/20, L3 12/20; the live tasks of all three formats pass).


## team_match (preview, 2026-10-13)

Format `team_match` (task `soccer-team-match-wallclock-program-standard`): one seat per robot of team A (A1, A2; one
side; head camera and own state each; the teammates talk only through the team channel) against the scripted team of
`match` (levels 1-3; the seeds of `match`). **Preview, not official** (`[formats.team_match] preview = true`): no hidden
seeds, `official_config = 0`, tools/check reports its camera gate as a WARN; it becomes official in a later soccer
revision once a model camera win is registered and the freeze checks pass (user decision 2026-10-13).

- Truth reference per seat (`reference_check.py`: one `Coach("reference")` per seat, its teammate as of its latest
  0.2-s message): L1 40/40, L2 39/40 (+5.3 goals), L3 21/40 (one-seat match: 40/40, 40/40, 22/40). Stand and random
  lose 0/20 at every level. Calibration passes.
- Camera reference: the swing-kick Reference loses L2 0/20 in both formats (team_match -5.2, match -3.9); with the
  in-walk kick (`StepReference`) team_match L1 13/20 (+1.0), L2 5/20 (-1.7), L3 0/20, match L2 3/20 (-2.8). In
  team_match the Reference plays the in-walk kick (its camera reference); `match` keeps the swing until the next soccer
  revision. The camera gate needs a registered model win.
- Team channel: 4000 messages per robot and match (was 3000: a robot talking every 0.2 s uses exactly 3000 in a full
  regular time of 600 s, so the shoot-out after it would be refused).
- Data and report: review `screening/soccer_team_match_zh.md` (Quest runs tm-g1t, tm-g1 .. tm-g4).
## 14. Coach formats with an overhead view: match_overhead, duel_overhead (2026-10-13)

Soccer formats now have two axes. Control is either **coach** (one seat commands its team: match, duel) or **team** (one seat per robot, team channel). Perception is either **onboard** (each robot's head camera) or **overhead** (fixed cameras over the field, like RoboCup SSL's shared vision). `match_overhead` and `duel_overhead` are coach + overhead: the same world, the same scripted levels and the same official levels (2, 3) as `match` and `duel`. Only the images differ: there are no head cameras.

**Overhead rig** (`sports/soccer/overhead.py`):
- Two cameras 6.0 m above the floor look straight down, one over each half. Each tile is 600 x 800 px and covers its half of the boarded area at head height (1.15 m), plus 0.4 m past the halfway line.
- Every team sees the field in its own frame: image right = its +x, tile 0 = its own half. For team B the cameras are team A's turned by 180 degrees, so both teams get the same pictures of the same situation.
- The images are ideal (no noise or distortion) and show the world 40 ms ago, as SSL's vision does. The calibration is in the game spec (`camera.overhead`).
- The observation is otherwise that of match and duel: the own robots' exact state and the referee's information.

**Overhead reference** (`builtin_players.Reference`, used when the spec has `camera.overhead`):
- It finds the orange ball and the opponents' colour in both tiles and casts the pixel's ray onto the plane of the ball's centre or the opponents' shoulders.
- Every robot goes for the ball and kicks with the in-walk kick, the truth reference's play.
- Detection during play: ball error median 0.03 m and p90 0.65 m (occlusion by the robots); opponent error median 0.06 m.

**Step 4** (Quest, in-process camera matches, 20 seeds per cell; the truth level is the same world as `match`, section 13):

| camera player (overhead) | L1 | L2 | L3 |
|---|---|---|---|
| reference (all-chase, in-walk kick) | 8/3/9 (7.5 : 6.6 goals) | 3/0/17 (6.5 : 9.2) | 2/0/18 (2.9 : 9.6) |
| chase_overhead (all-chase, swing kick; the simple baseline) | 4/2/14 | 1/1/18 | 0/0/20 |

- Simple baselines lose every official level: chase_overhead as above; stand and random lose every match at the truth level.
- Variants of the camera reference at L2 (20 seeds each) did not do better:
  - striker + cover: 1/0/19;
  - looser line-up of the in-walk kick: 2/2/16;
  - the same plus striker + cover: 0/0/20;
  - deciding every 2 control steps instead of 5: 1/0/9 (10 seeds).
- The gap to the truth reference (L2 40/40) is perception, latency and decision rate. So the "camera-only can win" evidence for `match_overhead` must come from a model match (step 5).

**Fairness, `duel_overhead`** (reference against reference, 20 seeds x both first orders): the first mover and seat A each won 22 of 40. The advantage is +5.0 ± 15.4 pp: no advantage beyond ±5 pp, but the interval is wide (as for `duel`).

**Stage 2 on the overhead formats** (core.testkit split by test on Quest): all pass.
- Determinism in process and fresh-process replay (match_overhead 5:7, duel_overhead 11:1).
- No draw; whitelist; observe is pure; malformed and fuzzed actions refused; solo quit; pre-start no-show; continuous score; seed private.
- Frame budget: 605 ms per observation (budget 1000 ms).
- Every exploit test.

**Hidden official seeds**: `match_overhead/L2`, `match_overhead/L3` and `duel_overhead` (20 each) were added to the registered set. The existing cells are unchanged. **Goldens**: `match_overhead-L1-s101` (reference vs L1, 10:4) and `duel_overhead-ref-s1-A` (13:4), on soccer-2v2-10, recorded with docker on a local host.

**Step 5 and stage 5 of the overhead formats.**
- **GPT-6 Astra won `match_overhead` L2 11:1** (mercy rule, seed 202, x10, the overhead cameras only, valid, 0 API errors). It is registered in `[[reference.camera_wins]]`.
- tools/check stage 5 for `match_overhead`:
  - truth reference: L2 20/20, L3 9/20;
  - camera reference: L2 0/5 (the registered model win stands in for it);
  - live tasks: the camera reference plays `soccer-match-overhead` through to the end (valid, deterministic), and nop gets success 0.
- Goldens re-recorded with docker on a local host, on soccer-2v2-10 (the realistic T1 motors of #129): `match_overhead-L1-s101` 10:4 and `duel_overhead-ref-s1-A` 13:4. The in-process recordings made on Quest do not replay bit for bit on the local hosts: the walking policy's torch kernels differ between the CPUs.
