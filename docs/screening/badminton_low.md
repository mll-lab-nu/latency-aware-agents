# Badminton, the low level (Agent as Robotics Engineer): screening

Branch `badminton-low` (on `screen-badminton`, with the shared `skill-in-loop` merged). The high level's screening is
`badminton.md`; steps 1 and 2 (the sport, the robot, B9) are shared and not repeated. Built as table tennis's low level.

## 1. Conclusion

Two rounds: round 1 on the old 5 m/s belt gantry, round 2 on the EGC-185-BS-KF-40P ball-screw gantry (2 m/s, 15 m/s^2
per axis; no automatic return; `move`; `act`'s optional `base`). Round 2's rows are provisional until the high level's
opponent and levels (board item BMOPP) and the shared JerkLimiter fix (#94) are settled: both change the cases' start
states and the servo, so the cases and rows are re-recorded then.

| criterion (docs/construction_zh.md, the low level's four) | round 2 evidence | result |
|---|---|---|
| feasible | our act from the toolkit only: API test 259/279 (0.93, threshold 0.8); competition test 7/8 | pass |
| non-trivial | the IK block (pose at a stance, no swing): API test 0/279, competition test 0/8 with 0 points | pass |
| extensible | all 9 joints (base 2 + arm 7) by joint targets at 100 Hz, checked only against the real limits; every tool optional | pass |
| measurable | weakened references: no rehearsal 40, rehearsal without the servo state 42, base held 14, nearest IK 248 (/279); round 1 model trials: Opus and Astra skills 148 / 149 of 154 and 8/8 (success rates saturate; point share 0.97 vs 0.79 and median errors 4 vs 13 mm separate them) | partly (success rates saturate for frontier models; continuous scores reported as the main ones) |

Configuration: toolkit 1.0.0 (`sports/badminton/seat/toolkit.toml`): the robot-only model cut out of the match scene, the
kinematics library `bm_kinematics.py` (FK, Jacobian, damped least-squares IK, multi-start, self-collision clearance, the
base's go-to law with per-axis limits from game spec), the servo model `bm_servo.py` (the robot's own 1 kHz servo law,
gains from game spec; a rollout from the observed state follows the simulator within a micrometre). Not given: the
platform's stroke controller. The low level's observation adds `q`, `qd`, `qd_target`, `servo`. Deliverable
`/app/output/skill/act.py`: `act(t, pos, normal, vel[, base], *, state, spec)`; contract `skill_api.py` (2 cm, 3 degrees
for either face, 0.15 m/s + 10 % at the contact step). Evaluated in an isolated container (`tools/skill_eval.py`); skill
in the loop with `RCB_SKILL_INLOOP`.

## 2. Cases (round 2)

12 matches of our truth-perception reference player with the platform's act (L1-L3, 4 seeds each; at most 50 legs per
match): 672 candidates (a leg's first act, and its last act when it re-planned: a new act from a moving arm), 406
feasible (the platform without its execution scatter delivers them within 15 mm, 2.5 degrees, 0.10 m/s + 8 %, base in
time, no self-collision) -> 279 hidden (54 re-plan) + 127 public, split by leg; every request carries `base` (the
platform's stance). E17 replays all 279 and passes.

## 3. Results

| implementation | round 2 API test | round 2 competition (L1, L2) | round 1 API test | round 1 competition |
|---|---|---|---|---|
| platform (high level, execution scatter) | 243/279 | 3/4, 0/4 (point share 0.48) | 125/154 | 4/4, 2/4 (0.56) |
| reference (toolkit only) | 259/279 | 4/4, 3/4 (0.64) | 131/154 | 4/4, 1/4 (0.54) |
| boundary (IK block) | 0/279 | 0/4, 0/4 (0.00) | 0/154 | 0/4, 0/4 (0.00) |
| reference, no rehearsal | 40/279 | | 24/154* | |
| reference, rehearsal without the servo state | 42/279 | | 7/154* | |
| reference, base held | 14/279 | | 24/154* | |
| reference, nearest IK solution | 248/279 | | 108/154* | |

(* round 1's variants were cut from an earlier version of the reference.) Isolation probe ok in both rounds. Round 1
model trials (skill in the loop, L1 seed 101): Opus 5.5 lost 1:7 (review time-out) with a skill of 148/154 and 8/8;
GPT-6 Astra won 11:2 (10 points from one short flat serve: an opponent shortcut, passed to the high level) with a skill of
149/154 and 8/8. No model trials on the new base until BMOPP / #94 are confirmed.

## 4. The base

Everything base-specific is data: the robot-only model is cut from the scene, the kinematics library takes the base's
limits from game spec, the servo model's base gains are one block. After the high level's base change: regenerate
`seat/robot_only.xml` (`python3 sports/badminton/robot_model.py`), refresh the toolkit's sha256, `evaluator.py record`
+ `make-cases`, `tools/skill_eval.py badminton platform|reference|boundary --record`.

## 5. Open points

Decisions taken as recommended (the walkthrough's section 3): the servo state in the observation, thresholds 0.8 / 0.05,
no automatic return at the low level either, competition test L1 + L2 with the point share as the main score, no
execution scatter at the low level (self-collision recorded only), no duel low task yet. Shared-code issues (not changed
here): skilleval's contract treats optional fields as required; CI S3 flags a servo-model copy in `seat/`.
