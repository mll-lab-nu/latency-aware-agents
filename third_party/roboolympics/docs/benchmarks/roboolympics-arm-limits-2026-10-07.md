# Real-robot arm limits (2026-10-07, review item C3)

The table-tennis prototypes (branch `tt-proto`, `docs/benchmarks/tabletennis-prototypes-20261007.md` §5) found that
our simulated Franka keeps its joint velocity limits on the targets but exceeds the real robot's acceleration, jerk and
torque-rate limits by 3-12x at p99. User decision (2026-10-07): every simulated arm is as capable as the real robot and
no more; a shared module carries the limits, and the sports adopt it one by one. Phase 1 (this note): the module, a
measurement tool, the fencing pilot, and measurements of the other arm sports. Phase 2 (tennis, badminton, table
tennis, basketball, air hockey) waits for their open branches.

## 1. The official limits

Every number is the manufacturer's; none is derived. The data files hold the full source strings.

**Franka Emika Panda (FER)**: `bodies/franka_panda/body.toml`. Sources: Franka Robotics, FCI documentation, "Control
Interface Specification and Robot Limits", section "Limits for Franka Emika Robot (FER)"
(https://frankarobotics.github.io/docs/robot_specifications.html, formerly `frankaemika.github.io/docs/control_parameters.html`;
page (c) 2023 Franka Robotics GmbH, read 2026-10-07), cross-checked against libfranka 0.9.2
`include/franka/rate_limiting.h` and franka_ros `franka_description/robots/panda/joint_limits.yaml`.

| joint | A1 | A2 | A3 | A4 | A5 | A6 | A7 |
|---|---|---|---|---|---|---|---|
| range (rad) | +-2.8973 | +-1.7628 | +-2.8973 | -3.0718..-0.0698 | +-2.8973 | -0.0175..3.7525 | +-2.8973 |
| velocity (rad/s) | 2.175 | 2.175 | 2.175 | 2.175 | 2.61 | 2.61 | 2.61 |
| acceleration (rad/s^2) | 15 | 7.5 | 10 | 12.5 | 15 | 20 | 20 |
| jerk (rad/s^3) | 7500 | 3750 | 5000 | 6250 | 7500 | 10000 | 10000 |
| torque (N m) | 87 | 87 | 87 | 87 | 12 | 12 | 12 |
| torque rate (N m/s) | 1000 | 1000 | 1000 | 1000 | 1000 | 1000 | 1000 |

Cartesian (Cartesian interfaces only, they do not bind joint-space control): translation 1.7 m/s, 13 m/s^2,
6500 m/s^3 (libfranka 0.9.2 says 2.0 m/s); rotation 2.5 rad/s, 25 rad/s^2, 12500 rad/s^3; elbow 2.175 rad/s,
10 rad/s^2, 5000 rad/s^3. In joint-space control Franka calls the range, velocity, acceleration and jerk limits
"necessary conditions" (a violation aborts the motion) and the torque and torque-rate limits "recommended"; in torque
control the torque rate is necessary.

**Franka Research 3 (FR3)**: `bodies/franka_fr3/body.toml`, same page, section "Limits for Franka Research 3" (also
libfranka 0.21.3, franka_description `robots/fr3/joint_limits.yaml`): velocity 2.62 (A1-A4), 5.26, 4.18, 5.26 rad/s;
acceleration 10 rad/s^2 and jerk 5000 rad/s^3 on every joint; torque 87 / 12 N m; torque rate 1000 N m/s;
position-based velocity limits (offset and deceleration per joint, stored, not applied yet); Cartesian 3.0 m/s,
9 m/s^2, 4500 m/s^3.

**KUKA LBR iiwa 14 R820**: `bodies/kuka_iiwa14/body.toml`. Source: KUKA Roboter GmbH, brochure "Sensitive robotics -
LBR iiwa" (EN, (c) 2017), table "Axis data": range +-170, +-120, +-170, +-120, +-170, +-120, +-175 deg; maximum velocity
85, 85, 100, 75, 130, 135, 135 deg/s (1.484, 1.484, 1.745, 1.309, 2.269, 2.356, 2.356 rad/s); maximum torque 320, 320,
176, 176, 110, 40, 40 N m. KUKA publishes no joint acceleration, jerk or torque-rate limit: those stay unset (not
enforced) until a manufacturer source exists.

**Which model each scene is.** Fencing, table tennis, tennis, badminton and basketball use MuJoCo Menagerie's
`franka_emika_panda` (`panda_nohand`): the Panda's kinematics, ranges and masses, so the Panda (FER) limits apply,
not the FR3's. Air hockey uses the Air Hockey Challenge model of the iiwa 14: the iiwa 14 R820 limits.

## 2. The shared module: `bodies/`

| file | what |
|---|---|
| `bodies/<body>/body.toml` | the limits as data (`[limits.arm]`: `range_lo_rad`, `range_hi_rad`, `speed_rad_s`, `accel_rad_s2`, `jerk_rad_s3`, `torque_nm`, `torque_rate_nm_s`; `[limits.cartesian]`; `[sources]`) for `franka_panda`, `franka_fr3`, `kuka_iiwa14` |
| `bodies/limits.py` | numpy only. `load(body)` -> `ArmLimits`; `plan_line` / `Line`: time-optimal straight joint move under velocity, acceleration and jerk (7-segment S-curve on the path parameter), optionally stretched to T; `JerkLimiter`: the robot's setpoint generator at the physics rate, follows any reference and never exceeds velocity / acceleration / jerk (a cascade of sqrt controllers with velocity feed-forward); `jerk_caps(M, lim)`: the jerk the torque-rate limit allows with the arm's inertia; `TorqueRateLimiter`: the servo's output stage (rate and size); `hermite`: rows -> smooth reference; `check_rows`: low-level rows against velocity and acceleration; `derivatives`, `stats`: measurement |
| `tools/arm_limits.py measure SPORT` | replays the sport's golden matches in its image (no rendering), records every physics step of every arm joint (positions, motor torques; and the commanded setpoints when the sport uses `JerkLimiter`), checks the replay still matches `expected.json`, prints per joint p50 / p95 / p99 / max of velocity, acceleration, jerk, torque and torque rate against the limits |
| `sport.toml [arm]` | `body = "franka_panda"`, `joints = "<regex with groups arm, j>"`: `tools/sportlib.assemble_sim` copies `bodies/limits.py` and `bodies/<body>/body.toml` into the task's `environment/sim/bodies/`; `task.toml` / README name the body as the robot; `tools/check` stage 1 checks the declaration |

**How it meets the unified body** (branch `unified-body`, `bodies/README.md` there). The names follow that design: a
top-level `bodies/` package, one directory per body with `body.toml`, `[limits.arm]` with the same key names
(`speed_rad_s`, `torque_nm`), sources in `[sources]`. This branch adds no `bodies/__init__.py` and no
`bodies/README.md` (a namespace package), so the two branches merge without add/add conflicts: unified-body brings
`bodies/__init__.py` (`load_body`, `planner`, `BODIES`), `hands/`, `wheeled_humanoid/`, `unitree_g1/` (its `[limits.arm]` uses the same keys), `wholebody.py` (unified-body at a3120787); this branch brings `limits.py` and the
three arm bodies. A trial merge (git merge-tree, arm-limits on 43bea0e1 with unified-body a3120787): nothing in
`bodies/` conflicts; five hunks do, all keep-both: `tools/generate.py` (the task.toml `robot` and README line:
`robot_name(d)` already covers `[body] name`), `tools/sportlib.py` (the docstring, `assemble_sim` calling both
`copy_arm_limits` and `copy_body`, the two functions side by side; unified-body's `copy_body` already creates
`bodies/` with `exist_ok` for this case); `tools/check.py` and `tasks/dataset.toml` merge cleanly (the digests are then
refreshed). After the merge: (1) `tools/sportlib.copy_body` also copies `bodies/limits.py`; (2) `[arm]` stays
for sports whose scene carries its own arm (the Franka sports) and `[body]` is for the mounted unified body, and
`tools/generate.robot_name` already reads either; (3) the wheeled humanoid's `Rig.apply` can put its 10 ms targets
through a `JerkLimiter` and its torques through a `TorqueRateLimiter` (its README lists acceleration, jerk and torque
rate as known gaps), once RB-Y1 acceleration / jerk / torque-rate numbers have a source (Rainbow's URDF gives none).

## 3. Measurements before (origin/main, golden matches, every 1 ms step, both arms pooled)

origin/main at 43bea0e1 (core 0.2.9); badminton re-measured after its serve PR (#14), the other sports are unchanged by it.
Worst joint's p99 / limit (and max / limit); "-" = no official limit. Measured is the joints' actual motion and the
motors' torque.

| sport | body | steps | velocity | acceleration | jerk | torque | torque rate |
|---|---|---|---|---|---|---|---|
| fencing (fencing-2) | Panda | 109580 | 1.38 (3.07) | 6.12 (171) | 12.4 (344) | 1.00 (1.00) | 18.3 (124) |
| table tennis | Panda | 65490 | 1.10 (2.79) | 6.58 (534) | 10.3 (2110) | 1.00 | 9.36 (174) |
| tennis | Panda | 125910 | 0.95 (1.55) | 4.00 (70.5) | 7.87 (142) | 1.00 | 4.64 (123) |
| badminton (with the robots' serve, PR #14) | Panda | 164000 | 1.51 (4.44) | 5.45 (184) | 6.33 (720) | 1.00 | 6.31 (129) |
| basketball (kinematic arm, 10 ms poses) | Panda | 12881 | 6.74 (6.87) | 19.2 (21.6) | 2.05 (2.52) | no motor | no motor |
| air hockey | iiwa 14 | 369500 | 1.93 (2.69) | - (p99 4.7-20 rad/s^2) | - | 0.43 (0.59) | - (p99 31-3658 N m/s) |

Table tennis reproduces the prototype's numbers (acceleration p99 22-49 rad/s^2 = 2.9-6.6x, prototype 3-12x; torque
rate up to 174k N m/s, prototype 100-135k; velocity briefly above the limit, p99 1.01-1.10x). Air hockey: torques are
far inside the iiwa's, but A2 and A4 exceed the iiwa's velocity limits (p99 2.87 and 2.43 rad/s against 1.48 and
1.31): its controller clips joint velocity at 3 rad/s, not at the datasheet values.

## 4. Fencing pilot (fencing-3)

**What changed** (`sports/fencing/fencing.py`, README "Arm limits"): the arm limits are the Panda's (fencing-2: 1.5 x
velocity and acceleration, no jerk or torque-rate limit). Planner: time-optimal S-curves at 95 % of the limits and of
the torque-rate jerk caps. Servo at 1 kHz: `JerkLimiter` setpoint, torque = M(q) qdd_setpoint + the cart's measured
acceleration through the coupling inertia + PD + bias, then `TorqueRateLimiter` (1 N m per ms, 87 / 12 N m); a joint
pushed more than 0.02 rad off its setpoint makes the setpoint follow the arm (without it the rate-limited servo wound
up and diverged after hits and cart starts). Low level: `traj` rows checked against the acceleration limits too
(refused at submission; a first row that does not join the current targets is rejected at the effect, the existing
rule). A full extension from on guard takes 0.55-0.75 s (fencing-2: 0.50-0.58 s).

**Limits respected** (golden matches re-recorded under fencing-3, `tools/arm_limits.py measure fencing`):

| | fencing-2 (before) | fencing-3 (after) |
|---|---|---|
| commanded setpoint velocity / acceleration / jerk, max / limit | (targets only speed-checked) | 0.999 / 1.000 / 0.70 |
| motor torque, max / limit | 1.00 | 1.00 |
| motor torque rate, p99 / max over the limit | 18.3 / 124 | 1.00 / 1.00 |
| actual joint velocity, worst p99 / limit | 1.38 | 1.08 |
| actual joint acceleration, worst p99 / limit (per joint) | 2.0-6.1 | 1.7-3.6 |
| actual joint jerk, worst p99 / limit (per joint) | 3.5-12.4 | 1.4-8.0 |

The actual motion still exceeds the acceleration and jerk limits where an external force acts. In the three golden
matches, samples with any joint above its acceleration limit: 81 % of those within 30 ms of an arm or blade contact,
51 % of those within 10 ms of a cart acceleration above 1 m/s^2 (the rail's go-to has no jerk limit: the cart steps
to 6-9 m/s^2 and shakes the arm, whose torque may change only 1 N m per ms), 8.7 % of the rest (servo vibration at
1 kHz). A real arm hit by a blade is also accelerated beyond its command limits.

**Reference performance** (camera-only reference against the scripted fencer, high level, in-process `core/local.py`
as `tools/golden.py record`):

| bouts | fencing-2 | fencing-3 |
|---|---|---|
| seeds 1-3 x levels 1-3 (9 bouts) | 9/9 won, touches 45:6 | 8/9 won, touches 44:21 (L1 3/3, L2 3/3, L3 2/3: seed 3 lost 4:5) |
| official seeds 101 (L1), 202 (L2), 303 (L3) | 3/3, 15:2 | 3/3, 15:6 |
| default configuration (seed 1, level 1) | 5:0 | 5:2 |
| duel reference v reference, seeds 1, 2 | A 5:0, B 5:0 | A 5:0, B 5:0 (the attacker/defender styles decide) |

A retuned reference (stop-hit look-ahead 0.3 -> 0.4 s, lead 0.25 -> 0.32 s) did worse (6/9, 0/3 at level 1) and was
not kept. Feasibility, the straight-arm running attack ("fleche") against the 13 other scripted strategies, 3 seeds x
both seats, full bouts: fencing-3 59/78 won (fencing-2 at 1.5 x limits: 42/78; the 2026-10-05 study with the Panda's
limits and the old controller: 66/78); it still loses to stop-hits (1/6), parries (0/6) and distance play (1/6), and
the camera-only reference defender scored against it every time it met it in the bouts above.

**Golden and checks**: `bout-high-s1-L1` 5:2, `bout-low-s2-L1` 5:4, `duel-high-s1-A` 5:0 re-recorded and replay 3/3;
the legacy records of the golden-data store retired (`golden/convert.py` RETIRED_ALL). `tools/check fencing`: stage 1
contract, stage 2 generic and exploit tests, stage 3 golden 3/3, stage 4 fairness (reference duels, N = 40: first side
0.0 pp, seat A 0.0 pp, +-15.5 pp, the usual realtime WARN), stage 5 live runs. The live bout (the real-time service
at slowdown 20, seed 1, level 1, camera-only reference over HTTP) is not deterministic in wall time: under fencing-3
the reference won 4 of 5 (5:4, 5:3, 5:4, and in `tools/check` runs one loss and one 5:3 win, valid, the replay deterministic); under fencing-2 it won 5:1. The
live duel and both nop runs pass. `ci_checks/run_all.sh`: all checks passed.

## 5. Phase 2: what each sport will take

All four Franka sports share the fencing structure (100 Hz joint targets, linear interpolation, PD + bias at 1 kHz,
targets speed-checked), so each needs the same five steps: (a) servo: `JerkLimiter` + inertia feed-forward +
`TorqueRateLimiter` + re-sync (fencing's `World.step` is the template); (b) planner: the stroke / throw planner made
acceleration- and jerk-aware; (c) low level: rows checked against acceleration (each sport's existing refusal rule);
(d) re-verify the reference and recalibrate the scripted opponents, re-record golden matches, fairness; (e) sport.toml
texts, README, tasks.

| sport | branch to wait for | specific work |
|---|---|---|
| table tennis | tt-serve (and the rail base of tt-proto) | strokes reach a paddle velocity at a contact time from rest in 0.25-0.4 s: acceleration binds (15 rad/s^2 needs 0.145 s to reach 2.175 rad/s; A2 7.5 rad/s^2 0.29 s). The Hermite swing must be timed under acceleration and jerk (search the swing start / duration, accept a lower contact speed); expect slower, later strokes: recalibrate the opponent's box and the 1.4 m/s paddle cap; the robot-only rehearsals must use the new servo. Rail base keeps its limits |
| tennis | tennis-serve | same as table tennis (`plan_racket` Hermite swing, rehearsals); `RACKET_GAIN = 5` already scales the delivered head speed, so a slower swing moves the gain question (declared deviation); rail base 4 m/s, 8 m/s^2 kept; low-level refusal "when they would take effect" gets the acceleration check |
| badminton | none (badminton-serve merged, PR #14): can start on main | same as tennis (`plan_stroke` / `plan_racket`, now also the serve swing); highest velocity excess (A2 p99 1.51, max 4.4x: PD overshoot in overhead swings; before the serve PR A1 p99 1.50, max 5.1x over 201990 steps) |
| basketball | reply-images | the arm is kinematic (keyframes, the ball launched at a set speed up to 14 m/s, far beyond what a Panda hand can deliver at about 2 m/s): retime the keyframe animation to S-curves within the limits (velocity p99 6.7x, acceleration 19x today: A4 swings 2 rad in 0.2 s); the launcher itself is a declared abstraction, an open decision whether it stays |
| air hockey | airhockey-ref | iiwa 14: clip the joint velocity at the datasheet values (85-135 deg/s, today 3 rad/s) in the IK/trajectory layer and re-time the 2 m/s / 10 m/s^2 Cartesian mallet path to them; torques already within; no acceleration / jerk / torque-rate limit without a KUKA source |

## 6. Open decisions

1. **Fencing balance.** With the real Panda the arm is slower against the footwork (the reason fencing-2 used 1.5 x).
   The reference still wins 11/12 in-process bouts but only 4/5 live bouts at the default configuration (a flaky
   `tools/check` stage 5), and the running attack wins 59/78 scripted bouts (it has counters). Keep the real limits
   (as decided) and strengthen the reference (or ease the level-1 fencer), or reopen D7.
2. **Measured vs commanded.** The commanded setpoints and the motor torques respect every limit; the joints' actual
   acceleration exceeds it under contact and cart starts. Accept (physical), or also limit the rail's jerk (a cart
   change: a jerk limit of a few hundred m/s^3 would remove most of the cart-driven part).
3. **Torque-rate limit in the feedback.** We limit the whole motor torque to 1000 N m/s (Franka: recommended in joint
   control, necessary in torque control); with the re-sync it is stable. The alternative is limiting only the
   feed-forward part (the planned motion) and leaving the disturbance response unlimited.
4. **iiwa acceleration / jerk / torque rate**: unpublished; enforce velocity and torque only, or adopt a documented
   assumption.
5. **Basketball launcher**: keep the abstract launcher (a declared deviation) or tie the release speed to what the arm
   can deliver.
6. **PR shape**: `bodies/` is not one of `ci_checks/check-pr-scope.sh`'s parts (it counts as legacy), so this branch is
   three parts (bodies + tools, sports/fencing, docs); split into a `bodies`/`tools` PR and a fencing PR, or add
   `bodies` as a part (the unified-body branch needs the same).
