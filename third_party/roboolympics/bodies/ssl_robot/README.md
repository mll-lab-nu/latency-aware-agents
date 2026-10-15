# bodies/ssl_robot

A **RoboCup Small Size League robot** after TIGERs Mannheim's v2020 generation, for the wheeled 5v5 soccer sport
(`sports/soccer_ssl`). Interface: `olympiad_review/screening/soccer_ssl_interfaces_zh.md` section 3 (plus
`substep` and `step_world`, below). Report: `olympiad_review/screening/soccer_ssl_body_zh.md`.

> **The model is our own approximation from published dimensions.** No SSL team publishes a URDF or MJCF; TIGERs'
> CAD has a non-commercial licence and ER-Force's none, so none is used. `body.py` builds simple solids from
> `body.toml [geometry]` (each value with its source); `tools/make_model.py` writes `model.xml` (one robot alone).
> **The dribbler is a contact model plus a capped grip** (declared deviation, "Dribbler" below).

## Use

```python
import mujoco
from bodies.ssl_robot import body as SSL
h = SSL.mount(spec, "blue_1", "blue", 1, pose=(-1.0, 0.5, 0.0))   # into a mujoco.MjSpec (world-frame pose)
model = spec.compile(); data = mujoco.MjData(model)
rig = SSL.rig(model, data, h, limits=None)        # None = body.toml; a dict may only TIGHTEN (ValueError otherwise)
rig.set_base_target(x, y, yaw, t=None, v_end=(0, 0))   # world frame -> {"status": ok|late|clipped, "eta"}
rig.kick(speed, mode="flat" | "chip", armed_for=0.0)   # -> "ok" | "clipped" | "kicker_not_ready" | "disarmed"
rig.dribble(rpm)                                       # -> "ok" | "clipped"
SSL.step_world(model, data, rigs)    # ONE 10 ms control step of all robots (the fast path, use this): every rig's
                                     # step, then 10 x (one vectorised PD for all + each rig's 1 kHz parts; mj_step)
rig.state()                          # own state only: pose, vel, vel_body, dribbler rpm / deflection / engaged,
                                     # kicker charge / armed, ball_in_beam (the break beam), target, eta
rig.take_events()                    # [{"event": "kicked" | "kick_expired" | "kicker_not_ready" | "kick_clipped" |
                                     #   "base_resync", "t": ..., ...}]
```

* The per-robot path `rig.step(m, d)` once per control step and `rig.substep(m, d)` before every `mj_step` gives
  bit-identical results (test) but is slower.
* `qpos` of the joints `<name>_x`, `<name>_y`, `<name>_yaw` is the robot's world pose; `qpos0` is the mount pose.
  The scene's `compiler.degree` is respected (MjSpec defaults to degrees).
* Contacts: the ball (sports/soccer_ssl/sim/ball.py) has priority 1; the chassis pieces and the kicker face have
  priority 2 (restitution about 0.6 with the ball, ER-Force; the kicker face damped), the dribbling bar priority 3
  (TPU). Robot against robot: the convex outline `<name>_hull_outer` on collision bit 2 (group 3, invisible); the
  thin pieces around the dribbler opening would let two fast robots slide into each other.
* The grip writes `data.qfrc_applied` (changes of its own contribution only). A world may clear `xfrc_applied`,
  never `qfrc_applied`.

## Specification (body.toml has every source)

| | value | source |
|---|---|---|
| Size, mass | 178 x 148 mm, 2.62 kg | TIGERs 2020 ETDP Table 1 |
| Front | flat chord 71.5 mm from the centre | Sumatra `center2DribblerDist` [config] |
| Base | 3 m/s, 3 m/s^2 (also braking: real 6, stricter), jerk 150 m/s^3 (est.), 20 rad/s, 50 rad/s^2 | TIGERs 2026 ETDP 3.2; Sumatra [config] |
| Drive force cap | 15.7 N per axis (2.62 kg x 6 m/s^2), yaw 1.26 N m | derived |
| Flat kick | max 8.5 m/s, sigma 3 % of the speed, direction 1 deg (est.) | TIGERs 2020; ZJUNlict 2024 |
| Chip kick | max 5.5 m/s at 45 deg (apex 0.8 m, first bounce 3.3 m measured) | Sumatra [config] |
| Capacitor | full in 2 s; a kick at v needs (v/vmax)^2 stored and uses 25 % x (v/vmax)^2 (a max kick: 0.5 s) | TIGERs 2020 3.2 |
| Dribbler | 14 mm TPU 60A bar on a damped hinge (8 deg swing), 17000 rpm, 0.031 N m, axis 36 mm high | TIGERs 2024, 2022, 2020; KIKS 2023 |
| Ball coverage | <= 19.8 % (the chip shovel stops the ball) | TIGERs 2020; rules 3.2.4 (20 %) |
| Break beam | across the opening 70 mm ahead of the centre at the ball's centre height (est.) | TIGERs 2020 2.3 |
| Vision pattern | centre 50 mm (blue / yellow), four 40 mm dots (pink / green) by id | rules 3.2.5, figures 5-6 |
| Side colour | a 40 mm band (0.085-0.125 m) all round but a +-15 deg camera window in front: 25 % of the side area; blue team dark navy, yellow team pale cream (one dark, one bright, not the pattern's saturated colours); visual only | rules 3.2.3 (>= 20 %, visible from all sides) |

## Dribbler

**Contact model** (always on): the bar spun by its motor through a P speed loop saturated at the torque limit
(TIGERs 2024 2.2), backspin pulls the ball against the bar; the carrier swings back on its damped hinge.
Measured with `tools/dribble_lab.py` (ZJUNlict's TestDribble replayed: hold the ball, accelerate at 3 m/s^2,
speed when the ball is 3 cm off; sport ball and carpet): forwards carried to 3 m/s; backwards lost at 0.93 m/s
(3000 rpm) / 2.73 m/s (8000 rpm); **sideways lost at 0.54-0.57 m/s and turning at 2.9-3.1 rad/s whatever the bar
friction (1.0-2.0), speed (300-15000 rpm), torque (0.005-0.06 N m), softness or the ball's floor contact**. A
spinning bar slides on the ball, so its kinetic friction points along the spin; the real soft bar's contact patch
cannot be had from a rigid contact. Receiving: 1 m/s caught, 2-5 m/s mostly not (3000 rpm).

**Capped grip** (`assist.py`, `body.toml [dribbler_assist]`, on by default): a horizontal spring-damper on the ball
towards its held point, force capped at 0.5 N (a push of about 0.75 N takes the ball; a robot pushes with up to
15.7 N), engaged when the ball reaches the beam at <= 5 m/s, released beyond the measured dribbling limits of real
robots, at a kick, or when the dribbler is off. The release speeds are set so that the same test loses the ball
at ZJUNlict's measured values:

| with the ball (3 m/s^2) | lost at (measured here, 3000 rpm) | published | grip releases at |
|---|---|---|---|
| backwards | 2.28 m/s | 2.3 m/s (ZJUNlict v2023; v2019 3.9) | 1.9 m/s |
| sideways | 2.61 m/s | 2.6 m/s (v2023; v2019 3.9) | 2.3 m/s |
| turning on the spot | 9.32 rad/s | 9.3 rad/s (v2023; v2019 12.9; KIKS 3.5) | 9.1 rad/s |
| forwards | carried to 3 m/s (the base's limit) | not published | none |
| stop from 1.5 m/s at 3 m/s^2 | held | not published | - |
| receiving 1-4 m/s (+-15 mm, +-10 deg) | 100 % caught, rebound <= 2 cm | TIGERs about 5 m/s | catch <= 5 m/s |
| receiving 5 m/s | 60 % (3000 rpm), 100 % (8000 rpm) | | |
| receiving 6-7 m/s | 0 % | | |

At 8000 rpm and above the contact alone holds the ball backwards beyond 2.3 m/s (2.8 m/s; at 10000 rpm to the
base's 3 m/s): within ZJUNlict's v2019 robot (3.9 m/s) but beyond v2023 (README of the report: a user decision).

## Speed

`tools/bench.py`: 10 robots, all driving to new random targets every second, dribblers on, one ball, this host
at load about 50: **3.1-3.5 x real time** (before: 0.41 x, the setpoint generator at 1 kHz per robot). The setpoint
generator (`RatePlanarBase`: bodies/servo.py PlanarBase with its tracking gains KP/KV 10/50 for a 10 ms step;
the library's 60/300 chatter at 10 ms) runs once per control step; the 1 kHz PD follows its knots by cubic Hermite
interpolation; `step_world` does one vectorised PD for all robots. `tools/planarbase_rate.py` compares the three:
the same peaks (speed <= 0.995, acceleration and jerk 1.000 of the limits), the same arrival times (+0.01-0.04 s
at 100 Hz), no overshoot; the library's gains at 100 Hz: 72 sign flips of the yaw acceleration, 2.4 cm overshoot.

## Admission

`python bodies/tests/test_ssl_robot.py` (15 tests). `bodies/conformance.py` is written for the arm robots'
interface (an XML mount with a hand, the 12-coordinate `u`, `robot_only_xml`, joint targets, self-collision of
arms) and does not apply to a planar robot; its items map to: 1 declared -> `test_limits_declared_with_sources`;
2 scene -> `test_ten_robots_one_scene_poses_and_patterns`; 3 servo + 4 limits -> `test_base_reaches_goals_within_limits`
(knots and interpolated setpoints within speed / acceleration / jerk, goals within 1 cm), `test_sport_may_only_tighten`;
5 appearance: one level of simple solids (n/a); 6 replay -> `test_deterministic_replay`, `test_group_path_equals_rig_path`;
7 assembly: `bodies.assemble("ssl_robot")` hashes body.toml, model.xml and the code as for every robot;
9 self-collision: one rigid chassis with a hinged dribbler (excluded pairs), no arms (n/a). The robot is
registered in `bodies.PLANAR` (not `BODIES`, whose library-wide checks call the arm interface).

## Known gaps

* PlanarBase keeps the base's velocity in the robot's frame, so turning while translating turns the world-frame
  velocity with the robot: an extra v x w (up to 1.5 x the acceleration limit in the tests) that it does not
  limit. A shared fix (bodies/servo.py), not done here.
* Braking is limited to 3 m/s^2 like accelerating (the real robot brakes at up to 6): PlanarBase has one limit.
* No tipping, no wheels (the base is a planar joint set; the drive force caps stand for the wheels' grip).
* The kick is an impulse (speed relative to the robot, the ball leaves without spin); the dribbler's grip is a
  capped force, not the TPU's deformation.
