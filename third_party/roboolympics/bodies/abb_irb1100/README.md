# bodies/abb_irb1100

The **ABB IRB 1100-4/0.58 on a two-axis gantry**: the robot of Google DeepMind's table-tennis system ("a 6 DoF ABB
1100 arm mounted on top of two Festo linear gantries", arXiv 2408.03906; the variant IRB 1100-4/0.58, arXiv
2309.03315). `load_body("abb_irb1100")`, the same API as the other robots with a model (`mount`, `rig`,
`robot_only_xml`, the hands on the flange body `mount`).

> **The model is an approximation from public specifications**, not the vendor's CAD: no licensed IRB 1100 model
> exists (no MuJoCo Menagerie, ROS-Industrial or ABB ROS 2 package; ABB's CAD has no redistribution licence; the one
> open model, work-r-labs/robots, carries meshes that are almost certainly derived from ABB's CAD). The kinematics
> are exact (ABB's dimensions), the shapes are simple solids of the published outer dimensions, the masses are
> estimates. All of it is our own work under the repository's licence (`LICENSE`).

| | |
|---|---|
| Model | `tools/make_model.py`: axis 2 at 327 mm, lower arm 280 mm, 10 mm offset, upper arm 300 mm, 64 mm to the flange (ABB product specification 3HAC064993-001 rev V, 2.3.2); ABB's axis directions and signs (2.6.1); ranges +-230, -115/+113, -205/+55, +-230, -125/+120, +-400 deg; 21.1 kg split over the links (estimate) |
| Gantry | DeepMind's: a 4 m lateral gantry carrying a 2 m forward gantry, the arm on top (2408.03906 Fig. 1). Travel forward -1.5..+0.5 m, lateral +-1.85 m from the home (DeepMind's simulated ranges). The arm's mounting surface 0.76 m above the floor (a table-tennis table's height, as the Franka sports' pedestal; DeepMind does not publish it) |
| u | the library's layout `[base_x, base_y, base_yaw, lift, waist, arm_0..6]`: `base_x` / `base_y` = the forward / lateral gantry axes (m, each a joint of its own, limited per axis), `arm_0..5` = axes 1..6; `base_yaw`, `lift`, `waist`, `arm_6` are not on this robot (always 0; `Rig.check` refuses a row that moves them) |
| Limits | `body.toml`, each with its source: axis speeds ABB (460, 360, 280, 560, 420, 750 deg/s); accelerations derived from ABB's category 0 stopping data for axes 1-3 (30.9, 20.3, 22.2 rad/s^2), estimated for axes 4-6 (the same acceleration / speed ratio); jerk 50 x the acceleration (estimate); torques from DeepMind's simulation of this robot (100, 100, 60, 20, 25, 100 N m); gantry 2 m/s (DeepMind), 8 m/s^2 and 200 m/s^3 (estimates: the table-tennis sport's rail values), forces 4102 / 1499 N (DeepMind's simulation) |
| Servo | `bodies/servo.py` (the library's one limit / servo path): jerk-limited setpoints for the two gantry axes and the six arm axes, PD + inertia feed-forward + gravity/Coriolis compensation, the output stage at the forces and torques, 1 kHz |
| Latency | `body.toml [controller.latency]`: DeepMind's measured means (ABB action 71 ms, gantry action 84 / 56 ms, ball observation 40 ms; 2309.03315 Table I, 2408.03906 Table X); for a sport that models them, the servo adds none |
| Self-collision | `[self_collision]`: the arm's links 1-6 and the paddle against the gantry (frame, beam, carriage) and the arm's base, the paddle against the arm's links (74 pairs with the paddle); link against link is off (ABB's axis ranges are the arm's mechanical design and the approximate solids are fatter than the real links at the elbow and wrist: they would block poses the real arm reaches) |
| Rest | `tools/rest_pose.py 0.40 -0.10 1.00`: the paddle's centre 0.40 m ahead of axis 1, 0.10 m to the right, 0.24 m above the mounting surface, its face forward |
| Checks | `python -m bodies.conformance abb_irb1100`; `python -m bodies.selfcollision check abb_irb1100` |

Not modelled: the real robot's internal cabling and its exact outer shape (simple solids); ABB's QuickMove motion
planner (the accelerations above are ABB's worst case, the arm stretched with its full load, enforced in every pose:
the real arm is faster in most poses with a 0.14 kg paddle); ABB's collision detection (a model-based torque supervision with a sensitivity
in per cent, no force threshold published: the self-collision warning uses the library's 20 N).

Table tennis on this body (B9, the serve-receivability test): `docs/benchmarks/abb-irb1100-tt-b9-20261010.md`.
