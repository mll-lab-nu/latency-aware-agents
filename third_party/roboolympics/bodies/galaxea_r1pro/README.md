# bodies/galaxea_r1pro

The **Galaxea R1 Pro** as a RoboOlympics body: the wheeled humanoid of the BEHAVIOR Challenge (OmniGibson's `R1Pro`).
Two 7-DOF Galaxea A2 arms on a 4-DOF torso (three pitch joints: "knee", "hip", the chest's counter-pitch; a yaw waist)
on a three-wheel vector-steering chassis; a fixed head with a stereo camera. Same API, controlled vector and servo law
as `bodies/wheeled_humanoid` (the RB-Y1), so `bodies/control/wholebody.py` and the wheeled sports' code drive it unchanged.

| | |
|---|---|
| Model | Galaxea's own R1 Pro URDF and meshes from [OpenGalaxea/GalaxeaManipSim](https://github.com/OpenGalaxea/GalaxeaManipSim) (`galaxea_sim/assets/r1_pro/`, commit `abe7f516`), **Apache-2.0** (`LICENSE`, "Copyright 2025 Galaxea"; that repository's README badge says MIT, its LICENSE file is Apache-2.0). No MJCF exists (not in MuJoCo Menagerie, none from Galaxea): `tools/make_model.py` converts the URDF and lists every change |
| Not used | the 2026 URDF in [userguide-galaxea/URDF](https://github.com/userguide-galaxea/URDF) (`urdf_r1pro_g1z_2026`: 0.35 + 0.35 m torso links, the newer G1Z gripper) has no licence file (its `package.xml` says "BSD" in an unfilled template): read for facts only; BEHAVIOR-1K's R1Pro asset (USD/URDF) comes with the BEHAVIOR dataset download under its own data licence, not redistributable here |
| Files | `body.toml` (DOF, limits with sources S1-S4, servo, rest pose, mounts), `body.py` (`Body`: `mount`, `rig`, `robot_only_xml`, `guard_root`; `Rig`: the wheeled humanoid's), `model.xml` (generated), `meshes/` (43 k faces, 2.2 MB, checked in), `meshes_fine/` (the URDF's 473 k faces, 23 MB: **not checked in**, `.gitignore`; `tools/make_model.py` writes them, and `lod="fine"` falls back to `meshes/` without them) |
| u | `[base_x, base_y, base_yaw, lift, waist, arm_0 .. arm_6]`: lift a drives torso_joint1 = a, torso_joint2 = -2a, torso_joint3 = -a (the chest stays upright, goes down 0.70 (1 - cos a) m and forward 0.10 sin a m; a <= 1.344: 0.54 m); waist = torso_joint4; arm_i = the hand arm's arm_joint(i+1) |
| Limits (r1p-2, the SDK's: audit G6 / D5) | base per axis of the robot's frame: 1.5 / 1.5 m/s, 2.5 forward / 1.0 sideways m/s^2, 3 rad/s, 1 rad/s^2 (jerks 25 x, estimate); lift 0.75 rad/s, 1.125 rad/s^2 and rad/s^3 (the SDK's torso joint caps 1.5 rad/s with acceleration and jerk 1.5 x, torso_joint2 moves at 2a); waist +-170 deg, 1.5 rad/s, 2.25 rad/s^2, 2.25 rad/s^3; arm {3, 3, 3, 3, 5, 5, 5} rad/s with acceleration and jerk 1.5 x (SDK joint control; the URDF's 7.1-10.5 rad/s are not what the controller allows), 55/55/25/25/18/18/18 N m (URDF); torso 304 N m (datasheet max). A body for slow sports: these limits rule out fast strokes |
| Mounts | wrist: `<side>_mount` where the gripper housing met arm_link7, +z away from the wrist; cameras: `head_left` / `head_right` (12 cm, 70 deg vertical); the head has no joint: `head_pitch` (default 20 deg down, the URDF's) and `head_yaw` are mount parameters |

Not modelled: the chassis' wheels (planar joints instead, as the RB-Y1's mecanum base), the torso's chest tilt (the
lift keeps the chest upright), the grippers (the hands replace them), the datasheet's end-effector limits (7.5 m/s,
10 m/s^2).

Check: `python -m bodies.conformance galaxea_r1pro` (the library's conformance test: limits with sources, two robots in one scene, the servo reaching targets, the limits enforced on the setpoints and torques, the appearance meshes, deterministic replay, the assembled configuration's hash).
