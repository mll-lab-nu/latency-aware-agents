# bodies/tianji_marvin_pro

> **TEST USE ONLY（仅供测试）.** `body.toml` says `status = "test-use"`: no published sport may use this robot
> (`tools/check` stage 1 refuses `[body] name = "tianji_marvin_pro"` with `publish = true`). Licence: the model and
> meshes come from the vendor-affiliated RoboTwin fork (KLMmotion/marvin_pro_robotwin) under that repository's MIT
> licence (`LICENSE`); **written confirmation from Tianji (天机智能) that the Marvin Pro assets may be redistributed
> under it is needed before any public release.**

The **Tianji Marvin Pro** (天机智能 Marvin Pro, Guangdong TianJi Intelligent Systems): a stationary dual-arm robot, two
7-DOF Marvin M6 arms mounted sideways on a fixed stand (X-shaped floor plate, column, chest, head camera block),
1.42 m, mains powered, no mobile base, lift or waist. Same API as `bodies/wheeled_humanoid` (`mount`, `rig`,
`robot_only_xml`, the hands on `<arm>_mount`, head cameras `head_left` / `head_right`); `load_body("tianji_marvin_pro")`.

| | |
|---|---|
| Model | the vendor's RoboTwin embodiment (KLMmotion/marvin_pro_robotwin `assets/embodiments/tianji/marvin_robot.urdf`, commit 1e30f003984ddb82d2994474b015f0933c131d94, MIT: `LICENSE`), converted URDF -> MJCF by `tools/make_model.py` (every change listed there). Not in MuJoCo Menagerie, not in BEHAVIOR-1K / OmniGibson (its robot assets: a1, fetch, franka, freight, husky, kinova, locobot, r1, r1pro, stretch, tiago, turtlebot, ur5e, vx300s) |
| Meshes | `meshes/` 61 k faces (decimated; the stand's 4,000 screw-sized parts dropped first), `meshes_fine/` 474 k faces, 23 MB, **not in git** (`tools/make_model.py` writes it; `lod="fine"` falls back to `meshes/` when absent) |
| u | the wheeled humanoid's layout `[base_x, base_y, base_yaw, lift, waist, arm_0..6]`; the first five are always 0 (`Rig.check` refuses a row that moves them), so `bodies/control/wholebody.py` and the sports' command layers run unchanged |
| Limits | `body.toml` with sources: joint ranges = URDF intersected with the vendor SDK's controller configuration (J1/J3/J5 +-170, J2 -120..-5.7, J4 -145..-5.7, J6 +-60, J7 +-90 deg), 180 deg/s every joint (datasheet, SDK), accelerations 450 / 900 deg/s^2 (SDK; enforced since mp-2), jerks 50 x the acceleration (estimate), torques 108/108/66/66/18/18/18 N m (URDF) |
| Servo | `bodies/servo.py` (the library's one limit / servo path): jerk-limited setpoints at the limits above, PD + inertia feed-forward + gravity/Coriolis compensation, clipped at the torques, 1 kHz |
| Rest | `tools/rest_pose.py`: the hand arm with a hand's face forward ahead-right of the chest; the free arm hanging beside the stand |
| Checks | `python -m bodies.conformance tianji_marvin_pro` (the library's conformance test: limits with sources, two robots in one scene, the servo reaching targets, the limits enforced, the appearance meshes, replay, the configuration hash) |

Gaps: no mobile base (the real product is a fixed stand); the wrist cameras of the real robot sit on its gripper,
which is replaced by the hand mount, so only the head camera is modelled; armature and damping are not published
(derived); the jerk and the torques are not published (estimate, URDF).
