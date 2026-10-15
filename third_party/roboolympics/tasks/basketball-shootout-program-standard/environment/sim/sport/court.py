"""MJCF for the basketball shootout scene (generated; `python3 court.py > scene/court.xml` dumps it).

World frame (metres): the hoop's centre is at (0, 0, 3.048); the court floor is z = 0; the shooter's half is y < 0
(the baseline is y = +1.60, behind the backboard). FIBA/NBA-like dimensions: rim inner diameter 0.4572 m
(tube 16 mm), rim centre 0.381 m in front of the backboard face, backboard 1.83 x 1.07 m (bottom edge 2.90 m),
lane 4.88 m wide, free-throw line 4.19 m from the hoop centre, three-point arc 7.24 m (6.71 m in the corners).

* Ball: size 7, radius 0.1195 m, mass 0.62 kg, thin-shell inertia. Air drag and the Magnus force are applied by
  bball.py (xfrc_applied); contacts are explicit <pair>s with hand-tuned restitution (floor ~0.8, rim ~0.6,
  backboard ~0.65, defender ~0.3), not system-identified.
* Shooter: a declared shooting machine (a launcher like Toyota's CUE, sport.toml D7) on a 1.0 m pedestal: a turret
  (yaw) carrying a fork, and a launch tube pivoting about its muzzle, the release point (MUZZLE_FWD in front of the
  turret axis, MUZZLE_H above the floor). Its two joints are display only: bball.py moves them along a display path
  to the commanded aim and sets the ball's release velocity itself; no machine part has a contact pair (it never
  touches the ball). The shooter's frame: +x forward, +y left.
* Defender: a kinematic (mocap) figure with raised arms that collides with the ball.
"""
from __future__ import annotations

import math

HOOP = (0.0, 0.0, 3.048)
RIM_R_IN = 0.2286
RIM_TUBE = 0.008
RIM_C = RIM_R_IN + RIM_TUBE          # centreline radius of the rim tube
BOARD_Y = 0.381                      # backboard front face
BASELINE_Y = 1.60
FT_Y = -4.19
THREE_R = 7.24
THREE_CORNER_X = 6.71
HALF_W = 7.62
BALL_R = 0.1195
BALL_M = 0.62
PEDESTAL_H = 1.0
N_RIM = 36
# the shooting machine (a declared launcher like Toyota's CUE; sport.toml D7): its muzzle, the release point, is
# MUZZLE_FWD in front of the turret axis at MUZZLE_H above the floor and swings with the turret's yaw
MUZZLE_FWD = 0.50
MUZZLE_H = 2.17
MACHINE_RGBA = ".22 .27 .36 1"
BARREL_RGBA = ".80 .82 .86 1"


def _solref(e: float, k: float = 4.0e5) -> str:
    """Direct (negative) solref for a spring-damper contact with restitution ~e."""
    ln = math.log(e)
    zeta = -ln / math.sqrt(math.pi ** 2 + ln ** 2)
    b = 2 * zeta * math.sqrt(k)
    return f"-{k:.0f} -{b:.2f}"


def _line(name, x0, y0, x1, y1, w=0.05):
    return (f'<geom name="{name}" type="capsule" fromto="{x0:.4f} {y0:.4f} 0.002 {x1:.4f} {y1:.4f} 0.002" '
            f'size="{w / 2}" rgba="0.97 0.97 0.97 1" contype="0" conaffinity="0" group="1"/>')


def _cam_axes(pos, target, roll_up=(0, 0, 1)):
    px, py, pz = pos
    d = [target[0] - px, target[1] - py, target[2] - pz]
    n = math.sqrt(sum(v * v for v in d)); d = [v / n for v in d]
    up = roll_up
    x = [d[1] * up[2] - d[2] * up[1], d[2] * up[0] - d[0] * up[2], d[0] * up[1] - d[1] * up[0]]
    n = math.sqrt(sum(v * v for v in x)); x = [v / n for v in x]
    y = [x[1] * d[2] - x[2] * d[1], x[2] * d[0] - x[0] * d[2], x[0] * d[1] - x[1] * d[0]]
    return " ".join(f"{v:.5f}" for v in x + y)


# fixed world cameras (the shooter's cameras move with the shooter; see bball.py CAMERAS)
WORLD_CAMS = {
    "broadcast": {"pos": (-10.5, -6.0, 6.0), "target": (0.0, -3.2, 1.2), "fovy": 50.0},
    "baseline": {"pos": (5.2, 1.4, 4.2), "target": (-1.0, -4.0, 1.0), "fovy": 60.0},
    "rim": {"pos": (3.0, -0.35, 3.35), "target": (0.0, 0.05, 3.1), "fovy": 32.0},
}
SHOOTER_CAM_POS = (-0.45, 0.0, 2.45)   # in the shooter's frame (+x forward, +y left, z up from the floor)
SHOOTER_CAM_PITCH_DEG = 8.0            # looking up by this much
SHOOTER_WIDE_FOVY = 55.0
SHOOTER_ZOOM_FOVY = 16.0


def build_xml() -> str:
    g = []
    # ---- court lines
    L = []
    L.append(_line("l_base", -HALF_W, BASELINE_Y, HALF_W, BASELINE_Y))
    L.append(_line("l_side_l", -HALF_W, BASELINE_Y, -HALF_W, -12.4))
    L.append(_line("l_side_r", HALF_W, BASELINE_Y, HALF_W, -12.4))
    L.append(_line("l_half", -HALF_W, -12.4, HALF_W, -12.4))
    L.append(_line("l_lane_l", -2.44, BASELINE_Y, -2.44, FT_Y))
    L.append(_line("l_lane_r", 2.44, BASELINE_Y, 2.44, FT_Y))
    L.append(_line("l_ft", -2.44, FT_Y, 2.44, FT_Y))
    n = 24
    for i in range(n):   # free-throw circle (full)
        a0, a1 = 2 * math.pi * i / n, 2 * math.pi * (i + 1) / n
        L.append(_line(f"l_ftc{i}", 1.8 * math.cos(a0), FT_Y + 1.8 * math.sin(a0), 1.8 * math.cos(a1),
                       FT_Y + 1.8 * math.sin(a1)))
    yc = -math.sqrt(THREE_R ** 2 - THREE_CORNER_X ** 2)
    L.append(_line("l_3c_l", -THREE_CORNER_X, BASELINE_Y, -THREE_CORNER_X, yc))
    L.append(_line("l_3c_r", THREE_CORNER_X, BASELINE_Y, THREE_CORNER_X, yc))
    a_end = math.asin(THREE_CORNER_X / THREE_R)
    n = 40
    for i in range(n):   # three-point arc (below the hoop, y < 0)
        a0 = -a_end + 2 * a_end * i / n; a1 = -a_end + 2 * a_end * (i + 1) / n
        L.append(_line(f"l_3a{i}", THREE_R * math.sin(a0), -THREE_R * math.cos(a0), THREE_R * math.sin(a1),
                       -THREE_R * math.cos(a1)))
    # ---- hoop
    H = []
    hx, hy, hz = HOOP
    for i in range(N_RIM):
        a0, a1 = 2 * math.pi * i / N_RIM, 2 * math.pi * (i + 1) / N_RIM
        H.append(f'<geom name="rim{i}" type="capsule" fromto="{hx + RIM_C * math.cos(a0):.5f} {hy + RIM_C * math.sin(a0):.5f} {hz} '
                 f'{hx + RIM_C * math.cos(a1):.5f} {hy + RIM_C * math.sin(a1):.5f} {hz}" size="{RIM_TUBE}" '
                 f'rgba="0.95 0.35 0.05 1" contype="0" conaffinity="0"/>')
    # bracket from the rim to the board
    H.append(f'<geom name="bracket" type="box" pos="0 {(RIM_C + BOARD_Y) / 2:.4f} {hz - 0.02}" '
             f'size="0.06 {(BOARD_Y - RIM_C) / 2:.4f} 0.02" rgba="0.95 0.35 0.05 1" contype="0" conaffinity="0"/>')
    # backboard (glass-ish, white border, red-orange shooter's square)
    H.append(f'<geom name="board" type="box" pos="0 {BOARD_Y + 0.015} 3.435" size="0.915 0.015 0.535" '
             f'rgba="0.92 0.94 0.97 1" contype="0" conaffinity="0"/>')
    for nm, pos, size in (("sq_t", (0, 3.048 + 0.45), (0.305, 0.025)), ("sq_b", (0, 3.048 + 0.0), (0.305, 0.025)),
                          ("sq_l", (-0.28, 3.048 + 0.225), (0.025, 0.25)), ("sq_r", (0.28, 3.048 + 0.225), (0.025, 0.25)),
                          ("bd_t", (0, 3.96), (0.915, 0.02)), ("bd_b", (0, 2.91), (0.915, 0.02)),
                          ("bd_l", (-0.9, 3.435), (0.02, 0.535)), ("bd_r", (0.9, 3.435), (0.02, 0.535))):
        H.append(f'<geom name="{nm}" type="box" pos="{pos[0]} {BOARD_Y - 0.001} {pos[1]}" size="{size[0]} 0.001 {size[1]}" '
                 f'rgba="0.85 0.1 0.05 1" contype="0" conaffinity="0" group="1"/>')
    # stanchion (visual)
    H.append(f'<geom name="pole" type="box" pos="0 {BASELINE_Y + 0.9} 1.8" size="0.1 0.1 1.8" rgba="0.2 0.22 0.3 1" '
             f'contype="0" conaffinity="0" group="1"/>')
    H.append(f'<geom name="arm_bar" type="box" pos="0 {(BOARD_Y + BASELINE_Y + 0.9) / 2 + 0.02} 3.45" '
             f'size="0.06 {(BASELINE_Y + 0.9 - BOARD_Y) / 2:.3f} 0.06" rgba="0.2 0.22 0.3 1" contype="0" conaffinity="0" group="1"/>')
    H.append(f'<geom name="pad" type="box" pos="0 {BASELINE_Y + 0.9} 0.6" size="0.5 0.35 0.6" rgba="0.15 0.2 0.45 1" '
             f'contype="0" conaffinity="0" group="1"/>')
    # net (visual only): 12 strings from the rim down to a smaller ring
    nn = 12
    for i in range(nn):
        a = 2 * math.pi * i / nn
        for s in (-1, 1):
            b = a + s * math.pi / nn
            H.append(f'<geom type="capsule" fromto="{RIM_C * math.cos(a):.4f} {RIM_C * math.sin(a):.4f} {hz - 0.01} '
                     f'{0.14 * math.cos(b):.4f} {0.14 * math.sin(b):.4f} {hz - 0.40}" size="0.004" rgba="1 1 1 0.9" '
                     f'contype="0" conaffinity="0" group="1"/>')
    for i in range(nn):
        a0, a1 = 2 * math.pi * i / nn, 2 * math.pi * (i + 1) / nn
        H.append(f'<geom type="capsule" fromto="{0.14 * math.cos(a0):.4f} {0.14 * math.sin(a0):.4f} {hz - 0.40} '
                 f'{0.14 * math.cos(a1):.4f} {0.14 * math.sin(a1):.4f} {hz - 0.40}" size="0.004" rgba="1 1 1 0.9" '
                 f'contype="0" conaffinity="0" group="1"/>')
    cams = []
    for name, c in WORLD_CAMS.items():
        cams.append(f'<camera name="{name}" pos="{c["pos"][0]} {c["pos"][1]} {c["pos"][2]}" '
                    f'xyaxes="{_cam_axes(c["pos"], c["target"])}" fovy="{c["fovy"]}"/>')
    p = math.radians(SHOOTER_CAM_PITCH_DEG)
    sx, sy, sz = SHOOTER_CAM_POS
    sh_axes = f"0 -1 0 {-math.sin(p):.6f} 0 {math.cos(p):.6f}"
    fork_h = MUZZLE_H - PEDESTAL_H - 0.12
    pairs = []
    ball_pair = lambda geom, e, fr: (f'<pair geom1="ball" geom2="{geom}" solref="{_solref(e)}" '
                                     f'friction="{fr} {fr} 0.005 0.0001 0.0001" solimp="0.95 0.99 0.001 0.5 2" condim="3"/>')
    pairs.append(ball_pair("floor", 0.80, 0.6))
    pairs.append(ball_pair("board", 0.65, 0.3))
    pairs.append(ball_pair("bracket", 0.5, 0.4))
    for i in range(N_RIM):
        pairs.append(ball_pair(f"rim{i}", 0.60, 0.5))
    for dg in ("d_torso", "d_head", "d_arm_l", "d_arm_r", "d_hand_l", "d_hand_r", "d_legs"):
        pairs.append(ball_pair(dg, 0.30, 0.6))

    xml = f"""<mujoco model="rcb_basketball_shootout">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.0005" integrator="implicitfast" cone="elliptic" gravity="0 0 -9.81"/>
  <size memory="20M"/>
  <visual>
    <headlight diffuse=".55 .55 .55" ambient=".35 .35 .35" specular=".2 .2 .2"/>
    <global offwidth="1280" offheight="960"/>
    <quality shadowsize="2048"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1=".62 .66 .74" rgb2="0.12 0.12 0.16" width="10" height="10"/>
    <texture type="2d" name="wood" builtin="checker" rgb1=".78 .58 .36" rgb2=".74 .54 .33" width="512" height="512"/>
    <material name="wood" texture="wood" texrepeat="30 30" reflectance="0.05"/>
    <texture type="2d" name="ballt" builtin="checker" rgb1=".93 .42 .08" rgb2=".80 .33 .05" width="64" height="64"/>
    <material name="ballm" texture="ballt" texrepeat="3 3"/>
  </asset>
  <worldbody>
    <light pos="0 -4 9" dir="0 0.2 -1" diffuse="0.6 0.6 0.6" specular="0.2 0.2 0.2" castshadow="true"/>
    <light pos="-6 -8 8" dir="0.5 0.6 -1" diffuse="0.3 0.3 0.3" castshadow="false"/>
    <geom name="floor" type="plane" size="12 16 0.1" pos="0 -4 0" material="wood" contype="0" conaffinity="0"/>
    {chr(10).join('    ' + s for s in L)}
    {chr(10).join('    ' + s for s in H)}
    {chr(10).join('    ' + s for s in cams)}

    <body name="ball" pos="0 -4 1.5">
      <freejoint name="ball"/>
      <geom name="ball" type="sphere" size="{BALL_R}" mass="{BALL_M}" shellinertia="true" material="ballm"
            contype="0" conaffinity="0"/>
    </body>

    <body name="defender" mocap="true" pos="0 -2 0">
      <!-- local frame: +x towards the shooter; right hand up in the shooter's face (on the line to the hoop), left hand high to the side -->
      <geom name="d_legs" type="capsule" fromto="0 0 0.1 0 0 0.9" size="0.16" rgba="0.1 0.12 0.35 1" contype="0" conaffinity="0"/>
      <geom name="d_torso" type="capsule" fromto="0 0 0.95 0 0 1.45" size="0.21" rgba="0.8 0.1 0.12 1" contype="0" conaffinity="0"/>
      <geom name="d_head" type="sphere" pos="0 0 1.78" size="0.12" rgba="0.55 0.38 0.26 1" contype="0" conaffinity="0"/>
      <geom name="d_arm_l" type="capsule" fromto="0.02 0.24 1.52 0.10 0.28 2.38" size="0.05" rgba="0.55 0.38 0.26 1" contype="0" conaffinity="0"/>
      <geom name="d_arm_r" type="capsule" fromto="0.02 -0.24 1.52 0.17 -0.05 2.39" size="0.05" rgba="0.55 0.38 0.26 1" contype="0" conaffinity="0"/>
      <geom name="d_hand_l" type="sphere" pos="0.11 0.29 2.46" size="0.085" rgba="0.55 0.38 0.26 1" contype="0" conaffinity="0"/>
      <geom name="d_hand_r" type="sphere" pos="0.19 -0.03 2.46" size="0.085" rgba="0.55 0.38 0.26 1" contype="0" conaffinity="0"/>
    </body>

    <body name="shooter" pos="0 -4.19 0">
      <geom name="pedestal" type="cylinder" size="0.2 {PEDESTAL_H / 2}" pos="0 0 {PEDESTAL_H / 2}" rgba=".25 .26 .3 1"
            contype="0" conaffinity="0"/>
      <geom name="base_plate" type="cylinder" size="0.4 0.02" pos="0 0 0.02" rgba=".2 .2 .22 1" contype="0" conaffinity="0"/>
      <camera name="shooter_wide" pos="{sx} {sy} {sz}" xyaxes="{sh_axes}" fovy="{SHOOTER_WIDE_FOVY}"/>
      <camera name="shooter_zoom" pos="{sx} {sy} {sz}" xyaxes="{sh_axes}" fovy="{SHOOTER_ZOOM_FOVY}"/>
      <body name="turret" pos="0 0 {PEDESTAL_H}">
        <!-- the shooting machine (display only: no contact pair, the release is set by bball.py) -->
        <inertial mass="20" pos="0 0 0.1" diaginertia="1 1 1"/>
        <joint name="turret_yaw" type="hinge" axis="0 0 1" range="-0.7 0.7" damping="1"/>
        <geom name="m_drum" type="cylinder" size="0.3 0.07" pos="0 0 0.07" rgba="{MACHINE_RGBA}" contype="0" conaffinity="0"/>
        <geom name="m_boom" type="box" size="{MUZZLE_FWD / 2:.4f} 0.22 0.04" pos="{MUZZLE_FWD / 2:.4f} 0 0.12" rgba="{MACHINE_RGBA}"
              contype="0" conaffinity="0"/>
        <geom name="m_fork_l" type="box" size="0.035 0.03 {fork_h / 2:.4f}" pos="{MUZZLE_FWD:.4f} 0.19 {0.12 + fork_h / 2:.4f}"
              rgba="{MACHINE_RGBA}" contype="0" conaffinity="0"/>
        <geom name="m_fork_r" type="box" size="0.035 0.03 {fork_h / 2:.4f}" pos="{MUZZLE_FWD:.4f} -0.19 {0.12 + fork_h / 2:.4f}"
              rgba="{MACHINE_RGBA}" contype="0" conaffinity="0"/>
        <body name="barrel" pos="{MUZZLE_FWD:.4f} 0 {MUZZLE_H - PEDESTAL_H:.4f}">
          <!-- pivots about the muzzle: the release point does not depend on the elevation; joint angle = -elevation -->
          <inertial mass="5" pos="-0.4 0 0" diaginertia="0.3 0.3 0.3"/>
          <joint name="barrel_pitch" type="hinge" axis="0 1 0" range="-1.45 -0.3" damping="1"/>
          <geom name="m_axle" type="cylinder" size="0.03 0.2" quat="0.7071068 0.7071068 0 0" rgba=".15 .15 .17 1"
                contype="0" conaffinity="0"/>
          <geom name="m_barrel" type="cylinder" fromto="-0.85 0 0 -0.14 0 0" size="0.15" rgba="{BARREL_RGBA}"
                contype="0" conaffinity="0"/>
          <geom name="m_band" type="cylinder" fromto="-0.20 0 0 -0.13 0 0" size="0.158" rgba=".12 .12 .14 1"
                contype="0" conaffinity="0"/>
          <site name="muzzle" pos="0 0 0" size="0.01" group="4"/>
        </body>
      </body>
    </body>
  </worldbody>
  <contact>
    {chr(10).join('    ' + s for s in pairs)}
  </contact>
</mujoco>
"""
    return xml


if __name__ == "__main__":
    print(build_xml())
