"""Generate fencing_scene.xml: an FIE piste with two identical robot fencers (python gen_scene.py out.xml).

Each fencer: a cart on a linear rail along the piste (one prismatic joint, the rail coordinate = world x of the cart),
a 0.90 m pedestal, a Franka Panda arm (MuJoCo Menagerie panda_nohand, Apache-2.0, assets/LICENSE.menagerie-franka)
and an epee held at the flange (grip 0.10 m, bell guard 13.5 cm diameter, blade 0.90 m ending in the point).
Fencer A stands at x < 0 facing +x, fencer B at x > 0 facing -x: B is A turned by 180 degrees about the vertical
axis through the centre of the piste, so both are right-handed and the two sides are exactly symmetric.

Contacts: every geom of A collides with every geom of B and with the floor (A: contype 1 / conaffinity 2, B: 2 / 1,
floor 0 / 3); a fencer's own geoms never collide with each other.
"""
import sys

PED_H = 0.90            # link0 height on the pedestal: shoulder (joint 2) at PED_H + 0.333 = 1.233 m
GRIP, GUARD_R, GUARD_D, BLADE, BLADE_R, TIP_R = 0.10, 0.0675, 0.04, 0.90, 0.004, 0.005
CART = (0.25, 0.22, 0.06)          # cart half sizes (x, y, z)
PISTE_HALF_L, PISTE_HALF_W, RUNBACK = 7.0, 0.75, 2.0
RAIL_LIM = 7.6
RGBA_A, RGBA_B = "0.92 0.50 0.42 1", "0.42 0.58 0.95 1"   # fencer A drawn red, B blue (each seat's cameras: itself red)                     # the rail ends 0.6 m behind the rear limit line (in the run-back)

g = []


def box(name, size, pos, rgba):
    g.append(f'    <geom name="{name}" type="box" size="{size[0]} {size[1]} {size[2]}" pos="{pos[0]} {pos[1]} {pos[2]}" '
             f'rgba="{rgba}" contype="0" conaffinity="0"/>')


WHITE = "0.96 0.96 0.96 1"
# the piste (t.18: 14 m x 1.5 m) and run-backs; the last 2 m before each rear limit in another colour (t.19.4)
box("piste_mid", (PISTE_HALF_L - 2.0, PISTE_HALF_W, 0.002), (0, 0, 0.0), "0.55 0.57 0.60 1")
for s in (-1, 1):
    box(f"piste_warn{s}", (1.0, PISTE_HALF_W, 0.002), (s * (PISTE_HALF_L - 1.0), 0, 0.0), "0.42 0.47 0.58 1")
    box(f"runback{s}", (RUNBACK / 2, PISTE_HALF_W, 0.002), (s * (PISTE_HALF_L + RUNBACK / 2), 0, 0.0), "0.35 0.36 0.38 1")
    box(f"onguard{s}", (0.025, PISTE_HALF_W, 0.0025), (s * 2.0, 0, 0.0005), WHITE)       # t.19.2
    box(f"rear{s}", (0.025, PISTE_HALF_W, 0.0025), (s * PISTE_HALF_L, 0, 0.0005), WHITE)  # t.19.3
for i in range(-7, 8):          # broken centre line (t.19.1)
    if i % 2 == 0:
        box(f"centre{i}", (0.025, 0.05, 0.0025), (0, i * 0.1, 0.0005), WHITE)
k = 0
for x in range(-9, 10):         # yellow distance marks every 1 m beside the piste (both sides)
    for s in (-1, 1):
        box(f"mk{k}", (0.04, 0.12, 0.0015), (x, s * (PISTE_HALF_W + 0.25), 0.0015), "1 0.85 0.1 1"); k += 1
for s in (-1, 1):                # side boundaries
    box(f"side{s}", (PISTE_HALF_L + RUNBACK, 0.02, 0.0025), (0, s * PISTE_HALF_W, 0.0005), WHITE)
geoms = "\n".join(g)


def fencer(p, x0, quat, ct, ca, rgba_body, rgba_trim):
    c = f'contype="{ct}" conaffinity="{ca}"'
    return f'''
    <body name="{p}cart" pos="0 0 0">
      <inertial pos="0 0 0.3" mass="40" diaginertia="4 4 2"/>
      <joint name="{p}rail" type="slide" axis="1 0 0" range="{-RAIL_LIM} {RAIL_LIM}" damping="0" ref="0"/>
      <body name="{p}mount" pos="0 0 0" quat="{quat}">
        <geom name="{p}cart_g" type="box" size="{CART[0]} {CART[1]} {CART[2]}" pos="0 0 {CART[2] + 0.02}" rgba="{rgba_trim}" {c} mass="0"/>
        <geom name="{p}pedestal" type="cylinder" size="0.12 {(PED_H - 0.14) / 2}" pos="0 0 {0.14 + (PED_H - 0.14) / 2}" rgba="{rgba_body}" {c} mass="0"/>
        <body name="{p}link0" childclass="panda" pos="0 0 {PED_H}">
          <inertial mass="0.629769" pos="-0.041018 -0.00014 0.049974" fullinertia="0.00315 0.00388 0.004285 8.2904e-7 0.00015 8.2299e-6"/>
          <geom name="{p}link0_c" mesh="link0_c" rgba="{rgba_body}" {c}/>
          <body name="{p}link1" pos="0 0 0.333">
            <inertial mass="4.970684" pos="0.003875 0.002081 -0.04762" fullinertia="0.70337 0.70661 0.0091170 -0.00013900 0.0067720 0.019169"/>
            <joint name="{p}joint1"/>
            <geom name="{p}link1_c" mesh="link1_c" rgba="{rgba_body}" {c}/>
            <body name="{p}link2" quat="1 -1 0 0">
              <inertial mass="0.646926" pos="-0.003141 -0.02872 0.003495" fullinertia="0.0079620 2.8110e-2 2.5995e-2 -3.925e-3 1.0254e-2 7.04e-4"/>
              <joint name="{p}joint2" range="-1.7628 1.7628"/>
              <geom name="{p}link2_c" mesh="link2_c" rgba="{rgba_body}" {c}/>
              <body name="{p}link3" pos="0 -0.316 0" quat="1 1 0 0">
                <joint name="{p}joint3"/>
                <inertial mass="3.228604" pos="2.7518e-2 3.9252e-2 -6.6502e-2" fullinertia="3.7242e-2 3.6155e-2 1.083e-2 -4.761e-3 -1.1396e-2 -1.2805e-2"/>
                <geom name="{p}link3_c" mesh="link3_c" rgba="{rgba_body}" {c}/>
                <body name="{p}link4" pos="0.0825 0 0" quat="1 1 0 0">
                  <inertial mass="3.587895" pos="-5.317e-2 1.04419e-1 2.7454e-2" fullinertia="2.5853e-2 1.9552e-2 2.8323e-2 7.796e-3 -1.332e-3 8.641e-3"/>
                  <joint name="{p}joint4" range="-3.0718 -0.0698"/>
                  <geom name="{p}link4_c" mesh="link4_c" rgba="{rgba_body}" {c}/>
                  <body name="{p}link5" pos="-0.0825 0.384 0" quat="1 -1 0 0">
                    <inertial mass="1.225946" pos="-1.1953e-2 4.1065e-2 -3.8437e-2" fullinertia="3.5549e-2 2.9474e-2 8.627e-3 -2.117e-3 -4.037e-3 2.29e-4"/>
                    <joint name="{p}joint5"/>
                    <geom name="{p}link5_c0" mesh="link5_c0" rgba="{rgba_body}" {c}/>
                    <geom name="{p}link5_c1" mesh="link5_c1" rgba="{rgba_body}" {c}/>
                    <geom name="{p}link5_c2" mesh="link5_c2" rgba="{rgba_body}" {c}/>
                    <body name="{p}link6" quat="1 1 0 0">
                      <inertial mass="1.666555" pos="6.0149e-2 -1.4117e-2 -1.0517e-2" fullinertia="1.964e-3 4.354e-3 5.433e-3 1.09e-4 -1.158e-3 3.41e-4"/>
                      <joint name="{p}joint6" range="-0.0175 3.7525"/>
                      <geom name="{p}link6_c" mesh="link6_c" rgba="{rgba_body}" {c}/>
                      <body name="{p}link7" pos="0.088 0 0" quat="1 1 0 0">
                        <inertial mass="7.35522e-01" pos="1.0517e-2 -4.252e-3 6.1597e-2" fullinertia="1.2516e-2 1.0027e-2 4.815e-3 -4.28e-4 -1.196e-3 -7.41e-4"/>
                        <joint name="{p}joint7"/>
                        <geom name="{p}link7_c" mesh="link7_c" rgba="{rgba_body}" {c}/>
                        <body name="{p}epee" pos="0 0 0.137" quat="0.7071068 0 0.7071068 0">
                          <!-- epee along the flange z axis: grip, bell guard (m.17: 13.5 cm), blade 0.90 m (m.16), point -->
                          <geom name="{p}grip" type="capsule" fromto="0 0 0 0 0 {GRIP}" size="0.016" mass="0.15" rgba="0.15 0.15 0.17 1" {c} group="1"/>
                          <geom name="{p}guard" type="cylinder" pos="0 0 {GRIP + GUARD_D / 2}" size="{GUARD_R} {GUARD_D / 2}" mass="0.20" rgba="0.80 0.80 0.84 1" {c} group="1"/>
                          <geom name="{p}blade" type="capsule" fromto="0 0 {GRIP + GUARD_D} 0 0 {GRIP + GUARD_D + BLADE - 2 * TIP_R}" size="{BLADE_R}" mass="0.40" rgba="0.75 0.77 0.80 1" {c} group="1"/>
                          <geom name="{p}tip" type="sphere" pos="0 0 {GRIP + GUARD_D + BLADE - TIP_R}" size="{TIP_R}" mass="0.005" rgba="{rgba_trim}" {c} group="1"/>
                          <!-- drawn only (no contacts, no mass): the blade drawn thicker and the point marked, for the cameras -->
                          <geom name="{p}blade_vis" type="capsule" fromto="0 0 {GRIP + GUARD_D} 0 0 {GRIP + GUARD_D + BLADE - 2 * TIP_R}" size="0.009" rgba="0.85 0.87 0.90 1" contype="0" conaffinity="0" mass="0" group="1"/>
                          <geom name="{p}tip_vis" type="sphere" pos="0 0 {GRIP + GUARD_D + BLADE - TIP_R}" size="0.018" rgba="1.0 0.9 0.2 1" contype="0" conaffinity="0" mass="0" group="1"/>
                          <site name="{p}tip_s" pos="0 0 {GRIP + GUARD_D + BLADE}" size="0.004" group="4"/>
                          <site name="{p}guard_s" pos="0 0 {GRIP + GUARD_D}" size="0.004" group="4"/>
                        </body>
                      </body>
                    </body>
                  </body>
                </body>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>'''


def actuators(p):
    # the rail motor: at most the Festo EGC-TB-185's 2500 N feed force (fencing.py RAIL_FMAX; fencing-3: 3000 N)
    return f'''    <motor name="{p}m_rail" joint="{p}rail" ctrlrange="-2500 2500"/>
    <motor class="panda" name="{p}m1" joint="{p}joint1"/>
    <motor class="panda" name="{p}m2" joint="{p}joint2"/>
    <motor class="panda" name="{p}m3" joint="{p}joint3"/>
    <motor class="panda" name="{p}m4" joint="{p}joint4"/>
    <motor class="panda" name="{p}m5" joint="{p}joint5" ctrlrange="-12 12"/>
    <motor class="panda" name="{p}m6" joint="{p}joint6" ctrlrange="-12 12"/>
    <motor class="panda" name="{p}m7" joint="{p}joint7" ctrlrange="-12 12"/>'''


xml = f'''<!--
  Robot epee fencing scene for robot_coding_bench (generated by gen_scene.py; see the task README).
  * World frame: piste surface z = 0, the piste's long axis is x, centre line x = 0; on-guard lines x = +-2,
    rear limits x = +-7 (FIE t.18-t.19: 14 m x 1.5 m), run-backs to x = +-9.
  * Fencer A (x < 0, facing +x) and fencer B (x > 0, facing -x, A turned by 180 degrees about z): cart on a rail
    along x (joint *rail = world x of the cart), 0.90 m pedestal, Franka Panda arm, epee (grip 0.10, guard
    0.135 diameter, blade 0.90; point = site *tip_s).
-->
<mujoco model="rcb_fencing">
  <compiler angle="radian" meshdir="assets" autolimits="true"/>
  <option integrator="implicitfast" timestep="0.001" cone="pyramidal"/>
  <size memory="40M"/>
  <visual>
    <headlight diffuse=".55 .55 .55" ambient=".3 .3 .3" specular=".2 .2 .2"/>
    <global offwidth="1280" offheight="960"/>
    <map znear="0.05" zfar="80"/>
    <quality shadowsize="0"/>
  </visual>
  <default>
    <geom friction="0.3 0.005 0.0001" solref="0.004 1" solimp="0.95 0.99 0.001"/>
    <default class="panda">
      <joint armature="0.1" damping="1" axis="0 0 1" range="-2.8973 2.8973"/>
      <geom type="mesh" rgba="0.92 0.93 0.94 1" group="2"/>
      <motor ctrlrange="-87 87"/>
    </default>
  </default>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1=".55 .6 .7" rgb2="0.2 0.22 0.28" width="10" height="10"/>
    <texture type="2d" name="hall" builtin="checker" rgb1=".26 .24 .22" rgb2=".24 .22 .20" width="300" height="300"/>
    <material name="hall" texture="hall" texrepeat="16 16"/>
    <mesh name="link0_c" file="link0.stl"/>
    <mesh name="link1_c" file="link1.stl"/>
    <mesh name="link2_c" file="link2.stl"/>
    <mesh name="link3_c" file="link3.stl"/>
    <mesh name="link4_c" file="link4.stl"/>
    <mesh name="link5_c0" file="link5_collision_0.obj"/>
    <mesh name="link5_c1" file="link5_collision_1.obj"/>
    <mesh name="link5_c2" file="link5_collision_2.obj"/>
    <mesh name="link6_c" file="link6.stl"/>
    <mesh name="link7_c" file="link7.stl"/>
  </asset>
  <worldbody>
    <light pos="-4 -3 9" dir="0.3 0.2 -1" diffuse="0.6 0.6 0.6" specular="0.2 0.2 0.2" castshadow="false"/>
    <light pos="4 3 9" dir="-0.3 -0.2 -1" diffuse="0.6 0.6 0.6" specular="0.2 0.2 0.2" castshadow="false"/>
    <geom name="floor" type="plane" size="30 30 0.1" pos="0 0 -0.003" material="hall" contype="0" conaffinity="3"/>
{geoms}
    <geom name="rail" type="box" size="{RAIL_LIM + 0.3} 0.05 0.008" pos="0 0 0.008" rgba="0.2 0.2 0.22 1" contype="0" conaffinity="0"/>
{fencer("A_", -2.0, "1 0 0 0", 1, 2, RGBA_A, "0.85 0.15 0.15 1")}
{fencer("B_", 2.0, "0 0 0 1", 2, 1, RGBA_B, "0.15 0.35 0.85 1")}
  </worldbody>
  <actuator>
{actuators("A_")}
{actuators("B_")}
  </actuator>
</mujoco>
'''
open(sys.argv[1], "w").write(xml)
