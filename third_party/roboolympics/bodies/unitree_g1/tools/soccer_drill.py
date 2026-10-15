#!/usr/bin/env python3
"""A soccer drill for the legged body (not part of any sport; the demonstration of bodies/unitree_g1 in the design
note): the G1 starts at a random pose, walks with Unitree's policy (velocity commands from a simple controller) around
the ball to a point behind it on the line to the goal, lines up, stops and kicks with its right foot
(rig.kick(target=the ball's back, direction=the shot line, speed)); a goal if the ball crosses the line between the
posts under the bar. The pitch: a 2 m x 1.2 m goal at x = 5 m, a 0.43 kg ball of 0.11 m radius.

    python -I tools/soccer_drill.py FIRST_SEED LAST_SEED [SPEED]      -> one JSON line per seed (goal, fell, ...)

(needs mujoco and numpy, as the simulator image has them)."""
import sys, math, json, numpy as np, mujoco as mj, xml.etree.ElementTree as ET
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from bodies import load_body
B = load_body('unitree_g1')
GOAL_X, GOAL_HALF, BAR = 5.0, 1.0, 1.2
WORLD = '''<mujoco model="soccer_drill_render">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.001" integrator="implicitfast"/>
  <visual><global offwidth="1920" offheight="1080"/></visual>
  <default/>
  <asset>
    <material name="pitch" rgba="0.24 0.47 0.27 1"/>
    <material name="line" rgba="0.93 0.93 0.93 1"/>
  </asset>
  <worldbody>
    <light pos="0 0 6" dir="0 0 -1" diffuse=".8 .8 .8" castshadow="false"/>
    <geom name="floor" type="plane" size="10 10 .1" material="pitch" conaffinity="5"/>
    <geom name="goal_line" type="box" pos="5 0 0.001" size="0.04 3 0.001" material="line" contype="0" conaffinity="0"/>
    <body name="ball" pos="1.2 -0.12 0.11">
      <freejoint name="ball"/>
      <geom name="geom_ball" type="sphere" size="0.11" mass="0.43" rgba="0.95 0.95 0.95 1" conaffinity="9"
            friction="0.5 0.02 0.002" solref="0.005 0.6"/>
    </body>
    <geom name="post_l" type="cylinder" fromto="5 1 0 5 1 1.2" size="0.05" rgba="1 1 1 1"/>
    <geom name="post_r" type="cylinder" fromto="5 -1 0 5 -1 1.2" size="0.05" rgba="1 1 1 1"/>
    <geom name="bar" type="cylinder" fromto="5 -1 1.2 5 1 1.2" size="0.05" rgba="1 1 1 1"/>
  </worldbody>
  <contact/>
</mujoco>
'''

def build():
    root = ET.fromstring(WORLD)
    B.mount(root, 'A_', (0, 0, 0), 0.0, hand=None)
    m = mj.MjModel.from_xml_string(ET.tostring(root, encoding='unicode'))
    return m, B.rig(m, 'A_', hand=None)

def wrap(a): return (a + math.pi) % (2 * math.pi) - math.pi

class Drill:
    BEHIND = 0.36      # ball centre ahead of the kicking foot's ankle at the kick
    def __init__(self, m, rig, foot='right', speed=2.5, aim=(GOAL_X, 0.0)):
        self.m, self.rig, self.foot, self.speed, self.aim = m, rig, foot, speed, np.array(aim, float)
        self.ball = mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, 'ball')
        self.fsite = mj.mj_name2id(m, mj.mjtObj.mjOBJ_SITE, 'A_%s_foot' % foot)
        self.state, self.t_state, self.cmd = 'approach', 0, np.zeros(3)
    def step(self, d, k):
        rig = self.rig
        p, R = rig.pelvis(d); yaw = math.atan2(R[1, 0], R[0, 0])
        b = d.xpos[self.ball][:2]
        dirv = self.aim - b; dirv /= np.linalg.norm(dirv); yaw_des = math.atan2(dirv[1], dirv[0])
        side = -1 if self.foot == 'right' else 1
        lat = np.array([-dirv[1], dirv[0]])            # left of the shot line
        foot_off = 0.11                                 # the kicking foot's ankle beside the pelvis line
        # where the pelvis should stand: behind the ball on the shot line, shifted so the kicking foot is on the line
        target = b - dirv * (self.BEHIND + 0.0) - lat * side * foot_off
        e = target - p[:2]
        far = np.linalg.norm(e)
        c, s = math.cos(yaw), math.sin(yaw)
        e_b = np.array([c * e[0] + s * e[1], -s * e[0] + c * e[1]])
        if self.state == 'approach':
            # far: face the target point and walk; near: face the shot direction and side-step in
            if far > 0.6:
                # aim at a point 0.5 m behind the target so the robot arrives along the shot line; keep 0.5 m from the
                # ball on the way (go round it on the robot's side)
                via = target - dirv * min(0.5, far - 0.3)
                rel = p[:2] - b
                s_al, l_al = rel @ dirv, rel @ lat
                seg = via - p[:2]
                tt = np.clip(((b - p[:2]) @ seg) / max(seg @ seg, 1e-9), 0, 1)
                if np.linalg.norm(p[:2] + tt * seg - b) < 0.5 and s_al > -0.6:
                    via = b - dirv * 0.6 + lat * (0.6 if l_al >= 0 else -0.6)
                ev = via - p[:2]; head = math.atan2(ev[1], ev[0])
                ev_b = np.array([c * ev[0] + s * ev[1], -s * ev[0] + c * ev[1]])
                wz = np.clip(1.5 * wrap(head - yaw), -0.8, 0.8)
                vx = np.clip(0.8 * ev_b[0], -0.3, 0.7) * max(0.0, math.cos(wrap(head - yaw)))
                vy = np.clip(0.8 * ev_b[1], -0.3, 0.3)
            else:
                wz = np.clip(1.5 * wrap(yaw_des - yaw), -0.6, 0.6)
                vx = np.clip(1.2 * e_b[0], -0.25, 0.35 if far > 0.25 else 0.15)
                vy = np.clip(1.2 * e_b[1], -0.25, 0.25)
            self.cmd = np.array([vx, vy, wz])
            if far < 0.035 and abs(wrap(yaw_des - yaw)) < 0.05:
                self.state, self.t_state = 'kick', k
                self.cmd = np.zeros(3)
                bc = d.xpos[self.ball]
                aim3 = np.array([dirv[0], dirv[1], 0.0])
                tgt = np.array([bc[0], bc[1], bc[2] - 0.03]) - aim3 * (0.11 + 0.105)   # the strike box touching the ball
                rig.kick(d, self.foot, speed=self.speed, target=tgt, direction=dirv)
        elif self.state == 'kick':
            self.cmd = np.zeros(3)
            if rig.mode == 'walk':
                self.state = 'done'
        return self.cmd

def run(seed, foot='right', speed=2.5, steps=2500, record=None, start=None):
    m, rig = build(); d = mj.MjData(m)
    rng = np.random.default_rng(seed)
    if start is None:
        bx, by = rng.uniform(1.0, 2.5), rng.uniform(-1.5, 1.5)
        rx, ry, ryaw = rng.uniform(-0.5, 0.3), rng.uniform(-1.0, 1.0), rng.uniform(-1.2, 1.2)
    else:
        (bx, by), (rx, ry, ryaw) = start
    jb = m.jnt_qposadr[mj.mj_name2id(m, mj.mjtObj.mjOBJ_JOINT, 'ball')]
    d.qpos[jb:jb + 3] = (bx, by, 0.11)
    rig.place(d, rig.rest, pos=(rx, ry, 0.793), yaw=ryaw); mj.mj_forward(m, d)
    dr = Drill(m, rig, foot, speed)
    u = rig.rest.copy(); res = dict(seed=seed, goal=False, fell=False, kicked=False)
    ball = dr.ball; vmax = 0.0; tk = None
    for k in range(steps):
        cmd = dr.step(d, k)
        u_prev = u.copy(); u = rig.rest.copy(); u[0:3] = cmd
        rig.begin_step(d, u)
        for j in range(1, 11):
            rig.apply(d, u_prev, u, j, u_prev); mj.mj_step(m, d)
        if record is not None and k % 2 == 0:
            record.append(d.qpos.copy())
        bv = np.linalg.norm(d.qvel[m.jnt_dofadr[mj.mj_name2id(m, mj.mjtObj.mjOBJ_JOINT, 'ball')]:][:3])
        if bv > vmax: vmax = bv
        if dr.state != 'approach' and tk is None: tk = k
        if rig.pelvis(d)[0][2] < 0.5 and not res['fell']:
            res['fell'] = True; res['fall_k'] = k; res['fall_phase'] = (rig.kicker.phases[min(rig.kicker.pi, len(rig.kicker.phases)-1)][0] if rig.kicker else rig.mode)
        bp = d.xpos[ball]
        if bp[0] >= GOAL_X and not res.get('crossed'):
            res['crossed'] = True; res['goal'] = bool(abs(bp[1]) < GOAL_HALF - 0.11 and bp[2] < BAR - 0.11)
            res['cross_y'], res['cross_z'] = float(bp[1]), float(bp[2])
        if dr.state == 'done' and (res.get('crossed') or bv < 0.05) and k > (tk or 0) + 150:
            if k > (tk or 0) + 400 or res.get('crossed'): break
    res.update(kicked=tk is not None, t_kick=None if tk is None else tk * 0.01, ball_vmax=round(vmax, 2),
               t_end=k * 0.01, ball_end=[round(float(x), 2) for x in d.xpos[ball]], pelvis_end=[round(float(x), 2) for x in rig.pelvis(d)[0]])
    return res

if __name__ == '__main__':
    a, b = int(sys.argv[1]), int(sys.argv[2]); speed = float(sys.argv[3]) if len(sys.argv) > 3 else 2.5
    for sd in range(a, b):
        print(json.dumps(run(sd, speed=speed)), flush=True)
