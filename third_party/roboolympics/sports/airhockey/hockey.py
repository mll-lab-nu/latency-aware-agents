"""Two physical 7-axis iiwa arms on an air hockey table: fixed-step Cartesian execution and the referee.

Robot dynamics are derived from AirHockeyChallenge (MIT, see scene/). The table is new and mesh-free. The referee
follows the USAA Official Rules of Air Hockey (sport.toml [rulebook]) with the deviations listed there
([[deviations]]): one game to 7 points, the seven-second rule (a foul: the puck goes to the opponent), dead-puck
face-offs, a safety cap on the playing time with a stated last resort (no draws) and a 0.35 m goal mouth (D10).

The arms are held to the real KUKA LBR iiwa 14's limits since revision v5 (2026-10-08, real-limits audit item 7;
bodies/kuka_iiwa14/body.toml, bodies/limits.py): the joint reference of every control step is retimed so that no axis
needs more than 0.95 x its data-sheet velocity (the Air Hockey Challenge's constraint factor), and the robot's 1 kHz
setpoint generator (limits.JerkLimiter) keeps the setpoint within that velocity, the AHC accelerations (10 x the
velocity limits) and the estimated jerks; the servo's commanded acceleration is clipped to the same accelerations
(v4 and before: no velocity limit on the position reference, +-50 rad/s^2). Both arms, the scripted opponent's too.

The mallet stays on the table since revision v6 (2026-10-10, Changhe's physical-realism audit): the Air Hockey
Challenge holds the end effector (the universal joint above the mallet) within +-2 cm of its height on the table
(MALLET_Z, Z_TOL), and a real arm can neither push the mallet through the table nor lift it over the puck. In v5 the
mallet's path was retimed against the joint velocities only and changed its acceleration at will; the setpoint
generator then limited each joint's acceleration and jerk on its own, which bends the joint-space path, so the mallet
left the table plane by up to 12 cm (its centre 11 cm below the surface, the rod 9 cm into it, 4 % of the control
steps more than 1 cm below). Since v6 the path itself is a jerk-limited motion on the table plane
(bodies/servo.PlanarBase) whose speed, acceleration and jerk caps are, at every step, what the joints allow there
(the joints' sensitivity to the mallet's motion, here, where the motion leads and, near the reach, radially, times
REF_V / REF_A / REF_J of their limits), that brakes before the workspace's edge, and whose IK keeps the arm near its
home posture in the null space: the joint reference stays inside the limits, and the setpoint generator follows it
without clipping single joints (Hockey.reference()). Hockey.height_violations counts the physics steps whose mallet
leaves the +-2 cm band (final.json: mallet_height_violations).
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

_ROOT = Path(__file__).resolve().parent.parent      # sim/ in a task (bodies/ next to sport/); sports/ in a checkout
sys.path.insert(0, str(_ROOT if (_ROOT / 'bodies').is_dir() else _ROOT.parent))
from bodies import limits as AL  # noqa: E402
from bodies import servo as SV  # noqa: E402

REVISION = 'airhockey-program-v7'      # v7 (2026-10-11): the arms' setpoint generator brakes within the jerk limit; v6 (2026-10-10): the mallet stays within +-2 cm of the table; v5: the real iiwa 14
                                        # limits; v4: goal mouth 0.35 m (v3: 0.40 m, v2: 0.25 m)
DT, PHYSICS_DT = .01, .001                  # 10 ms control step (eai-standard-realtime/1.0), 1 kHz physics
SUBSTEPS = round(DT / PHYSICS_DT)
POINTS_TO_WIN = 7                           # USAA I.1
SEVEN_S = 7.                                # USAA I.10: seven seconds to shoot the puck across the centre line
DEAD_S = 2.                                 # dead puck: out of both robots' reach this long -> face-off
LINE_S = 7.                                 # dead puck: resting on the centre line this long -> face-off
CAP_S = 120.                                # safety cap: at most this much playing time (game seconds) ...
MAX_RALLIES = 40                            # ... and at most this many rallies; then the last resort decides
CAM_W, CAM_H = 320, 240
LENGTH, WIDTH = 1.948, 1.038
GOAL_DEFAULT = .35                          # sport.toml deviation D10 (the AirHockeyChallenge table: .25)


def goal_width():
    """The goal mouth (m): GOAL_DEFAULT, or the experiment-only override RO_AIRHOCKEY_GOAL_M (0.2-0.6; the official
    tasks never set it; plugin.REVISION and final.json record a non-default width)."""
    v = os.environ.get('RO_AIRHOCKEY_GOAL_M', '').strip()
    if not v: return GOAL_DEFAULT
    g = float(v)
    if not (math.isfinite(g) and .2 <= g <= .6): raise ValueError('RO_AIRHOCKEY_GOAL_M must be within 0.2-0.6 m')
    return g


GOAL = goal_width()
PUCK_R, MALLET_R = .03165, .04815
CONTACT = PUCK_R + MALLET_R
PUCK_DAMPING = '.001'                       # N s/m on the 10 g puck: the air cushion's drag (about 10 % speed per second)
BASE_X = 1.51                               # robot bases at x = -1.51 (A) and +1.51 (B), y = 0
REACH = 1.45                                # mallet centre at most this far from its robot's base
X_NEAR, X_FAR, Y_MAX = .05, .92, .47        # mallet centre workspace in its own half: X_NEAR <= |x| <= X_FAR, |y| <= Y_MAX
HOME_X, SPOT_X = .82, .58                   # mallet home and the puck's placement spot (|x|), in the own half
MAX_SPEED, MAX_ACCEL = 2., 10.             # the mallet's Cartesian trajectory limits (m/s, m/s^2)
LEVELS = {                                  # the scripted opponent (seat B of the single-player game): mallet
    1: dict(speed=.4, delay=15, guard=None, gain=0., tol=.25, aim=0., lead=False),   # speed cap m/s, reaction delay
    2: dict(speed=.6, delay=10, guard=.80, gain=1., tol=.15, aim=0., lead=False),   # steps, guard line |x| m, gain
    3: dict(speed=1., delay=6, guard=.80, gain=1., tol=.08, aim=.07, lead=False),   # on the puck's y, line-up
    4: dict(speed=2., delay=3, guard=.80, gain=1., tol=.05, aim=.08, lead=True)}    # tolerance rad, aim (y at the
                                                                                    # goal), crossing prediction
SCENE = Path(__file__).with_name('scene')
KUKA = AL.load('kuka_iiwa14')               # the data sheet's ranges, velocities, torques; AHC accelerations; jerk (est.)
VEL_FACTOR = .95                            # the AHC constraint: joint velocities within 0.95 x the data sheet
ARM = AL.ArmLimits(KUKA.body, KUKA.model + ' (velocity x0.95, AHC)', KUKA.range_lo, KUKA.range_hi, VEL_FACTOR * KUKA.vel,
                   KUKA.acc, KUKA.jerk, KUKA.tau, None, KUKA.sources)
STRIKER_V, STRIKER_A = 3., 50.              # the two striker joints (not iiwa axes): the AHC model's PD as before
MALLET_Z = .0645                            # the universal joint above the mallet on the table (the AHC EE height)
Z_TOL = .02                                 # the AHC end-effector constraint: within +-2 cm of MALLET_Z (v6)
# v6 (2026-10-10): the mallet's path keeps the joint reference well inside the limits, so the setpoint generator
# follows it without clipping single joints (which bends the path off the table): Hockey.reference()
CART_ACCEL, CART_JERK = MAX_ACCEL, 150.      # the path's own acceleration (m/s^2, as in v5) and jerk (m/s^3) caps
REF_V, REF_A, REF_J = .95, .6, .3           # the shares of the joints' velocity / acceleration / jerk limits it may use
POSTURE = np.array([0, -.1961, 0, -1.8436, 0, .9704, 0, 0, 0.])   # the arms' posture at home; the IK pulls towards it
POSTURE_GAIN, POSTURE_STEP = .005, .0003    # in the null space (gain and rad per iteration): no drift to joint limits
NEAR_REACH = (1.35, 1.42)                   # m from the base: near the reach the radial joint speeds count (fully at 1.42)
LOOKAHEAD = .1                              # s: the path's limits also hold where its present motion leads
IK_TOL = .003                               # m: a path step the IK cannot reach on the table within this is not taken
EDGE_ROOM, EDGE_AIM = .015, .005            # m: the path plans to stop this far inside the workspace's edge, aims this far
PATH_LIMITS = dict(accel_m_s2=CART_ACCEL, jerk_m_s3=CART_JERK, yaw_rate_rad_s=1., yaw_accel_rad_s2=1., yaw_jerk_rad_s3=1.,
                   mode='norm')
OWN = {'A': -1., 'B': 1.}                   # the sign of x in each seat's own half
OTHER = {'A': 'B', 'B': 'A'}


def source_sha():
    h = hashlib.sha256(Path(__file__).read_bytes())
    h.update(Path(AL.__file__).read_bytes()); h.update((Path(AL.__file__).parent/'kuka_iiwa14'/'body.toml').read_bytes())
    h.update(Path(SV.__file__).read_bytes())
    for f in sorted(SCENE.glob('*.xml')):
        h.update(f.name.encode()); h.update(f.read_bytes())
    return h.hexdigest()


def scene_xml(seed=101):
    root = ET.Element('mujoco', model='visual_iiwa_airhockey')
    ET.SubElement(root, 'compiler', angle='radian', autolimits='true')
    ET.SubElement(root, 'option', timestep=str(PHYSICS_DT), integrator='implicitfast', cone='elliptic', iterations='40')
    visual = ET.SubElement(root, 'visual'); ET.SubElement(visual, 'global', offwidth='640', offheight='480')
    ET.SubElement(visual, 'headlight', ambient='.5 .5 .5', diffuse='.5 .5 .5')
    world = ET.SubElement(root, 'worldbody')
    ET.SubElement(world, 'light', pos='0 0 3', dir='0 0 -1', directional='true', castshadow='false')
    ET.SubElement(world, 'geom', name='table_surface', type='box', pos='0 0 -.045', size='1.07 .61 .045', rgba='.9 .92 .94 1', contype='0', conaffinity='0')
    ET.SubElement(world, 'geom', name='floor', type='plane', pos='0 0 -.15', size='4 4 .1', rgba='.16 .2 .24 1', contype='0', conaffinity='0')
    for side in (-1, 1):
        ET.SubElement(world, 'geom', name=f'wall_side_{side}', type='box', pos=f'0 {side*(WIDTH/2+.025)} .035', size='1.07 .025 .035', rgba='.2 .25 .3 1', friction='0 0 0', solref='.0075 .1', solimp='.99 .999 .001')
        for end in (-1, 1):
            half = (WIDTH-GOAL)/4
            ET.SubElement(world, 'geom', name=f'wall_end_{side}_{end}', type='box', pos=f'{end*(LENGTH/2+.025)} {side*(GOAL/2+half)} .035', size=f'.025 {half} .035', rgba='.2 .25 .3 1', friction='0 0 0', solref='.0075 .1', solimp='.99 .999 .001')
    # Four public calibration markers outside the puck lane, plus a center line.
    for i, (x, y) in enumerate([(-.9, -.46), (-.9, .46), (.9, -.46), (.9, .46)]):
        ET.SubElement(world, 'geom', type='cylinder', pos=f'{x} {y} .001', size='.015 .001', rgba='1 .75 .03 1', contype='0', conaffinity='0')
    ET.SubElement(world, 'geom', type='box', pos='0 0 .001', size='.002 .5 .001', rgba='.7 .72 .75 1', contype='0', conaffinity='0')
    base = ET.SubElement(world, 'body', name='puck_base')
    for name, kind, axis, damping in [('puck_x', 'slide', '1 0 0', PUCK_DAMPING), ('puck_y', 'slide', '0 1 0', PUCK_DAMPING), ('puck_yaw', 'hinge', '0 0 1', '0.000002')]:
        ET.SubElement(base, 'joint', name=name, type=kind, axis=axis, damping=damping, limited='false')
    ET.SubElement(base, 'inertial', pos='0 0 0', mass='.01', diaginertia='0.0000025 0.0000025 0.000005')
    ET.SubElement(base, 'geom', name='puck', type='cylinder', pos='0 0 .012', size=f'{PUCK_R} .01', rgba='.95 .03 .04 1', friction='0 0 0', condim='4', solref='.0075 .1')
    for seat, pos, axes in [('A', '0 -1.3 2.2', '1 0 0 0 .861 .509'), ('B', '0 1.3 2.2', '-1 0 0 0 -.861 .509')]:
        ET.SubElement(world, 'camera', name=f'front_{seat}', pos=pos, xyaxes=axes, fovy='48')
    ET.SubElement(world, 'camera', name='overview', pos='0 -2.3 2.5', xyaxes='1 0 0 0 .736 .677', fovy='55')
    # Merge upstream robot defaults/world/actuators without any external assets.
    for i in (1, 2):
        robot = ET.parse(SCENE/f'iiwa{i}.xml').getroot()
        for child in robot:
            if child.tag == 'compiler': continue
            if child.tag == 'worldbody':
                for body in child: world.append(body)
            elif child.tag == 'default':
                if i == 1: root.append(child)          # iiwa1.xml declares the shared classes (vis, robot)
            elif child.tag == 'actuator':
                acts = root.find('actuator')
                if acts is None: acts = ET.SubElement(root, 'actuator')
                for act in child: acts.append(act)
    return ET.tostring(root, encoding='unicode')


def workspace(seat, xy, margin=0.):
    """The nearest point of the seat's mallet workspace (own half, X_NEAR <= |x| <= X_FAR, |y| <= Y_MAX, at most
    REACH from its robot's base), shrunk by margin, to a point (approximately: box clip, then radial clip towards the
    base)."""
    u, v = OWN[seat]*float(xy[0]), float(xy[1])
    u, v = min(max(u, X_NEAR+margin), X_FAR-margin), min(max(v, -Y_MAX+margin), Y_MAX-margin)
    d = math.hypot(BASE_X-u, v)
    if d > REACH-margin:
        u, v = BASE_X-(BASE_X-u)*(REACH-margin)/d, v*(REACH-margin)/d
    return [OWN[seat]*u, v]


def reachable(seat, p):
    """Whether the seat's mallet can touch a puck centred at p (a mallet centre in the workspace within CONTACT)."""
    q = workspace(seat, p)
    return math.hypot(float(p[0])-q[0], float(p[1])-q[1]) <= CONTACT-.005


def in_workspace(seat, xy, eps=1e-4):
    u, v = OWN[seat]*float(xy[0]), float(xy[1])
    return X_NEAR-eps <= u <= X_FAR+eps and abs(v) <= Y_MAX+eps and math.hypot(BASE_X-u, v) <= REACH+eps


def edge_distance(seat, xy, u, far=1.):
    """How far the mallet centre can go from xy along the unit direction u before it leaves the seat's workspace
    (bisection; 0 if xy is already outside)."""
    if not in_workspace(seat, xy): return 0.
    lo, hi = 0., far
    if in_workspace(seat, np.asarray(xy)+u*far): return far
    for _ in range(16):
        mid = .5*(lo+hi)
        if in_workspace(seat, np.asarray(xy)+u*mid): lo = mid
        else: hi = mid
    return lo


def validate_action(body, seat):
    """None for `none`; else the target [x, y]. Non-finite, wrong half or outside the workspace -> ValueError (400)."""
    if body.get('type') == 'none': return None
    if body.get('type', 'target') != 'target': raise ValueError('action type must be target or none')
    values = [body.get('x'), body.get('y')]
    if any(type(v) not in (int, float) or not math.isfinite(v) for v in values): raise ValueError('x,y must be finite numbers')
    if seat not in ('A', 'B'): raise ValueError('unknown seat')
    if OWN[seat]*values[0] <= 0: raise ValueError('mallet must stay in own half')
    if not in_workspace(seat, values):
        raise ValueError(f'target outside the mallet workspace ({X_NEAR} <= |x| <= {X_FAR}, |y| <= {Y_MAX}, '
                         f'at most {REACH} m from the robot base at ({OWN[seat]*BASE_X:+.2f}, 0))')
    return [float(v) for v in values]


class Hockey:
    def __init__(self, seed=101, level=1, duel=False, first='A'):
        if level not in (1, 2, 3, 4) or first not in ('A', 'B'): raise ValueError('invalid configuration')
        self.seed, self.level, self.duel, self.first = seed, level, duel, first
        self.source_sha = source_sha(); self.model = mujoco.MjModel.from_xml_string(scene_xml())
        self.data = mujoco.MjData(self.model); self.ik = mujoco.MjData(self.model)
        self.rng = np.random.default_rng(seed)
        self.jl = {s: AL.JerkLimiter(ARM, PHYSICS_DT) for s in ('A', 'B')}     # the arms' 1 kHz setpoint generators
        # the output stage (limits.TorqueRateLimiter): |tau| within the data sheet's maximum torques; KUKA publishes no
        # torque rate (body.toml: NOT PUBLISHED), so no rate limit applies (the stage is stateless then)
        self.tl = {s: AL.TorqueRateLimiter(ARM, PHYSICS_DT) for s in ('A', 'B')}
        self.k = 0                  # control steps of play (the world's clock; frozen time does not count)
        self.point_k = 0            # control steps since the current point started
        self.results = []           # one entry per rally: how it ended (goal, foul, dead puck, time cap)
        self.done = self.awaiting_review = False
        self.decided_by = None      # 'score' | 'cap_points' | 'cap_last_goal' | 'cap_territory' | 'cap_first_possession'
        self.rail_contact_steps = self.contact_steps = 0
        self.height_violations = {'A': 0, 'B': 0}  # physics steps with the mallet outside MALLET_Z +- Z_TOL (v6)
        self.half_steps = {'A': 0, 'B': 0}      # steps the puck spent wholly in each half (the cap's territory measure)
        self.ids = {}; self.qadr = {}; self.dadr = {}; self.acts = {}; self.targets = {}
        for s, n in [('A', 1), ('B', 2)]:
            names = [f'iiwa_{n}/joint_{j}' for j in range(1, 8)]+[f'iiwa_{n}/striker_joint_{j}' for j in (1, 2)]
            self.ids[s] = [self.model.joint(name).id for name in names]
            self.qadr[s] = self.model.jnt_qposadr[self.ids[s]]; self.dadr[s] = self.model.jnt_dofadr[self.ids[s]]
            self.acts[s] = [self.model.actuator(name).id for name in names]
        self.ee = {'A': self.model.geom('iiwa_1/ee').id, 'B': self.model.geom('iiwa_2/ee').id}
        self.mallet_body = {'A': self.model.body('iiwa_1/striker_mallet').id, 'B': self.model.body('iiwa_2/striker_mallet').id}
        self.puck_geom = self.model.geom('puck').id
        self.wall_geoms = {i for i in range(self.model.ngeom) if (self.model.geom(i).name or '').startswith('wall_')}
        self.px = int(self.model.jnt_qposadr[self.model.joint('puck_x').id]); self.py = self.px+1
        self.pvx = int(self.model.jnt_dofadr[self.model.joint('puck_x').id])
        mujoco.mj_resetData(self.model, self.data)
        self.restart(self.first, 'start')

    # ---------------------------------------------------------------- state
    def position(self, seat): return self.data.geom_xpos[self.ee[seat]].copy()

    def puck(self): return np.array([self.data.qpos[self.px], self.data.qpos[self.py]])

    def puck_velocity(self): return np.array([self.data.qvel[self.pvx], self.data.qvel[self.pvx+1]])

    def restart(self, holder, why):
        """Play stops: both mallets go back to their home positions (instantly) and the referee places the puck at
        rest, at the holder's spot in its own half (seeded y), or at the table centre for a face-off (holder None)."""
        puck = [0., 0.] if holder is None else [OWN[holder]*SPOT_X, round(float(self.rng.uniform(-.18, .18)), 6)]
        self.data.qvel[:] = 0
        for s in ('A', 'B'):
            self.data.qpos[self.qadr[s]] = POSTURE
        self.data.qpos[self.px], self.data.qpos[self.py], self.data.qpos[self.py+1] = puck[0], puck[1], 0.
        mujoco.mj_forward(self.model, self.data)
        for s in ('A', 'B'):
            self.targets[s] = np.array([OWN[s]*HOME_X, 0.])
            self.data.qpos[self.qadr[s]] = self.solve(s, self.data.qpos[self.qadr[s]].copy(), self.targets[s], 80)
        self.data.ctrl[:] = 0
        self.commands = {s: list(self.targets[s]) for s in ('A', 'B')}
        self.qtarget = {s: self.data.qpos[self.qadr[s]].copy() for s in ('A', 'B')}
        self.cart_velocity = {s: np.zeros(2) for s in ('A', 'B')}
        self.path = {s: SV.PlanarBase(SV.BaseLimits.of(dict(speed_m_s=MAX_SPEED, **PATH_LIMITS)), PHYSICS_DT) for s in ('A', 'B')}
        for s in ('A', 'B'): self.path_reset(s, self.targets[s])
        self.side, self.side_steps = holder, 0            # possession: the half the puck is wholly in
        self.line_steps = self.dead_steps = 0
        self.last_touch = None
        self.script = dict(until=-1, target=None, hist=[])
        self.restart_k, self.restart_why = self.k, why
        for s in ('A', 'B'):
            self.jl[s].reset(self.data.qpos[self.qadr[s]][:7])
        mujoco.mj_forward(self.model, self.data)

    def path_reset(self, s, p, v=(0., 0.)):
        """Restart the seat's mallet path at p (table metres) moving with velocity v (m/s)."""
        self.targets[s], self.cart_velocity[s] = np.array(p, float), np.array(v, float)
        self.path[s].reset([*self.targets[s], 0.], [*self.cart_velocity[s], 0.])

    def solve(self, seat, q, target, iterations):
        adr = self.qadr[seat]; dof = self.dadr[seat]
        desired = np.diag([-1., 1., -1.]) if seat == 'A' else np.diag([1., -1., -1.])
        body = self.mallet_body[seat]
        jp = np.zeros((3, self.model.nv)); jr = np.zeros_like(jp)
        self.ik.qpos[:] = self.data.qpos
        ranges = self.model.jnt_range[self.ids[seat]]
        for _ in range(iterations):
            self.ik.qpos[adr] = q; mujoco.mj_kinematics(self.model, self.ik); mujoco.mj_comPos(self.model, self.ik)
            current = self.ik.xmat[body].reshape(3, 3)
            err = np.r_[np.array([target[0], target[1], MALLET_Z])-self.ik.xpos[body],
                        sum(np.cross(current[:, i], desired[:, i]) for i in range(3))*.5]
            mujoco.mj_jacBody(self.model, self.ik, jp, jr, body)
            jac = np.vstack([jp[:, dof], jr[:, dof]])
            pinv = jac.T@np.linalg.inv(jac@jac.T+.001*np.eye(6))
            delta = pinv@err
            delta += (np.eye(len(q))-pinv@jac)@np.clip(POSTURE_GAIN*(POSTURE-q), -POSTURE_STEP, POSTURE_STEP)  # v6: posture
            q = q+np.clip(delta, -.08, .08)
            q = np.clip(q, ranges[:, 0]+.01, ranges[:, 1]-.01)
        self.ik.qpos[adr] = q; mujoco.mj_kinematics(self.model, self.ik)
        self.ik_residual = float(np.linalg.norm(np.array([target[0], target[1], MALLET_Z])-self.ik.xpos[body]))
        return q

    def set_command(self, seat, command):
        if command is None: self.commands[seat] = workspace(seat, self.targets[seat]); return     # hold (v6: the path's
                                                                                         # position, not the lagging mallet)
        if OWN[seat]*command[0] <= 0: raise ValueError('mallet must stay in own half')
        self.commands[seat] = list(command)

    # ---------------------------------------------------------------- scripted opponent (single player, seat B)
    def scripted(self, seat='B'):
        """Trusted referee-side opponent, from the true puck position `delay` steps ago: guards its goal while the
        puck is in the other half or coming at it fast, else lines up behind the puck and strikes it at the other
        goal. Levels differ in mallet speed, reaction delay, how far the guard follows the puck, how precisely it lines
        up and where it aims (LEVELS)."""
        lv = LEVELS[self.level]; sg = OWN[seat]; sc = self.script
        sc['hist'].append(self.puck()); sc['hist'] = sc['hist'][-(lv['delay']+2):]
        P = sc['hist'][0]; W = (sc['hist'][1]-sc['hist'][0])/DT if len(sc['hist']) > 1 else np.zeros(2)
        p = np.array([sg*P[0], P[1]]); w = np.array([sg*W[0], W[1]])          # local: u > 0 is the own half
        m = self.position(seat)[:2]; m = np.array([sg*m[0], m[1]])
        p = p+w*lv['delay']*DT                                                  # now, by its own extrapolation
        world = lambda u, v: workspace(seat, [sg*u, v])
        if p[0] < -.02 or w[0] > .15:           # guard: at its guard line, following the puck's y (or its crossing)
            sc['until'] = -1
            if lv['guard'] is None:             # level 1 does not guard: it stays where it is (v6: where its path
                return workspace(seat, self.targets[seat])      # is; v5 chased the lagging mallet and could oscillate)
            y = p[1]
            if lv['lead'] and w[0] > .1:
                y = p[1]+w[1]*(lv['guard']-p[0])/w[0]
                r = WIDTH/2-PUCK_R; y = (y+r) % (4*r); y = (y if y < 2*r else 4*r-y)-r   # rail bounces folded in
            return world(lv['guard'], float(np.clip(y*lv['gain'], -.42, .42)))
        if self.k < sc['until']: return sc['target']
        aim = 0. if not lv['aim'] else (lv['aim'] if p[1] < 0 else -lv['aim'])  # the far corner of the goal
        d = np.array([-1.05, aim])-p; d /= np.linalg.norm(d)
        rel = p-m; dist = np.linalg.norm(rel)
        if .12 < dist < .3 and float(rel@d)/dist > math.cos(lv['tol']):         # lined up: strike through the puck
            sc['target'] = world(*(m+rel/dist*(dist+.25))); sc['until'] = self.k+40
            return sc['target']
        return world(*(p-d*.18))

    # ---------------------------------------------------------------- one control step
    def reference(self, s):
        """The seat's joint target at the end of this control step (v6). The mallet's path towards its command is a
        jerk-limited Cartesian motion on the table plane (bodies/servo.PlanarBase on the norm: CART_ACCEL, CART_JERK,
        at most MAX_SPEED; the scripted opponent: its level's speed) whose speed limit is, at every step, the speed at
        which no iiwa axis needs more than REF_V x its velocity limit (along the way to the command and along the
        present motion); the IK puts the joint target on the table plane. Should a step still need more than a
        joint's velocity limit, the path slows down along its line (v5's retiming). The joint reference so stays
        within the joints' velocity, acceleration and jerk limits with room for the setpoint generator, which then
        follows it without clipping single joints (in v5 that clipping bent the path off the table)."""
        qprev = self.qtarget[s]; p0 = self.targets[s]; g = self.path[s]
        target = np.array(self.commands[s])
        vmax = MAX_SPEED if s == 'A' or self.duel else LEVELS[self.level]['speed']
        sens = np.zeros(7)                                      # rad per metre of mallet motion, per iiwa axis
        cv0 = self.cart_velocity[s]
        for u in (target-p0, cv0):                              # along the way to the command and the present motion
            n = float(np.linalg.norm(u))
            if n < 1e-6: continue
            q = self.solve(s, qprev, p0+u/n*.01, 4)
            sens = np.maximum(sens, np.abs(q[:7]-qprev[:7])/.01)
        rad = np.array([p0[0]-OWN[s]*BASE_X, p0[1]]); r = float(np.linalg.norm(rad))
        near = min(1., max(0., (r-NEAR_REACH[0])/(NEAR_REACH[1]-NEAR_REACH[0])))
        if near > 0:                                            # near the reach the arm is nearly stretched: there any
            q = self.solve(s, qprev, p0+rad/r*.01, 4)          # motion may need the radial direction's joint speeds
            sens = np.maximum(sens, near*np.abs(q[:7]-qprev[:7])/.01)
        n = float(np.linalg.norm(cv0))
        if n > 1e-3:                                            # and where the present motion leads (LOOKAHEAD s)
            pa = np.array(workspace(s, p0+cv0*LOOKAHEAD)); qa = self.solve(s, qprev, pa, 4)
            q = self.solve(s, qa, pa+cv0/n*.01, 4)
            sens = np.maximum(sens, np.abs(q[:7]-qa[:7])/.01)
        sens = np.maximum(sens, 1e-9)
        acc = min(CART_ACCEL, REF_A*float(np.min(ARM.acc/sens)))
        jerk = min(CART_JERK, REF_J*float(np.min(ARM.jerk/sens)))
        speed = min(vmax, REF_V*float(np.min(ARM.vel/sens)))
        if n > 1e-6:                                            # it can still stop before the workspace's edge
            room = max(0., edge_distance(s, p0, cv0/n)-EDGE_ROOM)
            speed = min(speed, float(SV.brake_speed(room, SV.PlanarBase.BRAKE*acc, jerk)))
        lim = dict(PATH_LIMITS, speed_m_s=max(speed, 1e-3), accel_m_s2=acc, jerk_m_s3=jerk)
        g.lim = SV.BaseLimits.of(lim)
        aim = np.array(workspace(s, target, EDGE_AIM))         # the path aims a little inside the edge, so that it
        for _ in range(SUBSTEPS):                               # does not overshoot it on arrival
            pq, pv, _a = g.step([aim[0], aim[1], 0.])
        p, cv = pq[:2].copy(), pv[:2].copy()
        if not in_workspace(s, p):                              # never beyond the workspace (the arm's reach):
            w = np.array(workspace(s, p)); n = p-w; n /= max(1e-12, float(np.linalg.norm(n)))
            cv = cv-max(0., float(cv@n))*n                      # it stops at the edge, sliding along it
            p = w; g.reset([*p, 0.], [*cv, 0.])
        q = self.solve(s, qprev, p, 4)
        if self.ik_residual > IK_TOL:                            # the arm cannot get there on the table: stop
            p, cv = p0.copy(), np.zeros(2); g.reset([*p, 0.])
            q = self.solve(s, qprev, p, 4)
        need = float(np.max(np.abs(q[:7]-qprev[:7])/DT/ARM.vel))
        if need > 1.:                                           # still too fast for a joint: slow down on the line
            p = p0+(p-p0)/need*.98; cv = cv/need*.98
            q = self.solve(s, qprev, p, 4)
            g.reset([*p, 0.], [*cv, 0.])
        self.cart_velocity[s] = cv
        self.targets[s] = p
        return q

    def step(self):
        if self.done or self.awaiting_review: raise RuntimeError('world frozen')
        if not self.duel: self.set_command('B', self.scripted('B'))
        qprevious = {}
        for s in ('A', 'B'):
            qprevious[s] = self.qtarget[s].copy()
            self.qtarget[s] = self.reference(s)
        contact = set()
        mass = np.zeros((self.model.nv, self.model.nv))
        for substep in range(SUBSTEPS):
            mujoco.mj_forward(self.model, self.data)
            mujoco.mj_fullM(self.model, mass, self.data.qM)
            for s in ('A', 'B'):
                dof = self.dadr[s]
                q = self.data.qpos[self.qadr[s]]; v = self.data.qvel[dof]
                reference = qprevious[s]+(self.qtarget[s]-qprevious[s])*(substep+1)/SUBSTEPS
                rd = (self.qtarget[s]-qprevious[s])/DT
                qs, qds, qdds = self.jl[s].step(reference[:7], np.clip(rd[:7], -ARM.vel, ARM.vel))
                acceleration = np.empty(9)                      # the iiwa axes: the setpoint's feed-forward + PD
                acceleration[:7] = np.clip(qdds+100*(qs-q[:7])+20*(qds-v[:7]), -ARM.acc, ARM.acc)
                acceleration[7:] = np.clip(100*(reference[7:]-q[7:])+20*(np.clip(rd[7:], -STRIKER_V, STRIKER_V)-v[7:]),
                                           -STRIKER_A, STRIKER_A)
                torque = mass[np.ix_(dof, dof)]@acceleration+self.data.qfrc_bias[dof]
                torque[:7] = self.tl[s](torque[:7])
                bounds = self.model.actuator_ctrlrange[self.acts[s]]
                self.data.ctrl[self.acts[s]] = np.clip(torque, bounds[:, 0], bounds[:, 1])
            mujoco.mj_step(self.model, self.data)
            for s in ('A', 'B'):
                if abs(self.data.xpos[self.mallet_body[s]][2]-MALLET_Z) > Z_TOL: self.height_violations[s] += 1
            for i in range(self.data.ncon):
                c = self.data.contact[i]; g = (int(c.geom1), int(c.geom2))
                if self.puck_geom in g:
                    for s in ('A', 'B'):
                        if self.ee[s] in g: contact.add(s)
                    if g[0] in self.wall_geoms or g[1] in self.wall_geoms: contact.add('wall')
        self.k += 1; self.point_k += 1
        for s in ('A', 'B'):
            if s in contact: self.last_touch = s
        if contact & {'A', 'B'}: self.contact_steps += 1
        if 'wall' in contact: self.rail_contact_steps += 1
        self.referee()
        return contact

    def referee(self):
        """After each control step: a goal, a seven-second foul or a dead puck ends the rally; the world then freezes
        for the review and the next rally starts from the placement the rules give (ready())."""
        p = self.puck()
        if abs(p[0]) > LENGTH/2+PUCK_R and abs(p[1]) <= GOAL/2-PUCK_R:          # wholly in a goal (USAA I.2, V.1)
            scorer = 'A' if p[0] > 0 else 'B'
            return self.end_rally('goal', winner=scorer, next=OTHER[scorer],    # the scored-upon serves (I.4)
                                  own_goal=self.last_touch == OTHER[scorer])
        side = 'A' if p[0] < -PUCK_R else ('B' if p[0] > PUCK_R else None)   # on the centre line: nobody's (I.11)
        if side:
            self.half_steps[side] += 1
            self.line_steps = 0
            if side != self.side: self.side, self.side_steps = side, 0
            self.side_steps += 1
        else:
            self.line_steps += 1
        self.dead_steps = 0 if (reachable('A', p) or reachable('B', p)) else self.dead_steps+1
        if self.dead_steps*DT >= DEAD_S-1e-9:
            return self.end_rally('dead_puck', where='out_of_reach', next=None)
        if self.line_steps*DT >= LINE_S-1e-9:
            return self.end_rally('dead_puck', where='centre_line', next=None)
        if side and self.side_steps*DT >= SEVEN_S-1e-9:                  # a foul: forfeiture of the puck (VI.1)
            return self.end_rally('seven_second_foul', guilty=side, next=OTHER[side])
        if self.k*DT >= CAP_S-1e-9:
            return self.end_rally('time_cap', next=None)

    def end_rally(self, how, winner=None, guilty=None, next=None, **extra):
        self.results.append(dict(rally=len(self.results)+1, how=how, winner=winner, guilty=guilty,
                                 next='faceoff' if next is None else next, k=self.k, steps=self.point_k, **extra))
        sc = self.scores()
        if max(sc['A']['points'], sc['B']['points']) >= POINTS_TO_WIN:
            self.done, self.decided_by = True, 'score'
        elif how == 'time_cap' or self.k*DT >= CAP_S-1e-9 or len(self.results) >= MAX_RALLIES:
            self.done, self.decided_by = True, self.cap_decision()
        else:
            self.awaiting_review = True

    def cap_decision(self):
        sc = self.scores(); a, b = sc['A']['points'], sc['B']['points']
        if a != b: return 'cap_points'
        if any(r['how'] == 'goal' for r in self.results): return 'cap_last_goal'
        if self.half_steps['A'] != self.half_steps['B']: return 'cap_territory'
        return 'cap_first_possession'

    def ready(self):
        if not self.awaiting_review: raise ValueError('not in review')
        self.awaiting_review = False; self.point_k = 0
        nxt = self.results[-1]['next']
        self.restart(None if nxt == 'faceoff' else nxt, self.results[-1]['how'])

    # ---------------------------------------------------------------- result
    def scores(self):
        return {s: dict(points=sum(r['how'] == 'goal' and r['winner'] == s for r in self.results),
                        fouls=sum(r['guilty'] == s for r in self.results)) for s in ('A', 'B')}

    def winner(self):
        """'A' or 'B' once the game is over (never a draw), else None."""
        if not self.done: return None
        sc = self.scores(); a, b = sc['A']['points'], sc['B']['points']
        if a != b: return 'A' if a > b else 'B'
        goals = [r for r in self.results if r['how'] == 'goal']
        if goals: return goals[-1]['winner']                                    # the winner of the last point
        if self.half_steps['A'] != self.half_steps['B']:                        # 0:0: less time with the puck in
            return 'A' if self.half_steps['A'] < self.half_steps['B'] else 'B'  # one's own half (more pressure)
        return OTHER[self.first]                    # identical: the player who did not have the first possession

    def digest(self):
        state = np.empty(mujoco.mj_stateSize(self.model, mujoco.mjtState.mjSTATE_INTEGRATION))
        mujoco.mj_getState(self.model, self.data, state, mujoco.mjtState.mjSTATE_INTEGRATION)
        h = hashlib.sha256(state.tobytes())
        h.update(json.dumps(dict(k=self.k, point_k=self.point_k, results=self.results,
                                 commands=self.commands, targets={s: t.tolist() for s, t in self.targets.items()},
                                 cart_velocity={s: t.tolist() for s, t in self.cart_velocity.items()},
                                 path={s: [g.p.tolist(), g.v.tolist(), g.a.tolist()] for s, g in self.path.items()},
                                 height_violations=self.height_violations,
                                 qtarget={s: q.tolist() for s, q in self.qtarget.items()}, side=self.side,
                                 servo={s: [j.q.tolist(), j.qd.tolist(), j.qdd.tolist()] for s, j in self.jl.items()},
                                 side_steps=self.side_steps, line_steps=self.line_steps, dead_steps=self.dead_steps,
                                 half_steps=self.half_steps, last_touch=self.last_touch, rng=self.rng.bit_generator.state,
                                 script=dict(until=self.script['until'], target=self.script['target'],
                                             hist=[x.tolist() for x in self.script['hist']]),
                                 contacts=self.contact_steps, rails=self.rail_contact_steps, done=self.done,
                                 decided_by=self.decided_by, awaiting_review=self.awaiting_review), sort_keys=True).encode())
        return h.hexdigest()
