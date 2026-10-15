"""CPU MuJoCo quadrotor gate race. No route planner lives in the flight controller.

Axes: x forward, y left, z up; body-to-world quaternion is wxyz. SI units.
The vehicle and course are original procedural assets, not a Crazyflie digital twin.

Format (FAI Sporting Code Section 4, Volume F9 Drone Sports, 2026 edition, chapter C, class F9U; see
sports/drone/sport.toml [rulebook] / [[deviations]]):
  * Race     one flight of the open course (C.2 "an open loop to be flown once"), every gate in order;
  * Heat     the drones of all seats fly the same course at the same time, each in its own collision layer
             (a separate MuJoCo world: no drone-drone contact, and no seat sees the other's drone);
  * Event    single player: ROUNDS qualifying rounds (C.6.3, at least 3), ranked by the best result (C.6.3 a,
             Option 1: finishers by time, then by gates passed); duel: successive final races until one seat has
             won HEATS_TO_WIN heats (C.6.5.3), at most MAX_HEATS, a tied heat counts for nobody (C.6.5: a tie is
             split by an additional race).
The course (gates, panels) is fixed by the seed for the whole event (one circuit, C.2); the wind and the moving
shutters' periods and phases are seeded per round, so a round cannot be replayed open-loop from a previous one.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass

import mujoco
import numpy as np

VERSION = "drone-race/4.0"
DT = 0.01
SUBSTEPS = 5
MASS = 0.35
ARM = 0.105
YAW_COEFFICIENT = 0.014
MOTOR_MAX = 2.5
MOTOR_TAU = 0.025
RADIUS = 0.18
LEVELS = {
    1: dict(count=6, lateral=1.1, height_max=2.6, yaw=0.35, opening=1.4, limit=35.0, wind=0.0),
    2: dict(count=8, lateral=1.8, height_max=3.0, yaw=0.65, opening=1.15, limit=60.0, wind=1.6),
    3: dict(count=10, lateral=2.5, height_max=3.35, yaw=0.9, opening=0.85, limit=120.0, wind=3.2),
}
SHUTTERS = {
    1: dict(gates=[], amplitude=0.0, width=0.0, periods=[1.0, 1.0]),
    2: dict(gates=[3, 6], amplitude=1.15, width=0.75, periods=[5.2, 6.4]),
    3: dict(gates=[2, 4, 7, 9], amplitude=0.85, width=0.60, periods=[3.6, 5.0]),
}
SHUTTER_OFFSET = -0.12
SHUTTER_THICKNESS = 0.06
MAX_TIME = max(c["limit"] for c in LEVELS.values())
BOUNDS = dict(x=[-2.0, 52.0], y=[-5.8, 5.8], z=[0.15, 4.8])
TAG_PITCH = 0.12
TAG_OFFSET = 0.13
PANEL_WIDTH = 1.0
# L3 panels stand on the floor and reach above the arena ceiling (z = 4.8 m): no route over or under them.
PANEL_BOTTOM = 0.0
PANEL_HEIGHT = 5.0
ROUNDS = 3                 # single player: qualifying rounds (F9U C.6.3: at least 3)
HEATS_TO_WIN = 2           # duel: successive final races, won by the first seat with two heat wins (F9U C.6.5.3)
MAX_HEATS = 5              # duel: heat limit (a tied heat counts for nobody); then the stated tie-break
DUEL_LEVEL = 2             # the duel's course level
STATE_SPEC = mujoco.mjtState.mjSTATE_INTEGRATION


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def validate_action(action):
    if not isinstance(action, dict) or set(action) != {"velocity", "yaw_rate", "duration"}:
        raise ValueError("action requires only velocity [vx, vy, vz], yaw_rate and duration")
    v = action["velocity"]
    values = [*(v if isinstance(v, list) else []), action["yaw_rate"], action["duration"]]
    if len(values) != 5 or any(type(x) not in (int, float) or not math.isfinite(x) for x in values):
        raise ValueError("action values must be finite numbers")
    if np.linalg.norm(v[:2]) > 4.0 or abs(v[2]) > 2.0:
        raise ValueError("horizontal speed <= 4 m/s; vertical speed <= 2 m/s")
    if abs(action["yaw_rate"]) > 1.5 or not 0.05 <= action["duration"] <= 1.0:
        raise ValueError("yaw rate <= 1.5 rad/s; duration in [0.05, 1.0] game seconds")
    return {
        "velocity": [float(x) for x in v],
        "yaw_rate": float(action["yaw_rate"]),
        "duration": float(action["duration"]),
    }


@dataclass(frozen=True)
class Gate:
    center: np.ndarray
    yaw: float
    width: float = 1.6
    height: float = 1.6
    number: int = 1

    def local(self, point):
        d = np.asarray(point) - self.center
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        return np.array([c * d[0] + s * d[1], -s * d[0] + c * d[1], d[2]])

    def crossed(self, before, after):
        """The fraction of the segment before -> after at which the vehicle centre crosses the gate plane forwards
        inside the clearance margin, else None."""
        a, b = self.local(before), self.local(after)
        if not a[0] < 0 <= b[0]:
            return None
        fraction = -a[0] / (b[0] - a[0])
        p = a + fraction * (b - a)
        if abs(p[1]) <= self.width / 2 - RADIUS and abs(p[2]) <= self.height / 2 - RADIUS:
            return float(fraction)
        return None


@dataclass(frozen=True)
class Panel:
    center: np.ndarray
    width: float = PANEL_WIDTH
    height: float = PANEL_HEIGHT


def course(seed, level):
    if type(seed) is not int or not 0 <= seed < 2**32 or type(level) is not int or level not in LEVELS:
        raise ValueError("seed must be uint32; level must be 1, 2 or 3")
    rng = np.random.Generator(np.random.PCG64(seed))
    cfg = LEVELS[level]
    gates = []
    x = 3.7 + rng.uniform(-0.15, 0.15)
    for i in range(cfg["count"]):
        if i:
            x += rng.uniform(3.5, 4.3)
        y = (1 if i % 2 else -1) * cfg["lateral"] + rng.uniform(-0.25, 0.25)
        z = rng.uniform(1.0, cfg["height_max"])
        yaw = rng.uniform(-cfg["yaw"], cfg["yaw"])
        if i == 0:
            y, z, yaw = rng.uniform(-0.15, 0.15), 1.45 + rng.uniform(-0.07, 0.07), rng.uniform(-0.1, 0.1)
        if i:
            bearing = math.atan2(y - gates[-1].center[1], x - gates[-1].center[0])
            yaw = float(np.clip(bearing + rng.uniform(-0.16, 0.16), -cfg["yaw"], cfg["yaw"]))
        gates.append(Gate(np.array([x, y, z]), float(yaw), cfg["opening"], cfg["opening"], i + 1))
    return gates


def panels_for(gates, seed, level):
    if level != 3:
        return []
    rng = np.random.Generator(np.random.PCG64(seed ^ 0x61D09))
    panels = []
    for i in (2, 6):
        point = 0.5 * (gates[i].center + gates[i + 1].center)
        point[0] += rng.uniform(-0.15, 0.15)
        point[1] += rng.uniform(-0.12, 0.12)
        point[2] = PANEL_BOTTOM + PANEL_HEIGHT / 2
        panels.append(Panel(point))
    return panels


@dataclass(frozen=True)
class Shutter:
    gate: Gate
    amplitude: float
    width: float
    period: float
    phase: float

    def position(self, t):
        yaw = self.gate.yaw
        normal = np.array([math.cos(yaw), math.sin(yaw), 0.0])
        lateral = np.array([-math.sin(yaw), math.cos(yaw), 0.0])
        return (
            self.gate.center
            + SHUTTER_OFFSET * normal
            + self.amplitude * math.sin(2 * math.pi * t / self.period + self.phase) * lateral
        )


def shutters_for(gates, seed, level, round_no=1):
    cfg = SHUTTERS[level]
    rng = np.random.Generator(np.random.PCG64([seed, round_no, 0x53177]))
    return [
        Shutter(
            gates[n - 1],
            cfg["amplitude"],
            cfg["width"],
            float(rng.uniform(*cfg["periods"])),
            float(rng.uniform(0, 2 * math.pi)),
        )
        for n in cfg["gates"]
    ]


def scene_xml(gates, panels=(), shutters=()):
    parts = []
    for i, g in enumerate(gates):
        # Static physical IDs; no privileged next-target highlighting.
        geoms = []
        for y, z, sy, sz in [
            (-g.width / 2 - 0.045, 0, 0.045, g.height / 2 + 0.09),
            (g.width / 2 + 0.045, 0, 0.045, g.height / 2 + 0.09),
            (0, g.height / 2 + 0.045, g.width / 2, 0.045),
            (0, -g.height / 2 - 0.045, g.width / 2, 0.045),
        ]:
            geoms.append(
                f'<geom name="gate_{i}_{len(geoms)}" type="box" pos="0 {y} {z}" '
                f'size="0.045 {sy} {sz}" rgba="0.95 0.16 0.025 1"/>'
            )
        # Eight cells encode the gate number, then its four-bit complement, MSB first.
        # The black plate is above the frame so the orange silhouette remains intact.
        tag_z = g.height / 2 + 0.09 + TAG_OFFSET
        geoms.append(f'<geom type="box" pos="-0.057 0 {tag_z}" size="0.012 0.52 0.07" rgba="0.015 0.015 0.015 1"/>')
        bits = [(g.number >> n) & 1 for n in (3, 2, 1, 0)]
        bits += [1 - b for b in bits]
        for j, bit in enumerate(bits):
            if bit:
                geoms.append(
                    f'<geom type="box" pos="-0.073 {-(j - 3.5) * TAG_PITCH} {tag_z}" '
                    'size="0.006 0.045 0.045" rgba="0.95 0.95 0.95 1" contype="0" conaffinity="0"/>'
                )
        c = " ".join(map(str, g.center))
        parts.append(f'<body pos="{c}" euler="0 0 {g.yaw}">{"".join(geoms)}</body>')
    for i, panel in enumerate(panels):
        c = " ".join(map(str, panel.center))
        parts.append(
            f'<geom name="panel_{i}" type="box" pos="{c}" '
            f'size="0.04 {panel.width / 2} {panel.height / 2}" rgba="0.025 0.2 0.95 1"/>'
        )
    for i, shutter in enumerate(shutters):
        c = " ".join(map(str, shutter.position(0.0)))
        parts.append(
            f'<body name="shutter_body_{i}" mocap="true" pos="{c}" euler="0 0 {shutter.gate.yaw}">'
            f'<geom name="shutter_{shutter.gate.number}" type="box" '
            f'size="{SHUTTER_THICKNESS / 2} {shutter.width / 2} {shutter.gate.height / 2 + 0.09}" '
            'rgba="0.95 0.025 0.75 1"/></body>'
        )
    rotor_positions = [(ARM, ARM), (-ARM, ARM), (-ARM, -ARM), (ARM, -ARM)]
    sites, actuators, arms = [], [], []
    for i, (x, y) in enumerate(rotor_positions):
        sites.append(f'<site name="rotor{i}" pos="{x} {y} 0" size="0.015" rgba="0.2 0.2 0.2 1"/>')
        actuators.append(
            f'<general name="motor{i}" site="rotor{i}" gear="0 0 1 0 0 '
            f'{YAW_COEFFICIENT * (-1 if i % 2 else 1)}" ctrllimited="true" ctrlrange="0 {MOTOR_MAX}"/>'
        )
        arms.append(f'<geom type="capsule" fromto="0 0 0 {x} {y} 0" size="0.018" rgba="0.15 0.22 0.3 1"/>')
        arms.append(
            f'<geom type="cylinder" pos="{x} {y} 0.015" size="0.035 0.006" '
            'rgba="0.05 0.07 0.1 1" contype="0" conaffinity="0"/>'
        )
    return f'''<mujoco model="original_quadrotor_gate_race">
  <compiler angle="radian"/>
  <option timestep="0.002" gravity="0 0 -9.81" integrator="RK4" solver="Newton" iterations="20" tolerance="1e-10"/>
  <visual><global offwidth="640" offheight="480"/><map znear="0.01" zfar="80"/><quality shadowsize="1024"/></visual>
  <default><geom friction="0.7 0.01 0.001" solref="0.01 1"/></default>
  <asset><texture name="floor_tex" type="2d" builtin="checker" rgb1="0.20 0.27 0.3" rgb2="0.28 0.35 0.38" width="128" height="128"/>
  <material name="floor_mat" texture="floor_tex" texrepeat="30 10" reflectance="0.05"/></asset>
  <worldbody><light pos="8 0 12" dir="0 0 -1" diffuse="0.8 0.8 0.8"/>
    <geom name="floor" type="plane" size="60 8 0.1" material="floor_mat"/>
    <geom name="left_wall" type="box" pos="25 6 2.5" size="30 0.05 2.5" rgba="0.55 0.65 0.7 1"/>
    <geom name="right_wall" type="box" pos="25 -6 2.5" size="30 0.05 2.5" rgba="0.55 0.65 0.7 1"/>
    {"".join(parts)}
    <camera name="broadcast" pos="8 -15 10" xyaxes="1 0 0 0 0.55 0.83" fovy="65"/>
    <body name="drone" pos="0 0 1.45"><freejoint/>
      <inertial pos="0 0 0" mass="{MASS}" diaginertia="0.0015 0.0015 0.0025"/>
      <geom name="hull" type="box" size="0.055 0.04 0.025" rgba="0.08 0.3 0.85 1"/>
      {"".join(arms)}{"".join(sites)}
      <camera name="fpv" pos="0.12 0 0.035" xyaxes="0 -1 0 0 0 1" fovy="70"/>
    </body>
  </worldbody><actuator>{"".join(actuators)}</actuator>
</mujoco>'''


class Race:
    """One drone's flight of one round (a heat holds one Race per seat, each in its own MuJoCo world)."""

    def __init__(self, seed=101, level=1, round_no=1):
        if type(round_no) is not int or not 1 <= round_no <= max(ROUNDS, MAX_HEATS):
            raise ValueError("invalid round number")
        self.seed, self.level, self.round_no = seed, level, round_no
        self.gates = course(seed, level)
        self.cfg = LEVELS[level]
        self.panels = panels_for(self.gates, seed, level)
        self.wind_phase = np.random.Generator(np.random.PCG64([seed, round_no, 0xF17E])).uniform(0, 2 * math.pi, 3)
        self.shutters = shutters_for(self.gates, seed, level, round_no)
        self.xml = scene_xml(self.gates, self.panels, self.shutters)
        self.model = mujoco.MjModel.from_xml_string(self.xml)
        self.data = mujoco.MjData(self.model)
        self.body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "drone")
        self.camera = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "fpv")
        self.fingerprint = hashlib.sha256(
            (
                VERSION
                + mujoco.__version__
                + self.xml
                + canonical(
                    dict(
                        configuration=self.cfg,
                        round=round_no,
                        wind_phase=self.wind_phase.tolist(),
                        bounds=BOUNDS,
                        shutters=[
                            dict(
                                gate=h.gate.number, amplitude=h.amplitude, width=h.width, period=h.period, phase=h.phase
                            )
                            for h in self.shutters
                        ],
                    )
                )
            ).encode()
        ).hexdigest()
        self.motor = np.full(4, MASS * 9.81 / 4)
        self.mix = np.array(
            [
                [1] * 4,
                [ARM, ARM, -ARM, -ARM],
                [-ARM, ARM, ARM, -ARM],
                [YAW_COEFFICIENT, -YAW_COEFFICIENT, YAW_COEFFICIENT, -YAW_COEFFICIENT],
            ]
        )
        self.mix_inverse = np.linalg.inv(self.mix)
        self.command = {"velocity": [0.0] * 3, "yaw_rate": 0.0, "duration": 0.05}
        self.command_until = 0
        self.yaw_target = 0.0
        self.tick = self.sim_steps = self.passed = 0
        self.done = False
        self.reason = "running"
        self.gate_times = []
        self.events = []
        self.actions = []
        self.data.ctrl[:] = self.motor
        self._move_shutters(0.0)
        mujoco.mj_forward(self.model, self.data)

    def act(self, action):
        if self.done:
            raise ValueError("race is over")
        action = validate_action(action)
        self.command = action
        self.command_until = self.tick + max(1, math.ceil(action["duration"] / DT))
        self.actions.append({"tick": self.tick, "action": action})

    def _control(self):
        active = self.tick < self.command_until
        velocity = np.array(self.command["velocity"] if active else [0.0] * 3)
        self.yaw_target += (self.command["yaw_rate"] if active else 0.0) * DT
        acceleration = np.clip(2.6 * (velocity - self.data.qvel[:3]), [-4, -4, -3], [4, 4, 3])
        horizontal = np.linalg.norm(acceleration[:2])
        if horizontal > 4:
            acceleration[:2] *= 4 / horizontal
        force = MASS * (acceleration + [0, 0, 9.81])
        z = force / np.linalg.norm(force)
        heading = np.array([math.cos(self.yaw_target), math.sin(self.yaw_target), 0])
        y = np.cross(z, heading)
        y /= np.linalg.norm(y)
        desired = np.column_stack([np.cross(y, z), y, z])
        rotation = self.data.xmat[self.body].reshape(3, 3)
        error_world = sum(np.cross(rotation[:, i], desired[:, i]) for i in range(3)) / 2
        torque = 0.09 * (rotation.T @ error_world) - 0.018 * self.data.qvel[3:6]
        thrust = max(0.0, float(force @ rotation[:, 2]))
        return np.clip(self.mix_inverse @ np.r_[thrust, torque], 0, MOTOR_MAX)

    def wind(self, t):
        p = self.wind_phase
        return self.cfg["wind"] * np.array(
            [
                0.2 * math.sin(0.7 * t + p[0]),
                0.65 * math.sin(0.43 * t + p[1]) + 0.35 * math.cos(0.91 * t + p[2]),
                0.12 * math.sin(0.6 * t + p[2]),
            ]
        )

    def _finish(self, reason):
        self.done, self.reason = True, reason
        self.events.append({"event": reason, "time": float(self.data.time), "tick": self.tick})

    def _move_shutters(self, t):
        # Prescribed kinematic rail motion. A contact is terminal, not an impact-response task.
        for i, shutter in enumerate(self.shutters):
            self.data.mocap_pos[i] = shutter.position(t)

    def step(self):
        if self.done:
            raise ValueError("race is over")
        target = self._control()
        for _ in range(SUBSTEPS):
            before = self.data.qpos[:3].copy()
            t_before = float(self.data.time)
            self.motor += (target - self.motor) * (1 - math.exp(-self.model.opt.timestep / MOTOR_TAU))
            self.data.ctrl[:] = self.motor
            # Aerodynamic drag is a documented linear force, not a position constraint.
            self.data.xfrc_applied[self.body, :3] = -0.06 * (self.data.qvel[:3] - self.wind(t_before))
            self._move_shutters(t_before)
            mujoco.mj_step(self.model, self.data)
            self._move_shutters(float(self.data.time))
            mujoco.mj_forward(self.model, self.data)
            self.sim_steps += 1
            if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():
                raise RuntimeError("non-finite physics state")
            if any(
                self.model.geom_bodyid[c.geom1] == self.body or self.model.geom_bodyid[c.geom2] == self.body
                for c in self.data.contact[: self.data.ncon]
            ):
                self._finish("collision")
                break
            pos = self.data.qpos[:3]
            if not all(BOUNDS[axis][0] < pos[i] < BOUNDS[axis][1] for i, axis in enumerate(("x", "y", "z"))):
                self._finish("out_of_bounds")
                break
            fraction = self.gates[self.passed].crossed(before, pos)
            if fraction is not None:
                # Electronic timing: the instant the vehicle centre crosses the gate plane (interpolated in the step).
                crossing = t_before + fraction * (float(self.data.time) - t_before)
                self.passed += 1
                self.gate_times.append(crossing)
                self.events.append({"event": "gate", "gate": self.passed, "time": crossing, "tick": self.tick})
                if self.passed == len(self.gates):
                    self._finish("finished")
                    break
        self.tick += 1
        # mj_step can leave derived poses at the previous integration evaluation.
        mujoco.mj_forward(self.model, self.data)
        if not self.done and self.tick >= round(self.cfg["limit"] / DT):
            self._finish("time_limit")

    def state_hash(self):
        state = np.empty(mujoco.mj_stateSize(self.model, STATE_SPEC))
        mujoco.mj_getState(self.model, self.data, state, STATE_SPEC)
        judge = dict(
            tick=self.tick,
            round=self.round_no,
            configuration=self.cfg,
            sim_steps=self.sim_steps,
            passed=self.passed,
            done=self.done,
            reason=self.reason,
            gate_times=self.gate_times,
            command=self.command,
            command_until=self.command_until,
            yaw_target=self.yaw_target,
            events=self.events,
        )
        return hashlib.sha256(
            state.astype("<f8").tobytes() + self.motor.astype("<f8").tobytes() + canonical(judge).encode()
        ).hexdigest()

    def result(self):
        """The referee's result of this flight (F9U C.6.1: unfinished laps do not count; here gates are sectors)."""
        finished = self.reason == "finished"
        return dict(
            round=self.round_no,
            finished=finished,
            finish_time_s=self.gate_times[-1] if finished else None,
            gates_passed=self.passed,
            gates_total=len(self.gates),
            last_gate_time_s=self.gate_times[-1] if self.gate_times else None,
            gate_times=list(self.gate_times),
            reason=self.reason,
            ticks=self.tick,
        )

    def status(self):
        return dict(
            tick=self.tick,
            time_s=float(self.data.time),
            round=self.round_no,
            gates_passed=self.passed,
            gates_total=len(self.gates),
            done=self.done,
            reason=self.reason,
            finished=self.reason == "finished",
            time_limit_s=self.cfg["limit"],
            gate_times=list(self.gate_times),
            robot=dict(
                position=self.data.qpos[:3].tolist(),
                quaternion_wxyz=self.data.qpos[3:7].tolist(),
                velocity=self.data.qvel[:3].tolist(),
                angular_velocity_body=self.data.qvel[3:6].tolist(),
            ),
            command=dict(self.command),
            command_until_tick=self.command_until,
        )

    def spec(self):
        """Public geometry and limits. No seed: the course, wind and shutter phases follow from it."""
        return dict(
            protocol="eai-standard-realtime/1.0",
            version=VERSION,
            configuration=dict(level=self.level),
            timing=dict(
                control_dt_s=DT,
                physics_dt_s=float(self.model.opt.timestep),
                max_race_s=self.cfg["limit"],
            ),
            arena_bounds_m=BOUNDS,
            wind=dict(
                max_horizontal_speed_m_s=self.cfg["wind"] * 1.02,
                max_vertical_speed_m_s=self.cfg["wind"] * 0.12,
                description="smooth seeded wind through relative-air-velocity linear drag; new phases every round",
            ),
            moving_obstacles=dict(
                count=len(self.shutters),
                gate_numbers=SHUTTERS[self.level]["gates"],
                color="magenta",
                width_m=SHUTTERS[self.level]["width"],
                height_m=self.gates[0].height + 0.18,
                thickness_m=SHUTTER_THICKNESS,
                gate_normal_offset_m=SHUTTER_OFFSET,
                amplitude_m=SHUTTERS[self.level]["amplitude"],
                period_range_s=SHUTTERS[self.level]["periods"],
                motion="sinusoidal translation along each gate lateral axis; seeded period and phase, new every round",
                observation="infer current position and motion from timestamped RGB; no phase or state API",
            ),
            obstacles=dict(
                count=len(self.panels),
                color="blue",
                width_m=PANEL_WIDTH,
                height_m=PANEL_HEIGHT,
                bottom_z_m=PANEL_BOTTOM,
                thickness_m=0.08,
                orientation="vertical panels parallel to world yz, floor to above the ceiling; infer x/y from RGB",
            ),
            robot=dict(
                mass_kg=MASS,
                bounding_radius_m=RADIUS,
                max_motor_thrust_N=MOTOR_MAX,
                linear_drag_N_per_m_s=0.06,
                start_position_m=[0.0, 0.0, 1.45],
                velocity_frame="world: x forward, y left, z up",
            ),
            camera=dict(
                name="fpv",
                width=320,
                height=240,
                fovy_degrees=70,
                body_position=[0.12, 0, 0.035],
                camera_to_body_rotation=[[0, 0, -1], [-1, 0, 0], [0, 1, 0]],
                convention="MuJoCo/OpenGL camera: x right, y up, looks along -z",
            ),
            gates=dict(
                count=len(self.gates),
                opening_width_m=self.gates[0].width,
                opening_height_m=self.gates[0].height,
                border_m=0.09,
                frame_depth_m=0.09,
                centre_height_range_m=[1.0, self.cfg["height_max"]],
                max_abs_yaw_rad=self.cfg["yaw"],
                forward_spacing_range_m=[3.5, 4.3],
                tag=dict(
                    cell_pitch_m=TAG_PITCH,
                    cell_size_m=0.09,
                    centre_above_outer_frame_m=TAG_OFFSET,
                    front_offset_m=-0.034,
                    encoding="8 cells left-to-right from front: 4-bit gate number MSB first, then complement",
                ),
                crossing="in order, front to back, vehicle centre with 0.18 m edge clearance",
            ),
            action=dict(
                fields=["velocity", "yaw_rate", "duration"],
                horizontal_speed_max=4,
                vertical_speed_max=2,
                yaw_rate_max=1.5,
                duration_range=[0.05, 1.0],
                expiry="zero target velocity and hold heading; no course navigation",
            ),
        )


def placing_key(result):
    """Placing of one flight (F9U C.6.4 / C.6.5): finishers by time; then more gates passed (the sectors of the one
    lap), at equal gates the earlier time at the last gate; no gate passed comes last. Lower is better."""
    if result["finished"]:
        return (0, result["finish_time_s"], 0.0)
    g = result["gates_passed"]
    return (1, -g, result["last_gate_time_s"] if g else math.inf)


class Heat:
    """One round (single player) or heat (duel): one Race per seat on the same course with the same wind and shutters,
    stepped together on one clock. Checkpoints follow one rule, so the service and the replay compare them directly."""

    def __init__(self, seed, level, number, seats):
        self.number = number
        self.seats = tuple(seats)
        self.races = {s: Race(seed, level, number) for s in self.seats}
        self.k = 0
        self.done = False
        self.end = "running"
        self.checkpoints = {s: {"0": r.state_hash()} for s, r in self.races.items()}

    def act(self, seat, action):
        self.races[seat].act(action)

    def step(self):
        if self.done:
            raise ValueError("heat is over")
        for s in self.seats:
            r = self.races[s]
            if not r.done:
                r.step()
                if r.tick % 10 == 0 or r.done:
                    self.checkpoints[s][str(r.tick)] = r.state_hash()
        self.k += 1
        if all(r.done for r in self.races.values()):
            self.done, self.end = True, "all_done"
        elif len(self.seats) == 2 and self._decided():
            self.done, self.end = True, "decided"
            for r in self.races.values():
                if not r.done:
                    r._finish("heat_decided")      # still flying when the placing was decided

    def _decided(self):
        """Two seats: the placing can no longer change (one drone is down or home and the other cannot pass it, or the
        other is already ahead for good). The heat then ends at once."""
        for x, y in (self.seats, self.seats[::-1]):
            rx, ry = self.races[x], self.races[y]
            if not rx.done or ry.done:
                continue
            if rx.reason == "finished":
                if float(ry.data.time) > rx.gate_times[-1]:
                    return True            # y can no longer finish before x
            else:
                g = rx.passed
                if ry.passed > g or (g and ry.passed == g and ry.gate_times[g - 1] < rx.gate_times[g - 1]):
                    return True            # y is ahead of the stopped x and can only gain
        return False

    def results(self):
        return {s: r.result() for s, r in self.races.items()}

    def winner(self):
        """The heat's winner (duel), None when tied (equal placing keys)."""
        if len(self.seats) != 2:
            return None
        a, b = self.seats
        ka, kb = placing_key(self.races[a].result()), placing_key(self.races[b].result())
        return a if ka < kb else b if kb < ka else None

    def summary(self):
        return dict(number=self.number, ticks=self.k, end=self.end, winner=self.winner(), results=self.results())


class Event:
    """The whole contest: single player, ROUNDS qualifying rounds; duel, heats until HEATS_TO_WIN wins (MAX_HEATS)."""

    def __init__(self, seed, level, seats=("A",)):
        self.seed, self.level, self.seats = seed, level, tuple(seats)
        if len(self.seats) not in (1, 2):
            raise ValueError("one or two seats")
        self.kind = "single" if len(self.seats) == 1 else "duel"
        self.heat = Heat(seed, level, 1, self.seats)
        self.fingerprint = hashlib.sha256(
            canonical(
                dict(
                    version=VERSION,
                    kind=self.kind,
                    rounds=ROUNDS,
                    heats_to_win=HEATS_TO_WIN,
                    max_heats=MAX_HEATS,
                    course=self.heat.races[self.seats[0]].fingerprint,
                )
            ).encode()
        ).hexdigest()
        self.closed = []            # heat summaries
        self.closed_heats = []      # the Heat objects (actions and checkpoints, for the record)
        self.over = False

    def close_heat(self):
        """After the current heat is done: record it and either end the event or prepare the next heat."""
        if not self.heat.done or self.over:
            raise ValueError("no finished heat to close")
        self.closed.append(self.heat.summary())
        self.closed_heats.append(self.heat)
        if self.kind == "single":
            self.over = len(self.closed) >= ROUNDS
        else:
            wins = self.wins()
            self.over = max(wins.values()) >= HEATS_TO_WIN or len(self.closed) >= MAX_HEATS
        if not self.over:
            self.heat = Heat(self.seed, self.level, len(self.closed) + 1, self.seats)
        return self.closed[-1]

    def wins(self):
        return {s: sum(h["winner"] == s for h in self.closed) for s in self.seats}

    def flights(self, seat):
        return [h["results"][seat] for h in self.closed]

    def standing(self, seat):
        """A seat's results, best first (F9U C.6.3 a: the best result ranks; the 2nd and 3rd split ties)."""
        keys = sorted(placing_key(r) for r in self.flights(seat))
        finished = sorted(r["finish_time_s"] for r in self.flights(seat) if r["finished"])
        best = min(self.flights(seat), key=placing_key) if self.closed else None
        return dict(
            flights=len(self.closed),
            finished=len(finished),
            times_s=finished,
            best_time_s=finished[0] if finished else None,
            second_time_s=finished[1] if len(finished) > 1 else None,
            third_time_s=finished[2] if len(finished) > 2 else None,
            best_gates=best["gates_passed"] if best else 0,
            best_key=[[k[0], k[1], k[2] if math.isfinite(k[2]) else None] for k in keys],
        )

    def duel_winner(self):
        """(winner, how), never a draw. Two heat wins (F9U C.6.5.3; a tied heat is re-flown, C.6.5); after MAX_HEATS
        without, more heat wins; then the better best heat result, then the second best, ... (the F9U C.6.3 a
        tie-break order); then the winner of the last heat that had one; last resort, if no heat had a winner, a
        lot drawn from the seed."""
        if self.kind != "duel" or not self.over:
            return None, None
        a, b = self.seats
        w = self.wins()
        if w[a] != w[b]:
            return (a if w[a] > w[b] else b), ("heat_wins" if max(w.values()) >= HEATS_TO_WIN else "most_heat_wins")
        ka = sorted(placing_key(r) for r in self.flights(a))
        kb = sorted(placing_key(r) for r in self.flights(b))
        for x, y in zip(ka, kb):
            if x != y:
                return (a if x < y else b), "best_results"
        for h in reversed(self.closed):
            if h["winner"]:
                return h["winner"], "last_heat_winner"
        lot = int(np.random.Generator(np.random.PCG64([self.seed, 0x107])).integers(2))
        return self.seats[lot], "lot"

    def summary(self):
        out = dict(
            version=VERSION,
            kind=self.kind,
            level=self.level,
            seats=list(self.seats),
            over=self.over,
            heats=list(self.closed),
            standings={s: self.standing(s) for s in self.seats},
        )
        if self.kind == "duel":
            winner, how = self.duel_winner()
            out.update(wins=self.wins(), winner=winner, decided_by=how)
        return out
