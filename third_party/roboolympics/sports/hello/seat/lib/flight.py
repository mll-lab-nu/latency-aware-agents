"""hello's launcher flight model (the low level's toolkit, seat/toolkit.toml tool `flight_model`; the seat has it as
/opt/ro/lib/flight.py). The same physics as the simulator: a ball of radius 5 cm released at (0, 0, 1) m with the
throw's velocity, gravity 9.81 m/s^2, no drag, semi-implicit Euler steps of 2 ms (MuJoCo's default integrator).

Forward only: given a throw, where the ball first touches the ground. It does not aim: finding the throw that lands
at a requested point (the skill `aim`) is what a low-level player develops. Standard library only, deterministic.

    land(yaw_deg, pitch_deg, speed_m_s) -> {"x", "y", "t"}       first ground contact (ball centre), seconds after release
    path(yaw_deg, pitch_deg, speed_m_s) -> [(t, x, y, z), ...]   the flight, every 2 ms, until that contact
    within_limits(yaw_deg, pitch_deg, speed_m_s) -> bool         the launcher's ranges (the throw schema)
"""

import math

RELEASE = (0.0, 0.0, 1.0)
BALL_RADIUS = 0.05
GRAVITY = 9.81
DT = 0.002
LIMITS = {"yaw_deg": (-20.0, 20.0), "pitch_deg": (0.0, 60.0), "speed_m_s": (1.0, 12.0)}
MAX_T = 10.0


def velocity(yaw_deg, pitch_deg, speed):
    y, p = math.radians(yaw_deg), math.radians(pitch_deg)
    return speed * math.cos(p) * math.cos(y), speed * math.cos(p) * math.sin(y), speed * math.sin(p)


def within_limits(yaw_deg, pitch_deg, speed):
    return all(lo <= v <= hi for v, (lo, hi) in zip((yaw_deg, pitch_deg, speed), LIMITS.values()))


def path(yaw_deg, pitch_deg, speed):
    x, y, z = RELEASE
    vx, vy, vz = velocity(yaw_deg, pitch_deg, speed)
    out, t = [(0.0, x, y, z)], 0.0
    while z > BALL_RADIUS and t < MAX_T:
        vz -= GRAVITY * DT  # semi-implicit Euler: the velocity first, then the position
        x, y, z, t = x + vx * DT, y + vy * DT, z + vz * DT, t + DT
        out.append((t, x, y, z))
    return out


def land(yaw_deg, pitch_deg, speed):
    pts = path(yaw_deg, pitch_deg, speed)
    (t0, x0, y0, z0), (t1, x1, y1, z1) = pts[-2], pts[-1]
    f = (z0 - BALL_RADIUS) / (z0 - z1) if z0 != z1 else 1.0
    return {"x": x0 + f * (x1 - x0), "y": y0 + f * (y1 - y0), "t": t0 + f * (t1 - t0)}
