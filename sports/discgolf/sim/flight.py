"""Disc flight: a deterministic aerodynamic model of a spinning golf disc (pure Python floats, fixed step).

The model is the one of Hummel (2003, "Frisbee flight simulation and throw biomechanics", UC Davis MSc thesis) as
implemented in FrisPy (T. McClintock, MIT licence, https://github.com/tmcclintock/FrisPy): lift and drag from the angle
of attack, a pitching moment that grows with the angle of attack, and spin. This file is an independent
re-implementation (no FrisPy code) with two simplifications, both standard for spin-stabilised discs:
  * gyroscopic approximation: the disc's angular momentum is its spin, L = I_zz w_s n (n = the disc's normal), so an
    aerodynamic torque tau turns the normal at  dn/dt = tau_perp / (I_zz w_s)  (precession; the fast nutation, which
    the air damps within a few tenths of a second, is left out);
  * the spin decays under a small aerodynamic spin-down torque.
Forces, on the velocity relative to the air  u = v - w  (w = the wind, uniform):
  alpha = atan2(-u.n, |u - (u.n) n|)                       angle of attack (> 0: the air hits the underside)
  L = q A (CL0 + CLa alpha)   perpendicular to u, on the side of n
  D = q A (CD0 + CDa (alpha - ALPHA0)^2)   along -u          q = rho |u|^2 / 2
  M = q A d (CM0 + CMa alpha)   pitching moment, nose up > 0, about -(n x u_hat_in_plane)
  Ts = q A d CS (w_s d / (2 |u|))  spin-down torque about n
A clockwise spin (seen from above; a right-hand backhand) with a nose-up moment rolls the disc to its left (the
"fade"); a nose-down moment at high speed (CM0 < 0) rolls it to its right (the "turn"). Integration: classic RK4 with a
fixed step DT; collisions (trees, the basket, the ground) are tested on every step's segment.
"""
from __future__ import annotations

import math

RHO = 1.2           # kg/m^3
G = 9.81
DT = 0.01
MAX_T = 15.0
MASS = 0.175
DIAM = 0.211
R_DISC = DIAM / 2
AREA = math.pi * R_DISC ** 2
IZZ = 0.0013        # kg m^2 (rim-weighted golf disc)

# Disc models: aerodynamic coefficients (Hummel's form). Assumed values for golf discs (thinner, faster rims than the
# Ultrastar of Hummel 2003: lower drag; a stronger nose-down moment at speed for the driver, i.e. more high-speed turn).
DISCS = {
    "putter":   dict(CL0=0.15, CLa=2.30, CD0=0.090, CDa=1.10, ALPHA0=-0.035, CM0=-0.001, CMa=0.090, CS=-0.010),
    "midrange": dict(CL0=0.13, CLa=2.50, CD0=0.070, CDa=1.20, ALPHA0=-0.035, CM0=-0.003, CMa=0.100, CS=-0.010),
    "driver":   dict(CL0=0.12, CLa=2.70, CD0=0.050, CDa=1.30, ALPHA0=-0.035, CM0=-0.0045, CMa=0.100, CS=-0.010),
}
DISC_NAMES = list(DISCS)


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _norm(a):
    return math.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2])


def deriv(s, disc, wind):
    """s = (x, y, z, vx, vy, vz, nx, ny, nz, ws). Returns ds/dt and the angle of attack."""
    c = DISCS[disc] if isinstance(disc, str) else disc
    v = (s[3], s[4], s[5])
    n = (s[6], s[7], s[8])
    ws = s[9]
    u = (v[0] - wind[0], v[1] - wind[1], v[2] - wind[2])
    U = _norm(u)
    if U < 1e-6:
        return (v[0], v[1], v[2], 0.0, 0.0, -G, 0.0, 0.0, 0.0, 0.0), 0.0
    un = _dot(u, n)
    up = (u[0] - un * n[0], u[1] - un * n[1], u[2] - un * n[2])
    UP = _norm(up)
    alpha = math.atan2(-un, UP)
    q = 0.5 * RHO * U * U
    uh = (u[0] / U, u[1] / U, u[2] / U)
    # lift direction: n made perpendicular to u
    nu = _dot(n, uh)
    ld = (n[0] - nu * uh[0], n[1] - nu * uh[1], n[2] - nu * uh[2])
    LD = _norm(ld)
    CL = c["CL0"] + c["CLa"] * alpha
    CD = c["CD0"] + c["CDa"] * (alpha - c["ALPHA0"]) ** 2
    fl = q * AREA * CL / MASS / LD if LD > 1e-9 else 0.0
    fd = q * AREA * CD / MASS
    ax = fl * ld[0] - fd * uh[0]
    ay = fl * ld[1] - fd * uh[1]
    az = fl * ld[2] - fd * uh[2] - G
    # pitching moment about -(n x xa), xa = in-plane air direction
    dn = (0.0, 0.0, 0.0)
    dws = 0.0
    if abs(ws) > 1e-3:
        if UP > 1e-9:
            xa = (up[0] / UP, up[1] / UP, up[2] / UP)
            ya = _cross(n, xa)
            M = q * AREA * DIAM * (c["CM0"] + c["CMa"] * alpha)
            k = -M / (IZZ * ws)                    # dn/dt = tau / (Izz ws), tau = -M ya (perpendicular to n)
            dn = (k * ya[0], k * ya[1], k * ya[2])
        dws = q * AREA * DIAM * c["CS"] * (ws * DIAM / (2 * U)) / IZZ
    return (v[0], v[1], v[2], ax, ay, az, dn[0], dn[1], dn[2], dws), alpha


def rk4(s, disc, wind, dt=DT):
    k1, a = deriv(s, disc, wind)
    s2 = tuple(s[i] + 0.5 * dt * k1[i] for i in range(10))
    k2, _ = deriv(s2, disc, wind)
    s3 = tuple(s[i] + 0.5 * dt * k2[i] for i in range(10))
    k3, _ = deriv(s3, disc, wind)
    s4 = tuple(s[i] + dt * k3[i] for i in range(10))
    k4, _ = deriv(s4, disc, wind)
    out = [s[i] + dt / 6.0 * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]) for i in range(10)]
    nn = math.sqrt(out[6] ** 2 + out[7] ** 2 + out[8] ** 2)
    out[6] /= nn
    out[7] /= nn
    out[8] /= nn
    return tuple(out), a


def release_state(pos, aim_deg, speed, launch_deg, nose_deg, bank_deg, spin_rps):
    """The disc's state at release. aim: heading (deg, 0 = +x, counter-clockwise > 0); launch: the velocity's angle
    above horizontal; nose: the disc's pitch above the velocity (the angle of attack at release in still air); bank:
    the roll about the flight direction, > 0 = the disc's left edge down (seen from behind); spin_rps > 0 = clockwise
    seen from above (a right-hand backhand)."""
    psi, gam = math.radians(aim_deg), math.radians(launch_deg)
    th, be = math.radians(launch_deg + nose_deg), math.radians(bank_deg)
    f = (math.cos(psi), math.sin(psi), 0.0)
    left = (-math.sin(psi), math.cos(psi), 0.0)
    v = (speed * math.cos(gam) * f[0], speed * math.cos(gam) * f[1], speed * math.sin(gam))
    n = tuple(-math.sin(th) * f[i] + math.cos(th) * (math.cos(be) * (1.0 if i == 2 else 0.0) + math.sin(be) * left[i])
              for i in range(3))
    nn = _norm(n)
    n = (n[0] / nn, n[1] / nn, n[2] / nn)
    return (pos[0], pos[1], pos[2], v[0], v[1], v[2], n[0], n[1], n[2], -2.0 * math.pi * spin_rps)


def free_flight(s, disc, wind, max_t=MAX_T):
    """Flight without obstacles until z <= 0 (for tuning): list of states."""
    out = [s]
    t = 0.0
    while t < max_t and s[2] > 0:
        s, _ = rk4(s, disc, wind)
        out.append(s)
        t += DT
    return out
