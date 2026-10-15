"""Kinematics library of the air-hockey pilot's low-level toolkit (generic tools only: forward kinematics, Jacobians,
inverse kinematics; no skill). One KUKA LBR iiwa 14 with the striker (the robot-only model iiwa_mallet.xml, seat A:
base at (-1.51, 0, -0.1) in the table frame). numpy + mujoco.

    kin = Kin()                          # or Kin("/opt/robot/iiwa_mallet.xml")
    p, R = kin.fk(q)                     # the mallet centre (the striker_mallet body; flat on the table at z = Z) and
                                         # its rotation; q = 9 joints: the 7 iiwa axes + the 2 striker joints
    Jp, Jr = kin.jac(q)                  # 3 x 9 position / rotation Jacobians at the mallet centre
    q, res = kin.ik([x, y], q0)          # damped least squares: the mallet centre at (x, y, Z), the mallet flat
                                         # (its axis pointing down); res = the remaining position error (m)
    kin.limits                           # the joint limits the servo enforces (also in toolkit.toml)

The same inverse kinematics as the platform's own path controller (sports/airhockey/hockey.py Hockey.solve)."""
from __future__ import annotations

import os
from pathlib import Path

import mujoco
import numpy as np

Z = 0.0645                                     # the mallet body's height when the mallet sits on the table
DESIRED = np.diag([-1.0, 1.0, -1.0])           # seat A's flat mallet orientation (the body's axes in the table frame)
NQ = 9

# the joint limits of the servo (the 7 iiwa axes: 0.95 x KUKA data sheet velocity, AHC acceleration = 10 x data sheet
# velocity, jerk = 50 x acceleration (estimate); the 2 striker joints: the AHC model's servo)
VEL = np.array([1.40935, 1.40935, 1.65806, 1.24355, 2.15548, 2.23838, 2.23838, 3.0, 3.0])
ACC = np.array([14.8353, 14.8353, 17.4533, 13.0900, 22.6893, 23.5619, 23.5619, 50.0, 50.0])
JERK = np.array([741.765, 741.765, 872.665, 654.5, 1134.465, 1178.095, 1178.095, np.inf, np.inf])
RANGE_LO = np.array([-2.96706, -2.09440, -2.96706, -2.09440, -2.96706, -2.09440, -3.05433, -1.5708, -1.5708])
RANGE_HI = -RANGE_LO
TAU = np.array([320.0, 320.0, 176.0, 176.0, 110.0, 40.0, 40.0, 10.0, 10.0])


def default_model_path() -> str:
    tk = os.environ.get("AHM_TOOLKIT")
    for p in (os.environ.get("AHM_ROBOT_XML"), tk and str(Path(tk) / "iiwa_mallet.xml"), "/opt/robot/iiwa_mallet.xml",
              str(Path(__file__).resolve().parents[1] / "iiwa_mallet.xml")):
        if p and Path(p).is_file():
            return p
    raise FileNotFoundError("iiwa_mallet.xml (set AHM_ROBOT_XML)")


class Kin:
    def __init__(self, path: str | None = None):
        self.model = mujoco.MjModel.from_xml_path(path or default_model_path())
        self.data = mujoco.MjData(self.model)
        self.body = self.model.body("iiwa_1/striker_mallet").id
        self.jp = np.zeros((3, self.model.nv))
        self.jr = np.zeros((3, self.model.nv))
        self.limits = {"velocity_rad_s": VEL.tolist(), "acceleration_rad_s2": ACC.tolist(),
                       "jerk_rad_s3": [None if not np.isfinite(x) else float(x) for x in JERK],
                       "range_rad": [RANGE_LO.tolist(), RANGE_HI.tolist()], "torque_nm": TAU.tolist()}

    def _set(self, q):
        self.data.qpos[:NQ] = q
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)

    def fk(self, q):
        self._set(np.asarray(q, float))
        return self.data.xpos[self.body].copy(), self.data.xmat[self.body].reshape(3, 3).copy()

    def jac(self, q):
        self._set(np.asarray(q, float))
        mujoco.mj_jacBody(self.model, self.data, self.jp, self.jr, self.body)
        return self.jp[:, :NQ].copy(), self.jr[:, :NQ].copy()

    def ik(self, xy, q0, iters: int = 60, z: float = Z, tol: float = 1e-5):
        """q with the mallet centre at (x, y, z) and the mallet flat, from q0 (damped least squares, steps of at most
        0.08 rad per joint, within the joint ranges); returns (q, position residual in m)."""
        q = np.asarray(q0, float).copy()
        tgt = np.array([float(xy[0]), float(xy[1]), z])
        lo, hi = RANGE_LO + .01, RANGE_HI - .01
        res = 1.0
        for _ in range(iters):
            self._set(q)
            cur = self.data.xmat[self.body].reshape(3, 3)
            dp = tgt - self.data.xpos[self.body]
            res = float(np.linalg.norm(dp))
            err = np.r_[dp, sum(np.cross(cur[:, i], DESIRED[:, i]) for i in range(3)) * .5]
            if res < tol and np.linalg.norm(err[3:]) < 1e-4:
                break
            mujoco.mj_jacBody(self.model, self.data, self.jp, self.jr, self.body)
            J = np.vstack([self.jp[:, :NQ], self.jr[:, :NQ]])
            dq = J.T @ np.linalg.solve(J @ J.T + .001 * np.eye(6), err)
            q = np.clip(q + np.clip(dq, -.08, .08), lo, hi)
        self._set(q)
        return q, float(np.linalg.norm(tgt - self.data.xpos[self.body]))
