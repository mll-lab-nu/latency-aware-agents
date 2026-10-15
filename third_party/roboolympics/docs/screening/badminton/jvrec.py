"""Joint-velocity recorder for badminton (runs inside the sport image on an assembled sim at /srv).

    python3 jvrec.py EPISODE.jsonl OUT.npz            (replay a recorded episode, e.g. a golden)

Wraps bm.Phys.servo: per physics step of every live Phys (twins / planner copies excluded: an instance that ever
gets copy_from / load is a twin; the live ones are the longest series) it records the arm's physical joint velocity
(d.qvel), the setpoint velocity (JerkLimiter.qd after the step), the torque asked of the torque stage and the torque
it gave (TorqueRateLimiter in/out), the base's commanded acceleration (base_acc) and its physical acceleration
(d.qacc), the coupling torque M_arm,base @ base_acc, the RESYNC flag and the reflex state if the servo has one."""
import json, sys
import numpy as np
sys.path.insert(0, "/srv")
from core.episode import load_plugin, read_rows, replay
from bodies import limits as AL

REC = {}
TWIN = set()
LAST = {}
_trl_call = AL.TorqueRateLimiter.__call__


def trl_call(self, tau_des):
    LAST["in"] = np.array(tau_des, float).copy()
    out = _trl_call(self, tau_des)
    LAST["out"] = np.array(out, float).copy()
    return out


AL.TorqueRateLimiter.__call__ = trl_call
sport = load_plugin("/srv/sport/plugin.py")
BM = sys.modules["bm"]
_servo, _copy, _load = BM.Phys.servo, BM.Phys.copy_from, BM.Phys.load


def copy_from(self, o):
    TWIN.add(id(self)); return _copy(self, o)


def load(self, st):
    TWIN.add(id(self)); return _load(self, st)


def servo(self, q_des, qd_des, base_acc=None):
    d, va = self.d, self.va
    qd = d.qvel[va[2:]].copy()
    bqacc = d.qacc[va[:2]].copy()
    bqd = d.qvel[va[:2]].copy()
    LAST.clear()
    out = _servo(self, q_des, qd_des, base_acc)
    M = self._M
    ba = np.zeros(2) if base_acc is None else np.asarray(base_acc, float)
    cpl = M[np.ix_(va[2:], va[:2])] @ ba
    e = self.jl.q - d.qpos[self.qa[2:]]
    r = REC.setdefault(id(self), {"keep": self, "rows": []})
    rf = getattr(self, "reflex_at", None)
    r["rows"].append(np.r_[d.time, qd, self.jl.qd, LAST.get("in", np.zeros(7)), LAST.get("out", np.zeros(7)),
                           ba, bqacc, bqd, cpl, float(np.max(np.abs(e)) >= BM.RESYNC - 1e-12),
                           0.0 if rf is None else 1.0, np.asarray(qd_des[2:], float)])
    return out


BM.Phys.servo, BM.Phys.copy_from, BM.Phys.load = servo, copy_from, load
res = replay(sport, read_rows(sys.argv[1]))
live = sorted((r for k, r in REC.items() if k not in TWIN), key=lambda r: -len(r["rows"]))
n0 = len(live[0]["rows"]) if live else 0
live = [r for r in live if len(r["rows"]) >= 0.5 * n0]
arrs = {f"a{i}": np.array(r["rows"]) for i, r in enumerate(live)}
nref = {f"nreflex{i}": int(getattr(r["keep"], "n_reflex", 0)) for i, r in enumerate(live)}
np.savez_compressed(sys.argv[2], n=len(live), **arrs)
print(json.dumps({"replay": {k: res.get(k) for k in ("state_sha", "scores", "winner", "k")},
                  "live": [len(r["rows"]) for r in live], **nref}, default=float))
