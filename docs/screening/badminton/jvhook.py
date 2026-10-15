"""Online joint-velocity statistics for badminton's bm.Phys (import after bm; install(BM)).

Every Phys accumulates, per physics step (servo call), the arm's physical joint speed against its velocity limits
(bm.ARM_VEL): steps, steps over 1.0x / 1.05x, the max ratio, over-limit episodes (lengths), which joint, and the
same restricted to steps where |base acceleration| (physical) > 5 m/s^2. Stats live on the Phys object (attribute
_jv), so a deepcopy (B9 trials) carries them and the delta is the try's. stats(ph) returns a JSON-able summary."""
import numpy as np

V = np.array([2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61])


def _new():
    return {"steps": 0, "over": 0, "over105": 0, "max": 0.0, "run": 0, "eps": [], "byj": [0] * 7,
            "over_base5": 0, "steps_base5": 0, "sp_max": 0.0}


def install(BM):
    _servo = BM.Phys.servo

    def servo(self, q_des, qd_des, base_acc=None):
        d, va = self.d, self.va
        r = np.abs(d.qvel[va[2:]]) / V
        ba = float(np.hypot(*d.qacc[va[:2]]))
        out = _servo(self, q_des, qd_des, base_acc)
        s = self.__dict__.get("_jv")
        if s is None:
            s = self._jv = _new()
        s["steps"] += 1
        m = float(r.max())
        if ba > 5.0:
            s["steps_base5"] += 1
        if m > 1.0:
            s["over"] += 1
            s["run"] += 1
            s["byj"][int(np.argmax(r))] += 1
            if ba > 5.0:
                s["over_base5"] += 1
        elif s["run"]:
            s["eps"].append(s["run"]); s["run"] = 0
        if m > 1.05:
            s["over105"] += 1
        s["max"] = max(s["max"], m)
        s["sp_max"] = max(s["sp_max"], float(np.max(np.abs(self.jl.qd) / V)))
        return out

    BM.Phys.servo = servo


def stats(ph, since=None):
    s = dict(ph.__dict__.get("_jv") or _new())
    eps = list(s["eps"]) + ([s["run"]] if s["run"] else [])
    if since is not None:
        for k in ("steps", "over", "over105", "over_base5", "steps_base5"):
            s[k] -= since.get(k, 0)
        eps = eps[len(since.get("eps", [])):]
        s["byj"] = [a - b for a, b in zip(s["byj"], since.get("byj", [0] * 7))]
    return {"steps": s["steps"], "over": s["over"], "over105": s["over105"], "max": round(s["max"], 4),
            "eps": eps, "byj": s["byj"], "over_base5": s["over_base5"], "steps_base5": s["steps_base5"],
            "sp_max": round(s["sp_max"], 4), "n_reflex": int(getattr(ph, "n_reflex", 0))}


def snap(ph):
    s = ph.__dict__.get("_jv") or _new()
    return {k: (list(v) if isinstance(v, list) else v) for k, v in s.items()}
