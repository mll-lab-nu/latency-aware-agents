"""hello: the reference implementation of a sport's post-match analysis (core/diagnose.py; capability measurement
framework §7). Analysis only: no match imports it and it is never copied into a task (tools/sportlib.NOT_IN_SIM).

    python -m core.diagnose sports/hello/plugin.py RUN_DIR

Decision unit: one throw (one act). The claim read from the command (never asked of the player): the high level's
`aim {x, y}` names the ground point the player believes the target is at -> err_space = |aim - target| (the target's
true position from the replayed world at the act). The resting ball against the aim is the platform's execution
(launcher scatter + roll), a negative control for the player's abilities. The low level's throw has no aim point, so
it gives no claim (units only).

The metrics here are examples of the interface: none has been through a validity matrix, so every status is
"unvalidated" (VALIDITY below; a sport's real table cites its report and gives official / conditional only to
metrics that passed it).
"""
from __future__ import annotations

import math
import statistics

VALIDITY_REPORT = None          # the sport's validity matrix report (a link), None: no matrix run
VALIDITY = {                    # metric_id -> (capability, status)
    "aim_error_m": ("A1", "unvalidated"),
    "aim_bias_m": ("A2", "unvalidated"),
    "execution_error_m": ("A3", "behaviour"),
}


def units(rep):
    """One unit per accepted act (a throw), in order."""
    out = []
    for seat in rep["seats"]:
        for a in rep["acts"].get(seat, []):
            k = a["k_effect"] if a["k_effect"] is not None else a["k_recv"]
            out.append({"id": f"{seat}-{a['act_id']}", "seat": seat, "k0": k, "k1": k, "act_id": a["act_id"],
                        "effect": a["k_effect"] is not None})
    return sorted(out, key=lambda u: (u["k0"], u["seat"]))


def implied(rep, units):
    """The aim point of each high-level throw against the true target, and where the ball came to rest."""
    w = rep["world"]()
    acts = {(s, a["act_id"]): a for s, A in rep["acts"].items() for a in A}
    out = []
    for u in units:
        a = acts[(u["seat"], u["act_id"])]
        d = a["decision"] or {}
        if a["k_effect"] is None or d.get("type") != "aim":
            continue
        m = w.at(a["k_effect"])                        # the world the throw was made in
        tx, ty = m.target
        n_before = len(m.throws)
        m = w.at(a["k_effect"], after=True)            # the throw done
        rest = m.throws[n_before]["rest"] if len(m.throws) > n_before else None
        aim = [float(d["x"]), float(d["y"])]
        out.append({"unit": u["id"], "seat": u["seat"], "k_cmd": a["k_effect"], "act_id": a["act_id"],
                    "quantity": "target_xy", "implied": aim, "truth": [tx, ty],
                    "err_space": math.hypot(aim[0] - tx, aim[1] - ty), "err_vec": [aim[0] - tx, aim[1] - ty],
                    "err_time": None, "rest": rest,
                    "exec_err": None if rest is None else math.hypot(rest[0] - aim[0], rest[1] - aim[1])})
    return out


def metrics(units, claims):
    by: dict = {}
    for c in claims:
        by.setdefault(c["seat"], []).append(c)
    out = {}
    for seat, C in by.items():
        ex, ey = (statistics.fmean(c["err_vec"][i] for c in C) for i in (0, 1))
        execs = [c["exec_err"] for c in C if c["exec_err"] is not None]
        vals = {"aim_error_m": statistics.fmean(c["err_space"] for c in C), "aim_bias_m": math.hypot(ex, ey),
                "execution_error_m": statistics.fmean(execs) if execs else None}
        out[seat] = {m: {"value": v, "capability": VALIDITY[m][0], "status": VALIDITY[m][1], "n": len(C)}
                     for m, v in vals.items()}
    return out
