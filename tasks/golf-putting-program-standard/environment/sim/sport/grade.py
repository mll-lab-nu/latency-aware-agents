"""Mini-golf putting's part of reward.json (the common part is core/grading.py): numbers only, from final.json (the
verifier has no simulator) and the seats' metering files that the task uploads next to it. The key names are the
Season-1 graders' (scripts/olympiad/golf/*/tests/grade.py), unprefixed (the single round)."""

import json
import os
from pathlib import Path

SUCCESS_LINE = 9  # par; sport.toml [formats.single] success = { score_at_most = 9 }
HOLES = 4


def usage(seat):
    p = Path(os.environ.get("RCB_DATA", "/data")) / f"think_usage_{seat}.json"
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return {}


def evaluate(d):
    f = d.get("sport_fields") or {}
    seats = d["cfg"]["seats"]           # one format: the single round (the duel was retired on 2026-10-08)
    errs = d.get("errors") or []
    out = {
        "replay_error": int(not d.get("deterministic")),
        "sim_error": int(d.get("reason") == "technical_error" or any(e.get("where") != "replay" for e in errs)),
        "meter_error": int(any((d.get("meter_error") or {}).values())),
        "proxy_error": int(any((d.get("proxy_error") or {}).values())),
        "elapsed_s": float(d.get("elapsed_s") or 0.0),
    }
    for s in seats:
        sc = (f.get("scores") or {}).get(s) or {
            "total": 0,
            "holes": [None] * HOLES,
            "complete": False,
            "picked_up": [False] * HOLES,
            "par_of_holes_scored": 0,
        }
        st = (d.get("seats") or {}).get(s) or {}
        u = usage(s)
        out["total"] = int(sc["total"])
        out["complete"] = int(bool(sc["complete"]))
        out["clock_out"] = int(bool((f.get("clock_out") or {}).get(s)))
        out["n_shots"] = int(st.get("n_act", 0))
        for k in ("n_reviews", "refused_busy", "refused_not_model", "refused_turn"):
            out[k] = int(st.get(k, 0))
        out["clock_used_s"] = float((d.get("clock_used_s") or {}).get(s, 0.0))
        out["metered_requests"] = int(u.get("requests", 0))
        out["metered_output_tokens"] = int(u.get("output_tokens", 0))
        out["to_par"] = int(sc["total"]) - int(sc.get("par_of_holes_scored", 0))
        out["picked_up_holes"] = int(sum(bool(x) for x in sc.get("picked_up", [])))
        holes = list(sc.get("holes") or []) + [None] * HOLES
        for i in range(HOLES):
            out[f"score_h{i + 1}"] = int(holes[i] or 0)
        out["transient_api_errors"] = int((d.get("transient_api_errors") or {}).get(s, 0))
        out["unrecovered_api_errors"] = int((d.get("unrecovered_api_errors") or {}).get(s, 0))
    out["success_line"] = SUCCESS_LINE
    return out
