"""archery's part of reward.json (the common part is core/grading.py): numbers only, from final.json alone (the
verifier has no simulator). The keys are the legacy task's (archery-program-standard) wherever final.json holds the
value; the duel format and its keys are gone (review item B10)."""


def _single(d, f):
    seat = d["cfg"]["seats"][0]
    st = (d.get("seats") or {}).get(seat, {})
    arrows = [v for e in (f.get("arrow_scores") or []) for v in e]
    ends = list(f.get("end_scores") or []) + [0] * 4
    errs = bool(d.get("errors"))
    out = {"sim_error": int(errs), "replay_error": int(not d.get("deterministic")),
           "meter_error": int(bool((d.get("meter_error") or {}).get(seat))),
           "proxy_error": int(bool((d.get("proxy_error") or {}).get(seat))), "finalized": int(bool(d.get("finalized"))),
           "total": int(f.get("total") or 0), "max_total": int(f.get("max_total") or 120),
           "success_min_total": int(f.get("success_min_total") or 0), "round_complete": int(bool(f.get("round_complete"))),
           "clock_forfeit": int(bool(f.get("clock_forfeit"))), "started": int(bool((d.get("started") or {}).get(seat))),
           "n_x": int(f.get("n_x") or 0), **{f"end{i + 1}": int(ends[i]) for i in range(4)},
           "arrows_shot": int(f.get("arrows_shot") or 0), "misses": sum(1 for v in arrows if v == 0),
           "tens": sum(1 for v in arrows if v == 10), "clock_used_s": float((d.get("clock_used_s") or {}).get(seat, 0.0)),
           "elapsed_s": float(d.get("elapsed_s") or 0.0),
           "transient_api_errors": int((d.get("transient_api_errors") or {}).get(seat, 0)),
           "unrecovered_api_errors": int((d.get("unrecovered_api_errors") or {}).get(seat, 0))}
    for k in ("n_reviews", "n_act", "n_observe", "refused_busy", "refused_not_model"):
        out[k] = int(st.get(k, 0))
    return out


def evaluate(d):
    f = d.get("sport_fields") or {}
    return _single(d, f)
