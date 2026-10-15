"""basketball's part of reward.json (the common part is core/grading.py): numbers only, from final.json alone (the
verifier has no simulator). The keys are the legacy single task's (basketball-shootout-program-standard) wherever
final.json holds the value (the duel format and its keys are gone, review item B10); the legacy single's metered_* and
rate_limit_retries came from think_usage.json, which the v0.2 verifier does not read."""

RESULTS = ("rim", "backboard", "blocked", "airball")


def _single(d, f):
    seat = d["cfg"]["seats"][0]
    st = (d.get("seats") or {}).get(seat, {})
    counts = f.get("counts") or {}
    out = {"sim_error": int(bool(d.get("errors"))), "replay_error": int(not d.get("deterministic")),
           "meter_error": int(bool((d.get("meter_error") or {}).get(seat))),
           "proxy_error": int(bool((d.get("proxy_error") or {}).get(seat))), "finalized": int(bool(d.get("finalized"))),
           "made": int(f.get("made") or 0), "shots_taken": int(f.get("shots_taken") or 0),
           "shots_total": int(f.get("shots_total") or 10), "makes_needed": int(f.get("makes_needed") or 0),
           "complete": int(bool(f.get("complete"))), "clock_forfeit": int(bool(f.get("clock_forfeit"))),
           "started": int(bool((d.get("started") or {}).get(seat))),
           **{c: int(counts.get(c, 0)) for c in RESULTS},
           "clock_used_s": float((d.get("clock_used_s") or {}).get(seat, 0.0)),
           "elapsed_s": float(d.get("elapsed_s") or 0.0),
           "transient_api_errors": int((d.get("transient_api_errors") or {}).get(seat, 0)),
           "unrecovered_api_errors": int((d.get("unrecovered_api_errors") or {}).get(seat, 0))}
    for k in ("n_reviews", "n_act", "n_observe", "refused_busy", "refused_not_model"):
        out[k] = int(st.get(k, 0))
    return out


def evaluate(d):
    f = d.get("sport_fields") or {}
    return _single(d, f)
