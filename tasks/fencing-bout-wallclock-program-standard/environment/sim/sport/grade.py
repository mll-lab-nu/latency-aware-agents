"""fencing's part of reward.json (the common part is core/grading.py: success, valid_trial, technical_failure,
deterministic, seed, official_config, video_rendered, <seat>_won, <seat>_transient/unrecovered_api_errors, ...):
numbers only, from final.json (and the seat's metering summary think_usage_<seat>.json, a declared artifact of the
sim). The keys keep the legacy reward names of both tasks (scripts/olympiad/fencing/{bout,duel}/tests/grade.py);
the differences are listed in README.md."""
import json
import os
from pathlib import Path

DATA = Path(os.environ.get("RCB_DATA", "/data"))


def _usage(seat):
    try:
        return json.loads((DATA / f"think_usage_{seat}.json").read_text())
    except (OSError, ValueError):
        return {}


def _max(v):
    return round(max(v), 4) if isinstance(v, list) and v else 0.0


def evaluate(d):
    f = d.get("sport_fields") or {}
    seats = d["cfg"]["seats"]
    reason = d.get("reason")
    halts = f.get("halts") or []
    scores = d.get("scores") or {}
    lag = d.get("sim_lag") or {}
    stats = d.get("seats") or {}
    over = reason == "match_over"
    out = {"timing_invalid": 0, "bout_over": int(over), "extra_minute": int(bool(f.get("extra_minute"))),
           "halts": len(halts), "doubles": sum(1 for h in halts if h.get("double")),
           "corps_a_corps": sum(1 for h in halts if h.get("why") == "corps_a_corps"),
           "passivity_halts": sum(1 for h in halts if h.get("why") == "passivity"),
           "fencing_time_s": float(f.get("fencing_time_s") or 0.0), "slowdown": float(d.get("slowdown") or 0.0)}
    if len(seats) == 1:                                  # the bout against the scripted fencer (side B)
        s = seats[0]
        st, u = stats.get(s, {}), _usage(s)
        won = d.get("match_winner") == s and over
        rp = d.get("replay_scores") or {}
        pf, pa = int(scores.get(s, 0)), int(scores.get("B", 0))
        out.update(
            won=int(won), sim_error=int(bool(d.get("errors")) or reason == "technical_error"),
            replay_error=int(not d.get("deterministic")), meter_error=int(bool((d.get("meter_error") or {}).get(s))),
            proxy_error=int(bool((d.get("proxy_error") or {}).get(s))), finalized=int(bool(d.get("finalized"))),
            touches_for=pf, touches_against=pa, points_for=pf, points_against=pa,
            replay_touches_for=int(rp.get(s, -1)), replay_touches_against=int(rp.get("B", -1)),
            won_on_priority=int(won and f.get("how") == "priority"), level=float(f.get("level") or 0),
            n_act=float(st.get("applied", 0)), n_observe=float(st.get("observations", 0)),
            n_reviews=float(sum(1 for r in d.get("reviews") or [] if r.get("seat") == s)),
            refused_busy=float(st.get("refused_busy", 0)), refused_not_model=float(st.get("refused", 0)),
            dropped_acts=float(st.get("dropped", 0)), elapsed_s=float(d.get("elapsed_s") or 0.0),
            sim_lag_late_max_s=float(lag.get("late_max_s") or 0.0), sim_lag_late_p99_s=float(lag.get("late_p99_s") or 0.0),
            sim_step_ms_max=float(f.get("step_ms_max") or 0.0), plan_s_max=_max(st.get("plan_s")),
            act_s_max=_max(st.get("act_s")), metered_requests=int(u.get("requests", 0)),
            metered_output_tokens=int(u.get("output_tokens", 0)), metered_input_tokens=int(u.get("input_tokens", 0)),
            rate_limit_retries=int(u.get("rate_limit_retries", 0)),
            transient_api_errors=int((d.get("transient_api_errors") or {}).get(s, 0)),
            unrecovered_api_errors=int((d.get("unrecovered_api_errors") or {}).get(s, 0)))
        return out
    out.update(ready_forfeit=int(reason == "ready_timeout" and d.get("forfeiting") in seats
                                 and d.get("match_winner") is not None),
               won_on_priority=int(bool(d.get("valid_trial")) and f.get("how") == "priority"),
               steps=int(d.get("steps") or 0), sim_lag_max_s=float(lag.get("late_max_s") or 0.0),
               first_player_A=int(d.get("first") == "A"))
    for s in seats:
        st = stats.get(s, {})
        out[s + "_points"] = int(scores.get(s, 0))
        for k in ("attempts", "applied", "refused", "refused_busy", "dropped", "observations"):
            out[s + "_" + k] = int(st.get(k, 0))
    return out
