"""baseball's part of reward.json (the common part is core/grading.py): numbers only, from final.json (and, for the
legacy metering fields of format batting, the seat's think_usage_A.json artifact). The keys are the legacy tasks'
(baseball-batting-wallclock-program-standard, baseball-duel-wallclock-program-standard) wherever the v0.2 core
records the value; README.md lists the keys that changed.

Format batting (a solo score) follows the core's rules (decision B2 of 2026-10-07): success = a valid, complete game
(reason match_over) with at least 4 bases; a batter whose session ends after the start and before the end of the game
has a valid loss (core 0.2.6, reason abandoned, success 0, whatever the bases). The legacy service's rule (a valid game
at any end, success at 4 bases) is gone; `passed` still reports bases >= 4."""
import json
import os
from pathlib import Path

OFFICIAL_SEEDS = {1: [301, 311, 321], 2: [302, 312, 322], 3: [303, 313, 323], 4: [304, 314, 324]}
OFFICIAL_SLOWDOWNS = {"batting": [1, 2, 5, 10, 20], "duel": [10, 20]}
COUNTS = ("attempts", "applied", "refused", "dropped", "observations")


def usage(seat):
    try:
        return json.loads((Path(os.environ.get("RCB_DATA", "/data")) / f"think_usage_{seat}.json").read_text())
    except (OSError, ValueError):
        return {}


def _max(st, k):
    v = st.get(k) or []
    return float(max(v)) if isinstance(v, list) and v else 0.0


def evaluate(d):
    f, seats = d.get("sport_fields") or {}, d["cfg"]["seats"]
    lag = d.get("sim_lag") or {}
    slowdown = float(d.get("slowdown") or 0.0)
    if len(seats) == 1:
        return batting(d, f, seats[0], lag, slowdown)
    out = dict(game_over=int(d.get("reason") == "match_over"), ready_forfeit=int(d.get("reason") == "ready_timeout"),
               decided_by_last_pitch=int(d.get("decided_by") == "last_pitch"), timing_invalid=0,
               steps=int(d.get("steps") or 0), pitches=int(f.get("pitches") or 0),
               plate_appearances=int(f.get("plate_appearances") or 0), first_batter_A=int(d.get("first") == "A"),
               slowdown=slowdown, official_config=int(slowdown in OFFICIAL_SLOWDOWNS["duel"]),
               sim_lag_max_s=float(lag.get("late_max_s") or 0.0))
    for s in seats:
        st = (d.get("seats") or {}).get(s, {})
        out[s + "_bases"] = out[s + "_points"] = int((d.get("scores") or {}).get(s, 0))
        for k in COUNTS:
            out[s + "_" + k] = int(st.get(k, 0))
        out[s + "_rejected"] = int(st.get("rejected", 0))
        out[s + "_act_s_max"] = _max(st, "act_s")
        out[s + "_obs_age_s_max"] = _max(st, "obs_age_s")
    return out


def batting(d, f, a, lag, slowdown):
    st, u = (d.get("seats") or {}).get(a, {}), usage(a)
    errs = d.get("errors") or []
    bases, thr = int(f.get("bases") or 0), int(f.get("threshold") or 4)
    rep = (d.get("replay_scores") or {}).get(a)
    det = bool(d.get("deterministic"))
    meter = bool((d.get("meter_error") or {}).get(a))
    proxy = bool((d.get("proxy_error") or {}).get(a))
    sim_error = d.get("reason") == "technical_error" or any(e.get("where") != "replay" for e in errs)
    replay_error = not det or any(e.get("where") == "replay" for e in errs)
    level, seed = int(f.get("level") or 0), int(d.get("seed") or 0)
    out = dict(passed=int(bases >= thr), sim_error=int(sim_error), replay_error=int(replay_error),
               meter_error=int(meter), proxy_error=int(proxy), timing_invalid=0, bases=bases, threshold=thr,
               replay_bases=-1 if rep is None else int(rep), replay_decisions_equal=int(det),
               finalized=int(bool(d.get("finalized"))), revision=4, timing_lag_enforced=0,
               game_over=int(d.get("reason") == "match_over"), metered_requests=int(u.get("requests", 0)),
               metered_output_tokens=int(u.get("output_tokens", 0)), metered_input_tokens=int(u.get("input_tokens", 0)),
               rate_limit_retries=int(u.get("rate_limit_retries", 0)),
               transient_api_errors=int((d.get("transient_api_errors") or {}).get(a, 0)),
               unrecovered_api_errors=int((d.get("unrecovered_api_errors") or {}).get(a, 0)),
               level=float(level), slowdown=slowdown,
               official_config=int(seed in OFFICIAL_SEEDS.get(level, []) and slowdown in OFFICIAL_SLOWDOWNS["batting"]))
    for k in ("pa_done", "hits", "walks", "strikeouts", "home_runs", "pitches", "n_actions"):
        out[k] = float(f.get(k) or 0)
    out.update(n_decisions=float(st.get("applied", 0)), refused_not_model=float(st.get("refused", 0)),
               dropped_acts=float(st.get("dropped", 0)), rejected_acts=float(st.get("rejected", 0)),
               steps=float(d.get("steps") or 0), n_reviews=float(len(d.get("reviews") or [])),
               n_observe=float(st.get("observations", 0)), n_act=float(st.get("attempts", 0)),
               elapsed_s=float(d.get("elapsed_s") or 0.0), max_lag_s=0.0,
               sim_lag_late_max_s=float(lag.get("late_max_s") or 0.0),
               sim_lag_late_p99_s=float(lag.get("late_p99_s") or 0.0),
               sim_effective_slowdown=float(lag["run_wall_s"] / lag["run_sim_s"]) if lag.get("run_sim_s") else 0.0,
               sim_run_wall_s=float(lag.get("run_wall_s") or 0.0))
    return out
