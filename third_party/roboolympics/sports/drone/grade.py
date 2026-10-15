"""drone's part of reward.json (the common part is core/grading.py: success, valid_trial, deterministic, <seat>_won,
<seat>_*_api_errors, official_config ...): numbers only, from final.json alone (the verifier has no simulator). The
keys are the legacy tasks' (drone-racing / drone-duel-wallclock-program-standard) wherever final.json still has the
value; official_config is replaced by the sport's own rule (racing: the level's seeds; see sport.toml [official])."""

OFFICIAL_SEEDS = [1, 2, 3]
OFFICIAL_LEVELS = [1, 2, 3]
OFFICIAL_SLOWDOWNS = {1: [1, 2, 5, 10, 20], 2: [10, 20]}     # as the legacy tasks (racing, duel)
TIEBREAK = ("most_heat_wins", "best_results", "last_heat_winner", "lot")


def _t(x):
    return float(x) if x is not None else -1.0


def evaluate(d):
    f = d.get("sport_fields") or {}
    ev = f.get("event") or {}
    heats = ev.get("heats") or []
    seats = (d.get("cfg") or {}).get("seats") or ["A"]
    stats = d.get("seats") or {}
    lag = d.get("sim_lag") or {}
    errors = d.get("errors") or []
    reason = d.get("reason")
    out = dict(event_over=int(bool(ev.get("over"))), level=int(f.get("level") or 0),
               slowdown=float(d.get("slowdown") or 0.0), n_reviews=len(d.get("reviews") or []),
               n_actions=sum(int((stats.get(s) or {}).get("applied", 0)) for s in seats), ticks=int(d.get("steps") or 0),
               sim_error=int(any(e.get("where") == "world" for e in errors)),
               replay_error=int(not d.get("deterministic")), timing_invalid=0,
               sim_lag_max_s=float(lag.get("late_max_s", 0.0)), sim_lag_p99_s=float(lag.get("late_p99_s", 0.0)),
               official_config=int(int(d.get("seed") or 0) in OFFICIAL_SEEDS
                                   and int(f.get("level") or 0) in OFFICIAL_LEVELS
                                   and float(d.get("slowdown") or 0) in OFFICIAL_SLOWDOWNS[len(seats)]))
    for s in seats:
        st = stats.get(s) or {}
        out[s + "_n_actions"] = int(st.get("applied", 0))
        out[s + "_n_observe"] = int(st.get("observations", 0))
        out[s + "_n_reviews"] = sum(r.get("seat") == s for r in d.get("reviews") or [])
        out[s + "_dropped_acts"] = int(st.get("dropped", 0))
        out[s + "_meter_error"] = int(bool((d.get("meter_error") or {}).get(s)))
        out[s + "_proxy_error"] = int(bool((d.get("proxy_error") or {}).get(s)))
        out[s + "_mode_error"] = int((d.get("player_modes") or {}).get(s) not in ("model", "oracle"))
    if len(seats) == 1:                                       # format racing (the legacy single keys)
        st = (ev.get("standings") or {}).get("A") or {}
        flights = [h["results"]["A"] for h in heats]
        out.update(started=int(bool((d.get("joined") or {}).get("A"))), rounds_flown=len(flights),
                   rounds_finished=int(st.get("finished", 0)), best_time_s=_t(st.get("best_time_s")),
                   second_time_s=_t(st.get("second_time_s")), third_time_s=_t(st.get("third_time_s")),
                   best_gates=int(st.get("best_gates", 0)),
                   gates_total=int(flights[0]["gates_total"]) if flights else 0,
                   collisions=sum(r["reason"] == "collision" for r in flights),
                   out_of_bounds=sum(r["reason"] == "out_of_bounds" for r in flights),
                   time_limits=sum(r["reason"] == "time_limit" for r in flights),
                   n_observe=out["A_n_observe"],
                   transient_api_errors=int((d.get("transient_api_errors") or {}).get("A", 0)),
                   unrecovered_api_errors=int((d.get("unrecovered_api_errors") or {}).get("A", 0)),
                   meter_error=out["A_meter_error"], proxy_error=out["A_proxy_error"], mode_error=out["A_mode_error"])
        for i in range(3):
            r = flights[i] if i < len(flights) else None
            out[f"round{i + 1}_time_s"] = _t(r["finish_time_s"]) if r else -1.0
            out[f"round{i + 1}_gates"] = int(r["gates_passed"]) if r else 0
        return out
    wins = ev.get("wins") or {}
    decided = d.get("decided_by") or ""
    complete = d.get("match_winner") is not None
    out.update(ready_forfeit=int(reason == "ready_timeout" and complete),
               start_forfeit=int(reason == "no_show" and complete), heats=len(heats),
               tied_heats=sum(h.get("winner") is None for h in heats),
               decided_by_lot=int(decided == "lot"), decided_by_tiebreak=int(decided in TIEBREAK))
    for s in seats:
        st = (ev.get("standings") or {}).get(s) or {}
        flights = [h["results"][s] for h in heats]
        out[s + "_heat_wins"] = int(wins.get(s, 0))
        out[s + "_points"] = int(wins.get(s, 0))               # the duel's score: heat wins
        out[s + "_heats_finished"] = int(st.get("finished", 0))
        out[s + "_best_time_s"] = _t(st.get("best_time_s"))
        out[s + "_best_gates"] = int(st.get("best_gates", 0))
        out[s + "_collisions"] = sum(r["reason"] == "collision" for r in flights)
    return out
