"""racing's part of reward.json (the common part is core/grading.py: success, valid_trial, deterministic,
<seat>_won, <seat>_place, <seat>_*_api_errors, no_show, ...): numbers only, from final.json and the seats' metering
summaries (think_usage_<seat>.json, declared artifacts of the sim). The keys are the Season-1 racing grader's
(scripts/olympiad/racing/*/tests/grade.py) wherever final.json still has the value; README.md lists the differences.
Per-car keys cover every car of the race, so in the match format also the scripted car B (its API counters are 0)."""

import json
import os
from pathlib import Path

DATA = Path(os.environ.get("RCB_DATA", "/data"))
OFFICIAL_SEEDS = {1: [101, 111, 121], 2: [202, 212, 222], 3: [303, 313, 323], 4: [404, 414, 424]}
OFFICIAL_SLOWDOWNS = [1, 2, 5, 10, 20]
DUEL_SEEDS = [1, 2, 3]
DUEL_SLOWDOWNS = [10, 20]
RACE_SEEDS = [1, 2, 3]
RACE_SLOWDOWNS = [10, 20]
RACE_CARS_OFFICIAL = [4]
V02_DEFAULT_SEEDS = [1, 2, 3]  # the match task's default seed is 1 on the v0.2 core (legacy: 101)


def _usage(seat):
    try:
        return json.loads((DATA / f"think_usage_{seat}.json").read_text())
    except (OSError, ValueError):
        return {}


def official(fmt, seed, level, slowdown, cars) -> bool:
    if fmt == "race":
        return seed in RACE_SEEDS and slowdown in RACE_SLOWDOWNS and cars in RACE_CARS_OFFICIAL
    if fmt == "duel":
        return seed in DUEL_SEEDS and slowdown in DUEL_SLOWDOWNS
    return (seed in OFFICIAL_SEEDS.get(level, []) or seed in V02_DEFAULT_SEEDS) and slowdown in OFFICIAL_SLOWDOWNS


def evaluate(d):
    f = d.get("sport_fields") or {}
    seats = d["cfg"]["seats"]
    fmt = "match" if len(seats) == 1 else "duel" if len(seats) == 2 else "race"
    cars_n = int(f.get("cars") or max(2, len(seats)))
    cars = list("ABCDEF"[:cars_n])
    scores = f.get("race_scores") or {}
    results, crashes = f.get("results") or [], f.get("crashes") or []
    stats = d.get("seats") or {}
    lag = d.get("sim_lag") or {}
    reason = d.get("reason")
    valid = bool(d.get("valid_trial"))
    winner = d.get("match_winner")
    car_winner = "B" if winner == "opponent" else winner  # the race.py car that won
    seed, level = int(d.get("seed") or 0), int(f.get("level") or 1)
    slowdown = float(d.get("slowdown") or 0.0)
    transient = d.get("transient_api_errors") or {}
    unrecovered = d.get("unrecovered_api_errors") or {}
    out = dict(
        replay_error=int(not d.get("deterministic")),
        meter_error=int(any((d.get("meter_error") or {}).values())),
        proxy_error=int(any((d.get("proxy_error") or {}).values())),
        timing_invalid=0,
        official_config=int(official(fmt, seed, level, slowdown, cars_n)),
        level=level,
        slowdown=slowdown,
        first_player_A=int(d.get("first") == "A"),
        n_actions=sum(int((stats.get(s) or {}).get("applied", 0)) for s in seats),
        steps=int(d.get("steps") or 0),
        sim_steps=int(d.get("steps") or 0) * 5,
        heats=len(results),
        crashes=len(crashes),
        cars=cars_n,
        obstacle_crashes=sum(1 for c in crashes if c.get("fault_rule") == "obstacle"),
        won=int(valid and winner == "A"),
        race_over=int(reason == "match_over"),
        disqualification=int(f.get("end_reason") == "disqualification"),
        ready_forfeit=int(reason == "ready_timeout"),
        start_forfeit=int(reason == "no_show"),
        session_ended=int(reason in ("collected", "abandoned")),
        car_contact_steps=int(f.get("car_contact_steps") or 0),
        transient_api_errors=sum(int(v) for v in transient.values()),
        unrecovered_api_errors=sum(int(v) for v in unrecovered.values()),
        sim_lag_late_max_s=float(lag.get("late_max_s", 0.0)),
        sim_lag_late_p99_s=float(lag.get("late_p99_s", 0.0)),
        sim_effective_slowdown=float(lag.get("run_wall_s", 0.0)) / max(1e-9, float(lag.get("run_sim_s", 0.0)))
        if lag.get("run_sim_s")
        else 0.0,
    )
    if fmt == "match":  # the scripted car (no seat of its own)
        out["B_won"] = int(valid and winner == "opponent")
    for s in cars:
        st = stats.get(s) or {}
        for k, v in (scores.get(s) or {}).items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                out[f"{s}_{k}"] = v
        out[f"{s}_wall_contact_steps"] = int((f.get("wall_steps") or {}).get(s, 0))
        out[f"{s}_dropped_acts"] = int(st.get("dropped", 0))
        out[f"{s}_proxy_error"] = int(bool((d.get("proxy_error") or {}).get(s)))
        out[f"{s}_meter_error"] = int(bool((d.get("meter_error") or {}).get(s)))
        if s not in seats:  # the scripted car: no API, no seat counters
            out[f"{s}_transient_api_errors"] = out[f"{s}_unrecovered_api_errors"] = 0
        u = _usage(s) if s in seats else {}
        for k in ("requests", "input_tokens", "output_tokens"):
            out[f"{s}_metered_{k}"] = int(u.get(k, 0) or 0)
        out[f"{s}_n_observe"] = int(st.get("observations", 0))
        out[f"{s}_n_act"] = int(st.get("attempts", 0))
    if fmt == "race":
        out["winner_seat_index"] = cars.index(car_winner) if valid and car_winner in cars else -1
        for i, r in enumerate(results):
            for s, p in (r.get("positions") or {}).items():
                out[f"heat{i + 1}_{s}_position"] = int(p)
    return out
