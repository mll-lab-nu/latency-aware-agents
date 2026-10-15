"""hello's part of reward.json (the common part is core/grading.py): numbers only, from final.json alone (the
verifier has no simulator)."""


def evaluate(d):
    f = d.get("sport_fields") or {}
    throws = f.get("throws") or []
    out = {"balls": len(throws), "sudden_death_balls": sum(t["n"] >= 5 for t in throws)}
    for seat, side in (d.get("sides") or {}).items():
        mine = [t for t in throws if t["side"] == side]
        out[seat + "_points"] = int((d.get("scores") or {}).get(side, 0))
        out[seat + "_tens"] = sum(t["points"] == 10 for t in mine)
        out[seat + "_mean_miss_m"] = round(sum(t["r"] for t in mine) / len(mine), 4) if mine else 0.0
    return out
