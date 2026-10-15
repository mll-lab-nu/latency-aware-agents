"""Tables of the ABB B9 study (stdlib only): per result set the share of the library's serves returnable from each
stance, the best fixed stance (the B9 number: the receiver stands before it sees the serve), any stance, the stage
where the unreturnable ones stop, by serve class; the full simulation's impacts and self-collision records.

    python3 summarize.py [results dir]      -> results/summary.json and the markdown tables on stdout"""
import glob
import json
import sys
from collections import Counter
from pathlib import Path

R = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent / "results"
N_LIB = 284
STAGE_ZH = {"none": "无可回的击球（物理）", "time": "来不及", "reach": "够不着", "base": "龙门太慢", "pose": "击球姿态碰台",
            "arm": "手臂限位", "path": "挥拍路径碰台", "return": "散布下回不进", "ok": "可回"}


def load(prefix):
    rows = []
    for f in sorted(glob.glob(str(R / f"{prefix}_*.json"))):
        rows += json.loads(Path(f).read_text())
    return rows


def oracle_table(rows):
    if not rows:
        return None
    readies = sorted({k for r in rows for k in r["per"]})
    ok = {k: sum(1 for r in rows if r["per"].get(k, {}).get("stage") == "ok" or r["per"].get(k, {}).get("returned"))
          for k in readies}
    best = max(readies, key=lambda k: ok[k])
    anyk = sum(1 for r in rows if r.get("returnable"))
    stages = Counter()
    for r in rows:
        p = r["per"].get(best, {})
        if "stage" in p:
            stages[p["stage"]] += 1
        else:
            stages["ok" if p.get("returned") else (p.get("tries") or [{}])[-1].get("how", "no stroke passed the oracle")
                   if p.get("tries") else "no stroke passed the oracle"] += 1
    cls = Counter()
    clsok = Counter()
    for r in rows:
        c = r["cls"]
        key = c.get("length", "?")
        cls[key] += 1
        p = r["per"].get(best, {})
        clsok[key] += p.get("stage") == "ok" or bool(p.get("returned"))
    strict = None
    if rows and "tries" in next(iter(rows[0]["per"].values())):
        def good(p):
            return any(t["returned"] and (t.get("table_contact_n") or 0) <= 20.0 for t in p.get("tries", []))
        sk = {k: sum(1 for r in rows if good(r["per"].get(k, {}))) for k in readies}
        sb = max(readies, key=lambda k: sk[k])
        strict = dict(best_ready=sb, best_frac=round(sk[sb] / len(rows), 4))
    return dict(N=len(rows), strict_no_table_contact=strict,
                per_ready={k: round(ok[k] / len(rows), 4) for k in readies}, best_ready=best,
                best_frac=round(ok[best] / len(rows), 4), any_frac=round(anyk / len(rows), 4),
                stages_best=dict(stages), by_length={k: f"{clsok[k]}/{cls[k]}" for k in cls})


def fullsim_extras(rows):
    imp, sc, tab = [], [], 0
    for r in rows:
        for p in r["per"].values():
            for t in p.get("tries", []):
                if t.get("impact", {}).get("contact_ms"):
                    imp.append(t["impact"])
                if t.get("self_collision"):
                    sc.append(t["self_collision"])
                tab += (t.get("table_contact_n") or 0) > 0
    if not imp:
        return {}
    pk = sorted(i["peak_n"] for i in imp)
    ip = sorted(i["impulse_ns"] for i in imp)
    q = lambda a, f: a[min(len(a) - 1, int(f * len(a)))]
    return dict(strikes=len(imp), peak_force_n=dict(median=q(pk, 0.5), p95=q(pk, 0.95), max=pk[-1]),
                impulse_ns=dict(median=q(ip, 0.5), p95=q(ip, 0.95), max=ip[-1]),
                self_collision_events=sum(s["events"] for s in sc), self_collision_warnings=sum(s["warnings"] for s in sc),
                plays_with_robot_table_contact=tab)


def main():
    out = {}
    for f in sorted(glob.glob(str(R / "*_0.json"))):
        prefix = Path(f).name[:-len("_0.json")]
        rows = load(prefix)
        if rows and "returns" in rows[0]:              # rally_abb.py
            ret = [r["returns"] for r in rows]
            imp = sorted(i["peak_n"] for r in rows for i in r["impacts"])
            out[prefix] = dict(rallies=len(rows), returns=ret, total_returns=sum(ret), max_returns=max(ret),
                               rallies_with_2_or_more=sum(x >= 2 for x in ret), ends=dict(Counter(
                                   r["end"].split(" (")[0] for r in rows)),
                               impact_peak_n=dict(median=imp[len(imp) // 2], max=imp[-1]) if imp else None,
                               self_collision_events=sum(v["events"] for r in rows for v in r["self_collision"].values()),
                               rallies_with_robot_table_contact=sum(1 for r in rows if r["robot_table_n"] > 0))
            continue
        if not rows or "per" not in rows[0]:
            if rows and "reachable" in rows[0]:
                out[prefix] = dict(N=len(rows), reachable=round(sum(r["reachable"] for r in rows) / len(rows), 4))
            continue
        t = oracle_table(rows)
        if prefix.startswith("fs"):
            t.update(fullsim_extras(rows))
        out[prefix] = t
    (R / "summary.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    for k, v in out.items():
        print(k, json.dumps(v, ensure_ascii=False))


if __name__ == "__main__":
    main()
