"""Summaries of results/<tag>/{ref,oracle,cam}_*.jsonl (host side, stdlib).   python3 summarize.py tag [tag ...]
Writes results/summary.json (all tags found when no tag is given) and prints a table per tag."""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

R = Path(__file__).resolve().parent / "results"
TYPES = ("short", "mid", "long-flat", "long-high")


def load(tag, kind):
    out = {}
    for f in sorted((R / tag).glob(f"{kind}_*.jsonl")):
        for l in f.read_text().splitlines():
            if l.strip():
                r = json.loads(l)
                out[r["plan"]["i"]] = r
    return out


def fault_kind(r):
    f = r.get("fault", "")
    if f.startswith("not producible"):
        return "not producible (head-speed cap)"
    if "serve_fault:" in f:
        return f.split("serve_fault:")[1].split(":")[0]
    if "outside the service court" in f:
        return "out (lands outside the service court)"
    return f[:40]


def summarize(tag):
    s = {"tag": tag}
    for kind in ("ref", "oracle", "cam"):
        recs = load(tag, kind)
        if not recs:
            continue
        legal = {i: r for i, r in recs.items() if r.get("legal")}
        d = {"n": len(recs), "legal": len(legal), "knobs": next(iter(recs.values())).get("knobs")}
        d["faults"] = dict(Counter(fault_kind(r) for r in recs.values() if not r.get("legal")))
        by = defaultdict(lambda: [0, 0])
        sp = defaultdict(list)
        for r in legal.values():
            t = r["type"]
            by[t][0] += 1
            by[t][1] += bool(r.get("returned"))
            sp[t].append(r["flight"]["speed"])
        d["by_type"] = {t: {"legal": by[t][0], "returned": by[t][1],
                            "speed": [min(sp[t]), max(sp[t])] if sp[t] else None} for t in TYPES if by[t][0]}
        d["returned"] = sum(bool(r.get("returned")) for r in legal.values())
        d["frac"] = round(d["returned"] / len(legal), 4) if legal else None
        sp_all = [r["flight"]["speed"] for r in legal.values()]
        d["speed"] = [min(sp_all), max(sp_all)] if sp_all else None
        d["by_family"] = {f: {"attempted": sum(1 for r in recs.values() if r["plan"]["fam"] == f),
                              "legal": sum(1 for r in legal.values() if r["plan"]["fam"] == f)}
                          for f in ("low", "mid", "high")}
        unret = []
        for i, r in sorted(legal.items()):
            if not r.get("returned"):
                unret.append({"i": i, "plan": r["plan"], "type": r["type"], "speed": r["flight"]["speed"],
                              "land": r["flight"]["land"], "t_flight": r["flight"]["t_flight"],
                              "cause": r.get("cause") or r.get("rx_outcome")})
        d["unreturned"] = unret
        if kind == "oracle":
            d["wall_s"] = round(sum(r.get("wall_s", 0) for r in recs.values()))
        if kind != "oracle":
            d["rx_outcomes"] = dict(Counter(r.get("rx_outcome") for r in legal.values()))
        s[kind] = d
    return s


def main():
    tags = sys.argv[1:] or sorted(p.name for p in R.iterdir() if p.is_dir() and p.name not in ("logs",))
    allr = {}
    f = R / "summary.json"
    if f.exists():
        allr = json.loads(f.read_text())
    for t in tags:
        s = summarize(t)
        allr[t] = s
        print(f"== {t}")
        for kind in ("ref", "cam", "oracle"):
            d = s.get(kind)
            if not d:
                continue
            print(f"  {kind}: {d['returned']}/{d['legal']} legal returned ({d['frac']}), served {d['n']}, faults {d['faults']}")
            print(f"     by type {d['by_type']}")
            for u in d["unreturned"][:12]:
                print(f"     X {u['i']} {u['plan']['fam']} ({u['plan']['x']},{u['plan']['y']}) {u['type']} {u['speed']} m/s "
                      f"land {u['land']} t {u['t_flight']}: {u['cause']}")
    f.write_text(json.dumps(allr, indent=1))


if __name__ == "__main__":
    main()
