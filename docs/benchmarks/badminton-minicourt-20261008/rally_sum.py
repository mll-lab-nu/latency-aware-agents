"""Rally statistics of results/<tag>/duel_*.jsonl (host side, stdlib):  python3 rally_sum.py tag [tag ...]"""
import collections, glob, json, statistics, sys
from pathlib import Path
R = Path(__file__).resolve().parent / "results"
out = {}
for tag in sys.argv[1:]:
    recs = {}
    for f in sorted(glob.glob(str(R / tag / "duel_*.jsonl"))):
        for l in open(f):
            r = json.loads(l)
            recs.setdefault((r["seed"], r.get("first", "A")), r)
    hpp = [h for r in recs.values() for h in r["hits_per_point"]]
    how = collections.Counter()
    for r in recs.values():
        for h in r["how"]:
            k = ("serve_fault:" + h.split("serve_fault:")[1].split(":")[0]) if "serve_fault:" in h else \
                "unreturned serve" if "serve landed in" in h else "net" if "net" in h else \
                "own side" if "own side" in h else "landed in (not returned)" if "landed in" in h else \
                "out" if "out" in h else "other"
            how[k] += 1
    sp = [x["speed"] for r in recs.values() for x in r["hits"]]
    wins = collections.Counter(("A" if r["score"]["A"] > r["score"]["B"] else "B") for r in recs.values())
    d = dict(matches=len(recs), points=len(hpp), hits_per_point=round(statistics.mean(hpp), 2) if hpp else None,
             median=statistics.median(hpp) if hpp else None, max=max(hpp) if hpp else None,
             share_ge3=round(sum(h >= 3 for h in hpp) / max(1, len(hpp)), 2),
             share_ge4=round(sum(h >= 4 for h in hpp) / max(1, len(hpp)), 2),
             dist=dict(sorted(collections.Counter(hpp).items())), how=dict(how.most_common()),
             shot_speed_median=round(statistics.median(sp), 2) if sp else None, wins=dict(wins))
    out[tag] = d
    print(tag, json.dumps(d))
(R / "rally_summary.json").write_text(json.dumps(out, indent=1))
