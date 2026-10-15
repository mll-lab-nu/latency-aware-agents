"""Tables of the render benchmark (bench.py results/*.jsonl) under the match's time model: control step 10 ms of game
time, slowdown S (10 by default: one control step every 100 ms of wall time), one observation in flight per seat,
drawn by the seat's render worker. Per configuration: render and PNG time per observation (medians), the cost C =
render + PNG, the fastest observation rate (1 / C wall, S / C in game time), and the render worker's busy share when
the observations come at 30 / 60 / 75 Hz of game time (75 Hz: the real SSL vision, ZJUNlict 2020 ETDP; TIGERs 2026:
about 70 frames per second).

    python3 analyze.py [results dir] [--slowdown 10] [--glob 'bench_stub_*.jsonl']"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir", nargs="?", default=str(Path(__file__).resolve().parent / "results"))
    ap.add_argument("--slowdown", type=float, default=10.0)
    ap.add_argument("--glob", default="bench_*.jsonl")
    a = ap.parse_args()
    rows = []
    for f in sorted(glob.glob(str(Path(a.dir) / a.glob))):
        for line in open(f):
            rows.append(json.loads(line))
    S = a.slowdown
    print(f"slowdown {S:g}: a control step every {10 * S:g} ms wall; obs at 30 / 60 / 75 Hz game = every "
          f"{1000 * S / 30:.0f} / {1000 * S / 60:.0f} / {1000 * S / 75:.0f} ms wall\n")
    print("| tag | layout | tile | cams | MSAA | extra | render ms (p90) | PNG ms | PNG KB | C ms | max obs/s wall | "
          "max Hz game | busy @30 | @60 | @75 | peak MB |")
    print("|" + "---|" * 16)
    for r in rows:
        msaa = 0 if r.get("offsamples") == 0 else 4
        extra = ("shadows" if r.get("shadows") else "") + ("core renderer" if r.get("path") == "core" else "")
        c = r["render_ms"]["median"] + r["png_ms"]["median"]
        busy = [c / (1000 * S / hz) for hz in (30, 60, 75)]
        print(f"| {r['tag']} | {r['layout']} | {r['size'][0]}x{r['size'][1]} | {r['cams']} | {msaa} | {extra} | "
              f"{r['render_ms']['median']:.0f} ({r['render_ms']['p90']:.0f}) | {r['png_ms']['median']:.0f} | "
              f"{r['png_kb']:.0f} | {c:.0f} | {1000 / c:.1f} | {1000 * S / c:.0f} | "
              + " | ".join(f"{100 * b:.0f}%" for b in busy) + f" | {r['rss_mb']['peak']:.0f} |")


if __name__ == "__main__":
    main()
