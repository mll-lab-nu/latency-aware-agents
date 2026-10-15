#!/usr/bin/env python3
"""The model-free screening gates (docs/construction_zh.md step 4) over a batch of tools/screen matches.

    python3 tools/screen/gates.py SPORT --data DIR [--out report.md] [--json gates.json] [--boot 10000]

Reads DIR/runs/<variant>/L<level>_s<seed>/result.json (tools/screen/batch.py) and the sport's hooks
(sports/SPORT/screen/hooks.py: REFERENCE, REPLICAS, BASELINES, NECESSITY, SENSITIVITY, LEVELS; read without importing
them) and reports four gates, per level of the scripted opponent:

  reference      the reference player's wins and point share (P4: it completes meaningful matches; it should win the
                 default level)
  baselines      every simple baseline must lose the majority of its matches at every level (wins < n / 2);
                 the 95 % Wilson interval of its win rate is reported
  necessity      a program that uses only the non-core actions scores about nothing: point share <= --necessity-max
                 (default 0.05; the points it gets from the opponent's own faults count) and no win
  sensitivity    for each claimed capability, the two impairments (weak, strong) against the reference on the same
                 seeds: the mean paired difference of the point share with a 95 % bootstrap interval (resampled over
                 seeds); the replicas' difference is the noise floor. A capability passes a level when its strong
                 impairment's interval lies below 0 and the weak one does not score above the reference by more than
                 the noise floor; it is kept when it passes at least --keep-min levels (default: all levels but one)

Point share = points won / points played (the score, continuous); a match's result needs `summary.point_share` and
`summary.won` from the hooks' summary(), else the scores and winner of the result (seat A).
"""
from __future__ import annotations

import argparse
import ast
import json
import math
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def hooks_decl(sport: str) -> dict:
    tree = ast.parse((REPO / "sports" / sport / "screen" / "hooks.py").read_text())
    out = {}
    for n in tree.body:
        if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name):
            try:
                out[n.targets[0].id] = ast.literal_eval(n.value)
            except ValueError:
                pass
    return out


def load(data: Path) -> dict:
    """(variant, level) -> {seed: (point_share, won, result)}"""
    out: dict = {}
    for f in sorted((data / "runs").glob("*/*/result.json")):
        r = json.loads(f.read_text())
        s = r.get("summary") or {}
        sc = r.get("scores") or {}
        ps = s.get("point_share")
        if ps is None:
            a = sc.get("A", 0)
            tot = sum(v for v in sc.values() if isinstance(v, (int, float)))
            ps = a / tot if tot else 0.0
        won = s.get("won", r.get("winner") == "A")
        key = int(r["seed"]) if not r.get("first") else (int(r["seed"]), r["first"])   # duel: both first movers
        out.setdefault((r["variant"], r.get("level")), {})[key] = (float(ps), bool(won), r)
    return out


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def paired(a: dict, b: dict, boot: int, rng: random.Random) -> dict | None:
    """mean(b - a) over common seeds, 95 % bootstrap interval over seeds."""
    seeds = sorted(set(a) & set(b))
    if not seeds:
        return None
    d = [b[s][0] - a[s][0] for s in seeds]
    m = sum(d) / len(d)
    bs = sorted(sum(rng.choice(d) for _ in d) / len(d) for _ in range(boot))
    return {"n": len(d), "mean": m, "lo": bs[int(0.025 * boot)], "hi": bs[int(0.975 * boot) - 1]}


def fmt_ci(x: dict | None) -> str:
    return "—" if x is None else f"{x['mean']:+.3f} [{x['lo']:+.3f}, {x['hi']:+.3f}] (n={x['n']})"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("sport")
    ap.add_argument("--data", required=True, help="a batch directory, or several separated by commas")
    ap.add_argument("--out", default=None)
    ap.add_argument("--json", default=None)
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--necessity-max", type=float, default=0.05)
    ap.add_argument("--min-seeds", type=int, default=20, help="seeds per level a baseline cell needs (user rule: 20)")
    ap.add_argument("--keep-min", type=int, default=None)
    ap.add_argument("--levels", default=None, help="comma list (default: the hooks' LEVELS)")
    a = ap.parse_args(argv)
    H = hooks_decl(a.sport)
    R = {}
    for dd in a.data.split(","):              # several batch directories of the same revision
        for key, val in load(Path(dd)).items():
            R.setdefault(key, {}).update(val)
    levels = [int(x) for x in a.levels.split(",")] if a.levels else H.get("LEVELS", [None])
    ref = H.get("REFERENCE", "reference")
    rng = random.Random(0)
    L, J = [], {"reference": {}, "baselines": {}, "necessity": {}, "sensitivity": {}, "noise": {}}
    w = L.append

    w(f"# Model-free screening gates: {a.sport}\n\ndata: `{a.data}`\n")
    w("## Reference\n\n| level | wins | win rate [95 % CI] | mean point share |\n|---|---|---|---|")
    for lv in levels:
        x = R.get((ref, lv), {})
        k, n = sum(v[1] for v in x.values()), len(x)
        lo, hi = wilson(k, n)
        ps = sum(v[0] for v in x.values()) / n if n else float("nan")
        J["reference"][lv] = {"wins": k, "n": n, "ci": [lo, hi], "point_share": ps}
        w(f"| L{lv} | {k}/{n} | {k / max(1, n):.2f} [{lo:.2f}, {hi:.2f}] | {ps:.3f} |")

    rt = H.get("REFERENCE_TRUTH")
    if rt:
        w(f"\n## Opponent calibration: the truth-perception reference `{rt}` (wins the lowest level, falls with the level) "
          "and the camera reference (camera-only can win)\n\n| level | truth: wins (point share) | camera: wins (point share) |"
          "\n|---|---|---|")
        for lv in levels:
            xt, xc = R.get((rt, lv), {}), R.get((ref, lv), {})
            kt, nt = sum(v[1] for v in xt.values()), len(xt)
            kc, nc = sum(v[1] for v in xc.values()), len(xc)
            pt = sum(v[0] for v in xt.values()) / nt if nt else float("nan")
            pc = sum(v[0] for v in xc.values()) / nc if nc else float("nan")
            J.setdefault("calibration", {})[lv] = {"truth": [kt, nt, pt], "camera": [kc, nc, pc]}
            w(f"| L{lv} | {kt}/{nt} ({pt:.2f}) | {kc}/{nc} ({pc:.2f}) |")

    noise = {}
    for rep in H.get("REPLICAS", []):
        w(f"\n## Noise floor (replica `{rep}` − reference, point share)\n\n| level | difference [95 % CI] |\n|---|---|")
        for lv in levels:
            p = paired(R.get((ref, lv), {}), R.get((rep, lv), {}), a.boot, rng)
            noise.setdefault(lv, []).append(abs(p["mean"]) if p else 0.0)
            J["noise"].setdefault(rep, {})[lv] = p
            w(f"| L{lv} | {fmt_ci(p)} |")
    nf = {lv: max(v) if v else 0.0 for lv, v in noise.items()}

    truth_of = H.get("TRUTH_OF", {})           # baseline -> its truth-perception variant (baselines that perceive)
    min_seeds = a.min_seeds
    top = max(levels) if levels and levels[0] is not None else None
    w("\n## Simple baselines (the user's rule of 2026-10-11: each baseline with true perception first, >= %d seeds per "
      "level; it passes a level when it loses the majority there; a baseline that wins a level with true perception is "
      "re-run with the camera for that level and passes if it loses with the camera (\"this level mainly tests "
      "perception\"); at the highest formal level (L%s) it must lose with true perception)\n\n| baseline | "
      % (min_seeds, top) + " | ".join(f"L{lv} wins" for lv in levels) + " | pass |\n|---|" + "---|" * (len(levels) + 1))
    all_ok = True

    def cell(var, lv):
        x = R.get((var, lv), {})
        k, n = sum(v[1] for v in x.values()), len(x)
        ps = sum(v[0] for v in x.values()) / n if n else float("nan")
        return k, n, ps

    for b in H.get("BASELINES", []):
        cells, ok = [], True
        tv = truth_of.get(b, b)                # a baseline that does not perceive is its own truth version
        for lv in levels:
            k, n, ps = cell(tv, lv)
            lo, hi = wilson(k, n)
            rec = {"truth_variant": tv, "wins": k, "n": n, "ci": [lo, hi], "point_share": ps}
            enough = n >= min_seeds
            if n == 0:
                good, note = False, " (no runs)"
            elif k < n / 2:
                good, note = enough, "" if enough else f" (n<{min_seeds})"
            elif lv == top:
                good, note = False, " **FAIL** (wins the top level with true perception)"
            else:
                kc, nc, psc = cell(b, lv)
                rec.update(camera={"wins": kc, "n": nc, "point_share": psc})
                good = nc >= min_seeds and kc < nc / 2
                note = (f"; camera {kc}/{nc} ({psc:.2f}) - this level mainly tests perception" if good else
                        f"; camera {kc}/{nc} **FAIL**")
            rec["pass"] = good
            ok &= good
            J["baselines"].setdefault(b, {})[lv] = rec
            tag = "" if tv == b else " truth"
            cells.append(f"{k}/{n}{tag} ({ps:.2f}){note}")
        all_ok &= ok
        w(f"| {b} | " + " | ".join(cells) + f" | {'yes' if ok else 'NO'} |")
    w("\n(cells: wins/matches (mean point share); `truth` = the baseline with the ball's true position)")
    J["baselines_pass"] = all_ok

    w("\n## Skill necessity (gate: point share <= %.2f, no win)\n\n| program | " % a.necessity_max +
      " | ".join(f"L{lv}" for lv in levels) + " | pass |\n|---|" + "---|" * (len(levels) + 1))
    nec_ok = True
    for b in H.get("NECESSITY", []):
        cells, ok = [], True
        for lv in levels:
            x = R.get((b, lv), {})
            n = len(x)
            k = sum(v[1] for v in x.values())
            ps = sum(v[0] for v in x.values()) / n if n else float("nan")
            good = n > 0 and k == 0 and ps <= a.necessity_max
            ok &= good
            J["necessity"].setdefault(b, {})[lv] = {"wins": k, "n": n, "point_share": ps, "pass": good}
            cells.append(f"{ps:.3f}, {k}/{n} won" + ("" if good else " **FAIL**"))
        nec_ok &= ok
        w(f"| {b} | " + " | ".join(cells) + f" | {'yes' if ok else 'NO'} |")
    J["necessity_pass"] = nec_ok

    keep_min = a.keep_min if a.keep_min is not None else max(1, len(levels) - 1)
    w(f"\n## Capability sensitivity (impaired − reference, point share, paired seeds; a level passes when the strong "
      f"impairment's interval is below 0 and the weak one is not above the reference by more than the noise floor; "
      f"a capability is kept with >= {keep_min} of {len(levels)} levels)\n")
    w("| capability | impairment | " + " | ".join(f"L{lv}" for lv in levels) + " |\n|---|---|" + "---|" * len(levels))
    for cap, (weak, strong) in H.get("SENSITIVITY", {}).items():
        passed = 0
        per = {}
        for nm in (weak, strong):
            cells = []
            for lv in levels:
                p = paired(R.get((ref, lv), {}), R.get((nm, lv), {}), a.boot, rng)
                per[(nm, lv)] = p
                cells.append(fmt_ci(p))
            w(f"| {cap} | {nm} | " + " | ".join(cells) + " |")
        lv_ok = {}
        for lv in levels:
            ps, pw = per[(strong, lv)], per[(weak, lv)]
            ok = bool(ps and ps["hi"] < 0 and pw and pw["mean"] <= nf.get(lv, 0.0))
            lv_ok[lv] = ok
            passed += ok
        keep = passed >= keep_min
        J["sensitivity"][cap] = {"weak": weak, "strong": strong, "levels": {str(k): v for k, v in lv_ok.items()},
                                 "kept": keep,
                                 "diffs": {f"{k[0]}@L{k[1]}": v for k, v in per.items()}}
        w(f"| {cap} | **verdict** | " + " | ".join("pass" if lv_ok[lv] else "—" for lv in levels) +
          f" |  → {'KEPT' if keep else 'NOT KEPT'} ({passed}/{len(levels)})")
    duels = [x[2] for k, xs in R.items() for x in xs.values() if x[2].get("format") == "duel"]
    if duels:
        n = len(duels)
        fw = sum(1 for r in duels if r.get("winner") == r.get("first"))
        aw = sum(1 for r in duels if r.get("winner") == "A")
        w(f"\n## Fairness (format duel, the same policy on both seats, both first movers; N = {n})\n\n"
          "| advantage | wins | advantage (pp) [95 % CI] |\n|---|---|---|")
        for what, k in (("first mover", fw), ("seat A", aw)):
            lo, hi = wilson(k, n)
            J.setdefault("fairness", {})[what] = {"wins": k, "n": n, "pp": 100 * (k / n - 0.5),
                                                  "ci_pp": [100 * (lo - 0.5), 100 * (hi - 0.5)]}
            w(f"| {what} | {k}/{n} | {100 * (k / n - 0.5):+.1f} [{100 * (lo - 0.5):+.1f}, {100 * (hi - 0.5):+.1f}] |")
    text = "\n".join(L) + "\n"
    if a.out:
        Path(a.out).write_text(text)
    if a.json:
        Path(a.json).write_text(json.dumps(J, indent=1, default=str))
    print(text)


if __name__ == "__main__":
    main()
