"""Paired point-share differences (same seeds): paired.py BASE_TAG ABL_TAG [...pairs]; bootstrap 95 % CI."""
import json, sys, numpy as np
def sh(t):
    return {json.loads(l)["seed"]: (lambda r: r["score"][0] / max(1, sum(r["score"])))(json.loads(l)) for l in open(f"out/{t}.jsonl")}
args = sys.argv[1:]
D = []
for a, b in zip(args[::2], args[1::2]):
    A, B = sh(a), sh(b)
    d = [B[s] - A[s] for s in A if s in B]
    D += d
    print(f"{b:30s} - {a:30s}: n {len(d)} mean {np.mean(d):+.3f}")
D = np.array(D); rng = np.random.default_rng(0)
bs = [D[rng.integers(0, len(D), len(D))].mean() for _ in range(5000)]
print(f"pooled n {len(D)} mean {D.mean():+.3f} 95% CI [{np.percentile(bs,2.5):+.3f}, {np.percentile(bs,97.5):+.3f}]")
