#!/usr/bin/env python3
"""Provenance of bodies/unitree_g1/policy/falcon_g1_29dof.npz: the weights of FALCON's released G1 policy
(LeCAR-Lab/FALCON sim2real/models/falcon/g1_29dof.onnx, MIT; body.toml [walk.falcon], S8) as plain arrays, so that
the simulator runs it with numpy (falcon.py) and needs no onnxruntime. The graph: two actors (lower_body: legs +
waist, upper_body: arms), each Gemm + ELU three times and a Gemm, their outputs concatenated; the arrays keep its
names (lower_w0, lower_b0, ..., upper_b3; float32, unchanged). With them: a parity fixture (parity_obs: a few
observations, parity_action: onnxruntime's outputs for them) that falcon.Falcon.parity() reproduces.

    python -I tools/export_falcon.py PATH/TO/g1_29dof.onnx           write the npz (needs onnx, onnxruntime, numpy)
    python -I tools/export_falcon.py PATH/TO/g1_29dof.onnx --check   falcon.py vs onnxruntime on 1000 observations

The ONNX file must be the one body.toml names (onnx_sha256). The export is reproducible (same bytes every time).
"""

from __future__ import annotations

import hashlib
import sys
import tomllib
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
GRAPH = (["Gemm", "Elu"] * 3 + ["Gemm"]) * 2 + ["Concat"]
FIXTURE = 8  # observations in the parity fixture
TOL = 1e-5  # onnxruntime's float32 rounding against the float64 port, relative to max(1, |action|)


def weights(onnx_path: Path) -> dict:
    """The Gemm weights and biases of the two actors, checked against the graph's structure."""
    import onnx
    from onnx import numpy_helper

    g = onnx.load(str(onnx_path)).graph
    assert [n.op_type for n in g.node] == GRAPH, [n.op_type for n in g.node]
    init = {t.name: numpy_helper.to_array(t) for t in g.initializer}
    gemm = [n for n in g.node if n.op_type == "Gemm"]
    for n in g.node:
        a = {x.name: onnx.helper.get_attribute_value(x) for x in n.attribute}
        assert n.op_type != "Gemm" or (a.get("transB") == 1 and a.get("alpha", 1.0) == a.get("beta", 1.0) == 1.0)
        assert n.op_type != "Elu" or a.get("alpha", 1.0) == 1.0
    assert list(g.node[-1].input) == [gemm[3].output[0], gemm[7].output[0]]  # action = [lower, upper]
    out = {}
    for k, n in enumerate(gemm):
        actor = "lower" if k < 4 else "upper"
        assert n.input[1].startswith(f"actors.{actor}_body."), n.input[1]
        out[f"{actor}_w{k % 4}"] = init[n.input[1]].astype(np.float32)
        out[f"{actor}_b{k % 4}"] = init[n.input[2]].astype(np.float32)
    return out


def observations(n: int, seed: int) -> np.ndarray:
    """n observations, every term N(0, 1) (float32)."""
    return np.random.default_rng(seed).normal(0.0, 1.0, (n, 575)).astype(np.float32)


def onnx_actions(onnx_path: Path, obs: np.ndarray) -> np.ndarray:
    import onnxruntime as ort

    s = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    name = s.get_inputs()[0].name
    return np.concatenate([s.run(None, {name: row[None]})[0] for row in obs]).astype(np.float32)


def main(src: str, check: bool = False):
    decl = tomllib.loads((HERE / "body.toml").read_text())
    F = decl["walk"]["falcon"]
    src = Path(src)
    sha = hashlib.sha256(src.read_bytes()).hexdigest()
    if sha != F["onnx_sha256"]:
        sys.exit(f"{src}: sha256 {sha}, body.toml [walk.falcon] onnx_sha256 is {F['onnx_sha256']}")
    if not check:
        obs = observations(FIXTURE, 0)
        arrays = weights(src) | {"parity_obs": obs, "parity_action": onnx_actions(src, obs)}
        np.savez(HERE / F["weights"], source_sha256=np.array(sha), **arrays)
        print("wrote", F["weights"], "from", src, sha)
        return
    sys.path.insert(0, str(HERE.parents[1]))
    from bodies.unitree_g1.falcon import Falcon

    p = Falcon(decl)
    obs = observations(1000, 1)
    ref = onnx_actions(src, obs)
    got = np.array([p.forward(o.astype(float)) for o in obs])
    err, top = float(np.max(np.abs(got - ref))), float(np.max(np.abs(ref)))
    ok = err < TOL * max(1.0, top) and p.parity() < TOL * max(1.0, float(np.max(np.abs(p.fixture[1]))))
    print(
        f"1000 observations: max |numpy - onnxruntime| = {err:.3g} (max |action| {top:.3g}); "
        f"fixture {p.parity():.3g}: {'PASS' if ok else 'FAIL'}"
    )
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main(sys.argv[1], "--check" in sys.argv[2:])
