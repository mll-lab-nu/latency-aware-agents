#!/usr/bin/env python3
"""Provenance of bodies/unitree_g1/policy/agile_g1_velocity_height.npz: the weights and constants of NVIDIA WBC-AGILE's
released G1 velocity-height policy (nvidia-isaac/WBC-AGILE, Apache-2.0, the LEAPP bundle
agile/data/policy/velocity_height_g1/leapp/Velocity-Height-G1-History-v0: .onnx + .yaml; body.toml [walk.agile], S9)
as plain arrays, so that the simulator runs it with numpy (agile.py) and needs no onnxruntime. The graph: the
observation terms computed from the robot state, each term's 5-frame history rolled and the new frame appended, the
histories concatenated, Gemm + ELU three times and a Gemm (the action), targets = clip(offset + scale x action). Kept
unchanged (float32): the Gemm weights and biases (w0, b0, ..., w3, b3), the constants the graph uses (joint defaults,
joint speed scale, action clip, action scale / offset, target clip, gravity) and the gain outputs (kp, kd), plus the
bundle's joint order (obs_joints, action_joints, from its .yaml) and a parity fixture: graph inputs with histories
(parity_<input>) and onnxruntime's outputs for them (parity_<output>), which agile.Agile.parity() reproduces.

    python -I tools/export_agile.py PATH/TO/Velocity-Height-G1-History-v0.onnx           write the npz (needs onnx,
                                                                                         onnxruntime, numpy, pyyaml)
    python -I tools/export_agile.py PATH/TO/Velocity-Height-G1-History-v0.onnx --check   agile.py vs onnxruntime:
        1000 random graph evaluations (every output), and 200 ticks of agile.Agile.step() in Unitree's joint order,
        onnxruntime fed back its own outputs as the bundle's feedback_flow says (history started by Isaac Lab's rule);
        and the npz's weights, constants and joint order equal to what the export takes from the bundle

The ONNX file must be the one body.toml names (onnx_sha256), its .yaml beside it; body.toml's gains must be the graph's.
The export is reproducible (same bytes every time with the same onnxruntime build: the fixture holds its outputs).
"""

from __future__ import annotations

import hashlib
import sys
import tomllib
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
TERMS = ("velocity_height_commands", "base_ang_vel", "projected_gravity", "joint_pos", "joint_vel", "actions")
HIST = ["Slice", "Slice", "Concat", "Reshape", "Unsqueeze", "ScatterND", "Transpose"]  # roll a history, append a frame
CROSS = ["Mul", "Mul", "Sub"] * 3 + ["Concat"]
GRAPH = (
    HIST * 2
    + ["Slice", "Slice", "Split", "Split"]
    + CROSS
    + ["Mul", "Mul", "Sub", "Split"]
    + CROSS
    + ["Add"]
    + HIST
    + ["Sub"]
    + HIST
    + ["Sub", "Mul"]
    + HIST
    + ["Clip"]
    + HIST
    + ["Reshape"] * 6
    + ["Concat"]
    + ["Gemm", "Elu"] * 3
    + ["Gemm"]
    + ["Mul", "Add", "Max", "Min"]
)
FIXTURE = 8  # input sets in the parity fixture
TOL = 1e-5  # onnxruntime's float32 rounding against the float64 port, per element, relative to max(1, |output|)
FIXTURE_TOL = 1e-4  # agile.Agile.parity() on the stored fixture (absolute; conformance 3b's bar)


def rel(got, ref) -> float:
    """max |got - ref| / max(1, |ref|), element by element."""
    ref = np.asarray(ref, float)
    return float(np.max(np.abs(np.asarray(got, float) - ref) / np.maximum(1.0, np.abs(ref))))


def arrays(onnx_path: Path) -> dict:
    """The weights and constants, checked against the graph's structure."""
    import onnx
    from onnx import numpy_helper

    g = onnx.load(str(onnx_path)).graph
    nodes = g.node
    assert [n.op_type for n in nodes] == GRAPH, [n.op_type for n in nodes]
    init = {t.name: numpy_helper.to_array(t) for t in g.initializer}
    made = {o: n for n in nodes for o in n.output}
    attr = lambda n: {a.name: onnx.helper.get_attribute_value(a) for a in n.attribute}  # noqa: E731
    for k, t in enumerate(TERMS):  # each history: [h[1:], h[0:1]], the new frame scattered into the last slot, and
        sc = made[f"h_policy_{t}_out"]  # the histories concatenated in TERMS order (oldest frame first)
        assert sc.op_type == "ScatterND" and init[sc.input[1]].tolist() == [[-1]], t
        roll = made[sc.input[0]]
        first, rest = (made[i] for i in roll.input[::-1])
        assert [int(init[i][0]) for i in first.input[1:3]] == [0, 1] and int(init[rest.input[1]][0]) == 1, t
        assert first.input[0] == rest.input[0] == f"h_policy_{t}_in" and attr(roll)["axis"] == 0, t
        view = made[nodes[81].input[k]]
        assert made[view.input[0]].input[0] == sc.output[0] and attr(made[view.input[0]])["perm"] == [1, 0, 2], t
    n = {i: nodes[i] for i in (17, 28, 50, 58, 59, 67, 89, 90, 91, 92)}
    assert list(n[50].input[:1]) == ["robot_joint_pos"] and list(n[58].input[:1]) == ["robot_joint_vel"]
    assert n[59].input[0] == n[58].output[0] and n[67].input[0] == "last_action_in"
    assert n[89].input[0] == "last_action_out" and float(init[n[28].input[1]]) == 2.0
    lo, hi = float(init[n[67].input[1]]), float(init[n[67].input[2]])
    assert lo == -hi
    gemm = [x for x in nodes if x.op_type == "Gemm"]
    for x in nodes:
        a = attr(x)
        assert x.op_type != "Gemm" or (
            a.get("transB") == 1 and a.get("transA", 0) == 0 and a.get("alpha", 1.0) == a.get("beta", 1.0) == 1.0
        )
        assert x.op_type != "Elu" or a.get("alpha", 1.0) == 1.0
    assert gemm[3].output[0] == "last_action_out"
    out = {f"{wb}{k}": init[x.input[1 + j]].astype(np.float32) for k, x in enumerate(gemm) for j, wb in enumerate("wb")}
    c = lambda i, j=1: init[n[i].input[j]].astype(np.float32).reshape(-1)  # noqa: E731
    return out | {
        "joint_pos_default": c(50),
        "joint_vel_default": c(58),
        "joint_vel_scale": c(59).reshape(()),
        "action_clip": np.float32(hi),
        "action_scale": c(89),
        "action_offset": c(90),
        "target_min": c(91),
        "target_max": c(92),
        "gravity": init[n[17].input[0]].astype(np.float32).reshape(-1),
        "kp": init["joint_pos_kp_gains"].reshape(-1),
        "kd": init["joint_pos_kd_gains"].reshape(-1),
    }


def bundle(onnx_path: Path, A: dict) -> dict:
    """The joint order from the bundle's .yaml, checked against body.toml and the graph's conventions."""
    import yaml

    desc = yaml.safe_load(onnx_path.with_suffix(".yaml").read_text())
    ((name, M),) = desc["models"].items()
    ins, outs = ({x["name"]: x for x in M[k]} for k in ("inputs", "outputs"))
    assert M["parameters"]["sha256sum"] == A["onnx_sha256"]
    assert ins["robot_root_quat_w"]["element_names"][0] == ["qx", "qy", "qz", "qw"]
    joints = ins["robot_joint_pos"]["element_names"][0]
    legs = outs["joint_pos"]["element_names"][0]
    assert joints == ins["robot_joint_vel"]["element_names"][0] and len(joints) == 29
    assert legs == outs["joint_pos_kp_gains"]["element_names"][0] == outs["joint_pos_kd_gains"]["element_names"][0]
    assert all(ins[f"h_policy_{t}_in"]["shape"][0] == A["history"] for t in TERMS)
    assert float(desc["pipeline"]["configs"]["frequency"]) == A["rate_hz"]
    fb = {k.split("/")[1]: [v.split("/")[1] for v in vs] for k, vs in desc["pipeline"]["feedback_flow"].items()}
    assert fb == {o: [o[: -len("_out")] + "_in"] for o in [f"h_policy_{t}_out" for t in TERMS] + ["last_action_out"]}
    return {"obs_joints": np.array(joints), "action_joints": np.array(legs)}


def session(onnx_path: Path):
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.intra_op_num_threads = so.inter_op_num_threads = 1
    return ort.InferenceSession(str(onnx_path), sess_options=so, providers=["CPUExecutionProvider"])


def inputs(n: int, seed: int, history: int) -> dict:
    """n random input sets (float32): every term N(0, 1), unit quaternions, joint speeds x 3, last actions within
    +-15 (beyond the graph's clip), histories N(0, 1)."""
    r = np.random.default_rng(seed)
    q = r.normal(0.0, 1.0, (n, 1, 4))
    x = {
        "base_velocity": r.normal(0.0, 1.0, (n, 1, 4)),
        "robot_root_ang_vel_b": r.normal(0.0, 1.0, (n, 1, 3)),
        "robot_root_quat_w": q / np.linalg.norm(q, axis=-1, keepdims=True),
        "robot_joint_pos": r.normal(0.0, 1.0, (n, 1, 29)),
        "robot_joint_vel": r.normal(0.0, 3.0, (n, 1, 29)),
        "last_action_in": r.uniform(-15.0, 15.0, (n, 1, 12)),
    }
    for t, k in zip(TERMS, (4, 3, 3, 29, 29, 12)):
        x[f"h_policy_{t}_in"] = r.normal(0.0, 1.0, (n, history, 1, k))
    return {k: v.astype(np.float32) for k, v in x.items()}


def run(s, x: dict) -> dict:
    """onnxruntime on each input set: every output, stacked."""
    names = [o.name for o in s.get_outputs()]
    rows = [s.run(None, {k: v[i] for k, v in x.items()}) for i in range(len(x["base_velocity"]))]
    return {k: np.stack([r[j] for r in rows]) for j, k in enumerate(names)}


def squeeze(a: np.ndarray) -> np.ndarray:
    """(n, 1, k) -> (n, k); (n, 5, 1, k) -> (n, 5, k)."""
    return a.reshape(a.shape[0], -1, a.shape[-1]).squeeze(1) if a.ndim == 3 else a.reshape(a.shape[0], a.shape[1], -1)


def closed_loop(src: Path, p, b: dict, ticks: int = 200, seed: int = 2) -> float:
    """agile.Agile.step() (Unitree's joint order, quaternion w, x, y, z) against onnxruntime fed back its own outputs
    (inputs reordered by the .yaml's names, history started with the first frame, last action 0): max |difference| of
    the leg targets over the ticks and of the memory at the end, relative (rel)."""
    sys.path.insert(0, str(HERE.parents[1]))
    from bodies.unitree_g1.agile import TERMS as T, UNITREE

    assert T == TERMS
    s = session(src)
    names = [o.name for o in s.get_outputs()]
    ix = np.array([UNITREE.index(j) for j in b["obs_joints"]])
    legs = list(b["action_joints"])
    r = np.random.default_rng(seed)
    p.reset()
    mem, err = None, 0.0
    for _ in range(ticks):
        q, dq = r.normal(0.0, 0.5, 29), r.normal(0.0, 2.0, 29)
        quat, gyro = r.normal(0.0, 1.0, 4), r.normal(0.0, 1.0, 3)
        quat /= np.linalg.norm(quat)
        cmd = r.uniform(p.cmd_min, p.cmd_max)
        cmd *= max(1.0, 1.01 * p.cmd_min_norm / np.linalg.norm(cmd))  # the training ranges, not zeroed
        cmd = np.clip(cmd, p.cmd_min, p.cmd_max)
        ext = {
            "base_velocity": np.r_[cmd, p.height],
            "robot_root_ang_vel_b": gyro,
            "robot_root_quat_w": np.r_[quat[1:], quat[0]],
            "robot_joint_pos": q[ix],
            "robot_joint_vel": dq[ix],
        }
        ext = {k: v.astype(np.float32)[None] for k, v in ext.items()}
        if mem is None:  # Isaac Lab's history buffer after a reset: the first frame fills it
            zero = {
                f"h_policy_{t}_in": np.zeros((p.history_len, 1, k), np.float32)
                for t, k in zip(TERMS, (4, 3, 3, 29, 29, 12))
            }
            o = dict(zip(names, s.run(None, ext | zero | {"last_action_in": np.zeros((1, 12), np.float32)})))
            mem = {f"h_policy_{t}_in": np.repeat(o[f"h_policy_{t}_out"][-1:], p.history_len, 0) for t in TERMS}
            mem["last_action_in"] = np.zeros((1, 12), np.float32)
        o = dict(zip(names, s.run(None, ext | mem)))
        mem = {k[: -len("_out")] + "_in": v for k, v in o.items() if k.startswith(("h_policy_", "last_action"))}
        got = p.step(quat, gyro, q, dq, cmd)
        err = max(err, rel(got, o["joint_pos"].reshape(-1)[[legs.index(j) for j in UNITREE[:12]]]))
    ref = np.r_[mem["last_action_in"].ravel(), np.concatenate([mem[f"h_policy_{t}_in"].ravel() for t in TERMS]), 0.0]
    return max(err, rel(p.state(), ref))


def main(src: str, check: bool = False):
    decl = tomllib.loads((HERE / "body.toml").read_text())
    A = decl["walk"]["agile"]
    src = Path(src)
    sha = hashlib.sha256(src.read_bytes()).hexdigest()
    if sha != A["onnx_sha256"]:
        sys.exit(f"{src}: sha256 {sha}, body.toml [walk.agile] onnx_sha256 is {A['onnx_sha256']}")
    a, b = arrays(src), bundle(src, A)
    legs = list(b["action_joints"])
    sys.path.insert(0, str(HERE.parents[1]))
    from bodies.unitree_g1.body import LEGS

    u = [legs.index(j) for j in LEGS]
    if a["kp"][u].tolist() != np.float32(A["kp"]).tolist() or a["kd"][u].tolist() != np.float32(A["kd"]).tolist():
        sys.exit(f"body.toml [walk.agile] kp / kd are not the graph's: {a['kp'][u]} / {a['kd'][u]}")
    if not check:
        x = inputs(FIXTURE, 0, A["history"])
        y = run(session(src), x)
        fixture = {f"parity_{k}": squeeze(v) for k, v in (x | y).items() if not k.startswith("joint_pos_k")}
        np.savez(HERE / A["weights"], source_sha256=np.array(sha), **a, **b, **fixture)
        print("wrote", A["weights"], "from", src, sha)
        return
    from bodies.unitree_g1.agile import INPUTS, Agile

    p = Agile(decl)
    x = inputs(1000, 1, A["history"])
    y = {k: squeeze(v) for k, v in run(session(src), x).items()}
    x = {k: squeeze(v) for k, v in x.items()}
    err = 0.0
    for i in range(1000):
        ins = [x[k][i] for k in INPUTS + ("last_action_in",)] + [tuple(x[f"h_policy_{t}_in"][i] for t in TERMS)]
        tg, act, hist = p.graph(*ins)
        ref = [y["joint_pos"][i], y["last_action_out"][i]] + [y[f"h_policy_{t}_out"][i] for t in TERMS]
        err = max(err, max(rel(g, r) for g, r in zip([tg, act, *hist], ref)))
    loop = closed_loop(src, p, b)
    fx = p.parity()
    z = np.load(HERE / A["weights"])
    same = all(np.array_equal(z[k], v) for k, v in (a | b).items()) and str(z["source_sha256"]) == sha
    ok = err < TOL and loop < TOL and fx < FIXTURE_TOL and same
    print(
        f"1000 graph evaluations: max |numpy - onnxruntime| / max(1, |output|) = {err:.3g}; 200 ticks of step() fed "
        f"back: {loop:.3g}; fixture {fx:.3g}; npz = the bundle's arrays: {same}: {'PASS' if ok else 'FAIL'}"
    )
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main(sys.argv[1], "--check" in sys.argv[2:])
