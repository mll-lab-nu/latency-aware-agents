#!/usr/bin/env python3
"""Levels of detail of the library's robots (the players' images, renderer rule 6; user decision D1, 2026-10-08): a
robot's fine meshes (meshes_fine/, the unchanged upstream meshes, not in git) decimated to a triangle budget and
committed as meshes_<level>/ (e.g. meshes_25k/), so that a sport draws its players' images of the robot with the
appearance of the real robot at a small cost (a body's mount(..., lod="25k")).

    python -m bodies.lod build ROBOT [LEVEL]          e.g. build unitree_g1 25k  (default 25k)

The same method as renderers/presentation/lod.py: every visual mesh of model.xml decimated by the same ratio
(quadric error metrics, up to four passes where one stops short; a mesh under MIN_FACES keeps at least MIN_FACES triangles, and the ratio is chosen so that
the total, floors included, meets the budget), binary STL (vertices and faces
only, zero normals: MuJoCo computes them; the same bytes for the same mesh), and meshes_<level>/lod.json with the
tools' versions, the triangle counts and every file's sha256 (and the source file's). The levels are derived works of
the upstream meshes under the robot's licence (its LICENSE, NOTICE). Building needs
`pip install trimesh==5.1.1 fast-simplification==0.2.0` and the robot's meshes_fine/ (its tools/make_model.py); a
developer step, never run by a match.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
import tomllib
import xml.etree.ElementTree as ET
from pathlib import Path

HERE = Path(__file__).resolve().parent
MIN_FACES = 64
TOOLS = {"trimesh": "5.1.1", "fast-simplification": "0.2.0"}


def target_of(level: str) -> int:
    m = re.fullmatch(r"(\d+)k", level)
    if not m:
        raise ValueError(f"a level is <N>k (N thousand triangles per robot), not {level!r}")
    return int(m.group(1)) * 1000


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def sources(robot: str) -> dict[str, Path]:
    """model.xml's mesh stems -> their fine (upstream) files in meshes_fine/"""
    d = HERE / robot
    decl = tomllib.loads((d / "body.toml").read_text())
    fine = d / decl["body"]["meshes"]["fine"]
    files = {p.stem.lower(): p for p in fine.iterdir() if p.is_file()} if fine.is_dir() else {}
    out = {}
    for m in ET.parse(d / decl["body"]["model"]).getroot().find("asset").findall("mesh"):
        stem = Path(m.get("file")).stem
        if stem.lower() not in files:
            raise FileNotFoundError(f"{robot}: {stem} not in {fine} (run tools/make_model.py)")
        out[stem] = files[stem.lower()]
    return out


def build(robot: str, level: str = "25k", log=print) -> dict:
    import numpy as np
    import fast_simplification as fs
    import trimesh
    target = target_of(level)
    src = sources(robot)
    meshes = {}
    for stem, f in sorted(src.items()):
        t = trimesh.load(f, force="mesh", process=False, skip_materials=True)
        t.merge_vertices(merge_tex=True, merge_norm=True)
        meshes[stem] = (f, t)
    total = sum(len(t.faces) for _, t in meshes.values())
    ratio = min(1.0, target / total)
    for _ in range(20):                    # the floor of MIN_FACES taken into account: the total meets the budget
        floored = [len(t.faces) for _, t in meshes.values() if len(t.faces) * ratio < MIN_FACES]
        fixed = sum(min(n, MIN_FACES) for n in floored)
        rest = total - sum(floored)
        if rest <= 0:
            break
        ratio = min(1.0, max(target - fixed, 0) / rest)
    out = HERE / robot / f"meshes_{level}"
    out.mkdir(exist_ok=True)
    for old in out.glob("*.stl"):
        old.unlink()
    n_out, files = 0, {}
    for stem, (f, t) in meshes.items():
        nf = len(t.faces)
        keep = max(MIN_FACES, int(round(nf * ratio)))
        v, fc = np.asarray(t.vertices, np.float32), np.asarray(t.faces, np.int32)
        for _ in range(4):                 # a pass may stop short (open or non-manifold parts): pass again
            if len(fc) <= keep * 1.02:
                break
            v2, f2 = fs.simplify(v, fc, target_reduction=1 - keep / len(fc))
            if len(f2) >= len(fc):
                break
            v, fc = np.asarray(v2, np.float32), np.asarray(f2, np.int32)
        t = trimesh.Trimesh(v, fc, process=False)
        n_out += len(t.faces)
        dst = out / f"{stem}.stl"
        dst.write_bytes(_stl(t))
        files[dst.name] = dict(sha256=_sha(dst), triangles=int(len(t.faces)), source=f.name, source_sha256=_sha(f))
    man = dict(robot=robot, level=level, note=__doc__.split("\n\n")[0].replace("\n", " "), tools=TOOLS,
               target_triangles=target, source_triangles=int(total), triangles=int(n_out), files=files)
    (out / "lod.json").write_text(json.dumps(man, indent=1, sort_keys=True) + "\n")
    size = sum((out / k).stat().st_size for k in files)
    log(f"{robot} {level}: {total} -> {n_out} triangles, {len(files)} files, {size / 1e6:.2f} MB")
    return man


def _stl(t) -> bytes:
    """Binary STL (zero normals: MuJoCo computes them), the same bytes for the same mesh."""
    import numpy as np
    tri = np.asarray(t.vertices, np.float32)[np.asarray(t.faces, int)]
    rec = np.zeros(len(tri), dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
    rec["v"] = tri
    return b"\0" * 80 + np.uint32(len(tri)).astype("<u4").tobytes() + rec.tobytes()


def main(argv) -> int:
    if not argv or argv[0] != "build" or len(argv) not in (2, 3):
        print(__doc__)
        return 2
    build(argv[1], argv[2] if len(argv) == 3 else "25k")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
