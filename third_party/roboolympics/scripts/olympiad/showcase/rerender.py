#!/usr/bin/env python3
"""Re-render a finished RoboOlympics match with a renderer of your choice, in one command (host side; Docker does
the work). Matches themselves always run with the standard MuJoCo rendering; this tool only redraws a recorded match.

    python3 scripts/olympiad/showcase/rerender.py <trial dir> --sport tabletennis|tennis|soccer --out <dir>
        [--engine mujoco,mujoco-showcase,cycles,isaac] [--decor arena|stadium] [--t0 S] [--dur S] [--fps 50]
        [--k0 K --k1 K] [--sim <sim dir>] [--gpu 0|none] [--samples 64] [--stills 1.5,6]

Engines (comma-separated, default mujoco,mujoco-showcase):
  mujoco           the competition picture (MuJoCo rasterizer, the scene as the agents' match has it)
  mujoco-showcase  same rasterizer, better lights/materials/anti-aliasing           (GPU optional; CPU = OSMesa)
  cycles           Blender 4.2 Cycles path tracing                                  (needs an NVIDIA GPU)
  isaac            NVIDIA Isaac Sim 4.5 RTX path tracing (the robosuite/RoboCasa Omniverse route)  (needs an RTX GPU)
Every engine draws the same replayed states from the same broadcast camera, and every non-standard model passes
looks.same_scene() against the standard one (geometry, colours, cameras) before anything is drawn. Every frame of a
non-standard output is tagged "re-rendered: <engine>"; with --decor (promo-only surroundings outside the play
area, showcase engines only) also "re-rendered for presentation". If an engine's requirements are missing on this
machine, the tool stops before doing any work and prints what is needed.

<trial dir> is a Harbor trial directory of the match (…/<job>/<trial>/; its verifier/ holds trajectory.json /
episode.jsonl / replay_check.json). Nothing touches the trial dir. Step 1 (the sport's sim image, CPU) replays the
match with the match's own sim code and refuses unless the final state hash equals replay_check.json.
--sim defaults to tasks/<sport>-…/environment/sim of this checkout; its fingerprint must equal the match's.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
IMAGES = {"tabletennis": "ghcr.io/mll-lab-nu/eai-olympiad:0.1.0",
          "tennis": "ghcr.io/mll-lab-nu/eai-olympiad:0.1.0",
          "soccer": "ghcr.io/mll-lab-nu/eai-olympiad-soccer:0.1.0"}
EGL_IMAGE = os.environ.get("SHOWCASE_EGL_IMAGE", "eai-broadcast-soccer-egl:0.1.0")      # broadcast/Dockerfile.egl
BLENDER_IMAGE = os.environ.get("SHOWCASE_BLENDER_IMAGE", "rcb-showcase-blender:4.2.23")  # showcase/blender/Dockerfile
TASKS = {"tabletennis": "tabletennis-duel-*", "tennis": "tennis-duel-*", "soccer": "soccer-*"}


def run(cmd, **kw):
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, **kw)


def docker(image, mounts, cmd, gpu=None, name="rcb-showcase", env=()):
    c = ["docker", "run", "--rm", "--name", f"{name}-{os.getpid()}", "--network", "none",
         "--user", f"{os.getuid()}:{os.getgid()}", "-e", "HOME=/tmp", "-e", "MPLCONFIGDIR=/tmp"]
    if gpu is not None:
        c += ["--gpus", f"device={gpu}", "-e", "NVIDIA_DRIVER_CAPABILITIES=all"]
    for k, v in env:
        c += ["-e", f"{k}={v}"]
    for src, dst in mounts:
        c += ["-v", f"{src}:{dst}"]
    run(c + [image] + cmd)


ISAAC_IMAGE = os.environ.get("SHOWCASE_ISAAC_IMAGE", "nvcr.io/nvidia/isaac-sim:4.5.0")
ISAAC_CACHE = os.environ.get("SHOWCASE_ISAAC_CACHE", os.path.expanduser("~/.cache/rcb-showcase-isaac"))


def isaac(mounts, gpu, args):
    """Isaac Sim runs as root in its image; its shader caches persist in ISAAC_CACHE (the first start compiles
    shaders for ~10 min, later starts take ~20 s). The OptiX denoiser weights come from the host driver."""
    c = ["docker", "run", "--rm", "--name", f"rcb-showcase-isaac-{os.getpid()}", "--network", "none",
         "--gpus", f"device={gpu}", "-e", "NVIDIA_DRIVER_CAPABILITIES=all", "-e", "ACCEPT_EULA=Y",
         "-e", "PRIVACY_CONSENT=Y", "-e", "OMNI_KIT_ACCEPT_EULA=YES"]
    for sub, dst in (("kit", "/isaac-sim/kit/cache"), ("ov", "/root/.cache/ov"), ("glcache", "/root/.cache/nvidia/GLCache"),
                     ("computecache", "/root/.nv/ComputeCache"), ("logs", "/root/.nvidia-omniverse/logs"),
                     ("data", "/root/.local/share/ov/data")):
        os.makedirs(os.path.join(ISAAC_CACHE, sub), exist_ok=True)
        c += ["-v", f"{os.path.join(ISAAC_CACHE, sub)}:{dst}:rw"]
    if os.path.exists("/usr/share/nvidia/nvoptix.bin"):
        c += ["-v", "/usr/share/nvidia/nvoptix.bin:/usr/share/nvidia/nvoptix.bin:ro"]
    for src, dst in mounts:
        c += ["-v", f"{src}:{dst}"]
    run(c + ["--entrypoint", "/isaac-sim/python.sh", ISAAC_IMAGE] + args)
    run(["docker", "run", "--rm", "-v", f"{mounts[0][0]}:/s", "--entrypoint", "chown", ISAAC_IMAGE, "-R",
         f"{os.getuid()}:{os.getgid()}", "/s"])                # Isaac wrote as root


ENGINES = ("mujoco", "mujoco-showcase", "cycles", "isaac")
REQUIREMENTS = {
    "mujoco": "Docker and the sport's sim image ({sim}); a GPU is optional (EGL image {egl}, built from "
              "scripts/olympiad/broadcast/Dockerfile.egl; without it OSMesa on the CPU, ~10x slower).",
    "mujoco-showcase": "as mujoco.",
    "cycles": "an NVIDIA GPU with the NVIDIA Container Toolkit, and the image {blender} "
              "(docker build -t {blender} scripts/olympiad/showcase/blender; 1.8 GB). ~1-1.5 s per 1080p frame on an "
              "RTX 4090.",
    "isaac": "an NVIDIA RTX GPU (tested: RTX 4090, driver 580.178; Isaac Sim 4.5 needs >= 535.129) with the NVIDIA "
             "Container Toolkit; the image {isaac} (15 GB; docker pull works without an NGC login as of 2026-10-06; "
             "~20 GB free disk incl. caches); /usr/share/nvidia/nvoptix.bin from the driver for the denoiser. The "
             "first start compiles shaders for ~10 min (cached in {cache}); acceptance of the NVIDIA Omniverse EULA "
             "(passed as ACCEPT_EULA=Y).",
}


def image_present(name):
    return subprocess.run(["docker", "image", "inspect", name], capture_output=True).returncode == 0


def gpu_present(gpu):
    if gpu is None:
        return False
    r = subprocess.run(["nvidia-smi", "-i", str(gpu), "--query-gpu=name", "--format=csv,noheader"],
                       capture_output=True, text=True)
    return r.returncode == 0 and bool(r.stdout.strip())


def preflight(engines, sport, gpu):
    """Stop with the documented requirements if an engine cannot run on this machine."""
    fmt = dict(sim=IMAGES[sport], egl=EGL_IMAGE, blender=BLENDER_IMAGE, isaac=ISAAC_IMAGE, cache=ISAAC_CACHE)
    bad = []
    if subprocess.run(["docker", "version"], capture_output=True).returncode != 0:
        bad.append(("all", "Docker is not available"))
    elif not image_present(IMAGES[sport]):
        bad.append(("capture", f"missing sim image {IMAGES[sport]}"))
    for e in engines:
        if e == "cycles":
            if not gpu_present(gpu):
                bad.append((e, "no NVIDIA GPU (--gpu)"))
            if not image_present(BLENDER_IMAGE):
                bad.append((e, f"missing image {BLENDER_IMAGE}"))
        if e == "isaac":
            if not gpu_present(gpu):
                bad.append((e, "no NVIDIA GPU (--gpu)"))
            if not image_present(ISAAC_IMAGE):
                bad.append((e, f"missing image {ISAAC_IMAGE} (docker pull {ISAAC_IMAGE})"))
    if bad:
        msg = ["cannot run the requested engine(s) on this machine:"]
        for e, why in bad:
            msg.append(f"  [{e}] {why}")
        for e in dict.fromkeys(e for e, _ in bad):
            if e in REQUIREMENTS:
                msg.append(f"  {e} requires: " + REQUIREMENTS[e].format(**fmt))
        raise SystemExit("\n".join(msg))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("trial"); ap.add_argument("--sport", required=True, choices=sorted(IMAGES))
    ap.add_argument("--out", required=True)
    ap.add_argument("--engine", default="mujoco,mujoco-showcase",
                    help="comma-separated: " + ", ".join(ENGINES))
    ap.add_argument("--decor", help="promo-only surroundings outside the play area (tabletennis: arena, "
                    "soccer: stadium); showcase look only, every frame tagged 're-rendered for presentation'")
    ap.add_argument("--sim"); ap.add_argument("--gpu", default="0")
    ap.add_argument("--k0", type=int, default=0); ap.add_argument("--k1", type=int)
    ap.add_argument("--t0", type=float, default=0.0); ap.add_argument("--dur", type=float)
    ap.add_argument("--fps", type=float, default=50.0); ap.add_argument("--samples", type=int, default=64)
    ap.add_argument("--stills", default="")
    a = ap.parse_args()
    data = next((d for d in (os.path.join(a.trial, "verifier"), os.path.join(a.trial, "artifacts", "data"), a.trial)
                 if os.path.exists(os.path.join(d, "replay_check.json"))), None)
    if data is None:
        raise SystemExit("no replay_check.json in the trial (verifier/ or artifacts/data/)")
    sim = a.sim or next(iter(sorted(glob.glob(os.path.join(REPO, "tasks", TASKS[a.sport], "environment", "sim")))), None)
    if not sim:
        raise SystemExit("no sim directory: pass --sim")
    gpu = None if a.gpu == "none" else a.gpu
    engines = [e.strip() for e in a.engine.split(",") if e.strip()]
    if not engines or any(e not in ENGINES for e in engines):
        raise SystemExit(f"--engine: choose from {', '.join(ENGINES)}")
    if a.decor and engines == ["mujoco"]:
        raise SystemExit("--decor needs a showcase engine (mujoco-showcase, cycles or isaac): the competition "
                         "picture never gets decor")
    preflight(engines, a.sport, gpu)
    out = os.path.abspath(a.out)
    os.makedirs(out, exist_ok=True)
    work = tempfile.mkdtemp(prefix="showcase-", dir=out)
    try:
        shutil.copytree(sim, os.path.join(work, "sim"), ignore=shutil.ignore_patterns("__pycache__"))
        os.makedirs(os.path.join(work, "data"))
        for f in ("trajectory.json", "episode.jsonl", "replay_check.json"):
            if os.path.exists(os.path.join(data, f)):
                shutil.copy(os.path.join(data, f), os.path.join(work, "data", f))
        m = [(work, "/s"), (HERE, "/w:ro"), (out, "/o")]
        cap = ["python3", "/w/capture.py", a.sport, "/s/sim", "/s/data", "/s/frames.npz", "--k0", str(a.k0)]
        if a.k1 is not None:
            cap += ["--k1", str(a.k1)]
        docker(IMAGES[a.sport], m, cap, name="rcb-showcase-capture")
        win = ["--t0", str(a.t0), "--fps", str(a.fps)] + (["--dur", str(a.dur)] if a.dur else [])
        dec = ["--decor", a.decor] if a.decor else []
        tag = f"{a.sport}_{a.decor}" if a.decor else a.sport
        raster = [e for e in engines if e in ("mujoco", "mujoco-showcase")]
        if a.decor and "mujoco" in raster:              # decor never goes on the competition picture
            raster.remove("mujoco")
        if raster:
            q = "both" if len(raster) == 2 else ("standard" if raster == ["mujoco"] else "showcase")
            gl = [("MUJOCO_GL", "egl"), ("PYOPENGL_PLATFORM", "egl")] if gpu is not None and image_present(EGL_IMAGE) \
                else [("MUJOCO_GL", "osmesa"), ("PYOPENGL_PLATFORM", "osmesa")]
            docker(EGL_IMAGE if gl[0][1] == "egl" else IMAGES[a.sport], m,
                   ["python3", "/w/render.py", "/s/frames.npz", "/s/sim", "/o", "--quality", q,
                    "--stills", a.stills] + win + dec, gpu=gpu if gl[0][1] == "egl" else None, env=gl,
                   name="rcb-showcase-render")
        traced = [e for e in engines if e in ("cycles", "isaac")]
        if traced:
            if gpu is None:
                raise SystemExit("--engine cycles/isaac needs a GPU")
            docker(IMAGES[a.sport], m, ["python3", "/w/blender/export_scene.py", "/s/frames.npz", "/s/sim", "/s/export"]
                   + win + dec, name="rcb-showcase-export")
        for e in traced:
            if e == "cycles":
                docker(BLENDER_IMAGE, m, ["blender", "-b", "--factory-startup", "-P", "/w/blender/cycles_render.py",
                                          "--", "/s/export", "/s/cycles", "--samples", str(a.samples)], gpu=gpu,
                       name="rcb-showcase-cycles")
            else:
                isaac(m, gpu, ["/w/isaac/isaac_render.py", "/s/export", "/s/isaac", "--spp", str(a.samples)])
            docker(IMAGES[a.sport], m, ["python3", "/w/stamp.py", f"/s/{e}", e] + ([a.decor] if a.decor else []),
                   name="rcb-showcase-stamp")                 # every re-rendered frame carries the tag
            rep = "blender_report.json" if e == "cycles" else "isaac_report.json"
            docker(BLENDER_IMAGE, m, ["sh", "-c",
                                      f"ffmpeg -y -loglevel error -framerate {a.fps} -i /s/{e}/frame_%05d.png "
                                      "-c:v libx264 -preset slow -crf 18 -pix_fmt yuv420p -movflags +faststart "
                                      f"/o/{tag}_{e}.mp4 && cp /s/{e}/{rep} /o/{tag}_{e}_report.json"],
                   name="rcb-showcase-encode")
        if {"mujoco", "mujoco-showcase", "cycles"} <= set(engines) and not a.decor:
            docker(IMAGES[a.sport], m, ["python3", "/w/compose.py", a.sport, f"/o/{a.sport}_standard.mp4",
                                        f"/o/{a.sport}_showcase.mp4", "/s/cycles", f"/o/{a.sport}",
                                        "--fps", str(a.fps), "--stills", a.stills], name="rcb-showcase-compose")
        shutil.copy(os.path.join(work, "frames.npz"), os.path.join(out, f"{a.sport}_frames.npz"))
    finally:
        shutil.rmtree(work, ignore_errors=True)
    print(json.dumps(dict(out=out, files=sorted(os.listdir(out)))))


if __name__ == "__main__":
    sys.exit(main())
