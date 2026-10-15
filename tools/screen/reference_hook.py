"""The shared [reference] match hook of tools/check stage 5 (docs/screening/README.md "Reference check"): one match of
the reference against the scripted opponent, played by tools/screen/play.py, the loop of the step-4 screening (an
action takes effect D control steps after the observation it answers, lowered at that step; one action in flight).

A sport declares the hook in a file of its screening directory (not part of a task's simulator, tools/sportlib.py
NOT_IN_SIM), e.g. sports/<sport>/screen/reference.py:

    from reference_hook import hook                 # this file; tools/check stage 5 puts it next to its runner
    match = hook()                                  # the sport's [players] reference, camera = truth (no truth variant)

    match = hook("hooks.py", camera="reference", truth="reference_truth")    # the step-4 hooks' variants

and in sport.toml:

    [reference]
    match = "screen.reference:match"
    seeds = 20

What a sport chooses (everything else is the step-4 loop):

  hooks         the screening hooks file (tools/screen/README.md: make_player, augment_view, ...), relative to the
                caller's directory; None = this module's built-in hooks: the sport's [players] reference_class
                (variant "reference"), no truth variant
  camera        the hooks' camera-only reference variant (perception "camera")
  truth         the hooks' variant with the true state instead of the images (perception "truth", e.g. one that
                gets the true ball from augment_view); None: the camera variant plays the truth matches too, and the
                hook says so (result "perception": "camera"): a stand-in, not a truth reference; tools/check stage 5
                prints a NOTE, and FAILS once the sport is frozen ([official.hidden_seeds])
  read          result -> {"won", "scores": [ours, theirs], "report": {name: [count, of]}}; the default reads the
                hooks' summary() (won, report) and play.py's scores (seat A's side against the others, when numbers)
  delay, every, action_level, opts, max_steps
                play.py's options (defaults: the hooks' DELAY / ACTION_LEVEL; else delay 0, every 5, the high action
                level when the sport has one, else its first). The effect delay follows play.py's rule (core 0.2.18):
                a sport that declares sport.toml [timing] reaction_delay_steps plays with it, and a `delay` that
                differs is an error (play.py --sport-delay, like the hooks' DELAY); `delay` only counts for a sport
                that declares none

Levels and seeds come from sport.toml ([formats.<f>] / [official] levels, [reference] seeds / camera_seeds): tools/check
calls the hook once per (perception, level, seed).
"""
from __future__ import annotations

import importlib.util
import inspect
import json
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
SRV = "/srv"
NUMBER = (int, float)


# ------------------------------------------------------------------------------------- the built-in hooks (hooks=None)
# play.py loads this file as the hooks when the sport has none: the variant "reference" is the sport's [players]
# reference_class, unchanged (camera only).
def make_player(variant, spec, seat, seed, delay, opts):
    if variant != "reference":
        raise ValueError(f"the built-in hooks have one variant, 'reference' (not {variant!r})")
    import tomllib
    with open(os.path.join(SRV, "sport", "sport.toml"), "rb") as fh:
        ref = (tomllib.load(fh).get("players") or {}).get("reference_class", "builtin_players:Reference")
    mod, cls = ref.split(":")
    return getattr(importlib.import_module(mod), cls)(spec, seat, seed)


# ------------------------------------------------------------------------------------------------ the shared hook
def _load(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _play_module():
    """tools/screen/play.py (next to this file; loaded by path: a sport module named `play` cannot shadow it)."""
    return _load(os.path.join(HERE, "play.py"), "screen_play")


def default_action_level(srv: str = SRV) -> str:
    import tomllib
    with open(os.path.join(srv, "sport", "sport.toml"), "rb") as fh:
        lv = (tomllib.load(fh).get("modes") or {}).get("action_levels") or ["high"]
    return "high" if "high" in lv else lv[0]


def read_result(res: dict) -> dict:
    """play.py's result.json -> the hook's result. won: the hooks' summary() `won`, else play.py's winner is seat A's
    side; scores: summary() `scores` ([ours, theirs]), else play.py's scores (side -> number): A's against the sum of
    the others ([] when they are not numbers: tools/check then reports no point share); report: summary() `report`."""
    sm = res.get("summary") or {}
    won = bool(sm["won"]) if "won" in sm else res.get("winner") == "A"
    scores = list(sm.get("scores") or [])
    sc = res.get("scores") or {}
    if not scores and isinstance(sc, dict) and "A" in sc and all(isinstance(v, NUMBER) for v in sc.values()):
        scores = [sc["A"], sum(v for k, v in sc.items() if k != "A")]
    return {"won": won, "scores": [float(x) for x in scores], "report": dict(sm.get("report") or {})}


def hook(hooks: str | None = None, *, camera: str = "reference", truth: str | None = None, read=None,
         delay: int | None = None, every: int | None = None, action_level: str | None = None,
         opts: dict | None = None, max_steps: int | None = None, srv: str = SRV):
    """The [reference] match function: match(format, level, seed, perception) -> {won, scores, report, perception}."""
    if hooks is None:
        path = os.path.abspath(__file__)
        if truth is not None:
            raise ValueError("the built-in hooks have no truth variant: pass the sport's hooks file")
    else:
        base = os.path.dirname(os.path.abspath(inspect.stack()[1].filename))
        path = hooks if os.path.isabs(hooks) else os.path.join(base, hooks)
    read = read or read_result

    def match(format: str = "match", level: int | None = None, seed: int = 1, perception: str = "truth") -> dict:
        if perception not in ("truth", "camera"):
            raise ValueError(f"perception must be truth or camera, not {perception!r}")
        variant = truth if perception == "truth" and truth else camera
        res = play_one(path, variant, format, level, seed, delay=delay, every=every, action_level=action_level,
                       opts=opts, max_steps=max_steps, srv=srv)
        if not res.get("finished"):
            raise RuntimeError(f"the match did not finish ({res.get('k')} control steps)")
        out = dict(read(res))
        out["perception"] = "truth" if variant == truth else "camera"
        return out

    match.hooks, match.camera, match.truth = path, camera, truth
    return match


def play_one(hooks_path: str, variant: str, format: str, level, seed: int, *, delay=None, every=None,
             action_level=None, opts=None, max_steps=None, srv: str = SRV) -> dict:
    """One match of tools/screen/play.py in this process; its result.json."""
    play = _play_module()
    play.SRV = type(play.SRV)(srv)
    if action_level is None:
        sys.path.insert(0, srv)
        H = play.load_hooks(hooks_path)
        action_level = getattr(H, "ACTION_LEVEL", None) or default_action_level(srv)
    argv = ["--hooks", hooks_path, "--variant", variant, "--format", format, "--seed", str(int(seed)),
            "--action-level", action_level]
    if level is not None:
        argv += ["--level", str(int(level))]
    for flag, v in (("--sport-delay", delay), ("--every", every), ("--max-steps", max_steps)):
        if v is not None:
            argv += [flag, str(int(v))]
    for k, v in (opts or {}).items():
        argv += ["--opt", f"{k}={v}"]
    with tempfile.TemporaryDirectory(prefix="refhook-") as out:
        play.main(argv + ["--out", out])
        with open(os.path.join(out, "result.json")) as fh:
            return json.load(fh)
