"""One in-process match for the model-free screening checks (docs/construction_zh.md step 4): a player VARIANT that
the sport's screening hooks build (sports/<sport>/screen/hooks.py), against the sport's scripted opponent (format
match) or another variant (format duel), with an effect delay.

Runs inside the sport's image with the assembled simulator directory (tools/sportlib.assemble_sim) at /srv and the
hooks at /screen (tools/screen/batch.py starts it):

    python3 /tools/play.py --hooks /screen/hooks.py --variant V [--opp-variant W] --level L --seed S [--delay D] \
        --out DIR

The loop is core/local.py's real-time loop with one change, the latency model: an action takes effect D control steps
after the observation it answers, and it is lowered by the action layer at the step where it takes effect (as the live
service lowers an action at the control-step boundary where it takes effect); one action per seat in flight, a seat
observes every D steps while it has nothing in flight (the validity pilot's loop). D = 0 is core/local.py's loop (an
action takes effect at once, observed every `--every` steps).

D is the sport's declared reaction delay, sport.toml [timing] reaction_delay_steps (core 0.2.18), the rule the live
service plays by (core/realtime.py: an act takes effect at max(the next control-step boundary, the step of the seat's
latest world state + D)), so screening and the official matches agree. A sport that declares it may not set a
different hooks `DELAY` (the run stops). A sport that declares none keeps the hooks' `DELAY` (default 0) until it
adopts the declaration. `--sport-delay` is the sport's own screening delay given by its caller (the shared [reference]
hook, reference_hook.py `delay=`): it stands for the hooks' `DELAY` under the same rule. `--delay` overrides all of
them for an experiment (e.g. a latency sensitivity); result.json records `delay` and `delay_source`.

Writes DIR/result.json: the configuration, scores, winner, the sport's final fields, the hooks' summary(), wall time.
--episode also writes DIR/episode.jsonl (core 0.2.25): the replay rows (cmd, resume, end) and the diagnostic rows
(observe, act_received, effect, act_dropped, event; core/diaglog.py) of the live service, for the post-match diagnosis
(python -m core.diagnose PLUGIN DIR). The header records the screening variant ("screen"); a variant that changes the
world (hooks prepare / patch_match: "world_patched": true) replays exactly only with the same change
(core.diagnose.load(..., build=...)).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

SRV = Path("/srv")


def load_hooks(path: str):
    spec = importlib.util.spec_from_file_location("screen_hooks", path)
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(Path(path).resolve().parent))
    spec.loader.exec_module(mod)
    return mod


def _jd(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return str(o)


def screen_delay(decl: dict, hooks_delay, override) -> tuple[int, str]:
    """The effect delay D of a screening match and where it came from: --delay (an experiment), else the sport's
    declared reaction delay (sport.toml [timing] reaction_delay_steps; a different hooks DELAY is an error), else the
    hooks' DELAY (a sport that declares none, default 0)."""
    from core import spec as spec_mod
    declared = "reaction_delay_steps" in (decl.get("timing") or {})
    d = spec_mod.reaction_delay(decl)
    if declared and hooks_delay is not None and int(hooks_delay) != d:
        raise SystemExit(f"hooks DELAY = {hooks_delay} differs from the declared reaction delay D = {d} "
                         "(sport.toml [timing] reaction_delay_steps): remove DELAY from the hooks")
    if override is not None:
        return int(override), "--delay"
    if declared:
        return d, "sport.toml [timing] reaction_delay_steps"
    return int(hooks_delay or 0), "hooks DELAY (no declared reaction delay)"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--hooks", required=True)
    ap.add_argument("--variant", required=True)
    ap.add_argument("--opp-variant", default=None, help="format duel: the other seat's variant")
    ap.add_argument("--format", default=None)
    ap.add_argument("--level", type=int, default=None)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--first", default=None, help="format duel: the first mover / server")
    ap.add_argument("--delay", type=int, default=None, help="an experiment's effect delay (default: the declared D)")
    ap.add_argument("--sport-delay", type=int, default=None,
                    help="the sport's screening delay from its caller (the [reference] hook): as the hooks' DELAY")
    ap.add_argument("--every", type=int, default=5, help="observation period (steps) of a seat with nothing in flight")
    ap.add_argument("--action-level", default=None)
    ap.add_argument("--max-steps", type=int, default=200000)
    ap.add_argument("--opt", action="append", default=[], help="KEY=VALUE options for the hooks (repeatable)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--episode", action="store_true", help="also write DIR/episode.jsonl (replay + diagnostic rows)")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(SRV))
    opts = dict(o.split("=", 1) for o in a.opt)
    plugin = str(SRV / "sport" / "plugin.py")

    from core.episode import load_plugin
    from core import local
    from core import spec as spec_mod
    from core.sport import FINISHED, UNIT_OVER, MatchConfig, Refused, parse_action
    from core.outcome import decide

    sport = load_plugin(plugin)
    if str(SRV / "sport") not in sys.path:        # the hooks import the sport's own modules (its built-in players)
        sys.path.insert(1, str(SRV / "sport"))
    H = load_hooks(a.hooks)
    fmt = a.format or getattr(H, "FORMAT", "match")
    level = a.level
    decl = spec_mod.load(plugin)
    delay, delay_source = screen_delay(decl, getattr(H, "DELAY", None) if a.sport_delay is None else a.sport_delay,
                                       a.delay)
    alevel = a.action_level or getattr(H, "ACTION_LEVEL", "high")
    seats = ("A",) if fmt == "match" else ("A", "B")
    kw = dict(sport=sport.name, format=fmt, seats=seats, seed=a.seed, action_level=alevel)
    if level is not None:
        kw["level"] = level
    if a.first:
        kw["first"] = a.first
    cfg = MatchConfig(**kw)
    if hasattr(H, "prepare"):                     # process-wide settings of a variant (before the world is built)
        H.prepare(a.variant, opts)
    m = sport.build(cfg)
    if hasattr(H, "patch_match"):                 # an opponent change of a variant (e.g. a planted weakness)
        H.patch_match(m, a.variant, opts)
    layer = sport.action_layer(alevel)
    views = local.Views(sport, cfg, m, decl, True, plugin)
    from core.diaglog import jsonable, safe
    log = lambda ev, **row: None                  # noqa: E731
    if a.episode:
        from core import VERSION
        from core.episode import Episode
        (out / "episode.jsonl").unlink(missing_ok=True)
        ep = Episode(out / "episode.jsonl")
        ep.write("header", cfg=cfg.to_json(), sport=sport.name, revision=sport.revision, protocol=sport.protocol,
                 source_sha=None, versions={"core": VERSION, "sport": sport.revision}, local=True,
                 timing={"reaction_delay_steps": delay, "effect_delay_steps": 0},
                 screen={"variant": a.variant, "opp_variant": a.opp_variant, "opts": opts, "delay": delay,
                         "every": a.every, "world_patched": hasattr(H, "prepare") or hasattr(H, "patch_match")})
        log = ep.write
        views.log = safe(ep.write)
    dlog = safe(log)
    variants = {"A": a.variant}
    if fmt != "match":
        variants["B"] = a.opp_variant or a.variant
    players = {s: H.make_player(v, local.seat_spec(sport, cfg, s, plugin=plugin), s, a.seed, delay, opts)
               for s, v in variants.items()}
    plan = {s: sport.worker_init("plan", s, cfg) for s in seats} if getattr(layer, "worker", None) else {}
    pending: dict = {}                            # seat -> (k_eff, cmd, act_id)
    refused = {s: 0 for s in seats}
    rejected = {s: 0 for s in seats}
    n_act = {s: 0 for s in seats}
    t0 = time.time()

    def effect(seat, cmd, act_id=None):
        job = layer.snapshot(seat, cmd, m, m.k)
        prepared = None
        if job is not None:
            prepared = sport.worker_call(plan[seat], "plan", job) if plan else layer.prepare(job)
        low, info = layer.lower(seat, cmd, prepared, m)
        log("cmd", k=m.k, seat=seat, cmd=low)
        res = m.apply(seat, low) or {}
        if "rejected" in res:
            rejected[seat] += 1
        dlog("effect", seat=seat, act_id=act_id, k=m.k, **({"rejected": True} if "rejected" in res else {}))
        if hasattr(players[seat], "on_reply"):
            try:
                players[seat].on_reply(layer.reply(seat, cmd, info, m))
            except Exception:  # noqa: BLE001  (a reply the player cannot use changes nothing)
                pass

    while m.phase().state != FINISHED and m.k < a.max_steps:
        if m.phase().state == UNIT_OVER:
            for seat, x in pending.items():
                dlog("act_dropped", seat=seat, act_id=x[2], k=m.k, why="unit_over")
            pending.clear()
            log("resume", k=m.k)
            m.resume()
        for seat in seats:
            if seat in pending and pending[seat][0] <= m.k:
                _, cmd, act_id = pending.pop(seat)
                effect(seat, cmd, act_id)
        for seat in seats:
            if seat in pending or m.k % (delay or a.every) != 0:
                continue
            wants = getattr(H, "wants_images", None)
            views.render = True if wants is None else bool(wants(players[seat], m, seat))
            v = views.view(seat, {"phase": "running", "world": "running", "step": m.k, "t_sim": m.k * m.dt,
                                  "done": False})
            if hasattr(H, "augment_view"):        # e.g. a truth-perception variant: the true state added (screening only)
                H.augment_view(v, m, seat, players[seat])
            body = players[seat].act(v)
            if body is None:
                continue
            try:
                cmd = parse_action(layer, seat, body, m, {})
            except Refused:
                refused[seat] += 1
                continue
            n_act[seat] += 1
            act_id = views.acts.new(seat)
            dlog("act_received", seat=seat, k=m.k, act_id=act_id, obs_id_last=views.acts.last_obs.get(seat),
                 seen=m.k, entity=(cmd.get("entity") if isinstance(cmd, dict) else None), decision=cmd)
            if delay:
                pending[seat] = (m.k + delay, cmd, act_id)
            else:
                effect(seat, cmd, act_id)
        events = m.step()
        if events and a.episode:
            dlog("event", k=m.k, events=jsonable(events))
    for seat, x in pending.items():
        dlog("act_dropped", seat=seat, act_id=x[2], k=m.k, why="match_over")

    finished = m.phase().state == FINISHED
    log("end", k=m.k, reason="match_over" if finished else "unfinished")
    oc = decide(m, None, None, cfg.side_list(), cfg.side_of)
    ff = m.final_fields()
    res = {"sport": sport.name, "revision": sport.revision, "variant": a.variant, "opp_variant": variants.get("B"),
           "format": fmt, "level": level, "seed": a.seed, "first": a.first, "delay": delay,
           "delay_source": delay_source, "action_level": alevel,
           "opts": opts, "finished": finished, "k": int(m.k), "wall_s": round(time.time() - t0, 1),
           "scores": m.scores(), "winner": oc.winner, "decided_by": oc.decided_by, "refused": refused,
           "rejected": rejected, "acts": n_act, "final_fields": ff}
    if hasattr(H, "summary"):
        res["summary"] = H.summary(res, players)
    (out / "result.json").write_text(json.dumps(res, default=_jd))
    print(json.dumps({k: res[k] for k in ("variant", "level", "seed", "scores", "winner", "k", "wall_s")}, default=_jd))


if __name__ == "__main__":
    main()
