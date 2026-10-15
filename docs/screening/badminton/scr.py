"""Badminton screening (step 4, model-free): match-wallclock, perception-free (true shuttle + noise; study.PerfectRef).
Env: POLICY (ref|center|reflex|fixedracket|fixed|random|serveonly|tossonly|nop), ABL (none|a1_2|a1_5|a2_k05|a2_k0|
a4_half|a4_full|b2_rand|b2_fixed), LEVEL, SEEDS (comma list), SLOWDOWN (20), OPP (python file exec'd to patch bm, or
''), TAG (output name). Out: /work/out/<TAG>.jsonl (one line per match: summary + final fields). No model is run."""
import os as _os, sys as _sys
_sys.path[:0] = ["/repo", "/repo/sports/badminton/sim"]
import bm as _bm  # noqa: E402  (the base study: a candidate base installed before the plugin is imported)
if _os.environ.get("BASE"):
    _bm.set_base(_os.environ["BASE"])

import json, math, os, sys, time
sys.path.insert(0, "/repo/docs/benchmarks/badminton-minicourt-20261008")
import numpy as np
import study as S
PL, BM, BP = S.PL, S.BM, S.BP
from core.sport import MatchConfig

OPP = os.environ.get("OPP", "")
if OPP:
    exec(open(OPP).read(), {"BM": BM, "np": np, "math": math, "__name__": "opp"})

POL = os.environ.get("POLICY", "ref")
if os.environ.get("ZKILL"):                   # the reference's kill window top (1.75 = no kills)
    BP.Z_KILL = float(os.environ["ZKILL"])
if os.environ.get("KBONUS"):
    BP.KILL_BONUS = float(os.environ["KBONUS"])
if POL == "flatserve":                      # the reference, but every serve a flat fast short serve to the front
    _sc = BP.serve_choice                       # corner of the receiver's service court (Astra's hole, 10-12)
    def _flat_choice(spec_seed, st):
        kind, tgt = _sc(spec_seed, st)
        return "flat", np.array([np.sign(tgt[0]) * 0.92 * BP.COURT_X, 1.17])
    _sp = BP.serve_plan
    def _flat_plan(base_xy, kind, target, cap=None):
        nm = BP.NET_MARGIN
        BP.NET_MARGIN = 0.12 if kind == "flat" else nm
        try:
            return _sp(base_xy, kind, target, cap)
        finally:
            BP.NET_MARGIN = nm
    BP.SERVE_THETAS = dict(BP.SERVE_THETAS, flat=(40, 45, 50, 35, 55))
    BP.serve_choice, BP.serve_plan = _flat_choice, _flat_plan
    POL = "ref"
if POL == "ref2":                           # the reference with net play: short to the open side vs a back opponent
    import inspect as _insp
    _src = _insp.getsource(BP.choose)
    _a = "    back = opp is not None and opp[1] > 0.55 * COURT_Y"
    _b = "    plans = [(np.array([side * W, D]), DEEP_THETAS)]\n    if back:\n        plans += [(np.array([side * W, 0.5 * COURT_Y]), SMASH_THETAS), (np.array([side * W, Sh]), SHORT_THETAS)]"
    assert _a in _src and _b in _src, "choose changed"
    _src = _src.replace(_a, f"    back = opp is not None and opp[1] > {float(os.environ.get('BACK_Y', '1.0'))}")
    _src = _src.replace(_b, "    if back:\n        plans = [(np.array([side * W, Sh]), SHORT_THETAS), (np.array([side * W, 0.5 * COURT_Y]), SMASH_THETAS), (np.array([side * W, D]), DEEP_THETAS)]\n    else:\n        plans = [(np.array([side * W, D]), DEEP_THETAS)]")
    _c = "        side = -1.0 if opp[0] > 0 else 1.0           # the corner away from the opponent"
    assert _c in _src
    _src = _src.replace(_c, f"        side = (-1.0 if opp[0] > 0 else 1.0) * (1.0 if rng.random() < {float(os.environ.get('MIX', '1.0'))} else -1.0)")
    exec(compile(_src, "choose_ref2", "exec"), BP.__dict__)
    POL = "ref"
if POL == "refnowait":                   # the reference that never moves its base back (waits where it struck)
    BP.Reference.WAIT = False
ABL = os.environ.get("ABL", "none")
_orig_strokes, _orig_choose, _orig_backtrack = BP.strokes, BP.choose, BP.backtrack
K0 = BP.K
CENTRE = np.array([0.0, 0.6 * BP.COURT_Y])
N_FIX = np.array([0.0, math.cos(math.radians(55)), math.sin(math.radians(55))])
_LAST_OPP = {"xy": None}


def backtrack_spy(p, v, t, dt=0.001):
    r = _orig_backtrack(p, v, t, dt)
    _LAST_OPP["xy"] = np.asarray(r[:2], float).copy()
    return r


def strokes_center(p, v_in, target, thetas, vmax_head=None):
    return _orig_strokes(p, v_in, CENTRE, (45, 35, 55, 25, 60, 65, 70), vmax_head)


def strokes_reflex(p, v_in, target, thetas, vmax_head=None):
    # return to sender: the shuttle goes back to where the opponent played it from (our frame, its half y > 0)
    o = _LAST_OPP["xy"]
    tgt = CENTRE.copy() if o is None else np.array([float(np.clip(o[0], -0.85 * BP.COURT_X, 0.85 * BP.COURT_X)),
                                                    float(np.clip(o[1], 0.3 * BP.COURT_Y, 0.9 * BP.COURT_Y))])
    return _orig_strokes(p, v_in, tgt, (45, 35, 55, 25, 60, 65, 70), vmax_head)


_ALT = {"n": 0, "key": None}


def strokes_shortalt(p, v_in, target, thetas, vmax_head=None):
    # the tactic a model found (step 5): short and wide, alternating corners (no reading of the opponent)
    o = _LAST_OPP["xy"]
    key = None if o is None else (round(float(o[0]), 2), round(float(o[1]), 2))
    if key != _ALT["key"]:
        _ALT["key"], _ALT["n"] = key, _ALT["n"] + 1
    if POL == "shortrand":                  # short and wide, a random corner each shot (seeded by the shot)
        side = 1.0 if np.random.default_rng([_ALT["n"], 0 if key is None else int(abs(1000 * key[0] + 3000 * key[1])) % 100003]).random() < 0.5 else -1.0
    else:
        side = 1.0 if _ALT["n"] % 2 else -1.0
    return _orig_strokes(p, v_in, np.array([side * 0.8, 0.95]), (35, 25, 45, 15, 55, 5), vmax_head)


def strokes_fixed(p, v_in, target, thetas, vmax_head=None):
    return [(N_FIX.copy(), 2.0 * N_FIX)]


class _Heads:
    def random(self):
        return 0.0


def choose_b2(ts, P, V, t_now, base_xy, vmax, amax, rng, margin=0.12, opp=None, t_min=None):
    if ABL == "b2_fixed":
        rng = _Heads()                      # always the same corner, no reading of the opponent
    return _orig_choose(ts, P, V, t_now, base_xy, vmax, amax, rng, margin=margin, opp=None, t_min=t_min)


class Base(S.PerfectRef):
    camera = POL == "refcam" or os.environ.get("CAM") == "1"   # real camera perception (renders the tiles)

    def __init__(self, spec, seat, seed=0):
        noise = {"a1_2": 0.02, "a1_5": 0.05}.get(ABL, 0.003)
        super().__init__(spec, seat, seed, noise=noise)
        if os.environ.get("WAIT_XY"):          # the base study: another waiting point for the reference
            self.wait_xy = [float(x) for x in os.environ["WAIT_XY"].split(",")]
        if ABL in ("a2_k05", "a2_k0"):
            BP.K = BP.K * (0.5 if ABL == "a2_k05" else 0.0)
        if ABL == "a4_half":
            self.lead = self.lead // 2
        elif ABL == "a4_full":
            self.lead = 0

    def act(self, view):
        st = view.get("state") or {}
        serving = st.get("you_serve") and (st.get("serve") or {}).get("phase") in ("waiting", "open", "tossed")
        if POL == "nop":
            return None
        if POL == "tossonly":
            if serving and (st.get("serve") or {}).get("phase") == "open" and not getattr(self, "_tossed", {}).get(st.get("point")):
                self._tossed = {st.get("point"): True}
                b = st.get("robot_base_xy") or [0.12, -1.6]
                return {"type": "toss", "pos": [b[0] + 0.4, b[1] + 0.4, 1.0], "height": 0.0}
            return None
        if POL == "serveonly" and not serving:
            return None
        if POL in ("ref", "refnowait", "refcam", "center", "reflex", "fixedracket", "serveonly", "shortalt", "shortrand"):
            if POL in ("ref", "refnowait", "refcam", "serveonly") or serving:
                if ABL.startswith("b2_") and not serving:
                    BP.choose = choose_b2
                try:
                    return super().act(view)
                finally:
                    BP.choose = _orig_choose
            BP.strokes = {"center": strokes_center, "reflex": strokes_reflex, "fixedracket": strokes_fixed,
                          "shortalt": strokes_shortalt, "shortrand": strokes_shortalt}[POL]
            BP.backtrack = backtrack_spy
            try:
                return super().act(view)
            finally:
                BP.strokes, BP.backtrack = _orig_strokes, _orig_backtrack
        if st.get("world") != "running" or st.get("pending_commands"):
            return None
        if serving:
            return super().act(view)
        if st.get("ball_state") != "coming to you" or st.get("t_since_played") is None:
            self._sent_flight = None
            return None
        key = (st.get("point"), round(st["step"] * 0.01 - float(st["t_since_played"]), 3))
        if getattr(self, "_sent_flight", None) == key or float(st["t_since_played"]) < 0.2:
            return None
        self._sent_flight = key
        if POL == "fixed":
            return {"type": "act", "t": 0.9, "pos": [0.5, -1.1, 1.2], "normal": N_FIX.round(4).tolist(),
                    "vel": (2.0 * N_FIX).round(4).tolist()}
        g = getattr(self, "_g", None) or np.random.default_rng(self.seed_ * 31 + 7)
        self._g = g
        n = g.normal(size=3); n[1] = abs(n[1]); n /= np.linalg.norm(n)
        return {"type": "act", "t": float(g.uniform(0.4, 1.5)),
                "pos": [float(g.uniform(-1.0, 1.0)), float(g.uniform(-2.6, -0.5)), float(g.uniform(0.6, 2.0))],
                "normal": n.round(4).tolist(), "vel": (n * g.uniform(0.5, 3.0)).round(4).tolist()}


OPPLOG = []
_orig_opp_plan = BM.Match._opp_plan


def opp_plan_spy(self):
    o = self.opp_xy()
    OPPLOG.append({"k": int(self.k), "kind": self.leg.get("kind"), "p": [float(x) for x in self.s.sh_p],
                   "v": [float(x) for x in self.s.sh_v], "o": [float(x) for x in o]})
    r = _orig_opp_plan(self)
    OPPLOG[-1].update(miss=self.leg.get("opp_miss"), leave=bool(self.leg.get("opp_leave")),
                      pressure=self.leg.get("opp_pressure"))
    return r


BM.Match._opp_plan = opp_plan_spy


def run(level, seed):
    OPPLOG.clear()
    BP.K = K0
    cfg = MatchConfig(sport="badminton", format="match-wallclock", seats=("A",), seed=int(seed), level=int(level),
                      slowdown=float(os.environ.get("SLOWDOWN", "20")))
    m = PL.SPORT.build(cfg)
    layer = PL.SPORT.action_layer("high")
    pl = {"A": Base(S.seat_spec(cfg, m, "A"), "A", seed)}
    t0 = time.time()
    ref = S.play(cfg, m, layer, pl, max_steps=80000)
    g = m.g
    rally = [r for r in g.results if r["kind"] != "our_serve"]
    ff = m.final_fields()
    notes = [str((s.get("stroke") or "")) + " | " + str(s.get("base") or "") for s in g.shots]
    return {"policy": POL, "abl": ABL, "opp": os.path.basename(OPP) or "main", "base": BM.BASE_NAME, "level": level, "seed": seed,
            "score": list(g.score), "won": g.winner == 0, "n_points": len(g.points), "hits": g.hits,
            "rally_legs": len(rally), "rally_hit": sum(r["hit"] for r in rally),
            "rally_returned": sum(r["returned"] for r in rally),
            "serves": sum(r["kind"] == "our_serve" for r in g.results),
            "serves_good": sum(r["returned"] for r in g.results if r["kind"] == "our_serve"),
            "how": [p["how"][:90] for p in g.points], "refused": ref, "wall_s": round(time.time() - t0, 1),
            "n_acts": len(g.shots), "n_base_short": sum("could not reach" in n for n in notes),
            "n_speed_lim": sum("head speed limited" in n for n in notes),
            "n_pose_nd": sum("pose not delivered" in n for n in notes),
            "n_selfcol_note": sum("self-collision" in n for n in notes),
            "selfcol": ff.get("self_collision"), "opplog": list(OPPLOG),
            "final": {k: ff.get(k) for k in ("points", "results", "shots", "opponent", "serves")}}


if __name__ == "__main__":
    level = int(os.environ.get("LEVEL", "1"))
    tag = os.environ.get("TAG", f"{POL}_{ABL}_L{level}")
    path = f"/work/out/{tag}.jsonl"
    done = {json.loads(l)["seed"] for l in open(path) if l.strip()} if os.path.exists(path) else set()
    with open(path, "a") as fh:
        for s in [int(x) for x in os.environ["SEEDS"].split(",")]:
            if s in done:
                continue
            r = run(level, s)
            fh.write(json.dumps(r, default=float) + "\n"); fh.flush()
            print(POL, ABL, level, s, r["score"], r["won"], r["rally_returned"], "/", r["rally_legs"], r["wall_s"],
                  flush=True)
