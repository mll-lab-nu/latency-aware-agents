"""turnshot: the Season-1 shared turn-based server pattern on the v0.2 core (copy this file byte-identical).

Bowling, mini-golf putting and disc golf ran on one ~650-line HTTP service (scripts/olympiad/shared/<sport>/sim/
server.py) plus a per-sport rules module `sport.py` with one fixed interface (NAME, LOWER_WINS, ACTION_KEYS, TILES,
IMAGE_SIZE, IMAGE_LAYOUT, TIEBREAK_MAX, validate(body), decide(scores), spec(duel), Match(seats, order) with turn,
shoot, done, winner, start_tiebreak, n_tiebreaks, decision, views, summary, score, state_sha). On the v0.2 core the
core replaces server.py (core/turn.py: chess clocks, turns, reviews, the episode, the fresh-process replay) and this
file maps the rules module onto contract ② once:

  legacy server.py                              here
  the seed read from TURN_SEED at import        load_rules(sim_dir, seed): one fresh copy of sim/ per seed
  M.turn(ORDER); the review after a shot        to_move(); the core's review after each action
  validate(body); clocks with every shot (ms)   ShotLayer: the schema, then validate(); both chess clocks go into the
                                                low-level command (the replay sees them)
  M.shoot + the tie-break start in do_act       apply()
  every game done -> match_over                 phase()
  decision(): forfeit, decide, clock            the core's forfeit rule, then tiebreaks(): total, rules.decide,
                                                the chess clock at the last shot (the last resort)
  render_views: tiles into one PNG              camera_job(): the sport draws its own images (sim/ renderer)
  final.json (scores, actions, results ...)     final_fields() (legacy names; "scores" stays under sport_fields)
  state_sha per seat                            state_digest() = sha_json({seat: rules.Match.state_sha(seat)})

A sport = sim/ (its simulator and its rules module, copied unchanged from the legacy tree) + this file + a thin
plugin.py (the action schema, Sport.spec(), the players) + sport.toml. See sports/bowling/README.md.
"""
from __future__ import annotations

import importlib.util
import io
import json
import os
import sys
import threading
from pathlib import Path

from core.sport import FINISHED, PLAYING, MatchBase, Phase, Refused, SchemaLayer, check_command, sha_json

TIEBREAKS = ("total", "rulebook_tiebreak", "clock_left")      # sport.toml [tiebreaks] order of every turnshot duel
_LOCK = threading.RLock()
_RULES: dict = {}


def _under(mod, d: Path) -> bool:
    f = getattr(mod, "__file__", None)
    try:
        return bool(f) and Path(f).resolve().parent == d
    except (OSError, ValueError):
        return False


def load_rules(sim_dir, seed: int, rules_file: str = "sport.py"):
    """The rules module of sim_dir for one seed. The legacy modules read the seed from the environment at import
    (TURN_SEED, and what the rules module derives from it), so every seed gets its own fresh copy of the sim/ modules;
    the environment and sys.modules are restored afterwards. Cached per (sim_dir, seed)."""
    d = Path(sim_dir).resolve()
    key = (str(d), int(seed))
    with _LOCK:
        if key in _RULES:
            return _RULES[key]
        env = dict(os.environ)
        for name in [n for n, m in sys.modules.items() if _under(m, d)]:
            del sys.modules[name]
        sys.path.insert(0, str(d))
        os.environ["TURN_SEED"] = str(int(seed))
        try:
            spec = importlib.util.spec_from_file_location(f"turnshot_rules_{abs(hash(key))}", d / rules_file)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
        finally:
            sys.path.remove(str(d))
            for name in [n for n, m in sys.modules.items() if _under(m, d)]:
                del sys.modules[name]
            for k in [k for k in os.environ if k not in env]:
                del os.environ[k]
            os.environ.update(env)
        _RULES[key] = mod
        return mod


class TurnShotMatch(MatchBase):
    """Contract ② on a legacy rules module. Seats are sides (single: one seat; duel: two, each on its own copy of
    the world, e.g. its own lane). The rules module decides the turns, the scoring and its rulebook tie-break."""

    def __init__(self, cfg, rules):
        self.cfg, self.S = cfg, rules
        self.seats = list(cfg.seats)
        first = cfg.first if cfg.first in self.seats else self.seats[0]
        self.order = [first] + [s for s in self.seats if s != first]
        self.M = rules.Match(self.seats, self.order)
        self.k = 0
        self.shots: list[dict] = []                  # every shot in play order (seat, command, clocks, result)
        self.clock_out = {s: False for s in self.seats}
        self._png: dict = {}

    @property
    def duel(self) -> bool:
        return len(self.seats) > 1

    def other(self, seat):
        return next((s for s in self.seats if s != seat), None)

    # world
    def state_digest(self) -> str:
        return sha_json({s: self.M.state_sha(s) for s in self.seats})

    # actions
    def apply(self, seat, cmd):
        dec = {k: v for k, v in cmd.items() if k not in ("type", "clock_left_s")}
        res = self.M.shoot(seat, dec)
        self.k += 1
        self.shots.append({"seat": seat, "shot_n": self.k, "decision": dec, "clock_left_s": cmd.get("clock_left_s"),
                           "result": res})
        if self.duel and all(self.M.done(s) for s in self.seats) and self.M.winner() is None \
                and self.M.n_tiebreaks < self.S.TIEBREAK_MAX:
            self.M.start_tiebreak()                  # level after the game: the rulebook's extra play (no draws)
        return {"result": res}

    # referee
    def to_move(self):
        return None if self.over() else self.M.turn(self.order)

    def over(self) -> bool:
        return all(self.M.done(s) for s in self.seats)

    def phase(self):
        return Phase(FINISHED, reason="match_over") if self.over() else Phase(PLAYING)

    def scores(self):
        return {s: self.M.score(s, self.clock_out[s])["total"] for s in self.seats}

    def forfeit(self, seat, why):
        self.clock_out[seat] = why == "clock"
        return True

    def tiebreaks(self):
        if not self.duel:
            return []                                # solo: the score against the success line
        lower = bool(getattr(self.S, "LOWER_WINS", False))

        def total(m):
            t = m.scores()
            a, b = m.seats
            if t[a] == t[b]:
                return None
            return (a if t[a] < t[b] else b) if lower else (a if t[a] > t[b] else b)

        def rulebook(m):
            return m.M.winner()

        def clock(m):
            left = (m.shots[-1]["clock_left_s"] if m.shots else None) or {}
            o0, o1 = m.order
            return o0 if left.get(o0, 0.0) > left.get(o1, 0.0) else o1
        return list(zip(TIEBREAKS, (total, rulebook, clock)))

    # observation
    def observe(self, seat, query=None):
        done = self.M.done(seat)
        out = {"decision": None if done else self.M.decision(seat), "game_over": done}
        last = self.shots[-1] if self.shots else None
        if self.duel:
            o = self.other(seat)
            out["score"] = {"you": self.M.summary(seat), "opponent": self.M.summary(o)}
            out["first_player"] = "you" if self.order[0] == seat else "opponent"
            if last:
                out["last_shot"] = {"by": "you" if last["seat"] == seat else "opponent", **last["result"]}
        else:
            out["score"] = self.M.summary(seat)
            if last:
                out["last_shot"] = dict(last["result"])
        return out

    def views(self, seat) -> dict:
        """The camera images of seat's decision point ({tile name: HxWx3 array}); a plugin may override (cache)."""
        return self.M.views(seat)

    def tiles(self) -> list[dict]:
        return [{"camera": k, "x": x, "y": y, "width": w, "height": h} for k, (x, y, w, h) in self.S.TILES.items()]

    def camera_job(self, seat):
        if self.M.done(seat):
            return None, {}
        key = json.dumps(self.M.decision(seat), sort_keys=True, default=float)
        if self._png.get(seat, (None,))[0] != key:
            import numpy as np
            from PIL import Image
            ims = self.views(seat)
            W, H = self.S.IMAGE_SIZE
            canvas = np.zeros((H, W, 3), dtype=np.uint8)
            for k, (x, y, w, h) in self.S.TILES.items():
                if k in ims:
                    canvas[y:y + h, x:x + w] = ims[k][:h, :w, :3]
            b = io.BytesIO()
            Image.fromarray(canvas).save(b, format="PNG", compress_level=1)
            self._png[seat] = (key, b.getvalue())
        return {"png": self._png[seat][1]}, {"image_layout": self.S.IMAGE_LAYOUT, "tiles": self.tiles()}

    def final_fields(self):
        by = {s: [x for x in self.shots if x["seat"] == s] for s in self.seats}
        return {"scores": {s: self.M.score(s, self.clock_out[s]) for s in self.seats},
                "first_player": self.order[0], "order": self.order, "n_tiebreaks": self.M.n_tiebreaks,
                "clock_out": dict(self.clock_out), "complete": {s: self.M.done(s) for s in self.seats},
                "state_sha_by_seat": {s: self.M.state_sha(s) for s in self.seats},
                "actions": {s: [dict(x["decision"], shot_n=x["shot_n"], clock_left_s=x["clock_left_s"]) for x in v]
                            for s, v in by.items()},
                "results": {s: [x["result"] for x in v] for s, v in by.items()}}


class ShotLayer(SchemaLayer):
    """The one action level of a turnshot sport: the body is checked against the plugin's schema (strict: unknown
    fields refused), then by the rules module's own validate() (its rounding and messages), and both chess clocks
    at the shot (ms, as the legacy service recorded them) go into the command for the clock last resort."""

    def __init__(self, rules_of):
        super().__init__()
        self.rules_of = rules_of                     # match -> its rules module

    def parse(self, seat, body, match, ctx):
        check_command(match.schema(seat), body)
        dec = self.rules_of(match).validate(body)
        if isinstance(dec, str):
            raise Refused(dec)
        clk = ctx.get("clock_left_s") or {}
        return {"type": body["type"], **dec, "clock_left_s": {s: round(max(0.0, float(v)), 3) for s, v in clk.items()}}

    def reply(self, seat, cmd, info, match):
        return {"accepted": {k: v for k, v in cmd.items() if k not in ("type", "clock_left_s")}, **info}
