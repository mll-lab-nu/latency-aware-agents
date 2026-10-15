"""World-level matches of the built-in teams (no plugin, no rendering): the step-4 numbers before the plugin exists.

Two teams play in sim.world.World: a scripted opponent "L1".."L3" (opponents.ScriptedTeam: World.truth_view every
control step, its own reaction delay) or a built-in team in an agent's seat (a hooks.VARIANTS name or a
builtin_players.make name: truth view + its own robots' sensors every `delay` control steps, one batch of at most 5
commands per observation taking effect `delay` steps later, as tools/screen/play.py). Commands go through
World.execute (the agent's skills). The match is the referee's: two halves of `half_s` of playing
time; a level score after the second half is a draw here (the shoot-out is not played: `--shootout` plays it).

    python3 sports/soccer_ssl/screen/worldrun.py --blue reference --yellow L2 --seed 1 --half 120 --out R.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
SPORT = os.path.dirname(HERE)
if SPORT not in sys.path:
    sys.path.insert(0, SPORT)


class ServiceLatency:
    """LATENCY (2026-10-13): the timing a program meets in the official match service (core/realtime.py), measured with
    the camera reference through the service at slowdown x10 (review screening/latency_standard_zh.md): an act takes
    effect `effect()` control steps after the step of the observation it answers (the snapshot), and the program
    observes again `wait()` steps after the effect (core/client.py: act, then wait 0.05 s, then observe). Steps are
    fixed (SSL_LAT_EFFECT=2) or drawn from a distribution (SSL_LAT_EFFECT=1:0.1,2:0.7,3:0.2; seeded by the match seed
    and the team: deterministic). SSL_LAT_TRUTH=frame: a truth variant sees the ball and the opponents as the cameras
    show them (VISION_DELAY_STEPS old) and its own robots now, like a perfect reader of the images; else the truth of
    the observation step (the old harness). Without SSL_LAT_EFFECT: the declared D (hooks.DELAY = sport.toml [timing]
    reaction_delay_steps), wait 6, truth as the frame; SSL_LAT_OLD=1: the old model (`delay` steps, observe every
    `delay` steps)."""

    def __init__(self, effect, wait, truth_frame, seed, team):
        import random  # noqa: PLC0415

        self.e, self.w, self.truth_frame = effect, wait, truth_frame
        self.rng = random.Random(int(seed) * 1009 + (1 if team == "blue" else 2))

    @staticmethod
    def parse(spec):
        if ":" not in spec:
            return [(int(spec), 1.0)]
        out = [(int(a), float(b)) for a, b in (kv.split(":") for kv in spec.split(","))]
        tot = sum(p for _, p in out)
        return [(k, p / tot) for k, p in out]

    @classmethod
    def from_env(cls, seed, team):
        e = os.environ.get("SSL_LAT_EFFECT")
        if not e:
            return None
        return cls(cls.parse(e), cls.parse(os.environ.get("SSL_LAT_WAIT", "6")),
                   os.environ.get("SSL_LAT_TRUTH", "frame") == "frame", seed, team)

    def draw(self, dist):
        if len(dist) == 1:
            return dist[0][0]
        u, acc = self.rng.random(), 0.0
        for k, p in dist:
            acc += p
            if u < acc:
                return k
        return dist[-1][0]

    def effect(self):
        return self.draw(self.e)

    def wait(self):
        return max(1, self.draw(self.w))


class AgentSide:
    """a built-in team in the seat of an agent: it observes every `delay` control steps (truth + its own robots'
    sensors), and its commands, at most one batch of 5 per observation (Outbox), take effect `delay` steps later
    (tools/screen/play.py's latency model)."""

    def __init__(self, name, team, seed):
        import hooks  # noqa: PLC0415  (screen/hooks.py)
        import builtin_players as BP  # noqa: PLC0415

        spec = hooks.VARIANTS.get(name, dict(team=name))
        self.team = BP.make(spec["team"], team=team, seed=seed, **dict(spec.get("opts", {})))
        self.team.debug = bool(os.environ.get("SSL_DEBUG"))
        self.delay = max(1, int(spec.get("delay", hooks.DELAY)))
        self.lat = ServiceLatency.from_env(seed, team)   # LATENCY: an explicit service timing from the environment
        if self.lat is None and not os.environ.get("SSL_LAT_OLD"):
            # the default (10-13): the sport's declared reaction delay D (sport.toml [timing] reaction_delay_steps =
            # hooks.DELAY, core 0.2.18): an act takes effect D steps after its observation; the program observes again
            # 6 steps after the effect (the reference's loop); a truth variant sees the ball and the opponents as the
            # cameras show them. SSL_LAT_OLD=1: the harness before 10-13 (observe every DELAY steps, the truth now).
            self.lat = ServiceLatency([(self.delay, 1.0)], [(6, 1.0)], True, seed, team)
        self.next_obs = 0
        self.tv_hist = {}
        self.color = team
        self.box = BP.Outbox(5)
        self.inflight = None  # (k_effect, commands)
        self.BP = BP
        self.acts = 0

    def apply_due(self, w):
        if self.inflight is not None and self.inflight[0] <= w.k:
            for n, skill, args in self.inflight[1]:
                w.execute(self.color, n, skill, args)
            self.inflight = None
            if self.lat is not None:
                self.next_obs = w.k + self.lat.wait()

    def observe_now(self, w):
        if self.inflight is not None:
            return False
        if self.lat is None:
            return w.k % self.delay == 0
        return w.k >= self.next_obs

    def sent(self, w, cmds):
        if self.lat is None:
            if cmds:
                self.inflight = (w.k + self.delay, cmds)
            return
        if cmds:
            self.inflight = (w.k + self.lat.effect(), cmds)
        else:
            self.next_obs = w.k + self.lat.wait()

    def tick(self, w):
        if self.lat is not None and self.lat.truth_frame:
            self.tv_hist[w.k] = w.truth_view(self.color)
            self.tv_hist.pop(w.k - w.vision_delay - 1, None)
        self.apply_due(w)
        if self.observe_now(w):
            tv = w.truth_view(self.color)
            if self.lat is not None and self.lat.truth_frame:   # the ball and the opponents as the cameras show them
                old = self.tv_hist.get(max(0, w.k - w.vision_delay)) or self.tv_hist[min(self.tv_hist)]
                tv = dict(tv, ball=old["ball"], them=old["them"])
            own = {n: w.own_state((self.color, n)) for n in tv["us"]}
            view = self.BP.view_from_truth(tv, own)
            self.box.put(self.team.update(view), view)
            cmds = self.box.take()
            if cmds:
                self.acts += 1
            self.sent(w, cmds)

    @property
    def stats(self):
        return self.team.stats


class CameraSide(AgentSide):
    """the reference in an agent's seat that sees only what an agent sees: its own robots' state, the referee, and
    the overhead camera tiles of the frame VISION_DELAY_STEPS old (World.vision_frame), drawn by sim/camera.py and
    read by builtin_players' colour detector (TeamPlayer.view_from). Needs World(cameras=True)."""

    def __init__(self, name, team, seed):
        super().__init__(name, team, seed)
        self.player = None
        self.stats_cam = dict(frames=0, ball_seen=0, ball_err=[], opp_seen=0, opp_err=[])

    def tick(self, w):
        import mujoco  # noqa: PLC0415
        import numpy as np  # noqa: PLC0415
        from sim import camera  # noqa: PLC0415

        if self.player is None:
            cams = camera.camera_defs()
            self.player = self.BP.TeamPlayer.__new__(self.BP.TeamPlayer)
            self.player.spec, self.player.team, self.player.box = {}, self.team, self.box
            self.player.track = self.BP.CameraTracker()
            self.player.cams = {c["name"]: c for c in cams}
            self.player.color, self.player.last_view = self.color, None
            self.d2 = mujoco.MjData(w.model)
            self.cams = cams
        self.apply_due(w)
        if self.observe_now(w):
            k_frame, qpos = w.vision_frame()
            self.d2.qpos[:] = qpos
            mujoco.mj_forward(w.model, self.d2)
            tiles = camera.render(w.model, self.d2, w.camera_names)
            H, W = tiles[0].shape[:2]
            img = np.zeros((2 * H, 2 * W, 3), np.uint8)
            meta = []
            for i, (tile, cam) in enumerate(zip(tiles, self.cams)):
                x0, y0 = (i % 2) * W, (i // 2) * H
                img[y0:y0 + H, x0:x0 + W] = tile
                meta.append(dict(camera=cam["name"], x=x0, y=y0, w=W, h=H))
            s = w.sign(self.color)
            st = dict(robots={n: w.own_state((self.color, n)) for n in range(w.n)}, referee=w.referee_view(self.color),
                      t_sim=float(w.data.time), frame=dict(attack_sign=s, age_s=(w.k - k_frame) * 0.01))
            view = self.player.view_from(dict(state=st, observation=dict(tiles=meta)), img=img)
            if view is None:
                if self.lat is not None:
                    self.next_obs = w.k + self.lat.wait()
                return
            self._score(w, view)
            self.box.put(self.team.update(view), view)
            cmds = self.box.take()
            if cmds:
                self.acts += 1
            self.sent(w, cmds)

    def _score(self, w, view):
        tv = self.BP.view_from_truth(w.truth_view(self.color))
        S = self.stats_cam
        S["frames"] += 1
        e = self.BP.dist(view.ball, tv.ball)
        if e < 0.3:
            S["ball_seen"] += 1
        S["ball_err"].append(round(e, 4))
        for k, o in tv.opp.items():
            if not o.on:
                continue
            if k in view.opp:
                S["opp_seen"] += 1
                S["opp_err"].append(round(self.BP.dist(view.opp[k].p, o.p), 4))

    @property
    def stats(self):
        import numpy as np  # noqa: PLC0415

        out = dict(self.team.stats)
        S = self.stats_cam
        if S["frames"]:
            be = np.array(S["ball_err"])
            out["camera"] = dict(frames=S["frames"], ball_within_0_3m=S["ball_seen"],
                                 ball_err_median_m=float(np.median(be)), ball_err_p90_m=float(np.percentile(be, 90)),
                                 opp_detections=S["opp_seen"],
                                 opp_err_median_m=float(np.median(S["opp_err"])) if S["opp_err"] else None)
        return out


class ScriptedSide:
    def __init__(self, level, team, seed, **over):
        import numpy as np  # noqa: PLC0415
        from opponents import ScriptedTeam  # noqa: PLC0415

        self.t = ScriptedTeam(level, team=team, rng=np.random.default_rng(seed), **over)
        self.t.team.debug = bool(os.environ.get("SSL_DEBUG"))
        self.color = team

    def tick(self, w):
        for c in self.t.act(w.truth_view(self.color)):
            c = dict(c)
            skill = c.pop("type")
            n = int(c.pop("entity"))
            w.execute(self.color, n, skill, c)

    @property
    def stats(self):
        return self.t.stats


def side(name, team, seed):
    if name.startswith("L") and name[1:].isdigit():
        return ScriptedSide(int(name[1:]), team, seed)
    if name.startswith("L") and ":" in name:  # a level with parameter overrides, e.g. L3:swarm_screen=1,delay=0.15
        lv, opts = name[1:].split(":", 1)
        over = {}
        for kv in opts.split(","):
            k, v = kv.split("=")
            over[k] = float(v) if v.replace(".", "", 1).isdigit() else v
        return ScriptedSide(int(lv), team, seed, **over)
    if name.startswith("cam:"):
        return CameraSide(name[4:], team, seed)
    return AgentSide(name, team, seed)


def run_match(blue, yellow, seed, half_s=300.0, shootout=False, first=None, log_every=0.0, max_world_s=None):
    sys.path.insert(0, HERE)
    from sim.world import World  # noqa: PLC0415

    first = first or ("blue" if seed % 2 else "yellow")
    w = World(seed=seed, half_s=half_s, first_kickoff=first, cameras=blue.startswith("cam:") or yellow.startswith("cam:"))
    sides = {"blue": side(blue, "blue", seed * 7 + 1), "yellow": side(yellow, "yellow", seed * 7 + 2)}
    t0 = time.time()
    max_world_s = max_world_s or (2 * half_s * 2.5 + 200)
    poss = Counter()
    trace = []
    crash = None
    while True:
        R = w.referee
        if R.stage == "POST_GAME" or (R.stage == "SHOOTOUT" and not shootout) or w.data.time > max_world_s:
            break
        for s in sides.values():
            s.tick(w)
        if w.k % 5 == 0 and R.state == "RUNNING":
            bx = float(w.ball.pos[0])
            poss["ball_in_yellow_half" if bx * R.sign("blue") > 0 else "ball_in_blue_half"] += 1
        if log_every and w.k % int(log_every / 0.01) == 0:
            trace.append(dict(t=round(w.data.time, 2), ball=[round(float(x), 3) for x in w.ball.pos[:2]],
                              state=R.state, score=dict(R.score),
                              robots={f"{rid[0][0]}{rid[1]}": [round(c, 3) for c in w.robot_pose(rid)[0]]
                                      for rid in w.ids if w.on_field[rid]}))
        try:
            w.step()
        except Exception as e:  # noqa: BLE001  (a world / referee error ends the match; recorded)
            crash = f"{type(e).__name__}: {e}"
            break
    R = w.referee
    ev = R.events
    fouls = {t: Counter() for t in ("blue", "yellow")}
    goals = []
    for e in ev:
        typ = e.get("type")
        if typ == "GOAL":
            goals.append(dict(team=e.get("team"), t=round(float(e.get("t", 0.0)), 1)))
        if e.get("foul") or typ in ("BOT_CRASH_UNIQUE", "BOT_CRASH_DRAWN", "BOT_PUSHED_BOT", "BOT_HELD_BALL_DELIBERATELY",
                                    "BOT_DRIBBLED_BALL_TOO_FAR", "ATTACKER_DOUBLE_TOUCHED_BALL",
                                    "BOT_KICKED_BALL_TOO_FAST", "BOT_TOO_FAST_IN_STOP", "DEFENDER_TOO_CLOSE_TO_KICK_POINT",
                                    "ATTACKER_TOO_CLOSE_TO_DEFENSE_AREA", "ATTACKER_TOUCHED_BALL_IN_DEFENSE_AREA",
                                    "DEFENDER_IN_DEFENSE_AREA", "KEEPER_HELD_BALL", "BOUNDARY_CROSSING",
                                    "PREPARE_TIMEOUT"):
            team = e.get("team") or e.get("by_team")
            if team in fouls:
                fouls[team][typ] += 1
    score = dict(R.score)
    res = dict(blue=blue, yellow=yellow, seed=seed, half_s=half_s, first=first, score=score,
               winner=("blue" if score["blue"] > score["yellow"] else "yellow" if score["yellow"] > score["blue"]
                       else (R.winner if shootout else "draw")),
               stage=R.stage, world_s=round(float(w.data.time), 1), play_time=round(R.play_time, 1),
               wall_s=round(time.time() - t0, 1), goals=goals, fouls={t: dict(c) for t, c in fouls.items()},
               fouls_total={t: R.fouls[t] for t in R.fouls}, cards={t: dict(R.cards_given[t]) for t in R.cards_given},
               crash=crash, possession=dict(poss), stats={t: dict(sides[t].stats) for t in sides},
               event_types=dict(Counter(e.get("type") for e in ev)),
               log={t: list(getattr(sides[t].team if hasattr(sides[t], "team") else sides[t].t.team, "log", []))
                    for t in sides} if log_every else None,
               referee_events=[e for e in ev if e.get("type") not in ("BALL_IN_PLAY",)] if log_every else None)
    if trace:
        res["trace"] = trace
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--blue", required=True)
    ap.add_argument("--yellow", required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--half", type=float, default=300.0)
    ap.add_argument("--shootout", action="store_true")
    ap.add_argument("--first", default=None)
    ap.add_argument("--trace", type=float, default=0.0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    res = run_match(a.blue, a.yellow, a.seed, a.half, a.shootout, a.first, a.trace)
    s = json.dumps(res, default=str)
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w") as f:
            f.write(s)
    print(json.dumps({k: res[k] for k in ("blue", "yellow", "seed", "score", "winner", "world_s", "wall_s",
                                          "fouls_total", "stats")}, default=str))


if __name__ == "__main__":
    main()
