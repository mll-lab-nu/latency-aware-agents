"""Trusted launcher of a task's sim container: one metering proxy per seat (the unchanged think_proxy.py of players/
metering, copied by the generator to metering/think_proxy.py next to core/; seat i on port 18090 + i; model
credentials stay in this container) and the air-hockey pilot match service (pilots/airhockey_modes/rt_server.py).

Per-seat settings of a seat's proxy (the runner sets them; the compose file passes them through):
  RO_SEAT_SELECT_<seat> = "VAR=SOURCE,..."   VAR takes SOURCE's value (e.g. a second account's credential)
  RO_SEAT_ENV_<VAR>_<seat> = value          a non-empty value replaces VAR (e.g. the route of OpenAI-style calls)
"""
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SEATS = os.environ.get("RO_SEATS", "A,B").split(",")
DATA = os.environ.get("RCB_DATA", "/data")
PROXY = ROOT.parent / os.environ.get("RO_METER_PROXY", "metering/think_proxy.py")


def seat_env(seat: str, env: dict, seats=None) -> dict:
    """The environment of `seat`'s metering proxy."""
    seats = seats or SEATS
    env = dict(env)
    env["THINK_CLIENT_HOST"] = "seat_" + seat.lower() if len(seats) > 1 else env.get("THINK_CLIENT_HOST", "player")
    env["THINK_USAGE_FILE"] = f"{DATA}/think_usage_{seat}.json"
    env["THINK_CALLS_FILE"] = f"{DATA}/think_calls_{seat}.jsonl"
    for pair in filter(None, env.get(f"RO_SEAT_SELECT_{seat}", "").split(",")):
        var, _, src = pair.partition("=")
        if var and src in env:
            env[var] = env[src]
    for key in [k for k in env if k.endswith("_" + seat) and k.startswith("RO_SEAT_ENV_")]:
        if env[key]:
            env[key[len("RO_SEAT_ENV_"):-len(seat) - 1]] = env[key]
    return env


def main():
    if len(sys.argv) > 1:                       # one seat's proxy
        seat = sys.argv[1]
        env = seat_env(seat, dict(os.environ))
        os.execve(sys.executable, [sys.executable, str(PROXY), str(18090 + SEATS.index(seat))], env)
    Path(DATA).mkdir(exist_ok=True)
    for s in SEATS:
        open(f"{DATA}/think_calls_{s}.jsonl", "a").close()
    procs = []
    try:
        if PROXY.exists():
            for s in SEATS:
                procs.append(subprocess.Popen([sys.executable, __file__, s]))
        else:
            print(f"no metering proxy at {PROXY}: model seats cannot play", file=sys.stderr, flush=True)
        procs.append(subprocess.Popen([sys.executable, str(ROOT.parent / "pilots" / "airhockey_modes" / "rt_server.py")], cwd=ROOT.parent))
        while all(p.poll() is None for p in procs):
            time.sleep(.5)
        raise RuntimeError("a required service exited")
    finally:
        for p in procs:
            if p.poll() is None:
                p.terminate()
        for p in procs:
            try:
                p.wait(timeout=3)
            except subprocess.TimeoutExpired:
                p.kill()


if __name__ == "__main__":
    main()
