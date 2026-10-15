#!/usr/bin/env python3
"""Every model seat a run script starts carries its model lock THINK_MODEL_<seat> (user decision 2026-10-08, P21 point
2: the runners set the lock explicitly; the metering proxy's first-model lock is only the safety net).

  1. players.py: the python 3.10 reader of players.toml gives what tomllib gives (the run scripts' hosts have 3.10).
  2. run_single.sh / run_duel.sh / run_race.sh / run_team.sh, run with a FAKE Harbor (the pinned fork's, so it passes
     harbor_guard; test_harbor_guard.py) in a scratch copy of the scripts:
     every seat that runs a model (`--seat X=<agent>@<model>`) has a non-empty THINK_MODEL_X equal to players.toml's
     lock for that player, and the lock admits the model the seat actually runs (as the proxy compares them); every
     built-in seat (oracle, nop) has an empty one. An unknown player stops the run before Harbor starts.
  3. The metering proxy with a runner's lock (fake upstream, no model API is called): the seat's model is forwarded,
     any other model is refused (403) and never reaches the upstream.

    python3 scripts/olympiad/test_model_lock.py      (python >= 3.11 for check 1; standard library only)
"""
from __future__ import annotations

import fnmatch
import importlib.util
import itertools
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import traceback
from http.server import ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_harbor_guard import fake_harbor     # noqa: E402  the pinned fork's fake (harbor_guard in v02_run.sh)

ROOT = Path(__file__).resolve().parents[2]
RUNNERS = ("run_single.sh", "run_duel.sh", "run_race.sh", "run_team.sh", "v02_run.sh")
SUB_PLAYERS = ("astra-sub", "sol-sub", "luna-sub")                       # ChatGPT subscription (account_scope seat)
MODEL_PLAYERS = ("opus", "fable", "astra", "sol", "luna") + SUB_PLAYERS
DUEL_PLAYERS = MODEL_PLAYERS + ("oracle",)
SINGLE_PLAYERS = ("opus", "fable", "astra", "luna", "oracle", "nop") + SUB_PLAYERS     # run_single.sh has no sol
RACE_TEAM_PLAYERS = ("opus", "fable", "astra", "luna")                    # nor do run_race.sh / run_team.sh

def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


PLAYERS = load_module("ro_players", ROOT / "players" / "players.py")


def lock_admits(lock: str, model: str) -> bool:
    """think_proxy.model_allowed with THINK_MODEL = lock (names or fnmatch patterns, provider prefixes ignored)."""
    pats = [PLAYERS.bare_model(p) for p in lock.split(",") if p.strip()]
    return bool(pats) and any(fnmatch.fnmatchcase(PLAYERS.bare_model(model), p) for p in pats)


def scratch_repo(tmp: Path) -> Path:
    """The run scripts, players/ and the task directories with only their task.toml (run_race.sh reads the seats), so
    a runner finds everything and writes only here."""
    repo = tmp / "repo"
    (repo / "scripts" / "olympiad").mkdir(parents=True)
    for f in RUNNERS:
        shutil.copy2(ROOT / "scripts" / "olympiad" / f, repo / "scripts" / "olympiad" / f)
    (repo / "players").mkdir()
    for f in ("players.py", "players.toml"):
        shutil.copy2(ROOT / "players" / f, repo / "players" / f)
    for t in ROOT.glob("tasks/*/task.toml"):
        (repo / "tasks" / t.parent.name).mkdir(parents=True)
        shutil.copy2(t, repo / "tasks" / t.parent.name / "task.toml")
    (tmp / "fake_harbor").symlink_to(fake_harbor(tmp / "harbor_venv"))     # passes harbor_guard; records each trial
    sol_env = tmp / "sol.env"                     # placeholder values: nothing is called
    sol_env.write_text("AZURE_SOL_ENDPOINT=http://127.0.0.1:9\nAZURE_SOL_API_KEY=placeholder\n")
    (tmp / "codex.env").write_text("CODEX_ACCESS_TOKEN=placeholder-token\nCODEX_ACCOUNT_ID=placeholder-account\n")
    return repo


def run(tmp: Path, repo: Path, script: str, *args: str, extra_env: dict | None = None) -> tuple[int, list[dict]]:
    out = tmp / "harbor_calls.jsonl"
    out.unlink(missing_ok=True)
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("THINK_", "PLAYER_MODE_", "RCB_", "OPUS_ACCOUNT", "FABLE_ACCOUNT"))}
    env.update(HARBOR=str(tmp / "fake_harbor"), FAKE_HARBOR_OUT=str(out), SOL_ENV_FILE=str(tmp / "sol.env"),
               CODEX_ENV_FILE=str(tmp / "codex.env"))
    env.update(extra_env or {})
    # an inherited lock must not leak into a seat: the runner sets every seat itself
    env.update(THINK_MODEL_A="stale-model", THINK_MODEL_B="stale-model",
               CODEX_ACCESS_TOKEN_A="stale-token", CODEX_ACCESS_TOKEN_B="stale-token",
               THINK_CHATGPT_MODEL_A="stale-model", THINK_CHATGPT_MODEL_B="stale-model")
    p = subprocess.run(["bash", str(repo / "scripts" / "olympiad" / script), *args], cwd=repo, env=env,
                       capture_output=True, text=True, timeout=120)
    calls = [json.loads(x) for x in out.read_text().splitlines()] if out.exists() else []
    return p.returncode, calls


def seats_of(argv: list[str]) -> dict[str, str]:
    """{seat: agent spec} from the harbor arguments (`--seat A=module:Class@model` or `--seat A=oracle`)."""
    return {v.split("=", 1)[0]: v.split("=", 1)[1] for k, v in zip(argv, argv[1:]) if k == "--seat"}


def check_call(call: dict, players_by_seat: dict[str, str], toml: dict[str, dict]) -> None:
    seats = seats_of(call["argv"])
    assert set(players_by_seat) <= set(seats), (players_by_seat, seats)
    for seat, spec in seats.items():
        lock = call["env"].get(f"THINK_MODEL_{seat}")
        who = players_by_seat.get(seat, "nop")
        if "@" in spec:                                   # a model seat
            model = spec.rsplit("@", 1)[1]
            assert lock, f"seat {seat} ({who}, {model}) starts without THINK_MODEL_{seat}"
            assert lock == PLAYERS.think_model(toml[who]), (seat, who, lock)
            assert lock_admits(lock, model), f"seat {seat}: lock {lock!r} refuses its own model {model!r}"
            others = [p for p in MODEL_PLAYERS
                      if PLAYERS.bare_model(toml[p]["model"]) != PLAYERS.bare_model(toml[who]["model"])]
            for o in others:                              # and refuses every other player's model
                assert not lock_admits(lock, toml[o]["model"]), (seat, lock, toml[o]["model"])
        else:                                             # a built-in seat: no model, no lock
            assert spec in ("oracle", "nop") and lock == "", (seat, spec, lock)
        # the ChatGPT subscription (account_scope = "seat") reaches the *-sub seats only, never unsuffixed, with the
        # seat's pinned backend model (sol-sub: gpt-6.1-sol)
        token = call["env"].get(f"CODEX_ACCESS_TOKEN_{seat}")
        pin = call["env"].get(f"THINK_CHATGPT_MODEL_{seat}")
        if who in SUB_PLAYERS:
            assert token == "placeholder-token" and call["env"].get(f"CODEX_ACCOUNT_ID_{seat}") == "placeholder-account"
            assert spec == "players.harnesses.codex_chatgpt_think:CodexChatGPTThink@" + toml[who]["model"], spec
            assert pin == PLAYERS.chatgpt_model(toml[who]) and (pin == "gpt-6.1-sol") == (who == "sol-sub"), (who, pin)
            assert not pin or lock_admits(lock, pin), (who, lock, pin)          # the pinned model is the seat's own
            # the ChatGPT backend only (no OpenRouter fallback), at the API players' reasoning effort
            assert call["env"].get(f"THINK_OPENAI_ROUTE_{seat}") == "chatgpt", (seat, who, call["env"])
            assert f"{seat}:reasoning_effort=medium" in call["argv"], (seat, who, call["argv"])
        else:                                             # every runner clears the other seats' copies
            assert not token and not pin, (seat, who, "has a CODEX_ACCESS_TOKEN or THINK_CHATGPT_MODEL")
            assert call["env"].get(f"THINK_OPENAI_ROUTE_{seat}") != "chatgpt", (seat, who)
    assert "CODEX_ACCESS_TOKEN" not in call["env"] and "CODEX_ACCOUNT_ID" not in call["env"], "unsuffixed CODEX_*"


def main() -> int:
    bad = 0
    tmp = Path(tempfile.mkdtemp(prefix="model-lock-test-"))
    toml = {p["id"]: p for p in PLAYERS.load()}

    def check(name, fn):
        nonlocal bad
        try:
            n = fn()
            print(f"ok   {name}" + (f" ({n} runs)" if n else ""))
        except Exception:  # noqa: BLE001
            bad += 1
            print(f"FAIL {name}\n{traceback.format_exc()}")

    def simple_reader_matches_tomllib():
        if PLAYERS.tomllib is None:
            raise RuntimeError("needs python >= 3.11 (tomllib) to compare against")
        full = PLAYERS.load()
        simple = PLAYERS.load_simple()
        assert [{k: p[k] for k in PLAYERS.SIMPLE_FIELDS if k in p} for p in full] == simple, simple
        assert [PLAYERS.think_model(p) for p in full if p["model"] != "*"] == \
               [PLAYERS.think_model(p) for p in simple if p["model"] != "*"]
        for pid in MODEL_PLAYERS:
            assert PLAYERS.think_model(toml[pid]), pid

    def think_model_cli_on_every_player():
        for pid, p in toml.items():
            r = subprocess.run([sys.executable, str(ROOT / "players" / "players.py"), "think-model", pid],
                               capture_output=True, text=True)
            if p["model"] == "*":
                assert r.returncode == 1, (pid, r.stdout)
            else:
                assert r.returncode == 0 and r.stdout.strip() == PLAYERS.think_model(p), (pid, r.stdout, r.stderr)
        r = subprocess.run([sys.executable, str(ROOT / "players" / "players.py"), "think-model", "nobody"],
                           capture_output=True, text=True)
        assert r.returncode == 1

    repo = scratch_repo(tmp)

    def run_single_sets_the_lock():
        n = 0
        cases = [("tabletennis-match-wallclock-program-standard", {}), ("archery-program-standard", {})]
        for (task, extra), agent in itertools.product(cases, SINGLE_PLAYERS):
            rc, calls = run(tmp, repo, "run_single.sh", "--task", task, "--agent", agent, "--level", "1",
                            "--seed", "1", extra_env=extra)
            assert rc == 0 and len(calls) == 1, (task, extra, agent, rc, calls)
            check_call(calls[0], {"A": agent}, toml)
            n += 1
        return n

    def run_duel_sets_the_lock():
        n = 0
        for task in ("tabletennis-duel-wallclock-program-standard", "darts-duel-program-standard"):
            for a, b in itertools.product(DUEL_PLAYERS, DUEL_PLAYERS):
                rc, calls = run(tmp, repo, "run_duel.sh", "--task", task, "--a", a, "--b", b)
                assert rc == 0 and len(calls) == 1, (task, a, b, rc, calls)
                check_call(calls[0], {"A": a, "B": b}, toml)
                n += 1
        return n

    def run_race_sets_the_lock():
        n = 0
        for players in (["opus", "astra", "luna", "fable"], ["fable", "opus", "luna", "oracle"],
                        ["luna", "luna", "astra", "oracle"]):        # the race task's seats: A-D
            rc, calls = run(tmp, repo, "run_race.sh", "--players", ",".join(players))
            assert rc == 0 and len(calls) == 1, (players, rc, calls)
            check_call(calls[0], dict(zip("ABCD", players)), toml)
            n += 1
        for players in (["opus", "astra", "luna"], ["opus", "astra", "luna", "fable", "oracle"]):
            rc, calls = run(tmp, repo, "run_race.sh", "--players", ",".join(players))
            assert rc != 0 and not calls, (players, rc, calls)    # one player per seat, or no race
        return n

    def run_team_sets_the_lock():
        n = 0
        for a, b in itertools.product(RACE_TEAM_PLAYERS + ("oracle",), repeat=2):
            rc, calls = run(tmp, repo, "run_team.sh", "--a", a, "--b", b)
            assert rc == 0 and len(calls) == 1, (a, b, rc, calls)
            check_call(calls[0], {"A1": a, "A2": a, "B1": b, "B2": b}, toml)
            n += 1
        return n

    def unknown_player_stops_before_harbor():
        for script, args in (("run_duel.sh", ["--a", "opus", "--b", "gpt-unknown"]),
                             ("run_single.sh", ["--agent", "gpt-unknown", "--level", "1", "--seed", "1"]),
                             ("run_team.sh", ["--a", "gpt-unknown", "--b", "opus"]),
                             ("run_race.sh", ["--players", "opus,astra,luna,gpt-unknown"])):
            rc, calls = run(tmp, repo, script, *args)
            assert rc != 0 and not calls, (script, rc, calls)

    def proxy_refuses_another_model_with_a_runner_lock():
        tp = load_module("ro_test_think_proxy", ROOT / "players" / "metering" / "test_think_proxy.py")
        up = ThreadingHTTPServer(("127.0.0.1", 0), tp.Upstream)
        threading.Thread(target=up.serve_forever, daemon=True).start()
        base = f"http://127.0.0.1:{up.server_address[1]}"
        procs = []
        try:
            cases = {"opus": ("/v1/messages", "claude-opus-5-5", ["claude-fable-5-1", "claude-sonnet-5", "claude-haiku-5"]),
                     "sol": ("/v1/responses", "openai/gpt-6-sol", ["openai/gpt-6-astra", "openai/gpt-6-luna", "gpt-7"]),
                     "luna": ("/v1/responses", "openai/gpt-6-luna", ["openai/gpt-6-sol", "anthropic/claude-opus-5-5"])}
            for pid, (path, own, others) in cases.items():
                p, port = tp.start_proxy(tmp, base, THINK_MODEL=PLAYERS.think_model(toml[pid]))
                procs.append(p)
                hdr = {"anthropic-version": "2023-06-01"}
                body = (lambda m: {"model": m, "max_tokens": 5, "messages": []}) if path == "/v1/messages" else \
                       (lambda m: {"model": m, "input": "x"})
                for m in others:                          # first: a different model never becomes the lock
                    tp.SEEN.clear()
                    code, out = tp.call(port, "POST", path, body(m), hdr)
                    assert code == 403 and "model" in out["error"]["message"] and not tp.SEEN, (pid, m, code, out)
                tp.SEEN.clear()
                code, out = tp.call(port, "POST", path, body(own), hdr)
                assert code == 200 and len(tp.SEEN) == 1, (pid, own, code, out)
                blocked = [r for r in tp.calls(tmp, port) if r.get("blocked")]
                assert len(blocked) == len(others), blocked
        finally:
            for p in procs:
                p.terminate()
                p.wait(timeout=5)
            up.shutdown()

    tests = (simple_reader_matches_tomllib, think_model_cli_on_every_player, run_single_sets_the_lock,
             run_duel_sets_the_lock, run_race_sets_the_lock, run_team_sets_the_lock, unknown_player_stops_before_harbor,
             proxy_refuses_another_model_with_a_runner_lock)
    for f in tests:
        check(f.__name__, f)
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"{len(tests) - bad}/{len(tests)} passed")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
