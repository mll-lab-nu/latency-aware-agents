#!/usr/bin/env python3
"""The players list (players/players.toml): load it and check it. Standard library only (python >= 3.11; the run
scripts' `think-model` also works on python 3.10, with a reader of the few string fields it needs).

    python3 players/players.py check              # exit 1 on any problem
    python3 players/players.py list               # id, model, harness, version
    python3 players/players.py think-model ID     # the seat's model lock THINK_MODEL_<seat> (empty for built-ins)
    python3 players/players.py chatgpt-model ID   # the ChatGPT-backend model a subscription seat is pinned to
                                                  # (THINK_CHATGPT_MODEL_<seat>; empty: no pin)
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:          # python 3.10 (the run scripts' hosts): load_simple() reads what they need
    tomllib = None

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
HARNESSES = ("claude-code", "codex", "builtin")
PLAYS = ("program",)                   # direct (no-code) play: branch archive/direct-play
REQUIRED = ("id", "model", "harness", "agent_import_path", "play", "config_version")
BUILTIN_PATHS = ("oracle", "nop")
SECRET = re.compile(r"(sk-[A-Za-z0-9_-]{16,}|sk-ant-|ghp_[A-Za-z0-9]{20,}|eyJ[A-Za-z0-9_-]{20,}|/home/|/data/|/Users/)")


def load(path: Path = HERE / "players.toml") -> list[dict]:
    if tomllib is None:
        return load_simple(path)
    return tomllib.loads(path.read_text())["player"]


SIMPLE_FIELDS = ("id", "model", "deployment", "harness", "route")
_SIMPLE = re.compile(r'^(%s)\s*=\s*"([^"\\]*)"\s*(#.*)?$' % "|".join(SIMPLE_FIELDS))


def load_simple(path: Path = HERE / "players.toml") -> list[dict]:
    """Without tomllib: the [[player]] tables with their one-line plain string fields id, model, deployment, harness, route
    (a CI test keeps this equal to tomllib's reading of the same fields)."""
    players: list[dict] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line == "[[player]]":
            players.append({})
        elif players and (m := _SIMPLE.match(line)):
            players[-1][m.group(1)] = m.group(2)
    return players


def bare_model(name) -> str:
    """As the metering proxy compares models: 'openrouter/openai/gpt-x:online' -> 'gpt-x'."""
    return str(name or "").rsplit("/", 1)[-1].split(":", 1)[0].strip().lower()


def think_model(p: dict) -> str:
    """The model lock of a seat this player plays (THINK_MODEL_<seat>, players/metering/think_proxy.py): the model the
    harness names on the wire (provider prefixes dropped), plus the Azure deployment when its name differs (the pinned
    route serves that deployment either way). Empty for a built-in player (no model). User decision 2026-10-08 (P21, point 2):
    every run script sets it for every model seat; the proxy's first-model lock is only the safety net."""
    if p.get("harness") == "builtin":
        return ""
    if not p.get("model") or p.get("model") == "*":
        raise ValueError(f"{p.get('id')}: no fixed model to lock (model {p.get('model')!r})")
    names = [bare_model(p["model"])] + ([bare_model(p["deployment"])] if p.get("deployment") else [])
    return ",".join(dict.fromkeys(n for n in names if n))


def chatgpt_model(p: dict) -> str:
    """The model a ChatGPT subscription player's seat is pinned to (THINK_CHATGPT_MODEL_<seat>, think_proxy): its
    `deployment` (bare) for route "chatgpt"; empty otherwise (no pin)."""
    return bare_model(p.get("deployment")) if p.get("route") == "chatgpt" and p.get("deployment") else ""


def module_file(import_path: str) -> Path | None:
    mod = import_path.split(":", 1)[0]
    if not mod.startswith(("players.", "harbor_agents.")):
        return None
    p = ROOT.joinpath(*mod.split("."))
    return p.with_suffix(".py") if p.with_suffix(".py").is_file() else (p / "__init__.py" if p.is_dir() else None)


def validate(players: list[dict] | None = None, text: str | None = None) -> list[str]:
    players = load() if players is None else players
    text = (HERE / "players.toml").read_text() if text is None else text
    errs, seen = [], set()
    for i, p in enumerate(players):
        who = p.get("id", f"#{i}")
        errs += [f"{who}: missing field {f}" for f in REQUIRED if f not in p]
        if who in seen:
            errs.append(f"{who}: duplicate id")
        seen.add(who)
        if p.get("harness") not in HARNESSES:
            errs.append(f"{who}: harness must be one of {HARNESSES}")
        if not set(p.get("play", [])) <= set(PLAYS) or not p.get("play"):
            errs.append(f"{who}: play must be a non-empty subset of {PLAYS}")
        ip = p.get("agent_import_path", "")
        if p.get("harness") == "builtin":
            if ip not in BUILTIN_PATHS:
                errs.append(f"{who}: a built-in player's agent_import_path is one of {BUILTIN_PATHS}")
        else:
            if ":" not in ip or module_file(ip) is None:
                errs.append(f"{who}: agent_import_path {ip!r} is not module:Class under players/ or harbor_agents/")
            elif f"class {ip.split(':')[1]}" not in module_file(ip).read_text():
                errs.append(f"{who}: {ip!r}: no class {ip.split(':')[1]} in {module_file(ip).relative_to(ROOT)}")
            if not p.get("harness_version"):
                errs.append(f"{who}: harness_version is required (pinned) for a model player")
        if p.get("harness") != "builtin" and p.get("model") not in (None, "", "*") \
                and not think_model(p):                # the run scripts lock every model seat to it
            errs.append(f"{who}: empty model lock (THINK_MODEL)")
        for name in p.get("account_env", []):
            if not re.fullmatch(r"[A-Z][A-Z0-9_]*", name):
                errs.append(f"{who}: account_env holds variable names only, got {name!r}")
        if p.get("route") == "chatgpt" and p.get("account_scope") != "seat":
            errs.append(f"{who}: a ChatGPT subscription player (route chatgpt) needs account_scope = \"seat\"")
        if p.get("account_scope", "task") not in ("task", "seat"):
            errs.append(f"{who}: account_scope must be \"seat\" (or absent), got {p.get('account_scope')!r}")
    seat_only = {v for p in players if p.get("account_scope") == "seat" for v in p.get("account_env", [])}
    for p in players:                        # a seat-scoped variable is never also passed to every proxy
        shared = seat_only & set(p.get("account_env", [])) if p.get("account_scope", "task") != "seat" else set()
        if shared:
            errs.append(f"{p.get('id')}: account_env {sorted(shared)} is seat-scoped by another player "
                        "(account_scope = \"seat\")")
    if SECRET.search(text):
        errs.append("players.toml contains something that looks like a secret or a host path")
    return errs


def main(argv: list[str]) -> int:
    if argv[:1] == ["list"]:
        for p in load():
            print(f"{p['id']:12} {p['model']:36} {p['harness']:12} {p.get('harness_version', '-')}")
        return 0
    if argv[:1] == ["think-model"] and len(argv) == 2:
        found = [p for p in load() if p.get("id") == argv[1]]
        if not found:
            print(f"think-model: no player {argv[1]!r} in players.toml", file=sys.stderr)
            return 1
        try:
            print(think_model(found[0]))
        except ValueError as e:
            print(f"think-model: {e}", file=sys.stderr)
            return 1
        return 0
    if argv[:1] == ["chatgpt-model"] and len(argv) == 2:
        found = [p for p in load() if p.get("id") == argv[1]]
        if not found:
            print(f"chatgpt-model: no player {argv[1]!r} in players.toml", file=sys.stderr)
            return 1
        print(chatgpt_model(found[0]))
        return 0
    if argv[:1] == ["check"]:
        if tomllib is None:
            print("players check needs python >= 3.11 (tomllib)", file=sys.stderr)
            return 2
        errs = validate()
        for e in errs:
            print("FAIL [players]", e)
        print(f"players: {len(load())} player(s), {len(errs)} problem(s)")
        return 1 if errs else 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
