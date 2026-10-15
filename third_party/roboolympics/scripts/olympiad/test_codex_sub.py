"""Offline tests of the ChatGPT subscription players astra-sub, sol-sub (GPT-6.1 SOL) and luna-sub (players/players.toml):
their credentials CODEX_ACCESS_TOKEN / CODEX_ACCOUNT_ID are seat-scoped (account_scope = "seat"):
  * the generated compose passes them per seat only (RO_SEAT_ENV_CODEX_*_<seat>), never unsuffixed;
  * core/launch.py gives them to the astra-sub seat's metering proxy only; in a duel no value of the other seat's proxy
    environment holds them (no RO_SEAT_ENV_* / RO_SEAT_SELECT_* is left in any proxy), in both directions;
  * a *-sub seat is on route "chatgpt" end to end: its proxy has no task-wide credential (no OpenRouter key) and
    refuses every call outside the ChatGPT backend, so gpt-6-sol posted to /v1/responses never reaches OpenRouter;
  * the subscription's rate-limit headers (x-codex-*) reach think_usage and think_calls (codex_limits);
  * that proxy forwards the ChatGPT backend's Codex calls with the real credentials (fake upstream: no model is called),
    the other seat's proxy refuses them without reaching the upstream; no credential is logged;
  * sol-sub's seat sends Codex's family slug gpt-6-sol and its proxy pins every ChatGPT-backend call to gpt-6.1-sol
    (THINK_CHATGPT_MODEL_<seat>), under the seat's model lock;
  * the harness (players/harnesses/codex_chatgpt_think.py) gives the agent only a placeholder auth.json and a Codex
    config that talks to its seat's proxy (Harbor stubbed when it is not installed).
Standard library only (python >= 3.11: the generator reads tomllib):
    python3 scripts/olympiad/test_codex_sub.py"""
import base64
import importlib
import importlib.util
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import types
from dataclasses import dataclass, field
from http.server import ThreadingHTTPServer
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "olympiad"))
sol = importlib.import_module("test_sol_route")            # free_port, Proxy helpers, launch (core/launch.py)
TP_SPEC = importlib.util.spec_from_file_location("ro_test_think_proxy_cs", REPO / "players" / "metering" / "test_think_proxy.py")
tp = importlib.util.module_from_spec(TP_SPEC)
TP_SPEC.loader.exec_module(tp)
TOKEN, ACCOUNT = "sub-token-value", "sub-account-value"     # stand-ins: nothing real is used


def test_player_row_matches_the_runners():
    toml = (REPO / "players" / "players.toml").read_text()
    blocks = {re.search(r'id = "(\w[\w-]*)"', b).group(1): b for b in toml.split("[[player]]")[1:]}
    sub, astra = blocks["astra-sub"], blocks["astra"]
    assert 'model = "openai/gpt-6-astra"' in sub and 'account_scope = "seat"' in sub
    assert 'agent_import_path = "players.harnesses.codex_chatgpt_think:CodexChatGPTThink"' in sub
    assert 'account_env = ["CODEX_ACCESS_TOKEN", "CODEX_ACCOUNT_ID"]' in sub
    for key in ("harness_version", "agent_kwargs", "disabled_tools"):   # same as astra, effort medium included
        row = lambda b: re.search(key + r" = .*", b).group(0)
        assert row(sub) == row(astra), key
    assert 'reasoning_effort = "medium"' in sub and 'route = "chatgpt"' in sub
    seat = "players.harnesses.codex_chatgpt_think:CodexChatGPTThink@"
    single = (REPO / "scripts" / "olympiad" / "run_single.sh").read_text()
    assert seat + "$1" in single and "astra-sub) codex_sub openai/gpt-6-astra ;;" in single
    assert seat + "openai/gpt-6-astra" in (REPO / "scripts" / "olympiad" / "run_duel.sh").read_text()
    def cli(*a):
        return subprocess.run([sys.executable, str(REPO / "players" / "players.py"), *a], capture_output=True,
                              text=True).stdout.strip()
    assert cli("think-model", "astra-sub") == "gpt-6-astra" and cli("chatgpt-model", "astra-sub") == ""
    assert cli("think-model", "sol-sub") == "gpt-6-sol,gpt-6.1-sol" and cli("chatgpt-model", "sol-sub") == "gpt-6.1-sol"
    assert cli("think-model", "luna-sub") == "gpt-6-luna" and cli("chatgpt-model", "luna-sub") == ""
    for pid, model in (("sol-sub", "openai/gpt-6-sol"), ("luna-sub", "openai/gpt-6-luna")):
        b = blocks[pid]
        assert f'model = "{model}"' in b and 'account_scope = "seat"' in b and 'route = "chatgpt"' in b, pid
        for key in ("harness_version", "agent_import_path", "agent_kwargs", "disabled_tools", "account_env"):
            assert re.search(key + r" = .*", b).group(0) == re.search(key + r" = .*", sub).group(0), (pid, key)
        assert f"{seat}{model}" in (REPO / "scripts" / "olympiad" / "run_duel.sh").read_text(), pid
        assert f"{pid}) " in single and f"codex_sub {model} ;;" in single, pid
    assert 'deployment = "gpt-6.1-sol"' in blocks["sol-sub"]


def test_generated_compose_passes_codex_per_seat_only():
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run([sys.executable, str(REPO / "tools" / "generate.py"), "--out", tmp, "curling"], check=True,
                       capture_output=True)
        composes = {p.parent.parent.name: p.read_text() for p in Path(tmp).glob("*/environment/docker-compose.yaml")}
    assert composes and any("duel" in t for t in composes), list(composes)
    for task, text in composes.items():
        seats = re.search(r'RO_SEATS: "([^"]+)"', text).group(1).split(",")
        for s in seats:
            for v in ("CODEX_ACCESS_TOKEN", "CODEX_ACCOUNT_ID", "THINK_CHATGPT_MODEL"):
                assert f'RO_SEAT_ENV_{v}_{s}: "${{{v}_{s}:-}}"' in text, (task, v, s)
        assert not re.search(r"^\s+CODEX_(ACCESS_TOKEN|ACCOUNT_ID)\s*:", text, re.M), f"{task}: unsuffixed CODEX_*"
        assert re.search(r"^\s+CLAUDE_CODE_OAUTH_TOKEN:", text, re.M), task          # task-scoped ones unchanged


TASK_WIDE = {"OPENROUTER_API_KEY": "openrouter-value", "AZURE_AI_API_KEY": "azure-value",
             "AZURE_AI_ENDPOINT": "http://azure.invalid", "CLAUDE_CODE_OAUTH_TOKEN": "claude-value"}


def sim_env(upstream: str, seats: dict) -> dict:
    """The sim container's environment as the compose file and a run script set it: every per-seat variable as
    RO_SEAT_ENV_<VAR>_<seat> (empty unless the runner exported <VAR>_<seat>), the task-wide credentials unsuffixed
    (the compose file has no unsuffixed CODEX_*). seats: {seat: {VAR: value}} of what the runner exported."""
    env = {"PATH": os.environ.get("PATH", ""), "THINK_OPENAI_ROUTE": "azure", "THINK_CHATGPT_UPSTREAM": upstream,
           "THINK_ANTHROPIC_UPSTREAM": upstream, "THINK_UPSTREAM": upstream, **TASK_WIDE}
    for s, given in seats.items():
        env[f"PLAYER_MODE_{s}"] = "model"
        env[f"RO_SEAT_SELECT_{s}"] = ""
        for v in ("THINK_OPENAI_ROUTE", "THINK_MODEL", "THINK_CHATGPT_MODEL", "CODEX_ACCESS_TOKEN", "CODEX_ACCOUNT_ID",
                  *TASK_WIDE):
            env[f"RO_SEAT_ENV_{v}_{s}"] = given.get(v, "")
    return env


def duel_envs(upstream: str):
    """The proxies' environments of a duel astra-sub (A) vs astra (B, the same model on Azure: its proxy would accept a
    ChatGPT-backend call for gpt-6-astra if it had the subscription), as run_duel.sh sets it (route() and seat_codex:
    CODEX_*_A and THINK_OPENAI_ROUTE_A=chatgpt)."""
    env = sim_env(upstream, {"A": {"THINK_OPENAI_ROUTE": "chatgpt", "THINK_MODEL": "gpt-6-astra",
                                   "CODEX_ACCESS_TOKEN": TOKEN, "CODEX_ACCOUNT_ID": ACCOUNT},
                             "B": {"THINK_OPENAI_ROUTE": "azure", "THINK_MODEL": "gpt-6-astra"}})
    return sol.launch.seat_env("A", env, ["A", "B"]), sol.launch.seat_env("B", env, ["A", "B"])


def no_value_holds(env: dict, *secrets: str) -> list:
    return [(k, s) for k, v in env.items() for s in secrets if s in str(v)]


def test_launcher_gives_codex_to_its_seat_only():
    a, b = duel_envs("http://x")
    assert a["CODEX_ACCESS_TOKEN"] == TOKEN and a["CODEX_ACCOUNT_ID"] == ACCOUNT
    assert not no_value_holds(b, TOKEN, ACCOUNT), no_value_holds(b, TOKEN, ACCOUNT)     # B: none of A's subscription
    for e in (a, b):
        assert not [k for k in e if k.startswith(("RO_SEAT_ENV_", "RO_SEAT_SELECT_"))], sorted(e)
    assert "PLAYER_MODE_B" not in a and "PLAYER_MODE_A" not in b
    assert b["AZURE_AI_API_KEY"] == "azure-value"                     # an API seat keeps the task-wide credentials


def test_launcher_two_subscription_seats_hold_only_their_own():
    """Two *-sub seats (e.g. two ChatGPT accounts): neither proxy holds a value of the other's, in either direction."""
    ta, ia, tb, ib = "sub-token-of-A", "sub-account-of-A", "sub-token-of-B", "sub-account-of-B"
    env = sim_env("http://x", {"A": {"THINK_OPENAI_ROUTE": "chatgpt", "CODEX_ACCESS_TOKEN": ta, "CODEX_ACCOUNT_ID": ia},
                               "B": {"THINK_OPENAI_ROUTE": "chatgpt", "CODEX_ACCESS_TOKEN": tb, "CODEX_ACCOUNT_ID": ib,
                                     "THINK_CHATGPT_MODEL": "gpt-6.1-sol"}})
    a, b = sol.launch.seat_env("A", env, ["A", "B"]), sol.launch.seat_env("B", env, ["A", "B"])
    assert (a["CODEX_ACCESS_TOKEN"], a["CODEX_ACCOUNT_ID"]) == (ta, ia) and (b["CODEX_ACCESS_TOKEN"], b["CODEX_ACCOUNT_ID"]) == (tb, ib)
    assert not no_value_holds(a, tb, ib) and not no_value_holds(b, ta, ia), (no_value_holds(a, tb, ib), no_value_holds(b, ta, ia))
    assert b["THINK_CHATGPT_MODEL"] == "gpt-6.1-sol" and "THINK_CHATGPT_MODEL" not in a
    for e in (a, b):                                                  # a subscription seat: no task-wide credential
        assert not set(TASK_WIDE) & set(e) and not no_value_holds(e, *TASK_WIDE.values()), sorted(e)


def test_sub_seat_never_reaches_openrouter():
    """The review's case: a sol-sub seat with OPENROUTER_API_KEY in .env posts gpt-6-sol (which its lock admits) to
    /v1/responses: refused by its proxy, nothing reaches any upstream; the ChatGPT backend still works."""
    seen = []

    class Up(tp.Upstream):
        def _reply(self):
            seen.append(self.path)
            super()._reply()
        do_GET = do_POST = _reply

    up = ThreadingHTTPServer(("127.0.0.1", 0), Up)
    threading.Thread(target=up.serve_forever, daemon=True).start()
    try:
        env = sim_env(f"http://127.0.0.1:{up.server_address[1]}",
                      {"A": {"THINK_OPENAI_ROUTE": "chatgpt", "THINK_MODEL": "gpt-6-sol,gpt-6.1-sol",
                             "THINK_CHATGPT_MODEL": "gpt-6.1-sol", "CODEX_ACCESS_TOKEN": TOKEN,
                             "CODEX_ACCOUNT_ID": ACCOUNT}})
        a = sol.launch.seat_env("A", env, ["A"])
        assert "OPENROUTER_API_KEY" not in a and a["THINK_OPENAI_ROUTE"] == "chatgpt", sorted(a)
        body = {"model": "gpt-6-sol", "input": "x"}
        with sol.Proxy(a) as pa:
            r1 = tp.call(pa.port, "POST", "/v1/responses", body)
            r2 = tp.call(pa.port, "POST", "/v1/responses", dict(body, model="openai/gpt-6-sol"))
            r3 = tp.call(pa.port, "POST", "/v1/chat/completions", body)
            assert not seen, seen
            r4 = tp.call(pa.port, "POST", "/backend-api/codex/responses", body)
            recs = pa.calls(4)
        assert (r1[0], r2[0], r3[0], r4[0]) == (403, 403, 403, 200), (r1, r2, r3, r4)
        assert seen == ["/backend-api/codex/responses"], seen
        assert [r.get("route") for r in recs if not r.get("blocked")] == ["chatgpt"], recs
    finally:
        up.shutdown()


def test_each_proxy_with_its_own_credentials():
    seen = []

    class Up(tp.Upstream):
        def _reply(self):
            seen.append((self.path, self.headers.get("Authorization"), self.headers.get("chatgpt-account-id")))
            super()._reply()
        do_GET = do_POST = _reply

    up = ThreadingHTTPServer(("127.0.0.1", 0), Up)
    threading.Thread(target=up.serve_forever, daemon=True).start()
    try:
        a, b = duel_envs(f"http://127.0.0.1:{up.server_address[1]}")
        body = {"model": "gpt-6-astra", "input": "x"}
        fake = {"Authorization": "Bearer placeholder", "chatgpt-account-id": "placeholder-account"}
        with sol.Proxy(a) as pa, sol.Proxy(b) as pb:
            ca = tp.call(pa.port, "POST", "/backend-api/codex/responses", body, fake)
            cm = tp.call(pa.port, "GET", "/backend-api/codex/models?client_version=0.158.0", None, fake)
            cb = tp.call(pb.port, "POST", "/backend-api/codex/responses", body, fake)
            other = tp.call(pa.port, "POST", "/backend-api/codex/responses", {"model": "gpt-6-sol", "input": "x"}, fake)
            logs = json.dumps(pa.calls(3) + pb.calls(1))
        assert ca[0] == 200 and cm[0] == 200, (ca, cm)
        assert seen == [("/backend-api/codex/responses", f"Bearer {TOKEN}", ACCOUNT),
                        ("/backend-api/codex/models?client_version=0.158.0", f"Bearer {TOKEN}", ACCOUNT)], seen
        assert cb[0] == 500 and "CODEX_ACCESS_TOKEN" in json.dumps(cb[1]), cb      # B: no subscription, no upstream
        assert other[0] == 403, other                                              # the seat's model lock
        assert TOKEN not in logs and ACCOUNT not in logs and "placeholder-account" not in logs
    finally:
        up.shutdown()


def test_sol_sub_seat_is_pinned_to_gpt_6_1_sol():
    seen = []

    class Up(tp.Upstream):
        def _reply(self):
            n = int(self.headers.get("Content-Length") or 0)
            body = self.rfile.read(n) if n else b""
            seen.append(json.loads(body).get("model") if body else None)
            out = b'{"id": "r1", "usage": {"input_tokens": 3, "output_tokens": 2}}'
            self.send_response(200); self.send_header("Content-Type", "application/json")
            self.send_header("x-codex-primary-used-percent", str(40 + len(seen)))     # the subscription's usage
            self.send_header("Content-Length", str(len(out))); self.end_headers(); self.wfile.write(out)
        do_GET = do_POST = _reply

    up = ThreadingHTTPServer(("127.0.0.1", 0), Up)
    threading.Thread(target=up.serve_forever, daemon=True).start()
    try:
        a, _ = duel_envs(f"http://127.0.0.1:{up.server_address[1]}")
        a.update(THINK_MODEL="gpt-6-sol,gpt-6.1-sol", THINK_CHATGPT_MODEL="gpt-6.1-sol")    # run_duel.sh for sol-sub
        fake = {"Authorization": "Bearer placeholder"}
        with sol.Proxy(a) as pa:
            r1 = tp.call(pa.port, "POST", "/backend-api/codex/responses", {"model": "gpt-6-sol", "input": "x"}, fake)
            r2 = tp.call(pa.port, "POST", "/backend-api/codex/responses", {"model": "gpt-6.1-sol", "input": "x"}, fake)
            r3 = tp.call(pa.port, "POST", "/backend-api/codex/responses", {"model": "gpt-6-astra", "input": "x"}, fake)
            r4 = tp.call(pa.port, "GET", "/backend-api/codex/models?client_version=0.158.0", None, fake)
            recs = pa.calls(4)
        assert (r1[0], r2[0], r3[0], r4[0]) == (200, 200, 403, 200), (r1, r2, r3, r4)
        assert seen == ["gpt-6.1-sol", "gpt-6.1-sol", None], seen      # every model call reaches 6.1; lock holds
        calls = [r for r in recs if r.get("path") == "/backend-api/codex/responses" and not r.get("blocked")]
        assert [(r["model"], r.get("model_alias")) for r in calls] == [("gpt-6.1-sol", "gpt-6-sol"), ("gpt-6.1-sol", None)]
        # the rate-limit headers: per call in think_calls, the latest in think_usage
        assert [r.get("codex_limits") for r in calls] == [{"x-codex-primary-used-percent": "41"},
                                                          {"x-codex-primary-used-percent": "42"}], calls
        models = [r for r in recs if r.get("path") == "/backend-api/codex/models"]
        assert models and models[0].get("codex_limits") == {"x-codex-primary-used-percent": "43"}, models
        usage = json.loads((Path(pa.tmp) / "usage.json").read_text())
        assert usage.get("codex_limits") == {"x-codex-primary-used-percent": "43"}, usage
    finally:
        up.shutdown()


def harness_module():
    """players/harnesses/codex_chatgpt_think.py, with Harbor stubbed when it is not installed (CI)."""
    try:
        import harbor.agents.installed.codex  # noqa: F401
        stub = False
    except ImportError:
        stub = True

        class Codex:
            def __init__(self, *a, logs_dir=None, model_name=None, **kw):
                self.model_name, self.kw = model_name, kw

            def _build_effective_config(self, openai_base_url=None):
                return {"openai_base_url": openai_base_url} if openai_base_url else {}

        @dataclass(frozen=True)
        class ResolvedModelConnection:
            provider: str | None = None
            api_key: str | None = None
            base_url: str | None = None
            configured_base_url: str | None = None
            env: dict = field(default_factory=dict)

        for name in ("harbor", "harbor.agents", "harbor.agents.installed"):
            sys.modules.setdefault(name, types.ModuleType(name))
        sys.modules["harbor.agents.installed.codex"] = types.SimpleNamespace(Codex=Codex)
        sys.modules["harbor.agents.model_connection"] = types.SimpleNamespace(ResolvedModelConnection=ResolvedModelConnection)
    sys.path.insert(0, str(REPO))
    return importlib.import_module("players.harnesses.codex_chatgpt_think"), stub


def test_harness_gives_the_agent_placeholders_only():
    mod, _ = harness_module()
    os.environ.update(OPENAI_API_KEY="host-openai-key", OPENAI_BASE_URL="http://host-openai", CODEX_ACCESS_TOKEN=TOKEN)
    try:
        agent = mod.CodexChatGPTThink(logs_dir=Path(tempfile.mkdtemp()), model_name="openai/gpt-6-astra",
                                      think_proxy_url="http://sim:18091/v1", version="0.158.0")
        conn = agent.model_connection
        assert conn.api_key == mod.PLACEHOLDER_KEY and not conn.env and not conn.configured_base_url, conn
        cfg = agent._build_effective_config(None)
        assert cfg["model_provider"] == "chatgpt_think" and "openai_base_url" not in cfg
        assert cfg["model_providers"]["chatgpt_think"]["base_url"] == "http://sim:18091/backend-api/codex"
        assert cfg["model_providers"]["chatgpt_think"]["requires_openai_auth"] is True
        assert cfg["web_search"] == "disabled" and cfg["features"]["apps"] is False
        text = agent._resolve_auth_json_path().read_text()
        auth = json.loads(text)
        assert TOKEN not in text and "host-openai-key" not in text and auth["OPENAI_API_KEY"] is None
        payload = auth["tokens"]["id_token"].split(".")[1]
        claims = json.loads(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)))
        assert claims["https://api.openai.com/auth"]["chatgpt_account_id"] == mod.PLACEHOLDER_ACCOUNT
        assert claims["exp"] > claims["iat"] + 7 * 86400                       # no refresh during a match
    finally:
        for k in ("OPENAI_API_KEY", "OPENAI_BASE_URL", "CODEX_ACCESS_TOKEN"):
            os.environ.pop(k, None)


if __name__ == "__main__":
    names = [n for n in sorted(globals()) if n.startswith("test_")]
    bad = 0
    for n in names:
        try:
            globals()[n](); print("ok  ", n)
        except Exception:  # noqa: BLE001
            bad += 1; import traceback; traceback.print_exc(); print("FAIL", n)
    print(f"{len(names) - bad}/{len(names)} passed"); sys.exit(1 if bad else 0)
