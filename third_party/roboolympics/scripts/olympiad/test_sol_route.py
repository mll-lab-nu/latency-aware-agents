"""Unit tests of the per-seat Azure routes of the metering proxy (players/metering/think_proxy.py) as the core's launcher
(core/launch.py) sets them up: THINK_OPENAI_ROUTE_<seat>=azure_sol sends that seat's OpenAI-style calls to a second
Azure resource (AZURE_SOL_ENDPOINT / AZURE_SOL_API_KEY, passed to that seat only), the other seat stays on azure
(AZURE_AI_ENDPOINT / AZURE_AI_API_KEY / AZURE_DEPLOYMENTS) exactly as before, and a missing SOL credential never falls
back to the azure one. Two fake upstreams stand in for the resources. Standard library only (python >= 3.10):
    python3 scripts/olympiad/test_sol_route.py"""
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PROXY = REPO / "players" / "metering" / "think_proxy.py"
_spec = importlib.util.spec_from_file_location("ro_launch", REPO / "core" / "launch.py")
launch = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(launch)


def free_port():
    import socket
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close()
    return p


def fake_upstream(name):
    """A fake Azure resource: records (path, api-key header, authorization header, model) and streams a completed
    Responses answer with usage."""
    hits = []

    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            hits.append((self.path, self.headers.get("api-key"), self.headers.get("Authorization"), body.get("model")))
            self.send_response(200); self.send_header("Content-Type", "text/event-stream")
            self.send_header("Connection", "close"); self.end_headers()
            for d in ({"type": "response.created", "response": {"id": "r1"}},
                      {"type": "response.output_text.delta", "delta": name},
                      {"type": "response.completed", "response": {"id": "r1", "status": "completed", "output": [],
                       "usage": {"input_tokens": 10, "output_tokens": 2, "total_tokens": 12}}}):
                self.wfile.write(b"event: " + d["type"].encode() + b"\ndata: " + json.dumps(d).encode() + b"\n\n")

    srv = ThreadingHTTPServer(("127.0.0.1", free_port()), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, hits


def seat_envs(astra_url, sol_url, sol_key="sol-key"):
    """The sim container's environment of a duel astra (A) vs sol (B), as run_duel.sh and the compose file set it."""
    env = {"PATH": os.environ.get("PATH", ""), "THINK_OPENAI_ROUTE": "azure",
           "AZURE_AI_ENDPOINT": astra_url + "/models", "AZURE_AI_API_KEY": "astra-key",
           "AZURE_DEPLOYMENTS": "gpt-6-astra,gpt-6-astra-2",
           "AZURE_SOL_ENDPOINT": "", "AZURE_SOL_API_KEY": "", "AZURE_SOL_DEPLOYMENTS": "",
           "RO_SEAT_ENV_THINK_OPENAI_ROUTE_A": "azure", "RO_SEAT_ENV_THINK_OPENAI_ROUTE_B": "azure_sol",
           "RO_SEAT_ENV_AZURE_SOL_ENDPOINT_A": "", "RO_SEAT_ENV_AZURE_SOL_API_KEY_A": "",
           "RO_SEAT_ENV_AZURE_SOL_ENDPOINT_B": sol_url + "/", "RO_SEAT_ENV_AZURE_SOL_API_KEY_B": sol_key}
    return launch.seat_env("A", env, ["A", "B"]), launch.seat_env("B", env, ["A", "B"])


class Proxy:
    def __init__(self, env):
        self.tmp = tempfile.mkdtemp()
        self.port = free_port()
        env = dict(env, THINK_CLIENT_HOST="", THINK_USAGE_FILE=f"{self.tmp}/usage.json",
                   THINK_CALLS_FILE=f"{self.tmp}/calls.jsonl")
        self.proc = subprocess.Popen([sys.executable, str(PROXY), str(self.port)], env=env,
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for _ in range(100):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{self.port}/think/usage", timeout=1); break
            except Exception:  # noqa: BLE001
                time.sleep(.05)

    def call(self, model):
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}/v1/responses", method="POST",
                                     data=json.dumps({"model": model, "stream": True, "input": "hi"}).encode(),
                                     headers={"Content-Type": "application/json",
                                              "Authorization": "Bearer rcb-think-proxy-placeholder"})
        try:
            r = urllib.request.urlopen(req, timeout=30)
            return r.status, r.read().decode()
        except urllib.error.HTTPError as e:
            return e.code, e.read().decode()

    def calls(self, n=1):
        """The call log once it has n records (the proxy writes a record after its answer)."""
        path = Path(self.tmp) / "calls.jsonl"
        for _ in range(100):
            lines = path.read_text().splitlines() if path.exists() else []
            if len(lines) >= n:
                return [json.loads(l) for l in lines]
            time.sleep(.02)
        raise AssertionError(f"no {n} call records in {path}")

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.proc.terminate(); self.proc.wait()


def test_launcher_gives_sol_variables_to_its_seat_only():
    a, b = seat_envs("http://astra", "http://sol")
    assert a["THINK_OPENAI_ROUTE"] == "azure" and b["THINK_OPENAI_ROUTE"] == "azure_sol"
    assert a["AZURE_SOL_API_KEY"] == "" and a["AZURE_SOL_ENDPOINT"] == ""
    assert b["AZURE_SOL_API_KEY"] == "sol-key" and b["AZURE_SOL_ENDPOINT"] == "http://sol/"


def test_each_seat_reaches_its_own_resource():
    sa, hits_a = fake_upstream("astra")
    ss, hits_s = fake_upstream("sol")
    try:
        a, b = seat_envs(f"http://127.0.0.1:{sa.server_port}", f"http://127.0.0.1:{ss.server_port}")
        with Proxy(a) as pa, Proxy(b) as pb:
            sta, ta = pa.call("openai/gpt-6-astra")
            stb, tb = pb.call("openai/gpt-6.1-sol")
            ca, cb = pa.calls(), pb.calls()
        assert (sta, stb) == (200, 200) and "response.completed" in ta and '"sol"' in tb
        # Astra: Foundry endpoint without /models, its key as api-key, no bearer, prefix dropped, first pool replica
        assert hits_a == [("/openai/v1/responses", "astra-key", None, "gpt-6-astra")]
        # SOL: the Azure OpenAI endpoint (trailing slash dropped), its own key
        assert hits_s == [("/openai/v1/responses", "sol-key", None, "gpt-6.1-sol")]
        assert [c["route"] for c in ca] == ["azure"] and [c["route"] for c in cb] == ["azure_sol"]
        assert ca[0].get("azure_deployment") == "gpt-6-astra" and "azure_deployment" not in cb[0]   # SOL: no pool set
        assert all(c["completed"] and c["output_tokens"] == 2 for c in ca + cb)
        logs = json.dumps(ca + cb)
        assert "astra-key" not in logs and "sol-key" not in logs                  # credentials never logged
    finally:
        sa.shutdown(); ss.shutdown()


def test_sol_without_its_key_never_uses_the_astra_key():
    sa, hits_a = fake_upstream("astra")
    try:
        _, b = seat_envs(f"http://127.0.0.1:{sa.server_port}", f"http://127.0.0.1:{sa.server_port}", sol_key="")
        with Proxy(b) as pb:
            st, text = pb.call("openai/gpt-6.1-sol")
            calls = pb.calls()
        assert st == 500 and "AZURE_SOL_API_KEY" in text and "AZURE_SOL_ENDPOINT" in text and hits_a == []
        assert calls[0]["route"] == "azure_sol" and calls[0]["error"] == "missing_upstream_credentials"
    finally:
        sa.shutdown()


def test_sol_pool_uses_its_own_replicas():
    ss, hits_s = fake_upstream("sol")
    try:
        _, b = seat_envs("http://127.0.0.1:1", f"http://127.0.0.1:{ss.server_port}")
        b["AZURE_SOL_DEPLOYMENTS"] = "gpt-6.1-sol,gpt-6.1-sol-2"
        with Proxy(b) as pb:
            st, _ = pb.call("openai/gpt-6.1-sol")
            calls = pb.calls()
        assert st == 200 and hits_s[0][3] == "gpt-6.1-sol" and calls[0]["azure_deployment"] == "gpt-6.1-sol"
    finally:
        ss.shutdown()


def test_sol_route_pins_its_deployments_astra_does_not():
    """The SOL seat sends Codex's slug for the family (gpt-6-sol, which Codex has metadata for); its pinned route serves
    it with the resource's deployment. Astra's route forwards an unknown model unchanged, as before."""
    sa, hits_a = fake_upstream("astra")
    ss, hits_s = fake_upstream("sol")
    try:
        a, b = seat_envs(f"http://127.0.0.1:{sa.server_port}", f"http://127.0.0.1:{ss.server_port}")
        b["AZURE_SOL_DEPLOYMENTS"] = "gpt-6.1-sol"
        with Proxy(a) as pa, Proxy(b) as pb:
            assert pb.call("openai/gpt-6-sol")[0] == 200 and pa.call("openai/gpt-6-astra-x")[0] == 200
            cb, ca = pb.calls(), pa.calls()
        assert hits_s == [("/openai/v1/responses", "sol-key", None, "gpt-6.1-sol")]
        assert cb[0]["model_alias"] == "gpt-6-sol" and cb[0]["azure_deployment"] == "gpt-6.1-sol" and cb[0]["completed"]
        assert hits_a[0][3] == "gpt-6-astra-x" and "model_alias" not in ca[0]
    finally:
        sa.shutdown(); ss.shutdown()


def test_sol_player_config_matches_the_runner():
    """players.toml's sol = what run_duel.sh runs: Codex's gpt-6-sol slug (not the deployment name, which Codex 0.158.0
    has no metadata for), the deployment passed to the seat's pinned pool, the same harness settings as astra."""
    import re
    toml = (REPO / "players" / "players.toml").read_text()
    blocks = {re.search(r'id = "(\w[\w-]*)"', b).group(1): b for b in toml.split("[[player]]")[1:]}
    sol, astra = blocks["sol"], blocks["astra"]
    assert 'model = "openrouter/openai/gpt-6-sol"' in sol and 'deployment = "gpt-6.1-sol"' in sol
    for key in ("harness_version", "agent_import_path", "agent_kwargs"):
        assert re.search(key + r" = .*", sol).group(0) == re.search(key + r" = .*", astra).group(0), key
    sh = (REPO / "scripts" / "olympiad" / "run_duel.sh").read_text()
    assert "CodexOpenRouter@openrouter/openai/gpt-6-sol " in sh and "SOL_DEPLOYMENT:-gpt-6.1-sol" in sh
    assert '"AZURE_SOL_DEPLOYMENTS_$1=${vals[2]:-$SOL_DEPLOYMENT}"' in sh


def test_unknown_route_stays_openrouter():
    _, b = seat_envs("http://x", "http://y")
    b["THINK_OPENAI_ROUTE"] = "openrouter"
    with Proxy(dict(b, OPENROUTER_API_KEY="", THINK_UPSTREAM="http://127.0.0.1:1")) as pb:
        st, text = pb.call("openai/gpt-6-luna")
        calls = pb.calls()
    assert st == 500 and "OPENROUTER_API_KEY" in text and calls[0]["route"] == "openrouter"


if __name__ == "__main__":
    names = [n for n in sorted(globals()) if n.startswith("test_")]
    bad = 0
    for n in names:
        try:
            globals()[n](); print("ok  ", n)
        except Exception:  # noqa: BLE001
            bad += 1; import traceback; traceback.print_exc(); print("FAIL", n)
    print(f"{len(names) - bad}/{len(names)} passed"); sys.exit(1 if bad else 0)
