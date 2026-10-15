"""Tests of think_proxy.py's anti-cheat screen (audit 2026-10-08, G4) against a FAKE upstream (no model API is called):
path allowlist, model lock (explicit THINK_MODEL and first-model lock), provider-run tools stripped, refusals logged,
a THINK_OPENAI_ROUTE=chatgpt seat reaches the ChatGPT backend only, upstream redirects are returned, never followed.
Azure deployment failover on transient upstream failures (HTTP 408/500/502/504, a timeout or dropped connection, an
empty or server-failed stream) before any byte reached the agent; a failure after the stream began is not retried; the
last deployment's failure reaches the agent; no failover without AZURE_DEPLOYMENTS or on OpenRouter; 429 waits unchanged.

    python3 players/metering/test_think_proxy.py
Standard library only.
"""
from __future__ import annotations

import json
import os
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

PROXY = Path(__file__).resolve().parent / "think_proxy.py"
SEEN: list[dict] = []


class Upstream(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _reply(self):
        n = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(n) if n else b""
        SEEN.append({"method": self.command, "path": self.path, "auth": bool(self.headers.get("Authorization")),
                     "body": json.loads(body) if body else None})
        out = json.dumps({"id": "r1", "type": "message", "content": [{"type": "text", "text": "ok"}],
                          "usage": {"input_tokens": 3, "output_tokens": 2}}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    do_GET = do_POST = _reply


ELSEWHERE: list[dict] = []          # requests that reached the host an upstream redirect named


class Elsewhere(Upstream):
    def _reply(self):
        ELSEWHERE.append({"path": self.path, "auth": self.headers.get("Authorization"),
                          "account": self.headers.get("chatgpt-account-id")})
        super()._reply()

    do_GET = do_POST = _reply


AZ_PLAN: dict = {}                  # model (deployment) -> action, or a list of actions taken in turn
AZ_HITS: list[str] = []
HANG_S = 3.0                        # the Azure tests' proxy times out after THINK_UPSTREAM_TIMEOUT_S = 1


class FakeAzure(BaseHTTPRequestHandler):
    """A fake Azure (or OpenRouter) Responses endpoint; what it does for a model is AZ_PLAN[model]."""
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def ev(self, d):
        self.wfile.write(b"event: " + d["type"].encode() + b"\ndata: " + json.dumps(d).encode() + b"\n\n")
        self.wfile.flush()

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
        m = body.get("model")
        AZ_HITS.append(m)
        act = AZ_PLAN.get(m, "ok")
        if isinstance(act, list):
            act = act.pop(0) if len(act) > 1 else act[0]
        self.close_connection = True
        try:
            self._act(act, body)
        except OSError:
            pass                                             # the proxy gave up on us (a timeout case)

    def _act(self, act, body):
        if isinstance(act, int):                             # e.g. Azure's 408 "The operation was timeout."
            out = json.dumps({"error": {"code": "Timeout", "message": "The operation was timeout."}}).encode()
            self.send_response(act); self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(out))); self.end_headers(); self.wfile.write(out)
            return
        if act == "hang":                                    # no response headers within the proxy's timeout
            time.sleep(HANG_S)
            return
        if act == "reset":                                   # connection reset, no response
            self.connection.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
            self.connection.close()
            return
        done = {"type": "response.completed", "response": {"id": "r1", "status": "completed", "output": [],
                                                            "usage": {"input_tokens": 3, "output_tokens": 2}}}
        if not body.get("stream"):
            out = json.dumps(done["response"]).encode()
            self.send_response(200); self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(out))); self.end_headers(); self.wfile.write(out)
            return
        self.send_response(200); self.send_header("Content-Type", "text/event-stream")
        self.send_header("Connection", "close"); self.end_headers(); self.wfile.flush()
        if act == "empty":                                   # 200, then closed before any event
            return
        self.ev({"type": "response.created", "response": {"id": "r1", "status": "in_progress"}})
        if act == "hang_stream":                             # stalls before the first output event
            time.sleep(HANG_S)
            return
        if act == "server_failed":
            self.ev({"type": "response.failed", "response": {"id": "r1", "status": "failed", "instructions": "no timeout",
                                                              "error": {"code": "server_error", "message": "boom"}}})
            return
        self.ev({"type": "response.output_text.delta", "delta": "partial "})
        if act == "midstream_close":                         # the stream began, then the connection closed
            return
        if act == "midstream_hang":                          # the stream began, then stalled past the timeout
            time.sleep(HANG_S)
            return
        self.ev({"type": "response.output_text.delta", "delta": "done"})
        self.ev(done)


def redirecting(target: str):
    class Redirect(Upstream):
        def _reply(self):
            n = int(self.headers.get("Content-Length") or 0)
            self.rfile.read(n) if n else b""
            self.send_response(307)
            self.send_header("Location", target + self.path)
            self.send_header("Content-Length", "0")
            self.end_headers()

        do_GET = do_POST = _reply
    return Redirect


def raw_call(port, method, path, body=None, headers=None):
    """One request, no redirect followed (urllib would follow the proxy's 30x itself)."""
    import http.client
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    data = json.dumps(body).encode() if body is not None else None
    c.request(method, path, body=data, headers={"Content-Type": "application/json", **(headers or {})})
    r = c.getresponse()
    out = (r.status, dict(r.getheaders()), r.read())
    c.close()
    return out


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def start_proxy(tmp: Path, up: str, **env) -> tuple[subprocess.Popen, int]:
    port = free_port()
    e = {k: v for k, v in os.environ.items() if not k.startswith(("THINK_", "OPENROUTER", "CLAUDE_", "CODEX_", "AZURE"))}
    e.update(THINK_UPSTREAM=up, THINK_ANTHROPIC_UPSTREAM=up, THINK_CHATGPT_UPSTREAM=up,
             OPENROUTER_API_KEY="fake-openrouter", CLAUDE_CODE_OAUTH_TOKEN="fake-oauth", CODEX_ACCESS_TOKEN="fake-chatgpt",
             THINK_USAGE_FILE=str(tmp / f"usage_{port}.json"), THINK_CALLS_FILE=str(tmp / f"calls_{port}.jsonl"), **env)
    p = subprocess.Popen([sys.executable, str(PROXY), str(port)], env=e, stderr=subprocess.DEVNULL)
    for _ in range(100):
        try:
            socket.create_connection(("127.0.0.1", port), timeout=.2).close()
            return p, port
        except OSError:
            time.sleep(.05)
    raise RuntimeError("proxy did not start")


def call(port, method, path, body=None, headers=None):
    data = json.dumps(body).encode() if isinstance(body, (dict, list)) else body
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=data, method=method,
                                 headers={"Content-Type": "application/json", **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


def calls(tmp: Path, port: int) -> list[dict]:
    f = tmp / f"calls_{port}.jsonl"
    return [json.loads(x) for x in f.read_text().splitlines()]


def main() -> int:
    up = ThreadingHTTPServer(("127.0.0.1", 0), Upstream)
    threading.Thread(target=up.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{up.server_address[1]}"
    tmp = Path(tempfile.mkdtemp(prefix="think-proxy-test-"))
    procs, bad, n = [], 0, 0

    def check(name, fn):
        nonlocal bad
        try:
            fn()
            print(f"ok   {name}")
        except Exception:  # noqa: BLE001
            bad += 1
            print(f"FAIL {name}\n{traceback.format_exc()}")

    try:
        p1, first = start_proxy(tmp, base)                                   # no THINK_MODEL: first-model lock
        p2, locked = start_proxy(tmp, base, THINK_MODEL="gpt-6-*")           # explicit lock, a pattern
        p3, sub = start_proxy(tmp, base, THINK_MODEL="gpt-6-sol", THINK_OPENAI_ROUTE="chatgpt")   # a *-sub seat
        far = ThreadingHTTPServer(("127.0.0.1", 0), Elsewhere)
        threading.Thread(target=far.serve_forever, daemon=True).start()
        bounce = ThreadingHTTPServer(("127.0.0.1", 0), redirecting(f"http://127.0.0.1:{far.server_address[1]}"))
        threading.Thread(target=bounce.serve_forever, daemon=True).start()
        p4, redir = start_proxy(tmp, f"http://127.0.0.1:{bounce.server_address[1]}", THINK_MODEL="gpt-6-sol",
                                CODEX_ACCOUNT_ID="fake-account")
        procs += [p1, p2, p3, p4]

        def paths_outside_the_allowlist_are_refused_and_never_forwarded():
            SEEN.clear()
            for m, path in (("GET", "/"), ("GET", "/api/v1/key"), ("POST", "/v1/foo"), ("GET", "/v1/messages"),
                            ("POST", "/api/oauth/token"), ("POST", "/v1/files"), ("GET", "/backend-api/accounts")):
                code, out = call(first, m, path, {} if m == "POST" else None)
                assert code == 403 and out["error"]["type"] == "think_proxy_refused", (m, path, code, out)
            assert not SEEN, SEEN
            assert sum(1 for r in calls(tmp, first) if r.get("blocked")) >= 7

        def server_tools_are_stripped_client_tools_kept():
            SEEN.clear()
            body = {"model": "claude-opus-5-5", "max_tokens": 5, "messages": [{"role": "user", "content": "hi"}],
                    "tools": [{"type": "web_search_20250305", "name": "web_search"},
                              {"type": "web_fetch_20250910", "name": "web_fetch"},
                              {"type": "code_execution_20250825", "name": "code_execution"},
                              {"name": "mcp__game__game", "description": "d", "input_schema": {"type": "object"}}],
                    "mcp_servers": [{"type": "url", "url": "https://example.com/mcp", "name": "x"}]}
            code, out = call(first, "POST", "/v1/messages?beta=true", body, {"anthropic-version": "2023-06-01"})
            assert code == 200, (code, out)
            sent = SEEN[-1]
            assert sent["auth"] and sent["path"] == "/v1/messages?beta=true", sent
            assert [t.get("name") for t in sent["body"]["tools"]] == ["mcp__game__game"], sent["body"]
            assert "mcp_servers" not in sent["body"]
            rec = [r for r in calls(tmp, first) if r.get("model_call")][-1]
            assert set(rec["stripped"]) >= {"web_search_20250305", "web_fetch_20250910", "mcp_servers"}, rec

        def first_model_locks_the_seat():
            SEEN.clear()
            code, out = call(first, "POST", "/v1/messages", {"model": "claude-sonnet-5", "max_tokens": 5,
                                                              "messages": []}, {"anthropic-version": "2023-06-01"})
            assert code == 403 and "model" in out["error"]["message"], (code, out)
            code, _ = call(first, "POST", "/v1/messages/count_tokens", {"model": "claude-opus-5-5", "messages": []},
                           {"anthropic-version": "2023-06-01"})
            assert code == 200 and len(SEEN) == 1

        def explicit_lock_and_openrouter_web_options():
            SEEN.clear()
            body = {"model": "openai/gpt-6-luna:online", "input": "hi", "plugins": [{"id": "web"}],
                    "models": ["openai/gpt-7"], "web_search_options": {}, "tools": [
                        {"type": "web_search"}, {"type": "web_search_preview"}, {"type": "mcp", "server_url": "x"},
                        {"type": "file_search"}, {"type": "code_interpreter"},
                        {"type": "function", "name": "f", "parameters": {}}]}
            code, out = call(locked, "POST", "/v1/responses", body)
            assert code == 200, (code, out)
            sent = SEEN[-1]["body"]
            assert sent["model"] == "openai/gpt-6-luna" and [t["type"] for t in sent["tools"]] == ["function"], sent
            assert not {"plugins", "models", "web_search_options"} & set(sent), sent
            code, out = call(locked, "POST", "/v1/responses", {"model": "anthropic/claude-opus-5-5", "input": "x"})
            assert code == 403, (code, out)
            code, out = call(locked, "POST", "/v1/responses", {"model": "openai/gpt-6-sol", "input": "x"})
            assert code == 200, (code, out)

        def chatgpt_backend_paths_and_bad_bodies():
            SEEN.clear()
            code, _ = call(locked, "GET", "/backend-api/codex/models?client_version=1")
            assert code == 200 and SEEN[-1]["path"].startswith("/backend-api/codex/models")
            code, _ = call(locked, "POST", "/backend-api/codex/responses", {"model": "gpt-6-sol", "input": "x"})
            assert code == 200
            n = len(SEEN)
            code, _ = call(locked, "POST", "/v1/responses", b"not json")
            assert code == 400 and len(SEEN) == n
            code, _ = call(locked, "POST", "/v1/responses", [1, 2])
            assert code == 400 and len(SEEN) == n

        def chatgpt_route_reaches_the_chatgpt_backend_only():
            SEEN.clear()
            body = {"model": "gpt-6-sol", "input": "x"}
            for m, path, b, h in (("POST", "/v1/responses", body, {}), ("POST", "/v1/chat/completions", body, {}),
                                  ("POST", "/openai/v1/responses", body, {}), ("GET", "/v1/models", None, {}),
                                  ("POST", "/v1/messages", {"model": "gpt-6-sol", "max_tokens": 5, "messages": []},
                                   {"anthropic-version": "2023-06-01"})):
                code, out = call(sub, m, path, b, h)
                assert code == 403 and "route chatgpt" in out["error"]["message"], (path, code, out)
            assert not SEEN, SEEN                                  # nothing reached OpenRouter / Anthropic / Azure
            code, _ = call(sub, "POST", "/backend-api/codex/responses", body)
            assert code == 200 and SEEN[-1]["path"] == "/backend-api/codex/responses", SEEN
            blocked = [r for r in calls(tmp, sub) if r.get("blocked")]
            assert len(blocked) == 5 and all("route chatgpt" in r["blocked"] for r in blocked), blocked

        def upstream_redirects_are_returned_not_followed():
            ELSEWHERE.clear()
            for m, path, b in (("POST", "/backend-api/codex/responses", {"model": "gpt-6-sol", "input": "x"}),
                               ("POST", "/v1/responses", {"model": "gpt-6-sol", "input": "x"}),
                               ("GET", "/backend-api/codex/models?client_version=1", None)):
                status, headers, _ = raw_call(redir, m, path, b, {"Authorization": "Bearer placeholder"})
                assert status == 307 and headers.get("Location", "").endswith(path), (path, status, headers)
            assert not ELSEWHERE, ELSEWHERE                        # the credentials never left for the other host
            recs = [r for r in calls(tmp, redir) if not r.get("blocked")]
            assert [r["status"] for r in recs] == [307, 307, 307], recs

        def usage_counts_blocked_requests():
            code, u = call(first, "GET", "/think/usage")
            assert code == 200 and u["blocked_requests"] >= 8, u

        az = ThreadingHTTPServer(("127.0.0.1", 0), FakeAzure)
        az.daemon_threads = True
        threading.Thread(target=az.serve_forever, daemon=True).start()
        azbase = f"http://127.0.0.1:{az.server_address[1]}"
        azure_env = dict(THINK_OPENAI_ROUTE="azure", AZURE_AI_ENDPOINT=azbase + "/models", AZURE_AI_API_KEY="fake-azure",
                         THINK_UPSTREAM_TIMEOUT_S="1")
        p5, pool = start_proxy(tmp, base, AZURE_DEPLOYMENTS="dep-1,dep-2", **azure_env)
        p6, nopool = start_proxy(tmp, base, **azure_env)                     # Azure route, AZURE_DEPLOYMENTS empty
        p7, orouter = start_proxy(tmp, azbase, THINK_RETRY_WAITS_S="0.05,0.05", THINK_UPSTREAM_TIMEOUT_S="1")
        procs += [p5, p6, p7]

        def az_call(port, first_act, second_act=None, stream=True, model="dep-1"):
            """One Responses call; on the pool proxy, `first_act` is what its current deployment does and `second_act`
            what the other one does. Returns (status, body bytes, the call's record, deployments hit, usage)."""
            AZ_PLAN.clear(); AZ_HITS.clear()
            if port == pool:
                recs = [r for r in calls(tmp, pool) if r.get("azure_deployment")] if (tmp / f"calls_{pool}.jsonl").exists() else []
                cur = recs[-1]["azure_deployment"] if recs else "dep-1"
                AZ_PLAN[cur], AZ_PLAN["dep-2" if cur == "dep-1" else "dep-1"] = first_act, second_act or "ok"
            else:
                AZ_PLAN[model] = first_act
            status, _, out = raw_call(port, "POST", "/v1/responses", {"model": model, "stream": stream, "input": "hi"},
                                      {"Authorization": "Bearer placeholder"})
            time.sleep(.05)                                  # the record is written after the response is closed
            rec = [r for r in calls(tmp, port) if r.get("model_call")][-1]
            _, u = call(port, "GET", "/think/usage")
            return status, out, rec, list(AZ_HITS), u

        def azure_transient_failures_fail_over_before_the_first_byte():
            cases = [(408, "HTTP 408"), (504, "HTTP 504"), (500, "HTTP 500"), (502, "HTTP 502"), ("hang", "TimeoutError"),
                     ("reset", None), ("hang_stream", "TimeoutError"), ("empty", "stream closed before output"),
                     ("server_failed", "response.failed (server_error)")]
            for act, reason in cases:
                _, _, _, _, u0 = az_call(pool, "ok")
                status, out, rec, hits, u = az_call(pool, act)
                assert status == 200 and b"response.completed" in out and b"timeout" not in out.lower(), (act, status, out)
                assert rec["completed"] is True and rec["status"] == 200 and len(hits) == 2 and hits[0] != hits[1], (act, rec, hits)
                fo = rec.get("azure_failovers") or []
                assert len(fo) == 1 and fo[0]["from"] == hits[0] and rec["azure_deployment"] == hits[1], (act, rec)
                assert reason is None and fo[0]["reason"] in ("ConnectionResetError", "RemoteDisconnected") \
                    or fo[0]["reason"] == reason, (act, fo)
                assert u["azure_failovers"] == u0.get("azure_failovers", 0) + 1, (act, u0, u)
                assert u["failed_model_calls"] == u0["failed_model_calls"] == 0, (act, u)
            status, out, rec, hits, _ = az_call(pool, 408, stream=False)       # a non-streamed request too
            assert status == 200 and rec["completed"] and rec["azure_failovers"][0]["reason"] == "HTTP 408", (status, rec)

        def azure_failure_after_the_stream_began_is_not_retried():
            for act in ("midstream_close", "midstream_hang"):
                _, _, _, _, u0 = az_call(pool, "ok")
                status, out, rec, hits, u = az_call(pool, act)
                assert status == 200 and b"partial" in out and b"response.completed" not in out, (act, status, out)
                assert len(hits) == 1 and not rec.get("azure_failovers") and rec.get("completed") is False, (act, rec, hits)
                assert act == "midstream_close" or rec.get("error") == "TimeoutError", (act, rec)
                assert u["failed_model_calls"] == u0["failed_model_calls"] + 1, (act, u0, u)
                assert u.get("azure_failovers", 0) == u0.get("azure_failovers", 0), (act, u0, u)

        def azure_last_deployment_failure_reaches_the_agent():
            status, out, rec, hits, _ = az_call(pool, 408, 408)
            assert status == 408 and b"timeout" in out and len(hits) == 2, (status, out, hits)
            assert [f["reason"] for f in rec["azure_failovers"]] == ["HTTP 408"] and rec["status"] == 408, rec
            status, out, rec, hits, _ = az_call(pool, "hang", "hang")
            assert status == 502 and len(hits) == 2 and "TimeoutError" in rec.get("error", ""), (status, rec, hits)
            assert [f["reason"] for f in rec["azure_failovers"]] == ["TimeoutError"], rec

        def no_failover_without_a_pool_or_off_azure():
            for port, model in ((nopool, "dep-1"), (orouter, "openai/gpt-6-astra")):
                for act, want in ((408, 408), (504, 504), ("hang", 502)):
                    status, out, rec, hits, u = az_call(port, act, model=model)
                    assert status == want and len(hits) == 1 and not rec.get("azure_failovers"), (port, act, status, rec)
                    assert not u.get("azure_failovers") and rec.get("completed") is not True, (port, act, u)
            status, out, rec, hits, u = az_call(orouter, [429, 429, "ok"], model="openai/gpt-6-astra")   # 429 waits
            assert status == 200 and len(hits) == 3 and [r["status"] for r in rec["retries"]] == [429, 429], (status, rec)
            assert [r["wait_s"] for r in rec["retries"]] == [0.05, 0.05] and rec["completed"], rec

        for f in (paths_outside_the_allowlist_are_refused_and_never_forwarded, server_tools_are_stripped_client_tools_kept,
                  first_model_locks_the_seat, explicit_lock_and_openrouter_web_options,
                  chatgpt_backend_paths_and_bad_bodies, chatgpt_route_reaches_the_chatgpt_backend_only,
                  upstream_redirects_are_returned_not_followed, usage_counts_blocked_requests,
                  azure_transient_failures_fail_over_before_the_first_byte,
                  azure_failure_after_the_stream_began_is_not_retried, azure_last_deployment_failure_reaches_the_agent,
                  no_failover_without_a_pool_or_off_azure):
            check(f.__name__, f)
            n += 1
    finally:
        for p in procs:
            p.terminate()
            p.wait(timeout=5)
        up.shutdown()
    print(f"{n - bad}/{n} passed")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
