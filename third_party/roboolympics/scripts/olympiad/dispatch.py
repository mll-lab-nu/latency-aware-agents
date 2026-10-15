"""Model-account dispatcher for Olympiad runs, outside Harbor: which account a seat uses, how many matches run at once.

The accounts and limits live in a JSON file (default scripts/olympiad/accounts.json, or $OLYMPIAD_ACCOUNTS); it names
the .env variables that hold tokens, never a token. Two kinds of account:

  * "claude-subscription": a Claude subscription (Claude Code OAuth token). Its limits are usage windows (5 hours, 7 days,
    and per-model allowances such as Fable's), not request rates. Before a match, each candidate account gets a minimal
    request (max_tokens 1); its response headers carry the utilization of every window. The account is used only if
    the highest utilization, plus a reservation for each seat this process already runs on it, stays at or below
    `max_utilization`. The other host's seats show up in the utilization as they play.
  * "fixed": a key with a request-rate limit (Azure, OpenRouter). Measured capacity is expressed as the number of
    matches at once (the lane's `workers`); no probe.

Lanes group matches by the account that limits them (lane_of); each lane has `workers` threads per host. Nothing here
holds or rewrites a request: the run scripts pass the chosen account's .env to Harbor's native agent (PROTOCOL D11).
"""
import json
import os
import pathlib
import threading
import time
import urllib.error
import urllib.request

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CLAUDE_MODEL = {"opus": "claude-opus-5-5", "fable": "claude-fable-5-1"}
SEAT_ENV = {"opus": "OPUS_ACCOUNT", "fable": "FABLE_ACCOUNT"}     # read by run_single.sh / run_duel.sh


def load(path=None):
    path = pathlib.Path(path or os.environ.get("OLYMPIAD_ACCOUNTS") or HERE / "accounts.json")
    return json.loads(path.read_text())


class Dispatcher:
    def __init__(self, config=None, log=print):
        self.cfg = config or load()
        self.log = log
        self.lock = threading.Lock()
        self.inflight = {}            # account -> Claude seats this process runs on it now
        self.cache = {}               # (account, model) -> (time, ok); only refusals are cached
        q = self.cfg.get("quota", {})
        self.max_util = float(q.get("max_utilization", 0.70))
        self.per_seat = float(q.get("per_seat", 0.15))
        self.cache_s = float(q.get("refusal_cache_s", 120))

    # ---- lanes ---------------------------------------------------------------------------------------------------
    def lanes(self):
        return {name: int(v["workers"]) for name, v in self.cfg["lanes"].items()}

    def lane_of(self, players):
        """The first lane (in config order) that one of the players belongs to."""
        for name, v in self.cfg["lanes"].items():
            if any(p in v["players"] for p in players):
                return name
        raise ValueError(f"no lane for {players}")

    # ---- Claude subscriptions ------------------------------------------------------------------------------------
    def _token(self, account):
        var = self.cfg["accounts"][account]["token_env"]
        for line in (ROOT / ".env").read_text().splitlines():
            if line.startswith(var + "="):
                return line.split("=", 1)[1].strip()
        return None

    def utilization(self, account, model):
        """Highest utilization over the account's usage windows for this model (1.0 when refused or unreachable)."""
        tok = self._token(account)
        if not tok:
            return 1.0
        body = json.dumps({"model": model, "max_tokens": 1, "messages": [{"role": "user", "content": "ok"}],
                           "system": "You are Claude Code, Anthropic's official CLI for Claude."}).encode()
        req = urllib.request.Request("https://api.anthropic.com/v1/messages", data=body, method="POST", headers={
            "authorization": f"Bearer {tok}", "anthropic-version": "2023-06-01", "anthropic-beta": "oauth-2025-04-20",
            "content-type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                if r.status != 200:
                    return 1.0
                us = [float(v) for k, v in r.headers.items()
                      if k.lower().startswith("anthropic-ratelimit-unified-") and k.lower().endswith("-utilization")]
                return max(us or [0.0])
        except (urllib.error.URLError, OSError, ValueError):    # 429 / 5xx / network: no quota now
            return 1.0

    def _usable(self, account, player):
        model = CLAUDE_MODEL[player]
        hit = self.cache.get((account, model))
        if hit and time.time() - hit[0] < self.cache_s:
            return False
        util = self.utilization(account, model)
        with self.lock:
            busy = self.inflight.get(account, 0)
        ok = util + self.per_seat * busy <= self.max_util
        self.log(time.strftime("%H:%M:%S"), f"probe {account} {model}: utilization {util:.2f}, {busy} local seats "
                 f"in flight -> {'ok' if ok else 'wait'}", flush=True)
        if not ok:
            self.cache[(account, model)] = (time.time(), False)
        return ok

    def claude_env(self, players):
        """(env, accounts): env {OPUS_ACCOUNT/FABLE_ACCOUNT: slot} for the match's Claude players ({} if none) and the
        accounts they use; None when one of them has no account with headroom now."""
        env, used = {}, []
        for p in players:
            if p not in CLAUDE_MODEL:
                continue
            acc = next((a for a in self.cfg["prefer"][p] if self._usable(a, p)), None)
            if acc is None:
                return None
            env[SEAT_ENV[p]] = self.cfg["accounts"][acc]["slot"]
            used.append(acc)
        return env, used

    def hold(self, accounts):
        with self.lock:
            for a in accounts:
                self.inflight[a] = self.inflight.get(a, 0) + 1

    def release(self, accounts):
        with self.lock:
            for a in accounts:
                self.inflight[a] -= 1
