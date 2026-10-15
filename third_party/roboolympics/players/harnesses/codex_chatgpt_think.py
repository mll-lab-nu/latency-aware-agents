"""Codex on a ChatGPT subscription (metered, credential-isolated): players astra-sub, sol-sub, luna-sub (players.toml).

The agent's Codex is logged in "with ChatGPT" through a PLACEHOLDER auth.json (written here: unsigned JWT-shaped
tokens with a placeholder account; no real credential ever enters the agent container) and talks to a custom provider
at <think proxy>/backend-api/codex with `requires_openai_auth`, so it sends ChatGPT-backend requests to its seat's
metering proxy (players/metering/think_proxy.py in the task's sim sidecar). The proxy replaces the credentials with the
sidecar's real CODEX_ACCESS_TOKEN / CODEX_ACCOUNT_ID (seat-scoped: the run script exports CODEX_*_<seat>, the compose
file passes them to that seat's proxy only), forwards to https://chatgpt.com and meters the tokens.

The proxy URL is the seat's `think_proxy_url` (task.toml [[seats]] agent_kwargs, e.g. http://sim:18090/v1; a trailing
/v1 is dropped), else RCB_THINK_PROXY_URL. Through the run scripts:

    scripts/olympiad/run_single.sh --task curling-match-program-standard --agent astra-sub
"""
from __future__ import annotations

import base64
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

from harbor.agents.installed.codex import Codex
from harbor.agents.model_connection import ResolvedModelConnection

PLACEHOLDER_ACCOUNT = "00000000-0000-4000-8000-000000000000"
PLACEHOLDER_KEY = "rcb-think-proxy-placeholder"


def _think_url(override: str | None = None) -> str:
    url = (override or os.environ.get("RCB_THINK_PROXY_URL") or "").strip().rstrip("/")
    if url.endswith("/v1"):
        url = url[:-3]
    if not url:
        raise ValueError("CodexChatGPTThink needs its seat's think_proxy_url or RCB_THINK_PROXY_URL "
                         "(the sidecar's metering proxy, e.g. http://sim:18090)")
    return url


def _b64(obj: dict) -> str:
    return base64.urlsafe_b64encode(json.dumps(obj, separators=(",", ":")).encode()).rstrip(b"=").decode()


def _placeholder_jwt(claims: dict) -> str:
    """JWT-shaped and unsigned: Codex only decodes the payload (plan, account, expiry); the proxy drops it."""
    return ".".join((_b64({"alg": "none", "typ": "JWT"}), _b64(claims), "cmItcGxhY2Vob2xkZXI"))


def placeholder_auth(now: float | None = None) -> dict:
    """A ChatGPT-login auth.json for Codex without any real credential: refreshed "now" and valid for 30 days, so Codex
    does not try to refresh it during a match (its refresh token is a placeholder too, and a playing seat has no
    internet)."""
    now = int(now if now is not None else time.time())
    claims = {"iat": now, "exp": now + 30 * 86400, "sub": "rcb-placeholder",
              "https://api.openai.com/auth": {"chatgpt_plan_type": "pro", "chatgpt_account_id": PLACEHOLDER_ACCOUNT,
                                              "chatgpt_user_id": "user-rcb-placeholder",
                                              "user_id": "user-rcb-placeholder"}}
    return {"OPENAI_API_KEY": None,
            "tokens": {"id_token": _placeholder_jwt({**claims, "email": "placeholder@roboolympics.invalid"}),
                       "access_token": _placeholder_jwt(claims), "refresh_token": PLACEHOLDER_KEY,
                       "account_id": PLACEHOLDER_ACCOUNT},
            "last_refresh": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now))}


class CodexChatGPTThink(Codex):
    def __init__(self, *args, think_proxy_url: str | None = None, **kwargs):
        # think_proxy_url: this agent's metering proxy (a multi-seat task gives every seat its own);
        # default: RCB_THINK_PROXY_URL.
        self._think_proxy_url = think_proxy_url
        kwargs["web_search"] = "disabled"         # anti-cheat: no provider-run web search in any mode
        super().__init__(*args, **kwargs)

    @property
    def model_connection(self) -> ResolvedModelConnection:
        # Never resolve the host's OPENAI_* variables into the agent: it talks only to its seat's proxy (config below).
        return ResolvedModelConnection(provider="openai", api_key=PLACEHOLDER_KEY,
                                       base_url=f"{_think_url(self._think_proxy_url)}/backend-api/codex")

    def _resolve_auth_json_path(self) -> Path | None:
        path = Path(tempfile.mkdtemp(prefix="rcb-codex-auth-")) / "auth.json"
        path.write_text(json.dumps(placeholder_auth(), indent=2))
        return path

    def _build_effective_config(self, openai_base_url: str | None = None) -> dict[str, Any]:
        config = super()._build_effective_config(None)
        config.pop("openai_base_url", None)
        config["model_provider"] = "chatgpt_think"
        config["web_search"] = "disabled"
        # ChatGPT login turns on Codex's "apps" (ChatGPT connectors: a remote MCP server at chatgpt.com, provider-run
        # tools past the meter). Off: no tool reaches outside the game (Codex 0.158.0 tries it at start otherwise).
        config.setdefault("features", {})["apps"] = False
        config.setdefault("model_providers", {})["chatgpt_think"] = {
            "name": "ChatGPT subscription via the think proxy",
            "base_url": f"{_think_url(self._think_proxy_url)}/backend-api/codex",
            "wire_api": "responses",
            "requires_openai_auth": True,
            "supports_websockets": False,
        }
        return config

    async def run(self, instruction, environment, context) -> None:
        if not str(self.model_name or "").startswith("openai/"):
            raise ValueError("CodexChatGPTThink model must be openai/<model>, e.g. openai/gpt-6-astra")
        _think_url(self._think_proxy_url)
        await super().run(instruction, environment, context)
