# players/

A **player** is a model + the harness it runs in + its play configuration (program play: writes and runs its own
code; direct, no-code play was removed from main on 2026-10-10 and is kept on branch `archive/direct-play`). The design and the player rules are in
[docs/framework_zh.md](../docs/framework_zh.md): §3 (the four parts, "选手的玩法配置"), §4.1 (the player protocol:
the only channel to a match is the sim container's HTTP service, every model call goes through the metering
channel) and §9 (选手规则 1–4). A player knows no sport and never sees match internals.

## Layout

| Path | What |
|---|---|
| `players.toml` | the players list: one `[[player]]` per player (model, harness, pinned harness version, Harbor agent class, supported play modes, disabled tools, credential variable *names*, config version) |
| `players.py` | loader and checker: `python3 players/players.py check` / `list` (python >= 3.11); `think-model <id>` prints a player's model lock (also on python 3.10) |
| `harnesses/` | the Harbor agent adapters: `claude_think.py` (Claude Code), `codex_openrouter.py`, `codex_azure_think.py`, `codex_chatgpt_think.py` (Codex; `codex_chatgpt_think.py` = players `astra-sub` / `sol-sub` / `luna-sub`, Codex logged in with a ChatGPT subscription whose credentials only its seat's proxy holds (`account_scope = "seat"`)), `or_proxy.py` |
| `nocode/` | `nocode_audit.py`, the verifier's post-match no-code audit: every task ships it in `tests/`, `core/grading.py` applies it only to direct play (archived; the `game` MCP tool and the Codex gateway are on branch `archive/direct-play`) |
| `metering/think_proxy.py` | the metering proxy (it speaks each harness's provider API, so it lives here; the core keeps only `think_clock.py` / `api_errors.py`). Anti-cheat screen (2026-10-08): only the harnesses' model endpoints are forwarded, the seat's model is locked (`THINK_MODEL_<seat>`, set by every run script, see [Model lock](#model-lock); the first model used only as a safety net), provider-run tools (web search / fetch, code execution, remote MCP, OpenRouter plugins and `:online`) are stripped; refusals are logged as `blocked` in `think_calls_<seat>.jsonl`. Tests: `python3 players/metering/test_think_proxy.py` (fake upstream) |

`metering/think_proxy.py` is the source: `tools/generate.py` copies it into every task byte for byte
(`tools/generate.py check` keeps the copies identical), as it copies `nocode/nocode_audit.py` into every task's `tests/`.

Old paths keep working: `harbor_agents/<module>.py` are shims that alias the moved adapters (same module object), so
`harbor_agents.claude_think:ClaudeThink` behaves exactly as `players.harnesses.claude_think:ClaudeThink`.

## Model lock

Every model seat plays exactly one model: its metering proxy refuses (403, logged as `blocked`) any request that names
another model. The lock is set **explicitly** (user decision 2026-10-08, P21 point 2):

* **Value**: `python3 players/players.py think-model <id>`, from the player's row in `players.toml`: the model the
  harness names on the wire with provider prefixes dropped (`anthropic/claude-opus-5-5` -> `claude-opus-5-5`,
  `openrouter/openai/gpt-6-luna` -> `gpt-6-luna`), plus the Azure `deployment` when it differs (sol:
  `gpt-6-sol,gpt-6.1-sol`). Built-in players (`oracle`, `nop`) have none.
* **Set by every runner**: `run_single.sh`, `run_duel.sh`, `run_race.sh` and `run_team.sh` (and through them
  `run_round.py` / `dispatch.py`) call `seat_model <seat> <player>` (`scripts/olympiad/v02_run.sh`), which exports
  `THINK_MODEL_<seat>` for every seat they start; the task's compose passes it to that seat's proxy only
  (`RO_SEAT_ENV_THINK_MODEL_<seat>`, `core/launch.py`). A model player without a lock (or an unknown player) stops the
  run before Harbor starts; a value inherited from the caller's shell is always overwritten.
* **Safety net only**: without `THINK_MODEL` the proxy locks the seat to the first model it calls. No runner of ours
  relies on it; a new runner must call `seat_model` too.
* **Checked**: `ci_checks/check-model-lock.sh` (in `run_all.sh`) runs `scripts/olympiad/test_model_lock.py`: each
  runner with a fake Harbor, every model seat must carry the lock from `players.toml` and the lock must admit the
  seat's model and refuse the other players' models; then the proxy under those locks against a fake upstream.

Unchanged by this rule: a malformed action is the player's own refusal (counted, no forfeit), and provider-run web
search stays off for every player.

## Adding a player

1. **Adapter** (only for a new harness): a Harbor agent class in `players/harnesses/<name>.py`. It must route every
   model call through the task's metering proxy (`think_proxy_url` seat kwarg), pin the harness version, disable web
   search.
2. **One row** in `players.toml` (a new model on an existing harness needs only this). Credentials stay in `.env`;
   the row names the variables. Its `model` (and `deployment`) are the seat's model lock: check
   `python3 players/players.py think-model <id>` prints the name the harness sends.
3. **Checks**: `python3 players/players.py check`; a smoke match (the player against `oracle` on one task, valid trial).
