"""Offline checks of dispatch.py (no network): lanes, account choice, headroom with seats in flight, refusal cache."""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import dispatch  # noqa: E402

cfg = dispatch.load()
cfg["prefer"] = {"opus": ["claude-2", "claude-1"], "fable": ["claude-1"]}   # the logic under test, not the live choice
d = dispatch.Dispatcher(config=cfg, log=lambda *a, **k: None)
assert d.lanes() == {"astra": 2, "opus": 2, "fable": 2, "luna": 2}, d.lanes()
assert d.lane_of(["opus", "astra"]) == "astra" and d.lane_of(["fable", "opus"]) == "opus"
assert d.lane_of(["luna", "fable"]) == "fable" and d.lane_of(["luna"]) == "luna"

util = {("claude-2", "claude-opus-5-5"): 0.50, ("claude-1", "claude-opus-5-5"): 0.10,
        ("claude-1", "claude-fable-5-1"): 1.01}
d.utilization = lambda acc, model: util[(acc, model)]
assert d.claude_env(["astra", "luna"]) == ({}, [])
env, used = d.claude_env(["opus", "astra"])
assert env == {"OPUS_ACCOUNT": "2"} and used == ["claude-2"], (env, used)        # 0.50 <= 0.70: preferred account
d.hold(used)
d.hold(used)                                                                     # two seats: 0.50 + 0.30 > 0.70
env, used = d.claude_env(["opus", "luna"])
assert env == {"OPUS_ACCOUNT": "1"} and used == ["claude-1"], (env, used)        # falls back to the other account
assert d.claude_env(["fable", "luna"]) is None                                   # Fable's allowance used up
util[("claude-1", "claude-fable-5-1")] = 0.0
assert d.claude_env(["fable", "luna"]) is None                                   # refusal cached for 120 s
d.cache.clear()
assert d.claude_env(["fable", "luna"])[0] == {"FABLE_ACCOUNT": "1"}
d.release(["claude-2", "claude-2"])
assert d.inflight["claude-2"] == 0
print("PASS: lanes, preferred account, headroom with seats in flight, fallback, model allowance, refusal cache")
