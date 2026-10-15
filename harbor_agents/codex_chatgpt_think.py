"""Moved to players/harnesses/codex_chatgpt_think.py (v0.2 layout, docs/framework_zh.md §3). This shim keeps the old import path
`harbor_agents.codex_chatgpt_think` (Harbor --agent-import-path / --seat ...=harbor_agents.codex_chatgpt_think:Class) and `python harbor_agents/codex_chatgpt_think.py`
working: the old name IS the moved module (same object, every public and private name)."""
import importlib as _il
import runpy as _rp
import sys as _sys
from pathlib import Path as _P

_ROOT = _P(__file__).resolve().parents[1]
if __name__ == "__main__":
    _rp.run_path(str(_ROOT / "players" / "harnesses" / "codex_chatgpt_think.py"), run_name="__main__")
else:
    if str(_ROOT) not in _sys.path:
        _sys.path.insert(0, str(_ROOT))
    _sys.modules[__name__] = _il.import_module("players.harnesses.codex_chatgpt_think")
