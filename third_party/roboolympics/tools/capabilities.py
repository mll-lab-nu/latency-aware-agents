#!/usr/bin/env python3
"""The capabilities a sport tests (docs/capabilities_zh.md): sport.toml [capabilities], checked by tools/check stage 1,
and the sport x capability matrix the site is generated from.

    python3 tools/capabilities.py [--out FILE]     the matrix of every sports/*/sport.toml as JSON
    python3 tools/capabilities.py check            exit 1 if a sport's [capabilities] is missing or invalid

[capabilities] in sport.toml:
    a1 = 2      # 2 = primary, 1 = secondary, 0 = none (A1-A4, B1-B3 below, in this order)
    ...
    a4 = 0      # time and latency: realtime sports only (a turn-based sport declares 0)
    ...
    b3 = 0
    physics = ["spin_bounce"]     # the physics types tested, from PHYSICS (the doc's Chinese names: PHYSICS_ZH)
C1 (real-time programming) and C2 (learning during the match) apply to every sport and are not declared. The modes
are not repeated here: they come from [modes] (action levels x plays). Evaluation settings (modes, slowdown) cross
the abilities and never introduce new ones.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sportlib  # noqa: E402

CAPABILITIES = {"a1": "3D 空间理解", "a2": "物理动力学理解", "a3": "具身与执行", "a4": "时间与延迟",
                "b1": "规则", "b2": "策略与读对手", "b3": "团队协作"}
# C: the coding-agent abilities every sport tests; never declared per sport (docs/capabilities_zh.md v3)
GLOBAL = {"c1": "实时编程", "c2": "在比赛中学习"}
LEVELS = {2: "primary", 1: "secondary", 0: "none"}
PHYSICS = ("ballistics", "wind", "spin_bounce", "drag", "friction_rolling", "collision", "inertia_latency", "locomotion",
           "contact")
# the vocabulary's Chinese names (docs/framework_zh.md §8 "考的物理")
PHYSICS_NAMES_ZH = {"ballistics": "弹道", "wind": "风", "spin_bounce": "旋转与弹跳", "drag": "空气阻力",
                    "friction_rolling": "摩擦与滚动", "collision": "碰撞", "inertia_latency": "惯性与延迟",
                    "locomotion": "行走与移动", "contact": "接触"}
# the "考的物理" names of docs/capabilities_zh.md §6 -> PHYSICS (names that are perception, not physics, map to none)
PHYSICS_ZH = {"弹道": "ballistics", "风": "wind", "抛物线": "ballistics", "投篮": "ballistics",
              "旋转": "spin_bounce", "弹跳": "spin_bounce", "弯曲球路": "spin_bounce",
              "高阻力飞行": "drag", "气动": "drag",
              "摩擦": "friction_rolling", "滚动摩擦": "friction_rolling", "坡面": "friction_rolling",
              "弧线": "friction_rolling", "抓地": "friction_rolling",
              "碰撞": "collision", "碰撞链": "collision", "边框反弹": "collision", "反弹": "collision",
              "惯性": "inertia_latency", "旋翼延迟": "inertia_latency",
              "行走": "locomotion", "运球": "locomotion",
              "踢球": "contact", "剑尖接触": "contact",
              "远距离深度": None, "遮挡": None, "距离": None}


def problems(decl: dict) -> list[str]:
    c = decl.get("capabilities")
    realtime = (decl.get("sport") or {}).get("protocol") == "realtime"
    if not isinstance(c, dict):
        return ["sport.toml has no [capabilities] (a1..b3 = 2/1/0, physics = [...])"]
    out = [f"{k} must be 2, 1 or 0" for k in CAPABILITIES if c.get(k) not in LEVELS or isinstance(c.get(k), bool)]
    extra = sorted(set(c) - set(CAPABILITIES) - {"physics"})
    if extra:
        out.append(f"unknown keys {extra} (C1/C2 apply to every sport and the modes come from [modes]: not declared)")
    if not realtime and c.get("a4") not in (0, None):
        out.append("a4 (time and latency) must be 0: the sport has no realtime format")
    ph = c.get("physics")
    if not isinstance(ph, list) or not ph:
        out.append("physics must be a non-empty list")
    else:
        bad = sorted(set(map(str, ph)) - set(PHYSICS))
        if bad:
            out.append(f"physics {bad} not in the vocabulary {list(PHYSICS)}")
    if not any(c.get(k) == 2 for k in ("a1", "a2", "a3")):
        out.append("a sport tests physical-world understanding: at least one of a1-a3 must be 2")
    return out


def matrix() -> dict:
    rows = []
    for name in sportlib.names():
        d = sportlib.decl(name)
        c = d.get("capabilities") or {}
        md = d.get("modes") or {}
        rows.append({"sport": name, "title": d["sport"].get("title", name), "status": d["sport"].get("status"),
                     "protocol": d["sport"].get("protocol"), "formats": sorted(d.get("formats", {})),
                     "modes": {"action_levels": md.get("action_levels", []), "plays": md.get("plays", [])},
                     "capabilities": {k: c.get(k) for k in CAPABILITIES}, "vector": [c.get(k) for k in CAPABILITIES],
                     "physics": c.get("physics", []),
                     "problems": problems(d)})
    return {"capabilities": CAPABILITIES, "global": GLOBAL, "levels": LEVELS, "physics": list(PHYSICS), "physics_names_zh": PHYSICS_NAMES_ZH,
            "physics_zh": PHYSICS_ZH,
            "sports": rows}


def main(argv):
    if argv[:1] == ["check"]:
        bad = 0
        for r in matrix()["sports"]:
            for p in r["problems"]:
                bad = 1
                print(f"FAIL [capabilities] {r['sport']}: {p}")
        print("capabilities: every sport declares them" if not bad else "capabilities: fix sport.toml [capabilities]")
        return bad
    text = json.dumps(matrix(), ensure_ascii=False, indent=1)
    if "--out" in argv:
        Path(argv[argv.index("--out") + 1]).write_text(text + "\n")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
