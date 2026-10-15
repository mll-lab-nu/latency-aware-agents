#!/usr/bin/env bash
# Anti-cheat audit 2026-10-08 (G2, G6): in a RoboOlympics task (the v0.2 core: environment/docker-compose.yaml runs
# /srv/core/launch.py) every internal (seat) network runs with an isolated gateway
# (com.docker.network.bridge.gateway_mode_ipv4: isolated, Docker >= 28): a seat reaches the sim but no service on the
# host; the seats (player / seat_*) and the sim drop NET_RAW (no raw sockets, no SO_MARK); the sim does not forward
# packets (net.ipv4.ip_forward: "0"). A seat's skill container (skill_<seat>, the skill in the loop: the player's own
# code, skilleval/inloop.py) has no network, a read-only root, every capability dropped (the host adds back only
# SETUID / SETGID / DAC_READ_SEARCH) and no new privileges. Usage: check-seat-isolation.sh tasks/<task-dir>
t="$1"; python3 - "$t" <<'PY'
import re, sys, pathlib
t = pathlib.Path(sys.argv[1]); cf = t / "environment/docker-compose.yaml"; rc = 0
if not cf.is_file() or "/srv/core/launch.py" not in cf.read_text():
    sys.exit(0)
text = cf.read_text()
def fail(m):
    global rc; print(f"FAIL [seat isolation] {t}: {m}"); rc = 1
def blocks(section):
    """{name: block text} of the 2-space keys under a top-level section."""
    m = re.search(rf"^{section}:\n((?:(?:  .*)?\n)*)", text, re.M)
    out, name = {}, None
    for line in (m.group(1) if m else "").splitlines():
        k = re.match(r"^  (\S[^:]*):", line)
        if k:
            name = k.group(1); out[name] = line + "\n"
        elif name:
            out[name] += line + "\n"
    return out
nets, svcs = blocks("networks"), blocks("services")
internal = [n for n, b in nets.items() if re.search(r"^\s+internal: true", b, re.M)]
if not internal:
    fail("no internal (seat) network")
for n in internal:
    if not re.search(r"com\.docker\.network\.bridge\.gateway_mode_ipv4: \"?isolated\"?", nets[n]):
        fail(f"network {n}: internal without gateway_mode_ipv4: isolated (a seat could reach the host's services)")
for s, b in svcs.items():
    if s == "sim" or s == "player" or s.startswith("seat_"):
        if not re.search(r"^\s+cap_drop: \[[^\]]*\b(NET_RAW|ALL)\b", b, re.M):
            fail(f"service {s}: no cap_drop NET_RAW")
for s, b in svcs.items():
    if s.startswith("skill_"):
        for rule, pat in (("network_mode: none", r"^\s+network_mode: \"?none\"?\s*$"),
                          ("read_only: true", r"^\s+read_only: true\s*$"), ("cap_drop: [ALL]", r"^\s+cap_drop: \[ALL\]"),
                          ("no-new-privileges", r"no-new-privileges:true")):
            if not re.search(pat, b, re.M):
                fail(f"skill container {s}: no {rule}")
        extra = set(re.findall(r"[A-Z_]+", (re.search(r"^\s+cap_add: \[([^\]]*)\]", b, re.M) or [None, ""])[1]))
        if extra - {"SETUID", "SETGID", "DAC_READ_SEARCH"}:
            fail(f"skill container {s}: capabilities beyond the host's: {sorted(extra)}")
        if re.search(r"^\s+networks:", b, re.M) or "./sim" in b:
            fail(f"skill container {s}: a network or the simulator directory")
if "sim" in svcs and not re.search(r"net\.ipv4\.ip_forward: \"?0\"?", svcs["sim"]):
    fail('service sim: no sysctls net.ipv4.ip_forward: "0"')
sys.exit(rc)
PY
