"""Check (or, with --fix, refresh) the sha256 of the files the pilot's toolkit manifests list.
    python3 toolkit_check.py [--fix]"""
import hashlib
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    fix, bad = "--fix" in sys.argv, 0
    man = HERE / "toolkit" / "toolkit_low.toml"
    t = man.read_text()
    for key, rel in (("robot_model", "toolkit/iiwa_mallet.xml"), ("ik_library", "toolkit/lib/ahkin.py")):
        h = hashlib.sha256((HERE / rel).read_bytes()).hexdigest()
        m = re.compile(r'sha256 = "([0-9a-f]+)"').search(t, t.index(f"[{key}]"))
        ok = m and m.group(1) == h
        print(("ok  " if ok else "BAD ") + rel)
        if not ok:
            bad += 1
            if fix and m:
                t = t[:m.start(1)] + h + t[m.end(1):]
    if fix:
        man.write_text(t)
    return 0 if (bad == 0 or fix) else 1


if __name__ == "__main__":
    sys.exit(main())
