"""Run named test functions of sports/badminton/tests/test_badminton.py (env TESTS, comma list) in the image."""
import os, sys, time, traceback
sys.path.insert(0, "/repo"); os.chdir("/repo")
sys.path.insert(0, "/repo/sports/badminton/tests")
import test_badminton as T
ok = True
for name in os.environ["TESTS"].split(","):
    t0 = time.time()
    try:
        getattr(T, name)(); print("PASS", name, round(time.time() - t0, 1), flush=True)
    except Exception:
        ok = False; print("FAIL", name, round(time.time() - t0, 1)); traceback.print_exc()
sys.exit(0 if ok else 1)
