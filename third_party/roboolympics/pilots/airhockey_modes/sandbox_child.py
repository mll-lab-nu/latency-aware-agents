"""The sandbox's child (sandbox.py): loads the deliverable, answers JSON-line calls on fd 0 / 1. The deliverable's own
prints go to stderr."""
import importlib.util
import json
import os
import sys
import traceback


def main():
    path, cls = sys.argv[1], sys.argv[2]
    out = os.fdopen(os.dup(1), "w")
    os.dup2(2, 1)                                   # print() of the deliverable -> stderr
    for d in [os.path.dirname(path)] + [x for x in os.environ.get("AHM_LIB", "").split(os.pathsep) if x]:
        if d not in sys.path:
            sys.path.insert(0, d)
    obj = None

    def conv(o):
        try:
            import numpy as np
            if isinstance(o, np.ndarray):
                return o.tolist()
            if isinstance(o, np.generic):
                return o.item()
        except ImportError:
            pass
        raise TypeError(type(o).__name__)

    for line in sys.stdin:
        msg = json.loads(line)
        try:
            if msg["call"] == "init":
                spec = importlib.util.spec_from_file_location("deliverable", path)
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                obj = getattr(mod, cls)(msg["spec"])
                ret = None
            else:
                ret = getattr(obj, msg["call"])(*msg["args"])
            out.write(json.dumps({"ret": ret}, default=conv) + "\n")
        except Exception as e:  # noqa: BLE001
            tb = traceback.format_exc(limit=4)
            out.write(json.dumps({"err": f"{type(e).__name__}: {e}"[:400] + " | " + tb[-600:]}) + "\n")
        out.flush()


if __name__ == "__main__":
    main()
