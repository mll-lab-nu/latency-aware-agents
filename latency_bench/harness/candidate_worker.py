"""Sandbox entrypoint. Deliberately does not import the benchmark or its state."""

import importlib
import json
import sys


def main():
    sys.path.insert(0, "/candidate")
    module, name = sys.argv[1].split(":")
    function = getattr(importlib.import_module(module), name)
    payload = json.load(sys.stdin)
    snapshot, arguments = payload["snapshot"], payload["arguments"]
    if sys.argv[2] == "prompt":
        result = function(arguments["base"], snapshot)
    elif sys.argv[2] in {"request", "diagnostic"}:
        result = function(snapshot)
    else:
        result = function(snapshot, arguments)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
