"""Execute research-owned Python against public input in a networkless sandbox."""

from copy import deepcopy
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from latency_bench.policy.coding_agent import sandbox_command


class CandidateRuntime:
    """Isolate prompt, request and information-tool code from coordinator state."""

    def __init__(self, root, bubblewrap):
        self.root = Path(root).resolve()
        self.bubblewrap = bubblewrap

    def call(self, reference, operation, snapshot, arguments):
        public = deepcopy(snapshot)
        with tempfile.TemporaryDirectory(prefix="harness-public-") as temporary:
            workspace = Path(temporary)
            images = workspace / "images"
            images.mkdir()
            # Event images are also legal observations. No other host paths are mounted.
            observations = list(public["images"])
            observations.extend(event["observation"] for event in public["events"] if event["kind"] == "observation")
            copied = {}
            for item, source in [(item, item["path"]) for item in observations]:
                if source not in copied:
                    name = f"images/{len(copied)}.png"
                    shutil.copyfile(source, workspace / name)
                    copied[source] = "/workspace/" + name
                item["path"] = copied[source]
            mounts = ["--unshare-net", "--ro-bind", str(self.root), "/candidate",
                      "--ro-bind", str(Path(__file__).with_name("candidate_worker.py")), "/bridge.py"]
            if not Path(sys.prefix).is_relative_to("/usr"):
                mounts.extend(["--ro-bind", sys.prefix, sys.prefix])
            command = sandbox_command(self.bubblewrap, workspace, mounts,
                                      {"PYTHONDONTWRITEBYTECODE": "1"},
                                      [sys.executable, "-I", "/bridge.py", reference, operation])
            result = subprocess.run(command, input=json.dumps({"snapshot": public, "arguments": arguments}),
                                    capture_output=True, text=True, timeout=10)
            if result.returncode:
                raise RuntimeError(f"Candidate {operation} failed: {result.stderr[-4000:]}")
            return json.loads(result.stdout)
