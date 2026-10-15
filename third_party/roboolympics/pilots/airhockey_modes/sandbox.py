"""Run an untrusted deliverable (a skill implementation `class Impl(spec)`: move(view, x, y) / strike(view, req) / step(view); or a high-level
player `class Player(spec)`: act(view)) in a separate process: JSON lines over pipes, a clean environment (no
credentials), as an unprivileged user when the caller is root (it cannot read the evaluator, the hidden cases or the
simulator), a per-call time limit. The parent executes the returned commands; the child never touches the simulator.

    with Sandbox(path, "Impl", spec) as impl:  impl.call("strike", view, req) -> commands | raises SandboxError
"""
from __future__ import annotations

import json
import os
import select
import subprocess
import sys
from pathlib import Path

CHILD = Path(os.environ.get("AHM_SANDBOX_CHILD") or Path(__file__).resolve().parent / "sandbox_child.py")
NOBODY = 65534


class SandboxError(RuntimeError):
    pass


def _jsonable(o):
    try:
        import numpy as np
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, np.generic):
            return o.item()
    except ImportError:
        pass
    raise TypeError(type(o).__name__)


class Sandbox:
    def __init__(self, path, cls, spec, lib_dirs=(), call_timeout_s=20.0, init_timeout_s=120.0, user=None):
        self.path, self.cls = str(Path(path).resolve()), cls
        env = {"PATH": "/usr/local/bin:/usr/bin:/bin", "HOME": "/tmp", "OPENBLAS_NUM_THREADS": "1",
               "OMP_NUM_THREADS": "1", "MUJOCO_GL": "disable", "PYTHONDONTWRITEBYTECODE": "1",
               "AHM_LIB": os.pathsep.join(str(x) for x in lib_dirs),
               "AHM_TOOLKIT": str(spec.get("toolkit_dir", ""))}
        uid = NOBODY if (user is None and os.geteuid() == 0) else user
        pre = None
        if uid is not None:
            def pre():
                os.setgroups([])
                os.setgid(NOBODY)
                os.setuid(uid)
        self.p = subprocess.Popen([sys.executable, "-I", str(CHILD), self.path, cls], stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env,
                                  cwd=str(Path(self.path).parent), preexec_fn=pre)
        self.call_timeout_s = call_timeout_s
        self.stderr_tail = b""
        self._send({"call": "init", "spec": spec})
        self._recv(init_timeout_s)

    def _send(self, msg):
        try:
            self.p.stdin.write((json.dumps(msg, default=_jsonable) + "\n").encode())
            self.p.stdin.flush()
        except (BrokenPipeError, OSError) as e:
            raise SandboxError(f"the implementation's process ended ({e}); stderr: {self._err()}")

    def _err(self):
        try:
            r, _, _ = select.select([self.p.stderr], [], [], 0)
            if r:
                self.stderr_tail = (self.stderr_tail + os.read(self.p.stderr.fileno(), 65536))[-2000:]
        except (OSError, ValueError):
            pass
        return self.stderr_tail.decode(errors="replace")[-600:]

    def _recv(self, timeout):
        buf = b""
        fd = self.p.stdout.fileno()
        while not buf.endswith(b"\n"):
            r, _, _ = select.select([fd, self.p.stderr], [], [], timeout)
            if not r:
                raise SandboxError(f"time limit ({timeout} s) exceeded in one call")
            if self.p.stderr in r:
                self._err()
                if fd not in r:
                    continue
            chunk = os.read(fd, 1 << 20)
            if not chunk:
                raise SandboxError(f"the implementation's process ended; stderr: {self._err()}")
            buf += chunk
        msg = json.loads(buf)
        if "err" in msg:
            raise SandboxError(msg["err"])
        return msg.get("ret")

    def call(self, name, *args):
        self._send({"call": name, "args": list(args)})
        return self._recv(self.call_timeout_s)

    def close(self):
        try:
            self.p.kill()
            self.p.wait(timeout=5)
        except Exception:  # noqa: BLE001
            pass
        for f in (self.p.stdin, self.p.stdout, self.p.stderr):
            try:
                f.close()
            except Exception:  # noqa: BLE001
                pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()
