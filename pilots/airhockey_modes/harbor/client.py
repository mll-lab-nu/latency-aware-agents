"""Client of the air-hockey two-mode pilot's match service (http://sim:8000), standard library only.

    import sys; sys.path.insert(0, "/opt/robot"); from client import Game
    g = Game(); spec = g.spec(); g.start()
    obs = g.observe(); r = g.act([{"type": "move", "x": -0.7, "y": 0.1}]); g.wait(); g.ready("why", "plan")
"""
import json
import os
import urllib.error
import urllib.request

URL = os.environ.get("GAME_URL", "http://sim:8000")


class Game:
    def __init__(self, url=URL):
        self.url = url.rstrip("/")

    def _req(self, method, path, body=None, timeout=120):
        data = None if body is None else json.dumps(body).encode()
        rq = urllib.request.Request(f"{self.url}/{path}", data=data, method=method,
                                    headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(rq, timeout=timeout) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            try:
                return {"http_status": e.code, **json.loads(e.read())}
            except ValueError:
                return {"http_status": e.code, "error": str(e)}

    def spec(self):
        return self._req("GET", "spec")

    def status(self):
        return self._req("GET", "status")

    def start(self):
        return self._req("POST", "start", {})

    def observe(self):
        return self._req("POST", "observe", {})

    def act(self, commands):
        return self._req("POST", "act", {"commands": commands if isinstance(commands, list) else [commands]})

    def wait(self, timeout_s=60):
        return self._req("POST", "wait", {"timeout_s": timeout_s}, timeout=timeout_s + 30)

    def ready(self, why, plan):
        return self._req("POST", "ready", {"why": why, "plan": plan})
