from __future__ import annotations

import asyncio
import json
import threading
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from config import CHECKPOINTS
from demo_loop import DemoLoop

app = FastAPI(title="Realtime Inference Engine")

_loop: Optional[DemoLoop] = None
_lock = threading.Lock()


class StartRequest(BaseModel):
    env: str = "flappy"
    speed: float = 1.0
    latency: int = 0
    mode: str = "simulated"  # "paused" | "simulated" | "realtime"


class ConfigRequest(BaseModel):
    speed: Optional[float] = None


@app.get("/", response_class=HTMLResponse)
async def index():
    return (Path(__file__).parent / "static" / "index.html").read_text()


@app.post("/start")
async def start(req: StartRequest):
    global _loop
    with _lock:
        if _loop is not None:
            _loop.stop()
            _loop = None
        checkpoint = CHECKPOINTS.get(req.env, "")
        _loop = DemoLoop(
            env_name=req.env,
            checkpoint=checkpoint,
            speed=req.speed,
            latency=req.latency,
            mode=req.mode,
        )
        _loop.start()
    return {"ok": True, "env": req.env, "speed": req.speed,
            "latency": req.latency, "mode": req.mode}


@app.post("/stop")
async def stop():
    global _loop
    with _lock:
        if _loop is not None:
            _loop.stop()
            _loop = None
    return {"ok": True}


@app.post("/config")
async def config(req: ConfigRequest):
    # Only speed can be changed live; latency requires restart via /start
    with _lock:
        if _loop is not None and req.speed is not None:
            _loop.set_speed(req.speed)
    return {"ok": True}


@app.websocket("/ws")
async def ws(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            with _lock:
                loop = _loop
            if loop is not None:
                payload = loop.get_latest()
                if payload is not None:
                    await websocket.send_text(json.dumps(payload))
            await asyncio.sleep(0.033)
    except (WebSocketDisconnect, Exception):
        pass
