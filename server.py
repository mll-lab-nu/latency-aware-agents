from __future__ import annotations

import asyncio
import json
import threading
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from app_settings import DEFAULT_OUTPUT_ROOT, DEFAULT_PORT
from demo_loop import DemoLoop, HumanInputState

app = FastAPI(title="Human Game Interface")

_loop: Optional[DemoLoop] = None
_lock = threading.Lock()
_human_input = HumanInputState()


class StartRequest(BaseModel):
    env: str = "flappy"
    speed: float = 1.0
    run_mode: str = "eval"
    latency_frames: int = 0
    latency_start: Optional[int] = None
    latency_end: Optional[int] = None
    participant_name: str
    output_root: str = DEFAULT_OUTPUT_ROOT
    episodes_per_latency: int = 20
    max_steps_per_episode: Optional[int] = None
    session_id: Optional[str] = None


class ConfigRequest(BaseModel):
    speed: Optional[float] = None


class InputRequest(BaseModel):
    keys: list[str] = Field(default_factory=list)


@app.get("/", response_class=HTMLResponse)
async def index():
    return (Path(__file__).parent / "static" / "index.html").read_text()


@app.get("/settings")
async def settings():
    return {
        "default_output_root": DEFAULT_OUTPUT_ROOT,
        "default_port": DEFAULT_PORT,
    }


@app.post("/start")
async def start(req: StartRequest):
    global _loop
    latency_frames = max(0, int(req.latency_frames))
    latency_start = max(0, int(req.latency_start if req.latency_start is not None else latency_frames))
    latency_end = max(0, int(req.latency_end if req.latency_end is not None else latency_start))
    participant_name = req.participant_name.strip()
    if not participant_name:
        return {"ok": False, "error": "participant_name is required"}
    with _lock:
        _human_input.set_keys([])
        if _loop is not None:
            _loop.stop()
            _loop = None
        _loop = DemoLoop(
            env_name=req.env,
            speed=req.speed,
            latency_frames=latency_frames,
            run_mode=req.run_mode,
            latency_start=latency_start,
            latency_end=latency_end,
            human_input=_human_input,
            output_root=req.output_root or DEFAULT_OUTPUT_ROOT,
            participant_name=participant_name,
            session_id=req.session_id,
            episodes_per_latency=max(1, int(req.episodes_per_latency)),
            max_steps_per_episode=req.max_steps_per_episode,
        )
        _loop.start()
    return {
        "ok": True,
        "env": req.env,
        "speed": req.speed,
        "run_mode": _loop.run_mode,
        "latency_frames": latency_frames,
        "latency_start": _loop.latency_start,
        "latency_end": _loop.latency_end,
        "latencies": _loop.latencies,
        "participant_name": participant_name,
        "session_id": _loop.session_id,
        "episodes_per_latency": _loop.episodes_per_latency,
        "max_steps_per_episode": _loop.max_steps_per_episode,
        "num_episodes": _loop.num_episodes,
        "output_dir": str(_loop.output_dir),
        "session_root": str(_loop.session_root),
        "env_root": str(_loop.env_root),
        "mode": "human_realtime",
    }


@app.post("/stop")
async def stop():
    global _loop
    with _lock:
        _human_input.set_keys([])
        if _loop is not None:
            _loop.stop()
            _loop = None
    return {"ok": True}


@app.post("/config")
async def config(req: ConfigRequest):
    with _lock:
        if _loop is not None and req.speed is not None:
            _loop.set_speed(req.speed)
    return {"ok": True}


@app.post("/input")
async def input_state(req: InputRequest):
    _human_input.set_keys(req.keys)
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
