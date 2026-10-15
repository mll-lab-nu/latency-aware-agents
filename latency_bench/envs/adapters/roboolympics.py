"""Expose the frozen RoboOlympics single-seat matches to latency executors."""

from __future__ import annotations

import base64
import os
import sys
from dataclasses import replace
from importlib.metadata import distribution
from io import BytesIO

import numpy as np
from PIL import Image

from latency_bench.core.types import Action, Observation, StepResult
from latency_bench.envs.base import EnvAdapter
from latency_bench.utils.paths import REPO_ROOT


_SOURCE_ROOT = REPO_ROOT / "third_party" / "roboolympics"
sys.path.insert(0, str(_SOURCE_ROOT))

from core import spec  # noqa: E402 (the selected task's pinned upstream source)
from core.episode import load_plugin  # noqa: E402
from core.local import seat_spec  # noqa: E402
from core.observe import CameraWorker, filter_fields, players_scene  # noqa: E402
from core.outcome import decide  # noqa: E402
from core.sport import FINISHED, UNIT_OVER, MatchConfig, Refused, parse_action  # noqa: E402


_SPORTS = {
    "tabletennis": (100.0, "high", "match"),
    "badminton": (100.0, "high", "match-wallclock"),
    "soccer": (50.0, "low", "match"),
    "soccer_ssl": (100.0, "high", "match"),
}
_loaded_sport = None


def load_sport(name):
    """Load one sport per process because upstream uses unqualified module names."""
    global _loaded_sport
    if name not in _SPORTS:
        raise ValueError(f"Unsupported RoboOlympics sport: {name}")
    if _loaded_sport is None:
        if name == "soccer":
            # The upstream image copies these pinned package assets to /opt/sai_t1.
            os.environ["RCB_T1_ASSETS"] = str(
                distribution("sai-mujoco").locate_file("sai_mujoco/assets/robots/t1"),
            )
        plugin = _SOURCE_ROOT / "sports" / name / "plugin.py"
        _loaded_sport = (name, load_plugin(plugin))
    elif _loaded_sport[0] != name:
        raise ValueError("Run different RoboOlympics sports in separate processes (upstream module names overlap)")
    return _loaded_sport[1]


class RoboOlympicsAdapter(EnvAdapter):
    """Advance one native control step and preserve the public player view."""

    def __init__(self, *, sport: str, level: int, seed: int, max_env_steps: int, noop_action: Action):
        if noop_action.value is not None:
            raise ValueError("RoboOlympics noop_action must be null (advance without submitting a command)")
        self.noop_action = noop_action
        self.sport = load_sport(sport)
        self.plugin_path = _SOURCE_ROOT / "sports" / sport / "plugin.py"
        self.env_fps, action_level, match_format = _SPORTS[sport]
        self.OBSERVATION_TYPE = f"roboolympics_{sport}_v1"
        self.max_env_steps = max_env_steps
        self._cfg = MatchConfig(
            sport=sport, format=match_format, seats=("A",), seed=seed,
            level=level, action_level=action_level, play="program",
        )
        self._declaration = spec.load(self.plugin_path)
        self._allowed = spec.whitelist(self._declaration)
        self._renderer = None
        self._observation = None
        self.match = None
        self.env_step = 0

    def reset(self, seed: int | None = None) -> Observation:
        self.close()
        if seed is not None:
            self._cfg = replace(self._cfg, seed=seed)
        self.match = self.sport.build(self._cfg)
        self._layer = self.sport.action_layer(self._cfg.action_level)
        if self._cfg.sport == "badminton":
            # The HTTP planner's wall-time allowance is replaced by bench latency.
            self._layer.lead = 0
        if self.match.dt != self.frame_ms / 1000:
            raise ValueError(f"RoboOlympics native control step must be {self.frame_ms:g} ms")
        self.env_step = 0
        self._spec = seat_spec(
            self.sport, self._cfg, "A", m=self.match, plugin=self.plugin_path,
        )
        self._spec["timing"] = {"reaction_delay_steps": 0}
        if self._cfg.sport == "badminton":
            self._spec["act"].update(
                lead_steps=0, lead_game_s=0.0,
                rule="The command takes effect when the latency-bench scheduler releases it; t starts at that step.",
            )
            self._spec["skill_contract"]["act"]["notes"]["timing"] = self._spec["act"]["rule"]
        self._observation = None
        return self.observe()

    def step(self, action: Action) -> StepResult:
        before = self.match.scores()
        info = {"applied_action": action.value, "applied_action_name": action.name}
        if not action.is_noop:
            try:
                cmd = parse_action(self._layer, "A", action.value, self.match, {})
                job = self._layer.snapshot("A", cmd, self.match, self.env_step)
                prepared = self._layer.prepare(job) if job is not None else None
                low, planning = self._layer.lower("A", cmd, prepared, self.match)
                applied = self.match.apply("A", low)
                info["action_result"] = {**planning, **applied}
                if "rejected" in applied:
                    info.update(invalid_action=True, action_error=applied["rejected"])
            except Refused as exc:
                info.update(invalid_action=True, action_error=str(exc))
        info["events"] = self.match.step()
        self.env_step += 1
        self._observation = None
        after = self.match.scores()
        reward = (after["A"] - before["A"]) - (after["opponent"] - before["opponent"])
        phase = self.match.phase().state
        done = phase == FINISHED
        winner = decide(self.match, sides=("A", "opponent")).winner if done else None
        truncated = not done and self.env_step >= self.max_env_steps
        if phase == UNIT_OVER and not truncated:
            self.match.resume()
            info["soft_reset"] = True
        info.update(
            env_step=self.env_step, sim_time_ms=self.env_step * self.frame_ms,
            game_score=after["A"], task_metrics={
                "points_for": after["A"], "points_against": after["opponent"],
                "match_finished": done, "truncated": truncated,
                "success": winner == "A",
            },
        )
        if done:
            info["winner"] = winner
        # Executors request rendered observations at their configured cadence.
        return StepResult(observation=None, reward=float(reward), done=done, truncated=truncated, info=info)

    def observe(self) -> Observation:
        if self._observation is None:
            state, leaked = filter_fields(self.match.observe("A"), self._allowed)
            if leaked:
                raise ValueError(f"RoboOlympics observation contains undeclared fields: {leaked}")
            if self._cfg.sport in ("tabletennis", "badminton"):
                if self._renderer is None:
                    self._renderer = self.sport.worker_init("render", "A", self._cfg)
                job, camera_metadata = self.match.camera_job("A")
                image = self.sport.worker_call(self._renderer, "render", job)["png"]
            else:
                if self._renderer is None:
                    self._renderer = CameraWorker(
                        players_scene(self.sport, self.plugin_path, self._cfg), self._cfg.renderer,
                    )
                rendered = self._renderer.draw(
                    self.match.render_state(), self.match.cameras("A"),
                    self._declaration["observation"]["columns"],
                )
                image = rendered["png"]
                camera_metadata = {"tiles": rendered["tiles"]}
            view = {
                "state": {
                    **state, "seat": "A", "world": "running", "step": self.env_step,
                    "t_sim": self.env_step * self.frame_ms / 1000,
                    "done": self.match.phase().state == FINISHED,
                },
                "observation": {"image_png_b64": base64.b64encode(image).decode(), **camera_metadata},
            }
            self._observation = Observation(
                data=view, env_step=self.env_step, sim_time_ms=self.env_step * self.frame_ms,
                metadata={
                    "observation": self.OBSERVATION_TYPE, "spec": self._spec,
                    "episode_seed": self._cfg.seed, "unit": self.match.unit(),
                },
            )
        return self._observation

    def render_game_frame(self) -> np.ndarray:
        image = self.observe().data["observation"]["image_png_b64"]
        with Image.open(BytesIO(base64.b64decode(image))) as frame:
            return np.array(frame.convert("RGB"))

    def close(self) -> None:
        if self._renderer is not None:
            if self._cfg.sport in ("tabletennis", "badminton"):
                self._renderer.r.close()
            else:
                self._renderer.renderer.close()
            self._renderer = None
        self.match = None
        self._observation = None
