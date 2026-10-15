"""Expose RoboOlympics single-player table tennis to the latency executors."""

from __future__ import annotations

import base64
import sys
from dataclasses import replace
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
from core.observe import filter_fields  # noqa: E402
from core.outcome import decide  # noqa: E402
from core.sport import FINISHED, UNIT_OVER, MatchConfig, Refused, parse_action  # noqa: E402


PLUGIN_PATH = _SOURCE_ROOT / "sports" / "tabletennis" / "plugin.py"
SPORT = load_plugin(PLUGIN_PATH)


class RoboOlympicsTableTennisAdapter(EnvAdapter):
    """Advance one native control step and preserve the public player view."""

    OBSERVATION_TYPE = "roboolympics_tabletennis_v1"

    def __init__(self, *, level: int, seed: int, max_env_steps: int, noop_action: Action):
        if noop_action.value is not None:
            raise ValueError("RoboOlympics noop_action must be null (advance without submitting a command)")
        self.noop_action = noop_action
        self.env_fps = 100.0
        self.max_env_steps = max_env_steps
        self._cfg = MatchConfig(
            sport="tabletennis", format="match", seats=("A",), seed=seed,
            level=level, action_level="high", play="program",
        )
        self._allowed = spec.whitelist(spec.load(PLUGIN_PATH))
        self._layer = SPORT.action_layer("high")
        self._renderer = None
        self._observation = None
        self.match = None
        self.env_step = 0

    def reset(self, seed: int | None = None) -> Observation:
        self.close()
        if seed is not None:
            self._cfg = replace(self._cfg, seed=seed)
        self.match = SPORT.build(self._cfg)
        if self.match.dt != self.frame_ms / 1000:
            raise ValueError("RoboOlympics native control step must be 10 ms")
        self.env_step = 0
        self._spec = seat_spec(
            SPORT, self._cfg, "A", m=self.match, plugin=PLUGIN_PATH,
        )
        self._observation = None
        return self.observe()

    def step(self, action: Action) -> StepResult:
        before = self.match.scores()
        info = {"applied_action": action.value, "applied_action_name": action.name}
        if not action.is_noop:
            try:
                cmd = parse_action(self._layer, "A", action.value, self.match, {})
                low, planning = self._layer.lower("A", cmd, None, self.match)
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
            if self._renderer is None:
                self._renderer = SPORT.worker_init("render", "A", self._cfg)
            job, camera_metadata = self.match.camera_job("A")
            image = SPORT.worker_call(self._renderer, "render", job)["png"]
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
                metadata={"observation": self.OBSERVATION_TYPE, "spec": self._spec, "episode_seed": self._cfg.seed},
            )
        return self._observation

    def render_game_frame(self) -> np.ndarray:
        image = self.observe().data["observation"]["image_png_b64"]
        with Image.open(BytesIO(base64.b64decode(image))) as frame:
            return np.array(frame.convert("RGB"))

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.r.close()
            self._renderer = None
        self.match = None
        self._observation = None
