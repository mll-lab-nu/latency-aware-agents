from __future__ import annotations

import base64
import io
import threading
import time
from typing import Any, Callable, Dict, Optional

import numpy as np
from PIL import Image


class DemoLoop:
    """
    Runs a game + model loop in a background thread.

    mode="paused"    — sync inference, env waits for model (no sleep)
    mode="simulated" — sync inference, sleep to hit target FPS,
                       QueueLatencyWrapper applies K-frame delay from slider
    mode="realtime"  — async inference thread, game ticks at fixed FPS,
                       latency=0 in wrapper; effective_k = inf_ms / frame_ms
    """

    def __init__(
        self,
        env_name: str,
        checkpoint: str,
        speed: float = 1.0,
        latency: int = 0,
        mode: str = "simulated",
    ):
        self.env_name = env_name
        self.checkpoint = checkpoint
        self._speed = max(0.1, float(speed))
        self.latency = max(0, int(latency))
        self.mode = mode

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._latest: Optional[Dict[str, Any]] = None
        self._frame_lock = threading.Lock()

        # Real-time mode shared state
        self._rt_lock = threading.Lock()
        self._rt_action: Any = 0
        self._rt_inf_ms: float = 0.0
        self._rt_obs_buf: Any = None
        self._rt_new_obs = threading.Event()

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def start(self):
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        self._rt_new_obs.set()  # unblock inference thread if waiting
        if self._thread is not None:
            self._thread.join(timeout=10.0)

    def set_speed(self, speed: float):
        self._speed = max(0.1, min(8.0, float(speed)))

    def get_latest(self) -> Optional[Dict[str, Any]]:
        with self._frame_lock:
            return self._latest

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _target_step_time(self) -> float:
        """Seconds per game step (base = ~30 fps at 1×)."""
        return 0.033 / self._speed

    def _push(self, frame: np.ndarray, stats: Dict[str, Any]):
        img = Image.fromarray(frame.astype(np.uint8))
        w, h = img.size
        scale = 320 / max(w, h)
        img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.NEAREST)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=75)
        b64 = base64.b64encode(buf.getvalue()).decode()
        with self._frame_lock:
            self._latest = {"frame": b64, "stats": stats}

    def _stats(
        self,
        env_name: str,
        episode: int,
        step: int,
        reward: float,
        total_reward: float,
        inference_ms: float,
        effective_k: float,
    ) -> Dict[str, Any]:
        return {
            "env": env_name,
            "episode": episode,
            "step": step,
            "reward": round(float(reward), 3),
            "total_reward": round(float(total_reward), 3),
            "inference_ms": round(float(inference_ms), 2),
            "speed": round(self._speed, 2),
            "latency_frames": self.latency,
            "effective_k": round(float(effective_k), 2),
            "mode": self.mode,
        }

    def _get_render(self, env_or_venv, is_vec: bool) -> Optional[np.ndarray]:
        try:
            if is_vec:
                return env_or_venv.envs[0].unwrapped.render()
            return env_or_venv.unwrapped.render()
        except Exception:
            return None

    def _infer_sync(self, model_fn: Callable, obs) -> tuple:
        t = time.perf_counter()
        action = model_fn(obs)
        return action, (time.perf_counter() - t) * 1000

    def _maybe_sleep(self, t0: float):
        elapsed = time.perf_counter() - t0
        rem = self._target_step_time() - elapsed
        if rem > 0:
            time.sleep(rem)

    def _rt_inference_worker(self, model_fn: Callable, stop_event: threading.Event):
        """Background inference thread for real-time mode."""
        while not stop_event.is_set():
            if self._rt_new_obs.wait(timeout=0.1):
                self._rt_new_obs.clear()
                if stop_event.is_set():
                    break
                with self._rt_lock:
                    obs = self._rt_obs_buf
                if obs is None:
                    continue
                t = time.perf_counter()
                action = model_fn(obs)
                inf_ms = (time.perf_counter() - t) * 1000
                with self._rt_lock:
                    self._rt_action = action
                    self._rt_inf_ms = inf_ms

    # -------------------------------------------------------------------------
    # Generic episode loops
    # -------------------------------------------------------------------------

    def _loop_vec(self, venv, model_fn: Callable, env_name: str):
        """Episode loop for SB3-style vectorized envs (Flappy, VizDoom)."""
        episode = 0
        while not self._stop_event.is_set():
            episode += 1
            obs = venv.reset()
            total_reward = 0.0
            step = 0
            done = False

            inf_thread: Optional[threading.Thread] = None
            inf_stop: Optional[threading.Event] = None

            if self.mode == "realtime":
                inf_stop = threading.Event()
                with self._rt_lock:
                    self._rt_action = np.array([0])
                    self._rt_inf_ms = 0.0
                    self._rt_obs_buf = obs  # prime with first obs
                self._rt_new_obs.set()
                inf_thread = threading.Thread(
                    target=self._rt_inference_worker,
                    args=(model_fn, inf_stop),
                    daemon=True,
                )
                inf_thread.start()

            try:
                while not done and not self._stop_event.is_set():
                    t0 = time.perf_counter()

                    if self.mode == "realtime":
                        # Apply action pre-computed from previous obs
                        with self._rt_lock:
                            action = self._rt_action
                            inf_ms = self._rt_inf_ms
                        # Post new obs for next tick's inference
                        with self._rt_lock:
                            self._rt_obs_buf = obs
                        self._rt_new_obs.set()
                        frame_ms = self._target_step_time() * 1000
                        effective_k = inf_ms / frame_ms if frame_ms > 0 else 0.0
                    else:
                        action, inf_ms = self._infer_sync(model_fn, obs)
                        effective_k = float(self.latency)

                    obs, reward, done_vec, _ = venv.step(action)
                    done = bool(done_vec[0])
                    total_reward += float(reward[0])
                    step += 1

                    raw = self._get_render(venv, is_vec=True)
                    if raw is not None:
                        self._push(raw, self._stats(env_name, episode, step,
                                                    reward[0], total_reward,
                                                    inf_ms, effective_k))

                    if self.mode != "paused":
                        self._maybe_sleep(t0)
            finally:
                if inf_stop is not None:
                    inf_stop.set()
                    self._rt_new_obs.set()
                if inf_thread is not None:
                    inf_thread.join(timeout=2.0)

        venv.close()

    def _loop_gym(self, env, model_fn: Callable, env_name: str):
        """Episode loop for plain gym envs (Demon Attack / CleanRL)."""
        episode = 0
        while not self._stop_event.is_set():
            episode += 1
            obs, _ = env.reset()
            total_reward = 0.0
            step = 0
            done = False

            inf_thread: Optional[threading.Thread] = None
            inf_stop: Optional[threading.Event] = None

            if self.mode == "realtime":
                inf_stop = threading.Event()
                with self._rt_lock:
                    self._rt_action = 0
                    self._rt_inf_ms = 0.0
                    self._rt_obs_buf = obs  # prime with first obs
                self._rt_new_obs.set()
                inf_thread = threading.Thread(
                    target=self._rt_inference_worker,
                    args=(model_fn, inf_stop),
                    daemon=True,
                )
                inf_thread.start()

            try:
                while not done and not self._stop_event.is_set():
                    t0 = time.perf_counter()

                    if self.mode == "realtime":
                        with self._rt_lock:
                            action = self._rt_action
                            inf_ms = self._rt_inf_ms
                        with self._rt_lock:
                            self._rt_obs_buf = obs
                        self._rt_new_obs.set()
                        frame_ms = self._target_step_time() * 1000
                        effective_k = inf_ms / frame_ms if frame_ms > 0 else 0.0
                    else:
                        action, inf_ms = self._infer_sync(model_fn, obs)
                        effective_k = float(self.latency)

                    obs, reward, terminated, truncated, _ = env.step(int(action))
                    done = terminated or truncated
                    total_reward += float(reward)
                    step += 1

                    raw = self._get_render(env, is_vec=False)
                    if raw is not None:
                        self._push(raw, self._stats(env_name, episode, step,
                                                    reward, total_reward,
                                                    inf_ms, effective_k))

                    if self.mode != "paused":
                        self._maybe_sleep(t0)
            finally:
                if inf_stop is not None:
                    inf_stop.set()
                    self._rt_new_obs.set()
                if inf_thread is not None:
                    inf_thread.join(timeout=2.0)

        env.close()

    # -------------------------------------------------------------------------
    # Dispatch
    # -------------------------------------------------------------------------

    def _run(self):
        try:
            {"flappy": self._run_flappy,
             "demon_attack": self._run_demon_attack,
             "vizdoom": self._run_vizdoom}.get(self.env_name, self._unknown)()
        except Exception as e:
            import traceback
            print(f"[demo] {self.env_name} crashed: {e}")
            traceback.print_exc()

    def _unknown(self):
        print(f"[demo] unknown env: {self.env_name}")

    # -------------------------------------------------------------------------
    # Per-env setup  (build env + model_fn, dispatch to generic loop)
    # -------------------------------------------------------------------------

    def _run_flappy(self):
        from envs.flappy import make_demo_env
        from models.sb3_agent import load_sb3

        env_latency = 0 if self.mode == "realtime" else self.latency
        venv = make_demo_env(latency=env_latency)
        model = load_sb3(self.checkpoint) if self.checkpoint else None

        if model is not None:
            def model_fn(obs):
                return model.predict(obs, deterministic=True)[0]
        else:
            def model_fn(obs):
                return np.array([venv.action_space.sample()])

        self._loop_vec(venv, model_fn, "flappy")

    def _run_demon_attack(self):
        import torch
        from envs.demon_attack import make_demo_env
        from models.cleanrl_agent import load_cleanrl

        env_latency = 0 if self.mode == "realtime" else self.latency
        env = make_demo_env(latency=env_latency)
        model = (
            load_cleanrl(self.checkpoint,
                         obs_shape=env.observation_space.shape,
                         num_actions=env.action_space.n)
            if self.checkpoint else None
        )

        if model is not None:
            def model_fn(obs):
                t = torch.as_tensor(obs, dtype=torch.uint8).unsqueeze(0)
                return model.get_deterministic_action(t).cpu().item()
        else:
            def model_fn(obs):
                return env.action_space.sample()

        self._loop_gym(env, model_fn, "demon_attack")

    def _run_vizdoom(self):
        from envs.vizdoom import make_demo_env
        from models.sb3_agent import load_sb3

        env_latency = 0 if self.mode == "realtime" else self.latency
        venv = make_demo_env(latency=env_latency)
        model = load_sb3(self.checkpoint) if self.checkpoint else None

        if model is not None:
            def model_fn(obs):
                return model.predict(obs, deterministic=True)[0]
        else:
            def model_fn(obs):
                return np.array([venv.action_space.sample()])

        self._loop_vec(venv, model_fn, "vizdoom")
