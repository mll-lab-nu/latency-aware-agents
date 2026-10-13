"""Observation I/O and process lifetime shared by native coding-agent policies."""

from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
import queue
import signal
import subprocess
import tempfile
import threading
import time
import uuid

from latency_bench.core.types import Observation, PolicyOutput
from latency_bench.data.prompts import BASE_PROMPTS
from latency_bench.envs.base import ENV_RAW_RGB_FRAME_STACK_INFO_KEY
from latency_bench.policy.base import PolicyRunner
from latency_bench.policy._starvla_input import model_rgb_images, observation_data_to_hwc_uint8_frames


class InvalidAgentAction(ValueError):
    """A completed native turn did not return an action allowed by the contract."""


def decode_action(text: str, actions: dict):
    payload = json.loads(text)
    # The agent's response is an external action API, not a permissive text parser.
    if set(payload) != {"action"} or type(payload["action"]) is not int:
        raise ValueError('Expected exactly {"action": <integer action id>}')
    return actions[payload["action"]]


def sandbox_command(bubblewrap, workspace, mounts, environment, command):
    """Confine a native agent to system runtime files and its episode workspace."""
    prefix = [
        bubblewrap, "--unshare-user", "--unshare-pid", "--unshare-ipc", "--unshare-uts",
        "--die-with-parent", "--new-session", "--ro-bind", "/usr", "/usr",
        "--symlink", "usr/bin", "/bin", "--symlink", "usr/sbin", "/sbin",
        "--symlink", "usr/lib", "/lib", "--symlink", "usr/lib64", "/lib64",
        "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp",
        "--dir", "/home/ubuntu", "--dir", "/opt",
    ]
    for path in ("/etc/ssl", "/etc/resolv.conf", "/etc/hosts", "/etc/nsswitch.conf", "/etc/passwd", "/etc/group"):
        prefix.extend(["--ro-bind", path, path])
    prefix.extend(["--bind", str(workspace), "/workspace", "--chdir", "/workspace", "--clearenv"])
    prefix.extend(mounts)
    environment = {
        "HOME": "/home/ubuntu", "SHELL": "/bin/bash", "LANG": "C.UTF-8",
        "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
        **environment,
    }
    for key, value in environment.items():
        prefix.extend(["--setenv", key, value])
    return prefix + ["--", *command]


class CodingAgentPolicyRunner(PolicyRunner):
    def __init__(self, config: dict, action_resolver):
        self.config = config
        self.settings = config["policy"]
        self.reference_input = self.settings["type"] == "coding_agent"
        if self.reference_input:
            from latency_bench.policy.coding_agent_contract import AgentActionCodec, build_reference_observer

            self.observe, self.task_prompt, self.task_contract, self.input_contract = build_reference_observer(config, action_resolver)
            self.codec = AgentActionCodec(self.settings["env_action_space"], action_resolver, self.task_contract)
        else:
            self.actions = {
                index: action_resolver.resolve(ref)
                for index, ref in enumerate(self.settings["actions"])
            }
        self.output = Path(config["logging"]["output_dir"]) / "agent"
        self.output.mkdir(parents=True, exist_ok=True)
        self.profile = (json.loads(Path(config["latency"]["profile_path"]).read_text())
                        if "profile_path" in config["latency"] else None)
        self.process = None
        self.runtime = None

    def reset_state(self, slot_id: int | None = None) -> None:
        self.close()

    def prepare_episode(self, episode_id: int) -> None:
        started = time.perf_counter()
        self._start_episode(episode_id)
        (self.episode_dir / "preparation.json").write_text(json.dumps({
            "episode_id": episode_id,
            "startup_ms": (time.perf_counter() - started) * 1000,
            "measured_before_episode_clock": True,
        }, indent=2) + "\n")

    def _start_episode(self, episode_id: int) -> None:
        self.episode_id = episode_id
        self.decision_id = 0
        self.session_id = str(uuid.uuid4())
        self.episode_dir = self.output / f"episode_{episode_id:03d}"
        self.episode_dir.mkdir()
        self.workspace = Path(self.settings["workspace_root"]) / self.session_id
        self.workspace.mkdir(parents=True)
        (self.workspace / "observations").mkdir()
        self.runtime = tempfile.TemporaryDirectory(
            prefix=self.provider + "-", dir=self.settings["runtime_root"],
        )
        command = self._sandbox_command(self.workspace, Path(self.runtime.name))
        self.stderr = (self.episode_dir / "stderr.log").open("w")
        self.process = subprocess.Popen(
            command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.stderr,
            text=True, start_new_session=True,
        )
        self.lines = queue.Queue()
        self.reader = threading.Thread(
            target=self._read_lines, args=(self.process.stdout, self.lines), daemon=True,
        )
        self.reader.start()
        self._initialize_session()
        (self.episode_dir / "session.json").write_text(json.dumps({
            "provider": self.provider, "session_id": self.session_id, "pid": self.process.pid, "command": command,
            "model": self.settings["model"], "effort": self.settings["effort"],
            "workspace": str(self.workspace),
        }, indent=2) + "\n")

    @staticmethod
    def _read_lines(stream, lines) -> None:
        try:
            for line in stream:
                lines.put(line)
        finally:
            lines.put(None)

    def _initialize_session(self) -> None:
        return None

    def _finish_session(self) -> None:
        return None

    def _receive_event(self, deadline, events):
        try:
            line = self.lines.get(timeout=max(0, deadline - time.monotonic()))
        except queue.Empty as error:
            raise TimeoutError(f"{self.provider} did not finish this request") from error
        if line is None:
            raise RuntimeError(f"{self.provider} exited before returning an action")
        events.write(line)
        events.flush()
        return json.loads(line)

    def _initial_description(self) -> str:
        if self.reference_input:
            return self._reference_description()
        env = self.config["env"]
        if self.config["executor"]["mode"] == "realtime":
            timing = (
                "The game continues while you think. Observation-to-action delays are your "
                "actual wall-clock response time, including input processing, network requests, "
                "generation and decoding. Busy-worker observations are dropped.\n"
            )
        else:
            summary = self.profile["summary"]
            timing = (
                "Observation-to-action delays and worker occupancy "
                "are sampled from a measured OpenVLA single-RTX-3090 profile with temporal "
                f"correlation (mean {summary['mean_ms']:.2f} ms, P95 {summary['p95_ms']:.2f} ms). "
                "Busy-worker observations are dropped. Your wall-clock thinking time does not "
                "advance this virtual environment; the sampled delay does.\n"
            )
        return (
            BASE_PROMPTS["flappy"] + "\n"
            "Observation: one 224x224 RGB image, plus the seven-element state vector "
            "provided below. The state vector is always zero, as in the reference VLA input. "
            "You may use the observations and actions from this episode and your native tools.\n"
            "Action space: integer 0 = NOOP, 1 = FLAP. Each command applies for one raw "
            "environment frame; subsequent frames use NOOP until another command arrives. "
            "The latest ready command supersedes older commands.\n"
            f"The environment runs at {env['env_fps']} FPS and observations are scheduled "
            f"at {env['obs_fps']} FPS. " + timing +
            f"The episode ends on collision or after {self.config['evaluation']['eval_max_steps']} "
            "raw frames.\n"
            'For each observation, return only a JSON object {"action": 0} or {"action": 1} '
            "as your final response, without Markdown or explanation.\n"
        )

    def _reference_description(self):
        env = self.config["env"]
        latency = self.config["latency"]
        timing = {key: latency[key] for key in (
            "method", "fixed_latency_ms", "mean_latency_ms", "std_latency_ms",
            "min_latency_ms", "max_latency_ms", "per_episode",
        ) if key in latency}
        if self.profile is not None:
            timing["profile_summary"] = self.profile["summary"]
        contract = {"actions": self.codec.description(), "env_fps": env["env_fps"],
                    "observation": self.input_contract,
                    "obs_fps": env["obs_fps"], "scheduler": self.config["scheduler"],
                    "worker_capacity": self.config["executor"]["simulated_worker_capacity"],
                    "latency": timing, "max_raw_frames": self.config["evaluation"]["eval_max_steps"]}
        if "state_space" in env:
            contract["state_labels"] = env["state_space"]["labels"]
        if self.task_contract is not None:
            contract["video_keys"] = self.task_contract["video_keys"]
            contract["action_normalization"] = self.task_contract["normalization"]["action"]
        output = ('{"action": <one native action>}' if self.codec.horizon == 1 else
                  '{"actions": <array of ' + str(self.codec.horizon) + ' native actions>}')
        return (
            self.task_prompt + "\n"
            "Images and state use the selected VLA input preprocessing, including normalization. "
            "Images retain checkpoint order: oldest to newest for temporal images; the declared "
            "video order for simultaneous views, possibly stitched as in the checkpoint. "
            "Use observations, actions and native tools within this episode.\n"
            "Each action/chunk head applies at the raw environment frame rate. Return actions "
            "in native environment units and the declared order. Frozen action-prefix entries "
            "are normalized inputs and remain committed; only the suffix can be changed.\n"
            "Virtual delays and worker occupancy follow the configured latency sampler. Busy-worker "
            "observations are dropped. Your wall-clock thinking time does not advance the virtual "
            "environment. No future sampled delay is supplied.\n"
            "Contract: " + json.dumps(contract) + "\n"
            "Return only " + output + " as your final response, without Markdown or explanation.\n"
        )

    def _observation_input(self, observation):
        if self.reference_input:
            return self.observe(observation)
        frames = observation_data_to_hwc_uint8_frames(
            observation.metadata[ENV_RAW_RGB_FRAME_STACK_INFO_KEY],
        )
        return {"image": model_rgb_images(frames[-1:], tuple(self.config["env"]["obs_resize"])),
                "state": self.settings["observation_state"]}

    def predict_batch(self, observations):
        # shortcut: one native episode at a time; separate auth homes before parallel eval.
        observation, = observations
        return [self.predict(observation)]

    def predict(self, observation: Observation) -> PolicyOutput:
        started = time.perf_counter()
        if self.process is None:
            self._start_episode(observation.metadata["episode_id"])
        call_dir = self.episode_dir / f"decision_{self.decision_id:05d}"
        call_dir.mkdir()
        record = {
            "episode_id": self.episode_id, "decision_id": self.decision_id,
            "obs_id": observation.metadata["obs_id"], "env_step": observation.env_step,
            "sim_time_ms": observation.sim_time_ms, "session_id": self.session_id,
            "pid": self.process.pid, "requested_model": self.settings["model"],
            "effort": self.settings["effort"], "started_unix_ns": time.time_ns(),
        }
        try:
            example = self._observation_input(observation)
            images = []
            for index, image in enumerate(example["image"]):
                suffix = "" if index == 0 else f"_{index}"
                image_file = io.BytesIO()
                image.save(image_file, format="PNG")
                image_bytes = image_file.getvalue()
                (call_dir / f"observation{suffix}.png").write_bytes(image_bytes)
                (self.workspace / f"observation{suffix}.png").write_bytes(image_bytes)
                path = f"observations/{self.decision_id:05d}{suffix}.png"
                (self.workspace / path).write_bytes(image_bytes)
                images.append({"path": "/workspace/" + path, "bytes": image_bytes})
            record["image_sha256s"] = [hashlib.sha256(image["bytes"]).hexdigest() for image in images]
            record["image_sha256"] = record["image_sha256s"][0]
            inputs = {}
            for key in ("state", "action_prefix", "action_prefix_mask"):
                if key in example:
                    value = example[key]
                    inputs[key] = value if isinstance(value, list) else value.tolist()
            (call_dir / "observation.json").write_text(json.dumps(inputs) + "\n")
            prompt = self._initial_description() if self.decision_id == 0 else ""
            if self.reference_input:
                prompt += "Current images, in order: " + json.dumps([image["path"] for image in images]) + ". "
                prompt += "Observation: " + json.dumps(inputs) + "."
            else:
                prompt += "Current observation attached (also available as /workspace/observation.png). "
                prompt += "State: " + json.dumps(self.settings["observation_state"]) + "."
            answer, result = self._request_action(prompt, images, call_dir)
            record["result"] = result
            try:
                output = (self.codec.decode(answer, observation) if self.reference_input else
                          PolicyOutput(action=decode_action(answer, self.actions), raw_output=answer))
            except (ValueError, KeyError, TypeError) as error:
                raise InvalidAgentAction(str(error)) from error
            action = output.action
            record["action"] = action.value
            if output.action_chunk is not None:
                record["action_chunk"] = output.action_chunk.tolist()
            record["status"] = "completed"
            record["wall_ms"] = (time.perf_counter() - started) * 1000
            print(
                f"[coding-agent] episode={self.episode_id} decision={self.decision_id} "
                f"frame={observation.env_step} action={action.value} wall_ms={record['wall_ms']:.1f}",
                flush=True,
            )
            self.decision_id += 1
            output.metadata.update({
                "policy_type": self.provider, "agent_model": self.settings["model"],
                "agent_effort": self.settings["effort"], "agent_session_id": self.session_id,
                "agent_wall_ms": record["wall_ms"], "agent_call_dir": str(call_dir),
                "input_image_sha256": record["image_sha256"],
                "input_image_sha256s": record["image_sha256s"],
            })
            return output
        except BaseException as error:
            record.update(status="failed", error=repr(error))
            raise
        finally:
            record["finished_unix_ns"] = time.time_ns()
            record["wall_ms"] = (time.perf_counter() - started) * 1000
            (call_dir / "call.json").write_text(json.dumps(record, indent=2) + "\n")

    def close(self) -> None:
        try:
            if self.process is not None:
                try:
                    try:
                        self.process.stdin.close()
                    except BrokenPipeError:
                        pass  # A failed CLI may have already closed its input.
                    try:
                        self.process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(self.process.pid, signal.SIGTERM)
                        try:
                            self.process.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            os.killpg(self.process.pid, signal.SIGKILL)
                            self.process.wait()
                    self.reader.join(timeout=5)
                    self.process.stdout.close()
                finally:
                    self.stderr.close()
                    self.process = None
        finally:
            if self.runtime is not None:
                self._finish_session()
                self.runtime.cleanup()
                self.runtime = None
