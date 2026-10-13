"""Paper environments retain their VLA observations and native action semantics."""

from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import yaml
from gymnasium import spaces

from latency_bench.core.actions import ActionResolver
from latency_bench.core.config import load_config
from latency_bench.core.types import Action, Observation
from latency_bench.envs.base import ENV_RAW_RGB_FRAME_STACK_INFO_KEY as RGB
from latency_bench.envs.factory import environment_integration, resolve_env_config
from latency_bench.envs.tasks.contracts import gymnasium_action_space_contract, gymnasium_task_contract
from latency_bench.policy.coding_agent_config import adapt_reference_config
from latency_bench.policy.coding_agent_contract import (
    AgentActionCodec, bind_coding_agent_action_space, build_reference_observer, prepare_coding_agent_config,
)

ROOT = Path(__file__).resolve().parents[2]
REFERENCES = {
    "flappy": "configs/experiments/flappy/starvla_eval_zero_latency.yaml",
    "deadly_corridor": "configs/experiments/deadly_corridor/starvla_eval_zero_latency.yaml",
    "demon_attack": "configs/experiments/demon_attack/starvla_eval_zero_latency.yaml",
    "asterix": "configs/experiments/asterix/starvla_eval_zero_latency.yaml",
    "atlantis": "configs/experiments/atlantis/starvla_eval_zero_latency.yaml",
    "air_raid": "configs/examples/gymnasium/air_raid/starvla_eval.yaml",
    "ant": "configs/examples/gymnasium/ant/starvla_eval.yaml",
    "walker2d": "configs/examples/gymnasium/walker2d/starvla_eval.yaml",
    "hopper": "configs/examples/gymnasium/hopper/starvla_eval.yaml",
    "inverted_pendulum": "configs/examples/gymnasium/inverted_pendulum/starvla_eval.yaml",
}


def agent_config(reference, tmp_path):
    client = load_config(ROOT / "configs/experiments/coding_agent/flappy_codex_luna_low.yaml")
    protocol = {key: deepcopy(client[key]) for key in ("experiment", "executor", "latency", "evaluation", "logging")}
    protocol["latency"] = {"method": "zero"}
    protocol["logging"]["output_dir"] = str(tmp_path / "output")
    return adapt_reference_config(reference, client["policy"], provider="codex", protocol=protocol)


@pytest.mark.parametrize("task,transport", [(task, False) for task in REFERENCES] + [("ant", True)])
def test_ten_standard_environments_keep_reference_pixels_state_and_actions(tmp_path, monkeypatch, task, transport):
    reference = load_config(ROOT / REFERENCES[task])
    reference["policy"]["checkpoint"] = "checkpoints/not_loaded.pt"
    resolve_env_config(reference)
    env = reference["env"]
    cfg = reference["policy"]
    cfg.pop("latency_prompt_map_path", None)
    cfg.pop("gymnasium_training_task_contract", None)
    cfg["root"] = str(tmp_path)
    if transport:
        cfg["state_source"] = "transport"
    include_state = "state_space" in env or task == "flappy"
    state_dim = len(env["state_space"]["labels"]) if "state_space" in env else 7
    native_count = {"flappy": 2, "deadly_corridor": 7, "demon_attack": 6,
                    "asterix": 9, "atlantis": 4, "air_raid": 6, "inverted_pendulum": 5}
    action_dim = len(env["action_space"]["labels"]) if "action_space" in env else native_count[task]
    model_action_dim = 6 if task == "asterix" else action_dim
    model = {"framework": {"name": "QwenOFT", "action_model": {"state_dim": state_dim, "action_env_dim": model_action_dim}},
             "datasets": {"vla_data": {"include_state": include_state, "image_mode": "single"}},
             "rl_games": {"env_eval": {"image_size": 24}}}
    if environment_integration(reference) == "gymnasium":
        model["datasets"]["vla_data"]["robot_type"] = "rl_games_gymnasium_native"
        cfg["unnorm_key"] = "test_task"
        def statistics(dim):
            return {"min": [-2] * dim, "max": [2] * dim, "mean": [0] * dim,
                    "std": [1] * dim, "q01": [-2] * dim, "q99": [2] * dim}
        (tmp_path / "dataset_statistics.json").write_text(json.dumps({
            "test_task": {"state": statistics(state_dim), "action": statistics(model_action_dim)},
        }))
        # Test delegation at the external training-transform boundary. Its actual
        # normalization is exercised by test_generic_gymnasium_starvla_policy.
        class TrainingTransform:
            unnorm_key = "test_task"

            @classmethod
            def from_statistics(cls, model_cfg, saved_statistics, *, unnorm_key, robot_type):
                assert unnorm_key == "test_task" and robot_type == "rl_games_gymnasium_native"
                assert saved_statistics[unnorm_key]["state"] == statistics(state_dim)
                assert model_cfg["datasets"]["vla_data"]["include_state"] == include_state
                return cls()

            def apply_state(self, state):
                np.testing.assert_array_equal(state, np.ones((1, state_dim)))
                return np.full((1, state_dim), 0.375, dtype=np.float32)

        monkeypatch.setitem(sys.modules, "deployment.model_server.policy_norm_processor",
                            SimpleNamespace(PolicyNormProcessor=TrainingTransform))
    sidecar = tmp_path / "config.full.yaml"
    sidecar.write_text(yaml.safe_dump(model))
    cfg["model_config"] = str(sidecar)
    if environment_integration(reference) == "gymnasium":
        manifest = {"integration_name": "gymnasium", "gymnasium_task": gymnasium_task_contract(env),
                    "active_action_dim": model_action_dim, "uses_state": include_state}
        if include_state:
            manifest.update(state_labels=env["state_space"]["labels"], state_dim=state_dim,
                            state_normalization={"min": [0] * state_dim, "max": [2] * state_dim})
        path = tmp_path / "manifest.json"
        path.write_text(json.dumps(manifest))
        cfg["task_manifest_path"] = str(path)
    config = agent_config(reference, tmp_path)
    prepare_coding_agent_config(config)
    resolver = ActionResolver(config["env"])
    observer, prompt, contract, input_contract = build_reference_observer(config, resolver)
    frames = np.random.default_rng(2).integers(0, 256, (4, 32, 40, 3), dtype=np.uint8)
    observation = Observation({"transport": np.ones(state_dim)} if transport else None, 4, 100,
                              {RGB: frames, "gymnasium_state": np.ones(state_dim),
                                            "latency_ms": 987654321, "numeric_obs": "PRIVATE"})
    example = observer(observation)
    height, width = config["env"]["obs_resize"]
    np.testing.assert_array_equal(example["image"][0], Image.fromarray(frames[-1]).resize((width, height)))
    assert "987654321" not in prompt and set(example) <= {"image", "state"}
    assert (input_contract["state"] is not None) == include_state
    if include_state:
        # The transformed value must reach the agent, for native and history state.
        expected = 0.375 if environment_integration(reference) == "gymnasium" else 0.0
        np.testing.assert_allclose(example["state"], np.full((1, state_dim), expected), atol=1e-6)
        if environment_integration(reference) == "gymnasium":
            assert input_contract["state"]["normalization"] == statistics(state_dim)
    else:
        assert "state" not in example
    if task == "deadly_corridor":
        space, value, noop = spaces.MultiBinary(7), [1, 0, 1, 0, 0, 1, 1], [0] * 7
    elif "action_space" in env:
        spec = gymnasium_action_space_contract(env)
        space = spaces.Box(np.asarray(spec["low"], dtype=np.float32), np.asarray(spec["high"], dtype=np.float32))
        value, noop = [0.25] * action_dim, resolver.noop_action.value
    else:
        space, value, noop = spaces.Discrete(native_count[task]), native_count[task] - 1, resolver.noop_action.value
    bind_coding_agent_action_space(config, SimpleNamespace(gym_env=SimpleNamespace(action_space=space), noop_action=Action(noop)))
    codec = AgentActionCodec(config["policy"]["env_action_space"], resolver, contract)
    output = codec.decode(json.dumps({"action": value}), observation)
    assert output.action.value == value
    assert output.action_chunk is None
    assert space.contains(np.asarray(value, dtype=space.dtype) if isinstance(value, list) else value)
    if task == "flappy":
        assert output.action.is_oneshot
    if task == "asterix":
        assert codec.description()["n"] == 9  # The factorized VLA head has six logits.


@pytest.mark.parametrize("task,horizon,state_dim,action_dim,views", [
    ("mikasa", 1, 7, 7, 2), ("mikasa", 8, 63, 7, 2),
    ("balance", 8, 198, 43, 1), ("balance", 8, 550, 43, 1), ("balance", 16, 902, 43, 1),
])
def test_robot_multiview_state_full_chunks_and_frozen_prefix(tmp_path, task, horizon, state_dim, action_dim, views):
    path = ("configs/examples/mikasa/official_starvla_profile.yaml" if task == "mikasa" else
            "configs/examples/gymnasium/g1_sonic_balance_simple/starvla_profile.yaml")
    reference = load_config(ROOT / path)
    contract = {"task": task, "robot_type": "latency_" + task, "prompt": "Do the task.",
                "state_dim": state_dim, "action_dim": action_dim, "action_horizon": horizon,
                "state_labels": [f"s{i}" for i in range(state_dim)], "video_keys": [f"camera{i}" for i in range(views)],
                "normalization": {"state": {"min": [0] * state_dim, "max": [2] * state_dim},
                                  "action": {"min": [-2] * action_dim, "max": [2] * action_dim}}}
    if task == "balance" and state_dim > 198:
        contract["robot_type"] = f"latency_balance_profile_h{horizon}"
    contract_path = tmp_path / "contract.json"
    contract_path.write_text(json.dumps(contract))
    model_path = tmp_path / "config.full.yaml"
    model_path.write_text(yaml.safe_dump({"framework": {"name": "QwenOFT"}, "datasets": {
        "vla_data": {"include_state": True, "obs_image_size": [24, 32], "image_mode": "single"}}}))
    reference["policy"].update(task_contract_path=str(contract_path), model_config=str(model_path))
    config = agent_config(reference, tmp_path)
    prepare_coding_agent_config(config)
    if task == "balance" and state_dim > 198:
        assert config["env"]["name"] == "g1_sonic_balance_simple"
        assert config["env"]["action_context_horizon"] == horizon
    resolver = ActionResolver(config["env"])
    observer, _, actual_contract, input_contract = build_reference_observer(config, resolver)
    cfg = reference["policy"]
    metadata = {cfg["state_info_key"]: np.full(state_dim, 0.5), cfg["image_views_info_key"]: [
        np.full((8, 8, 3), 40 * i, dtype=np.uint8) for i in range(views)]}
    if horizon > 1:
        metadata.update(action_prefix=np.full((horizon, action_dim), 0.5),
                        action_prefix_mask=np.arange(horizon) < 2)
    observation = Observation(None, 0, 0, metadata)
    example = observer(observation)
    assert len(example["image"]) == views
    assert input_contract["state"]["normalization"] == contract["normalization"]["state"]
    np.testing.assert_array_equal(example["state"], np.full((1, state_dim), -0.5))
    for i, image in enumerate(example["image"]):
        np.testing.assert_array_equal(image, np.full((24, 32, 3), 40 * i))
    # Physical limits deliberately differ from [-1, 1]; no normalized-action assumption.
    space = spaces.Box(-2, 2, (action_dim,), dtype=np.float32)
    bind_coding_agent_action_space(config, SimpleNamespace(gym_env=SimpleNamespace(action_space=space),
                                                          noop_action=Action(np.zeros(action_dim))))
    codec = AgentActionCodec(config["policy"]["env_action_space"], resolver, actual_contract)
    value = [1.5] * action_dim
    payload = {"action": value} if horizon == 1 else {"actions": [value] * horizon}
    output = codec.decode(json.dumps(payload), observation)
    assert output.action_chunk.shape == (horizon, action_dim)
    if horizon > 1:
        np.testing.assert_array_equal(example["action_prefix"], np.full((horizon, action_dim), 0.25))
        np.testing.assert_array_equal(output.action_chunk[:2], np.full((2, action_dim), 0.5))
        np.testing.assert_array_equal(output.action_chunk[2:], np.full((horizon - 2, action_dim), 1.5))
    else:
        np.testing.assert_array_equal(output.action.value, value)
    json.dumps(codec.description())


@pytest.mark.parametrize("kind,bad", [
    ("box", [True, 0]), ("box", ["0", 0]), ("box", [float("nan"), 0]),
    ("box", [float("inf"), 0]), ("box", [1.01, 0]), ("box", [0]),
    ("multibinary", [1.0, 0]), ("multibinary", [True, 0]), ("multibinary", [2, 0]),
    ("multidiscrete", [0, 3]), ("multidiscrete", [0.5, 1]),
])
def test_invalid_vectors_fail_at_external_boundary(kind, bad):
    spec = {"type": kind, "noop": [0, 0], "shape": [2], "nvec": [3, 3], "start": [0, 0],
            "low": [-1, -1], "high": [1, 1], "dtype": "float32"}
    codec = AgentActionCodec(spec, ActionResolver({"noop_action": [0, 0]}), None)
    with pytest.raises(ValueError):
        codec.decode(json.dumps({"action": bad}), Observation(None, 0, 0))


def test_factorized_deadly_tuple_space_and_missing_or_wrong_chunks():
    config = {"policy": {"type": "coding_agent"}, "env": {"name": "deadly_corridor"}}
    space = spaces.Tuple([spaces.Discrete(n) for n in (3, 3, 3, 2)])
    bind_coding_agent_action_space(config, SimpleNamespace(gym_env=SimpleNamespace(action_space=space),
                                                          noop_action=Action((0, 0, 0, 0))))
    codec = AgentActionCodec(config["policy"]["env_action_space"], ActionResolver({"noop_action": [0]*4}), None)
    observation = Observation(None, 0, 0)
    assert codec.decode('{"action": [2, 1, 2, 1]}', observation).action.value == [2, 1, 2, 1]
    assert codec.decode('{"action": [0, 0, 0, 0]}', observation).action.is_noop
    codec.horizon = 8
    for text in ('{}', '{"action": [0,0,0,0]}', '{"actions": []}', '{"actions": "bad"}'):
        with pytest.raises(ValueError):
            codec.decode(text, observation)


def test_composition_keeps_reference_and_requires_explicit_protocol(tmp_path):
    reference = load_config(ROOT / REFERENCES["asterix"])
    before = deepcopy(reference)
    config = agent_config(reference, tmp_path)
    assert reference == before
    assert config["env"] == reference["env"]
    assert config["scheduler"] == reference["scheduler"]
    assert config["policy"]["reference_policy"] == reference["policy"]
    assert "suites" not in config["evaluation"]
    assert config["executor"]["parallel_envs"] == 1
    config["executor"]["parallel_envs"] = 2
    with pytest.raises(ValueError, match="one parallel environment"):
        prepare_coding_agent_config(config)


@pytest.mark.parametrize("mode", ["packed", "multi", "stitch", "kv"])
def test_reference_temporal_windows_and_transport_are_shared(tmp_path, mode):
    reference = load_config(ROOT / REFERENCES["flappy"])
    reference["policy"]["checkpoint"] = "checkpoints/not_loaded.pt"
    reference["env"].update(env_fps=30, obs_fps=15, frame_stack=1, obs_resize=[16, 24])
    model = {"framework": {"name": "QwenOFT", "action_model": {"state_dim": 6},
                           "kv_memory": {"enabled": mode == "kv"}},
             "datasets": {"vla_data": {"include_state": True, "pack_image_sequence": mode == "packed",
                          "image_sequence_length": 3, "image_mode": "stitch" if mode == "stitch" else "multi",
                          "num_obs_frames": 3, "stitch_grid": [1, 3]}}}
    path = tmp_path / "model.yaml"
    path.write_text(yaml.safe_dump(model))
    reference["policy"].update(model_config=str(path), state_source="transport")
    reference["policy"].pop("latency_prompt_map_path", None)
    config = agent_config(reference, tmp_path)
    prepare_coding_agent_config(config)
    assert config["env"]["frame_stack"] == (1 if mode == "kv" else 5)
    observer, _, _, input_contract = build_reference_observer(config, ActionResolver(config["env"]))
    frames = np.stack([np.full((8, 8, 3), i * 20, dtype=np.uint8) for i in range(9)])
    transport = np.arange(6, dtype=np.float32)
    observation = Observation({"transport": transport}, 9, 300, {RGB: frames, "slot_id": 0})
    example = observer(observation)
    np.testing.assert_array_equal(example["state"], transport.reshape(1, 6))
    assert "slot_id" not in example
    assert input_contract["state"]["format"] == "transport"
    selected = frames[-1:] if mode == "kv" else frames[-5::2]
    if mode == "stitch":
        from latency_bench.policy._starvla_input import get_stitch_frames

        np.testing.assert_array_equal(example["image"][0], get_stitch_frames()(list(selected), grid=(1, 3), size=(224, 224)))
    else:
        assert len(example["image"]) == len(selected)
        for actual, frame in zip(example["image"], selected):
            np.testing.assert_array_equal(actual, Image.fromarray(frame).resize((24, 16)))
    del observation.data["transport"]
    with pytest.raises(KeyError, match="transport"):
        observer(observation)


def test_deadly_agent_uses_multibinary_adapter(monkeypatch):
    from latency_bench.envs.adapters import deadly_corridor
    from latency_bench.envs.factory import build_env_adapter

    monkeypatch.setattr(deadly_corridor, "DeadlyCorridorVlaEnvAdapter", lambda **kw: kw)
    config = {"policy": {"type": "coding_agent", "action_layout": "multibinary_7"},
              "env": {"name": "deadly_corridor"}}
    result = build_env_adapter(config, Action([0] * 7))
    assert result["export_env_raw_rgb_frames"] is True
    assert result["noop_action"] is None


@pytest.mark.parametrize("history", [False, True])
def test_first_party_factory_runs_serial_episode_and_history_path(tmp_path, monkeypatch, history):
    from latency_bench.envs.adapters.dummy import DummyEnvAdapter
    from latency_bench.eval.driver import run_from_config
    from latency_bench.executors.env_step_backend import SerialEnvStepBackend
    from latency_bench.policy.codex import CodexPolicyRunner

    reference = load_config(ROOT / REFERENCES["flappy"])
    reference["policy"]["checkpoint"] = "checkpoints/not_loaded.pt"
    reference["env"].update(frame_stack=1, obs_resize=[16, 24])
    if history:
        reference["env"]["action_history_decisions"] = 2
        reference["policy"]["state_source"] = "transport"
    path = tmp_path / "model.yaml"
    path.write_text(yaml.safe_dump({"framework": {"name": "QwenOFT", "action_model": {"state_dim": 6}},
                                   "datasets": {"vla_data": {"include_state": True, "image_mode": "single"}}}))
    reference["policy"]["model_config"] = str(path)
    reference["policy"].pop("latency_prompt_map_path", None)
    config = agent_config(reference, tmp_path)
    config["evaluation"].update(episodes=2, max_env_steps=3)
    config["logging"]["video"]["enabled"] = False

    class PixelEnv(DummyEnvAdapter):
        action_space = spaces.Discrete(2)

        def _make_observation(self, metadata=None):
            observation = super()._make_observation(metadata)
            observation.metadata[RGB] = np.zeros((1, 8, 8, 3), dtype=np.uint8)
            observation.metadata["secret"] = "PRIVATE_SENTINEL"
            return observation

    sessions, prompts = [], []
    def start(self, episode_id):
        self.episode_id, self.decision_id, self.session_id = episode_id, 0, f"episode-{episode_id}"
        self.process = SimpleNamespace(pid=123)
        self.episode_dir = self.output / self.session_id
        self.episode_dir.mkdir()
        self.workspace = tmp_path / self.session_id
        (self.workspace / "observations").mkdir(parents=True)
        sessions.append(self.session_id)

    def request(self, prompt, images, call_dir):
        prompts.append(prompt)
        assert len(images) == 1
        assert "PRIVATE_SENTINEL" not in prompt
        return '{"action": 1}', {"test": True}

    monkeypatch.setattr(CodexPolicyRunner, "_start_episode", start)
    monkeypatch.setattr(CodexPolicyRunner, "_request_action", request)
    monkeypatch.setattr(CodexPolicyRunner, "close", lambda self: setattr(self, "process", None))
    env = PixelEnv(max_steps=3, noop_action=Action(0, is_noop=True))
    kwargs = {"env_backend": SerialEnvStepBackend([env])} if history else {"env": env}
    metrics = run_from_config(config, **kwargs)
    assert len(metrics) == 2 and sessions == ["episode-0", "episode-1"]
    assert env.closed and len(prompts) == 6
    first = json.loads((tmp_path / "output/agent/episode-0/decision_00000/observation.json").read_text())
    np.testing.assert_array_equal(first["state"], np.zeros((1, 6)))
    if history:
        last = json.loads((tmp_path / "output/agent/episode-0/decision_00002/observation.json").read_text())
        assert np.asarray(last["state"]).sum() > 0
