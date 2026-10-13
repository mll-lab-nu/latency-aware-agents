"""Task inference must use current state and return the full action chunk."""

import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from latency_bench.core.types import Observation
from latency_bench.integrations.starvla.task_contracts import TASK_SPECS
from latency_bench.policy.action_prefix import encode_prefix_state
from latency_bench.policy.starvla_tasks import TaskStarVlaPolicyRunner


@pytest.mark.parametrize("prompt_mode", ["raw", "latency_neutral"])
def test_task_checkpoint_uses_full_input_config_and_latency_prompt(tmp_path, monkeypatch, prompt_mode):
    """Long-lived task-policy contract: explicit input config and prompt modes affect inference."""
    from latency_bench.policy.starvla_tasks import build_task_starvla_policy

    base_prompt = "Intercept the rolling ball and grasp it to stop it."
    latency_prompt = base_prompt + " Current action latency is 2 raw frames (100.00 ms)."
    full_config = tmp_path / "config.full.yaml"
    full_config.write_text(yaml.safe_dump({
        "framework": {"name": "QwenOFT"},
        "datasets": {"vla_data": {"obs_image_size": [160, 192], "image_mode": "single", "include_state": True}},
    }))
    contract_path = tmp_path / "task_contract.json"
    contract_path.write_text(json.dumps({
        "task": "mikasa_intercept_grab_fast", "state_dim": 7, "prompt": base_prompt,
        "normalization": {"state": {"min": [0] * 7, "max": [2] * 7},
                          "action": {"min": [-1] * 7, "max": [1] * 7}},
    }))
    prompt_map = tmp_path / "prompts.json"
    prompt_map.write_text(json.dumps({"2": {"prompt": latency_prompt}}))
    captured = []
    framework = SimpleNamespace(predict_action=lambda *, examples: (
        captured.extend(examples) or {"normalized_actions": np.zeros((len(examples), 1, 7))}
    ))
    framework.to = lambda **kwargs: framework
    framework.eval = lambda: framework

    def unused_partial_config(path):
        raise AssertionError("Explicit input config must take precedence over the accessed-only checkpoint config")

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(bfloat16=object()))
    monkeypatch.setitem(sys.modules, "starVLA.model.framework.base_framework", SimpleNamespace(
        baseframework=SimpleNamespace(from_pretrained=lambda path, **kwargs: framework)))
    monkeypatch.setitem(sys.modules, "starVLA.model.framework.share_tools", SimpleNamespace(
        read_mode_config=unused_partial_config, apply_config_compat=lambda cfg: cfg))
    policy = build_task_starvla_policy({"executor": {"inference_devices": ["cpu"]}, "policy": {
        "source": "local", "root": str(tmp_path), "checkpoint": "steps_5000_pytorch_model.pt", "model_config": full_config.name,
        "backbone_path": str(tmp_path / "backbone"), "task_contract_path": str(contract_path),
        "state_info_key": "state", "image_views_info_key": "views",
        "latency_prompt_map_path": str(prompt_map), "prompt_mode": prompt_mode,
    }})
    output = policy.predict(Observation(None, 0, 0, {
        "state": np.zeros(7), "views": [np.zeros((8, 8, 3), dtype=np.uint8)] * 2, "latency_raw_frames": 2,
    }))
    assert [image.size for image in captured[0]["image"]] == [(192, 160), (192, 160)]
    assert captured[0]["lang"] == (latency_prompt if prompt_mode == "raw" else base_prompt)
    assert output.action_chunk.shape == (1, 7)


@pytest.mark.parametrize('spec', TASK_SPECS.values(), ids=TASK_SPECS)
def test_task_policy_normalizes_current_state_and_decodes_full_command(spec):
    def predict_action(*, examples: list):
        result = np.stack([np.full((spec.action_horizon, spec.action_dim), e['state'][0, 0]) for e in examples])
        return {'normalized_actions': result}

    policy = TaskStarVlaPolicyRunner(
        SimpleNamespace(predict_action=predict_action),
        policy_config={'state_info_key': 'state', 'image_views_info_key': 'views'},
        model_config={'datasets': {'vla_data': {'obs_image_size': [224, 224], 'image_mode': 'single'}}},
        contract={'task': spec.name, 'state_dim': spec.state_dim, 'prompt': 'task',
                  'normalization': {'state': {'min': [0] * spec.state_dim, 'max': [2] * spec.state_dim},
                                    'action': {'min': [-2] * spec.action_dim, 'max': [4] * spec.action_dim}}},
    )
    observations = [Observation(None, 0, 0, {'state': np.full(spec.state_dim, value), 'latency_raw_frames': 2,
                    'views': [np.zeros((8, 8, 3), dtype=np.uint8) for _ in spec.video_keys]})
                    for value in [0, 2]]
    for observation, expected in zip(observations, [-2, 4]):
        example = policy._example(observation)
        assert example['state'].shape == (1, spec.state_dim)
        assert len(example['image']) == len(spec.video_keys)
        assert example['lang'] == 'task'
        output = policy.predict(observation)
        assert output.action_chunk.shape == (spec.action_horizon, spec.action_dim)
        np.testing.assert_array_equal(output.action_chunk, np.full((spec.action_horizon, spec.action_dim), expected))
        np.testing.assert_array_equal(output.action.value, [expected] * spec.action_dim)
    del observations[0].metadata['state']
    with pytest.raises(KeyError, match='state'):
        policy.predict(observations[0])


def test_task_policy_normalizes_and_forwards_action_prefix():
    spec = TASK_SPECS["balance_profile_h16"]
    captured = []

    def predict_action(*, examples: list):
        captured.extend(examples)
        return {
            "normalized_actions": np.zeros(
                (len(examples), spec.action_horizon, spec.action_dim), dtype=np.float32
            )
        }

    policy = TaskStarVlaPolicyRunner(
        SimpleNamespace(predict_action=predict_action),
        policy_config={"state_info_key": "state", "image_views_info_key": "views"},
        model_config={
            "datasets": {
                "vla_data": {"obs_image_size": [224, 224], "image_mode": "single"}
            }
        },
        contract={
            "task": spec.name,
            "state_dim": spec.state_dim,
            "prompt": "task",
            "normalization": {
                "state": {
                    "min": [0] * spec.state_dim,
                    "max": [2] * spec.state_dim,
                },
                "action": {
                    "min": [-2] * spec.action_dim,
                    "max": [4] * spec.action_dim,
                },
            },
        },
    )
    prefix = np.stack([
        np.full(spec.action_dim, -2, dtype=np.float32),
        np.full(spec.action_dim, 4, dtype=np.float32),
        *[
            np.zeros(spec.action_dim, dtype=np.float32)
            for _ in range(spec.action_horizon - 2)
        ],
    ])
    prefix_mask = np.asarray(
        [True, True, *([False] * (spec.action_horizon - 2))]
    )
    observation = Observation(
        None,
        0,
        0,
        {
            "state": np.zeros(spec.state_dim, dtype=np.float32),
            "views": [np.zeros((8, 8, 3), dtype=np.uint8)],
            "action_prefix": prefix,
            "action_prefix_mask": prefix_mask,
        },
    )

    output = policy.predict(observation)

    np.testing.assert_array_equal(captured[0]["action_prefix"][:2, 0], [-1, 1])
    np.testing.assert_array_equal(captured[0]["action_prefix_mask"], prefix_mask)
    np.testing.assert_array_equal(output.action_chunk[:2], prefix[:2])
    np.testing.assert_array_equal(output.action_chunk[2:], 1.0)


@pytest.mark.parametrize("model_source", ["live", "checkpoint"])
@pytest.mark.parametrize("task", ["balance_profile_h8", "mikasa_conditioned_h8"])
def test_task_policy_loads_contract_for_model_source(tmp_path, monkeypatch, model_source, task):
    from latency_bench.policy.starvla import build_live_starvla_policy

    spec = TASK_SPECS[task]
    contract = {
        "task": spec.name,
        "state_dim": spec.state_dim,
        "prompt": "balance",
        "normalization": {
            "state": {"min": [0] * spec.state_dim, "max": [2] * spec.state_dim},
            "action": {"min": [-1] * spec.action_dim, "max": [1] * spec.action_dim},
        },
    }
    contract_path = tmp_path / "task_contract.json"
    if task == "mikasa_conditioned_h8":
        contract["normalization"]["state"]["min"][7:] = [-1] * 56
        contract["normalization"]["state"]["max"][7:] = [1] * 56
    contract_path.write_text(json.dumps(contract))
    framework = SimpleNamespace()

    model_cfg = {"datasets": {"vla_data": {
        "obs_image_size": [224, 224], "image_mode": "single"
    }}}
    config = {"policy": {
        "task_contract_path": str(contract_path),
        "state_info_key": "state",
        "image_views_info_key": "views",
    }}
    if model_source == "live":
        policy = build_live_starvla_policy(
            framework=framework,
            model_cfg=model_cfg,
            config=config,
            action_resolver=None,
        )
    else:
        from latency_bench.policy.starvla_tasks import build_task_starvla_policy

        config["policy"].update(
            source="local", root=str(tmp_path), checkpoint="checkpoint.pt",
            backbone_path=str(tmp_path / "prepared_backbone"),
        )
        config["executor"] = {"inference_devices": ["cpu"]}
        loaded = {}

        def from_pretrained(checkpoint_path, *, backbone_path):
            loaded.update(checkpoint=checkpoint_path, backbone=backbone_path)
            return framework

        framework.to = lambda **kwargs: framework
        framework.eval = lambda: framework
        monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(bfloat16=object()))
        monkeypatch.setitem(
            sys.modules, "starVLA.model.framework.base_framework",
            SimpleNamespace(baseframework=SimpleNamespace(from_pretrained=from_pretrained)),
        )
        monkeypatch.setitem(
            sys.modules, "starVLA.model.framework.share_tools",
            SimpleNamespace(read_mode_config=lambda path: (model_cfg, {}),
                            apply_config_compat=lambda cfg: cfg),
        )
        policy = build_task_starvla_policy(config)
        assert loaded == {
            "checkpoint": str(tmp_path / config["policy"]["checkpoint"]),
            "backbone": config["policy"]["backbone_path"],
        }

    assert isinstance(policy, TaskStarVlaPolicyRunner)
    assert policy.framework is framework
    captured = []
    def predict_action(*, examples):
        captured.extend(examples)
        return {"normalized_actions": np.zeros((1, 8, spec.action_dim))}
    framework.predict_action = predict_action
    prefix = np.full((8, spec.action_dim), 0.75)
    observation = Observation(None, 0, 0, {
        "state": np.zeros(spec.state_dim), "views": [np.zeros((8, 8, 3), dtype=np.uint8)],
        "action_prefix": prefix, "action_prefix_mask": np.arange(8) < 2,
    })
    if task == "mikasa_conditioned_h8":
        observation.metadata["state"] = encode_prefix_state(
            np.zeros(7), prefix, observation.metadata["action_prefix_mask"])
    output = policy.predict(observation)
    assert captured[0]["image"][0].size == (224, 224)
    if task == "mikasa_conditioned_h8":
        assert captured[0]["state"].shape == (1, 63)
        np.testing.assert_array_equal(captured[0]["state"][0, 7:21], 0.75)
        np.testing.assert_array_equal(captured[0]["state"][0, 21:], 0)
    np.testing.assert_array_equal(output.action_chunk[:2], prefix[:2])
    np.testing.assert_array_equal(output.action_chunk[2:], 0.0)
