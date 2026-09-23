"""Task inference must use current state and return the full action chunk."""

import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from latency_bench.core.types import Observation
from latency_bench.data.starvla_tasks import TASK_SPECS
from latency_bench.policy.starvla_tasks import TaskStarVlaPolicyRunner


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
    observations = [Observation(None, 0, 0, {'state': np.full(spec.state_dim, value),
                    'views': [np.zeros((8, 8, 3), dtype=np.uint8) for _ in spec.video_keys]})
                    for value in [0, 2]]
    for observation, expected in zip(observations, [-2, 4]):
        example = policy._example(observation)
        assert example['state'].shape == (1, spec.state_dim)
        assert len(example['image']) == len(spec.video_keys)
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

    policy.predict(observation)

    np.testing.assert_array_equal(captured[0]["action_prefix"][:2, 0], [-1, 1])
    np.testing.assert_array_equal(captured[0]["action_prefix_mask"], prefix_mask)


@pytest.mark.parametrize("model_source", ["live", "checkpoint"])
def test_task_policy_loads_contract_for_model_source(tmp_path, monkeypatch, model_source):
    from latency_bench.policy.starvla import build_live_starvla_policy

    spec = TASK_SPECS["balance_profile_h8"]
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
            checkpoint_path=str(tmp_path / "checkpoint.pt"),
            backbone_path=str(tmp_path / "prepared_backbone"),
            device="cpu",
        )
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
            SimpleNamespace(read_mode_config=lambda path: (model_cfg, {})),
        )
        policy = build_task_starvla_policy(config)
        assert loaded == {
            "checkpoint": config["policy"]["checkpoint_path"],
            "backbone": config["policy"]["backbone_path"],
        }

    assert isinstance(policy, TaskStarVlaPolicyRunner)
    assert policy.framework is framework
