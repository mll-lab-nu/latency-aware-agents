"""Small inputs for the profiling module's long-lived boundary contracts."""

import pytest


@pytest.fixture
def profile_config(tmp_path):
    return {
        'experiment': {
            'name': 'measurement',
            'seed': 7,
        },
        'env': {
            'name': 'dummy',
            'env_fps': 50,
            'obs_fps': 10,
            'noop_action': 'noop',
        },
        'policy': {
            'type': 'constant',
            'action': 'noop',
        },
        'executor': {
            'mode': 'realtime',
            'inference_devices': ['cpu'],
            'sync_cuda': False,
        },
        'scheduler': {
            'hold_policy': 'hold',
            'ordering_policy': 'latest_ready',
        },
        'evaluation': {
            'episodes': 2,
            'max_env_steps': 10,
        },
        'logging': {
            'output_dir': str(tmp_path / 'output'),
        },
        'profiling': {
            'model_id': 'user_model',
            'workload_id': 'user_task',
            'repeats': 2,
        },
    }


@pytest.fixture
def local_model_config(profile_config, tmp_path):
    root = tmp_path / "model"
    (root / "checkpoints").mkdir(parents=True)
    (root / "checkpoints/model.pt").write_bytes(b"measured weights")
    (root / "config.yaml").write_text("{}\n")
    profile_config["profiling"]["model_artifact"] = {
        "source": "local", "local_root": str(root),
        "checkpoint": "checkpoints/model.pt", "model_config": "config.yaml",
    }
    return profile_config
