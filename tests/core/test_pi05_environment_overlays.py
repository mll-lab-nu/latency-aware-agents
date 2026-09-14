from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import pytest

from latency_bench.core.config import load_config

ROOT = Path(__file__).resolve().parents[2]
CONFIGS = ROOT / 'configs/examples/gymnasium'
SOURCES = CONFIGS / 'pi05_sources'
sys.path.insert(0, str(ROOT / 'third_party/starVLA'))
from examples.rl_games.scripts import launch_train


@pytest.mark.parametrize('env,state_dim,action_dim', [
    ('air_raid', 0, 6), ('inverted_pendulum', 4, 5), ('hopper', 11, 3),
    ('half_cheetah', 17, 6), ('walker2d', 17, 6), ('ant', 27, 8), ('humanoid', 376, 17),
])
def test_pinned_pi05_environment_overlay(env, state_dim, action_dim):
    path = CONFIGS / env / 'pi05_train.yaml'
    overlay = load_config(path)
    manifest = json.loads((path.parent / overlay['source_manifest']).read_text())
    source_map = json.loads((SOURCES / 'source_map.json').read_text())
    for source in source_map['files'][env]:
        data = (SOURCES / source['path']).read_bytes()
        assert hashlib.sha256(data).hexdigest() == source['sha256']
        assert len(data) == source['bytes']
    assert manifest['active_action_dim'] == action_dim
    assert manifest['uses_state'] == (state_dim != 0)
    if state_dim:
        assert manifest['state_dim'] == state_dim
        assert len(manifest['state_labels']) == state_dim
    profile = load_config(SOURCES / env / 'profile_latency/profiling/sessions/0/generated.yaml')
    assert overlay['env'] == profile['env']
    assert overlay['source_clock']['obs_stride_raw_frames'] == manifest['obs_stride_raw_frames']
    assert overlay['env']['env_fps'] == (50 if env == 'air_raid' else 10)
    assert overlay['env']['obs_fps'] == (12.5 if env == 'air_raid' else 10)
    assert all(value is None for value in overlay['deferred_bindings'].values())
    cfg = launch_train.compose_training_config(
        config_name='train', model='pi05', env='gymnasium', init='scratch', mode='single',
        overrides=[
            'rl_games.gymnasium.task_contract=' + launch_train._hydra_value(manifest['gymnasium_task']),
            'rl_games.env_eval.enabled=false',
            'rl_games.env_eval.mid_train.enabled=false',
            'rl_games.env_eval.post_train.enabled=false',
            *overlay['hydra_overrides'],
        ],
    )
    assert cfg.framework.name == 'QwenPI_v3'
    assert cfg.base_model.repo_id == 'Qwen/Qwen3-VL-4B-Instruct'
    action = cfg.framework.action_model
    assert (action.state_dim, action.action_dim, action.action_env_dim) == (state_dim, action_dim, action_dim)
    assert action.state_encoding == 'continuous_projector'
    assert action.action_horizon == 1
    assert action.future_action_window_size == action.past_action_window_size == 0
    assert action.num_inference_timesteps == 4
    assert cfg.datasets.vla_data.include_state == (state_dim != 0)
    assert cfg.initialization.checkpoint_hf_repo_id is None
    assert cfg.initialization.checkpoint_local_dir is None
    assert cfg.trainer.pretrained_checkpoint is None
    assert cfg.trainer.is_resume is False
