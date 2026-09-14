"""Task inference must use current state and return the full action chunk."""

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
