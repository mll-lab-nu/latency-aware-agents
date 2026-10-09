"""Long-lived random-stream reproducibility contract for runtime seeding."""

import random

import numpy as np
import pytest
import torch

from latency_bench.utils.seeding import seed_everything


@pytest.mark.parametrize("device", [
    "cpu",
    pytest.param("cuda", marks=[
        pytest.mark.gpu,
        pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable"),
    ]),
])
def test_seed_everything_reproduces_python_numpy_and_torch_streams(device):
    def draw(seed):
        assert seed_everything(seed) == seed
        return [random.random(), *np.random.random(3), *torch.rand(3, device=device).cpu().tolist()]

    first = draw(17)
    other = draw(19)
    assert first == draw(17)
    assert first != other
