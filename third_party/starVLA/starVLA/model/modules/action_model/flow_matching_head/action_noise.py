"""Sample action trajectories with optional observation-specific random streams."""
import torch


def sample_action_noise(shape, *, dtype, device, seeds):
    if seeds is None:
        return torch.randn(shape, dtype=dtype, device=device)
    return torch.stack([
        torch.randn(shape[1:], dtype=dtype, device=device,
                    generator=torch.Generator(device=device).manual_seed(seed))
        for seed in seeds
    ])
