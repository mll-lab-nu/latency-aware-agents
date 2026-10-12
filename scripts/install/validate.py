"""Run the runtime checks used by the Bash installation entrypoints."""

import argparse
import os
import sys


def check_torch() -> None:
    import torch
    import torchaudio
    import torchvision
    import triton

    assert sys.version_info[:2] == (3, 10), sys.version
    assert torch.__version__ == "2.11.0+cu128", torch.__version__
    assert torchvision.__version__ == "0.26.0+cu128", torchvision.__version__
    assert torchaudio.__version__ == "2.11.0+cu128", torchaudio.__version__
    assert triton.__version__ == "3.6.0", triton.__version__
    assert torch.version.cuda == "12.8", torch.version.cuda
    torch.ones(1, device="cuda").add_(1)
    torch.cuda.synchronize()
    print(f"torch={torch.__version__} cuda={torch.version.cuda} gpu={torch.cuda.get_device_name(0)}")


def check_env(name: str) -> None:
    import gymnasium as gym

    if name == "flappy":
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        import flappy_bird_gymnasium

        env = gym.make("FlappyBird-v0", render_mode="rgb_array", use_lidar=False)
    elif name == "atari":
        import ale_py

        env = gym.make("ALE/DemonAttack-v5", render_mode="rgb_array")
    else:
        import vizdoom.gymnasium_wrapper

        env = gym.make("VizdoomDeadlyCorridor-MultiBinary-v1", render_mode="rgb_array", frame_skip=4)
    try:
        env.reset(seed=42)
        env.step(env.action_space.sample())
        frame = env.render()
        assert frame is not None, name
    finally:
        env.close()
    print(f"ok-{name}-env")


def check_flash_attention() -> None:
    import torch
    from flash_attn import __version__, flash_attn_func

    inputs = [torch.randn(2, 128, 8, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True) for _ in range(3)]
    output = flash_attn_func(*inputs, causal=True)
    output.float().square().mean().backward()
    torch.cuda.synchronize()
    assert torch.isfinite(output).all()
    for tensor in inputs:
        assert torch.isfinite(tensor.grad).all()
    print(f"[flash_attn/check] OK {__version__}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", choices=("torch", "base", "starvla", "flappy", "atari", "deadly_corridor", "flash-attn", "flash-attn-path", "python"))
    parser.add_argument("--tier", choices=("use", "dev"), default="use")
    args = parser.parse_args()
    if args.target == "python":
        assert sys.version_info[:2] == (3, 10), sys.version
    elif args.target == "torch":
        check_torch()
    elif args.target == "base":
        check_torch()
        import datasets
        import pyarrow
        import sample_factory
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        from training.common.sf_compat import validate_sample_factory_version

        validate_sample_factory_version(sample_factory, compat_label="installation")
        print(f"ok-base datasets={datasets.__version__} pyarrow={pyarrow.__version__}")
    elif args.target == "starvla":
        check_torch()
        import omegaconf
        import starVLA
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        if args.tier == "dev":
            import hydra
        print("ok-starVLA")
    elif args.target == "flash-attn":
        check_flash_attention()
    elif args.target == "flash-attn-path":
        import torch
        import flash_attn_2_cuda

        print(flash_attn_2_cuda.__file__)
    else:
        check_env(args.target)


if __name__ == "__main__":
    main()
