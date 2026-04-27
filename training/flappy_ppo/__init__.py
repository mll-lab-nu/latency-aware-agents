from __future__ import annotations


__all__ = ["train"]


def __getattr__(name: str):
    if name == "train":
        from training.flappy_ppo.ppo_loop import train

        return train
    raise AttributeError(name)
