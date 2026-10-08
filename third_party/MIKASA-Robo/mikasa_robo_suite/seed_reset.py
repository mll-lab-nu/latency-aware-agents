import torch


def reset_seeded_slot(env, *, slot_id: int, seed: int):
    env_idx = torch.tensor([slot_id], device=env.unwrapped.device)
    return env.reset(seed=seed, options={"env_idx": env_idx})


def reset_seeded_slots(env, seeds: list[int]):
    observation = None
    for slot_id, seed in enumerate(seeds):
        observation, _ = reset_seeded_slot(env, slot_id=slot_id, seed=seed)
    return observation
