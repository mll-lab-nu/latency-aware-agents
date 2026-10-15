#!/usr/bin/env python3
"""Provenance of bodies/unitree_h1/policy/motion.npz: the weights of Unitree's walking policy (unitree_rl_gym
deploy/pre_train/h1_2/motion.pt (the H1-2 walking policy), BSD-3-Clause; a TorchScript LSTM actor) as plain arrays, so that the simulator runs it
with numpy (body.py WalkPolicy) and needs no torch. The network: LSTM(47 -> 64, one layer), then Linear(64 -> 32),
ELU, Linear(32 -> 12); the gates in PyTorch's order (input, forget, cell, output).

    python -I tools/export_policy.py PATH/TO/motion.pt          (needs torch; not in the image)
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
KEYS = {"w_ih": "memory.weight_ih_l0", "w_hh": "memory.weight_hh_l0", "b_ih": "memory.bias_ih_l0",
        "b_hh": "memory.bias_hh_l0", "w0": "actor.0.weight", "b0": "actor.0.bias", "w2": "actor.2.weight",
        "b2": "actor.2.bias"}


def main(src: str):
    import torch
    net = torch.jit.load(src)
    sd = net.state_dict()
    arrays = {k: sd[v].numpy().astype(np.float32) for k, v in KEYS.items()}
    sha = hashlib.sha256(Path(src).read_bytes()).hexdigest()
    np.savez(HERE / "policy" / "motion.npz", source_sha256=np.array(sha), **arrays)
    print("wrote policy/motion.npz from", src, sha)


if __name__ == "__main__":
    main(sys.argv[1])
