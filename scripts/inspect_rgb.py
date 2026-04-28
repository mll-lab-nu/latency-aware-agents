#!/usr/bin/env python3
import argparse
import pickle
from pathlib import Path

import numpy as np
from PIL import Image


def load_first_obs(pkl_path):
    with open(pkl_path, "rb") as f:
        obj = pickle.load(f)

    if isinstance(obj, list) and len(obj) > 0:
        first = obj[0]

        if isinstance(first, dict):
            if "base_observations" in first and len(first["base_observations"]) > 0:
                return np.asarray(first["base_observations"][0])
            if "observations" in first and len(first["observations"]) > 0:
                return np.asarray(first["observations"][0])

        if isinstance(first, list) and len(first) > 0 and isinstance(first[0], dict):
            step0 = first[0]
            for key in ["obs", "observation", "state"]:
                if key in step0:
                    return np.asarray(step0[key])

    raise ValueError("Could not find first observation in pickle.")


def to_uint8(arr):
    arr = np.asarray(arr)
    if arr.dtype == np.uint8:
        return arr
    arr = arr.astype(np.float32)
    if arr.min() >= 0 and arr.max() <= 1:
        arr = arr * 255.0
    return np.clip(arr, 0, 255).astype(np.uint8)


def is_true_rgb(img_hwc):
    # img_hwc is H x W x 3
    r = img_hwc[:, :, 0]
    g = img_hwc[:, :, 1]
    b = img_hwc[:, :, 2]
    return not (np.array_equal(r, g) and np.array_equal(g, b))


def save_rgb(img_hwc, path):
    Image.fromarray(img_hwc).save(path)
    print(f"Saved: {path}")


def save_gray(img_2d, path):
    Image.fromarray(img_2d).save(path)
    print(f"Saved: {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pkl", required=True, help="Path to rollout pickle")
    parser.add_argument("--out-dir", default="inspect_frames", help="Output directory")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    obs = load_first_obs(args.pkl)
    obs = to_uint8(obs)

    print("shape:", obs.shape)
    print("dtype:", obs.dtype)
    print("min:", obs.min(), "max:", obs.max())

    # Case 1: CHW
    if obs.ndim == 3 and obs.shape[1] == 84 and obs.shape[2] == 84:
        c, h, w = obs.shape
        print("Detected CHW format")

        if c == 3:
            img = np.transpose(obs, (1, 2, 0))
            save_rgb(img, out_dir / "frame_0.png")
            print("frame_0 is RGB:", is_true_rgb(img))
            return

        if c == 12:
            print("Interpreting as 4 RGB frames stacked in CHW")
            frames = obs.reshape(4, 3, h, w)
            for i in range(4):
                img = np.transpose(frames[i], (1, 2, 0))
                save_rgb(img, out_dir / f"frame_{i}.png")
                print(f"frame_{i} is RGB:", is_true_rgb(img))
            return

        if c == 4:
            print("Interpreting as 4 grayscale frames")
            for i in range(4):
                save_gray(obs[i], out_dir / f"frame_{i}.png")
            return

    # Case 2: HWC
    if obs.ndim == 3 and obs.shape[0] == 84 and obs.shape[1] == 84:
        h, w, c = obs.shape
        print("Detected HWC format")

        if c == 3:
            save_rgb(obs, out_dir / "frame_0.png")
            print("frame_0 is RGB:", is_true_rgb(obs))
            return

        if c == 12:
            print("Interpreting as 4 RGB frames stacked in HWC")
            frames = obs.reshape(h, w, 4, 3)
            for i in range(4):
                img = frames[:, :, i, :]
                save_rgb(img, out_dir / f"frame_{i}.png")
                print(f"frame_{i} is RGB:", is_true_rgb(img))
            return

        if c == 4:
            print("Interpreting as 4 grayscale frames")
            for i in range(4):
                save_gray(obs[:, :, i], out_dir / f"frame_{i}.png")
            return

    raise ValueError(f"Unsupported shape: {obs.shape}")


if __name__ == "__main__":
    main()