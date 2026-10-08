#!/usr/bin/env python
"""Inspect a trained StarVLA VLA: action-head decision + a VLM reasoning trace.

For N sampled frames from a rl_games dataset (default: flappy_200ep), this:
  1. runs the **action head** (the policy's *real* decision), and
  2. generates a free-form **reasoning trace** from the same VLM backbone.

Output is a CSV with one row per sample:
    image_frame_path, true_label, action_from_vlm, reasoning_trace
(plus a few extra diagnostic columns: action_id, action_probs, latency_ms).

IMPORTANT — honesty caveat
    The action comes from the action head on the `🔍` token's hidden state. The
    reasoning text is produced by a *separate* `generate()` pass on the same
    backbone, so it is **post-hoc narration**, not a causal chain-of-thought: the
    model was action-SFT'd (small LM-head loss), not trained to reason. Use it to
    *probe* what the VLM can say, not as a faithful explanation of the action.

Run this on a GPU box with the trained checkpoint and the starVLA env installed.
Example:
    python examples/rl_games/scripts/inspect_reasoning_trace.py \
        --ckpt-path /path/to/openvla_flappy_mixed_checkpoint \
        --dataset-subdir flappy_fix_latency_0_200ep \
        --split val --num-samples 20 \
        --output-csv /workspace/outputs/reasoning_trace_flappy.csv
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image

# action_id -> label, in the order listed in the flappy prompt ("NOOP, FLAP").
ENV_ACTION_LABELS = {
    "flappy": ["NOOP", "FLAP"],
    "demon_attack": ["NOOP", "FIRE", "RIGHT", "LEFT", "RIGHTFIRE", "LEFTFIRE"],
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--ckpt-path",
        default="",
        help="Local trained StarVLA (QwenOFT) checkpoint dir. Optional if --hf-repo-id is given.",
    )
    p.add_argument("--output-csv", required=True, help="Where to write the results CSV.")
    # --- optional: auto-download the checkpoint from HuggingFace ---
    p.add_argument(
        "--hf-repo-id",
        default="",
        help="HF model repo to download the checkpoint from, e.g. talha15032/openvla_bridge_flappy_latency_mixed_exp2.",
    )
    p.add_argument(
        "--hf-include",
        default="",
        help="Glob pattern of files to download from the repo (like `hf download --include`), "
        "e.g. 'steps_5000_state/**'. Default: download the whole repo.",
    )
    p.add_argument("--hf-revision", default=None, help="Optional repo revision/branch/commit.")
    p.add_argument(
        "--download-dir",
        default="",
        help="Where to download the HF checkpoint. Default: <csv dir>/_ckpt/<repo-name>.",
    )
    p.add_argument(
        "--ckpt-subpath",
        default="",
        help="Subdirectory inside the downloaded repo that is the actual checkpoint dir to load "
        "(e.g. 'steps_5000_state'). Default: repo root.",
    )
    p.add_argument("--hf-token", default=None, help="HF token (else uses HF_TOKEN/HUGGINGFACE_HUB_TOKEN env).")
    p.add_argument(
        "--checkpoint-kind",
        default="trained",
        choices=["trained", "bridge"],
        help="'trained': state-dir checkpoint whose config must be reconstructed (your flappy run). "
        "'bridge': a self-contained checkpoint repo that ships config.yaml + dataset_statistics.json + a .pt "
        "(e.g. StarVLA/Qwen3VL-OFT-Bridge-RT-1); its stale base_vlm path is auto-rewritten to the local base VLM.",
    )
    # The synced HF checkpoint repo only contains the model state (steps_N_state/),
    # NOT config.yaml / dataset_statistics.json. read_mode_config needs both at the
    # run dir, so supply them (from the training run dir, e.g.
    # results/Checkpoints/<run_id>/config.yaml). The script arranges the layout
    # read_mode_config expects (run_dir/config.yaml + run_dir/checkpoints/steps_N_state).
    p.add_argument(
        "--config-yaml",
        default="",
        help="Path to the run's config.yaml. If omitted for an HF-downloaded ckpt, it is reconstructed.",
    )
    p.add_argument(
        "--dataset-statistics-json",
        default="",
        help="Path to the run's dataset_statistics.json. If omitted, a stub is written (fine for discrete actions).",
    )
    # --- auto-reconstruction of config.yaml / dataset_statistics.json (when not provided) ---
    # The synced HF repo lacks these; if the original run dir is gone we recompose
    # config.yaml from the same Hydra config the run used and stub the stats.
    p.add_argument("--base-vlm", default="", help="Local path to the base Qwen3-VL model (for from_pretrained).")
    p.add_argument(
        "--base-vlm-repo",
        default="Qwen/Qwen3-VL-4B-Instruct",
        help="HF repo to download the base VLM from when --base-vlm is not given. "
        "Must match the checkpoint's vocab (Qwen/Qwen3-VL-4B-Instruct = 151936; the "
        "-Action variant is 153984 and will mismatch QwenOFT checkpoints).",
    )
    p.add_argument("--base-vlm-dir", default="", help="Where to download the base VLM. Default: <csv dir>/_base_vlm.")
    p.add_argument("--recon-model", default="openvla", help="Hydra model group for config reconstruction.")
    p.add_argument("--recon-init", default="bridge", help="Hydra init group for config reconstruction.")
    p.add_argument("--recon-mode", default="mixed_latency", help="Hydra mode group for config reconstruction.")
    p.add_argument("--recon-config-name", default="train", help="Hydra config name for reconstruction.")
    p.add_argument("--recon-run-id", default="reconstructed_run", help="Dummy run_id for reconstruction.")
    p.add_argument("--recon-workspace-dir", default="/workspace", help="workspace_dir for reconstruction.")
    p.add_argument(
        "--recon-override",
        action="append",
        default=[],
        help="Extra Hydra override(s) for reconstruction (repeatable), e.g. architecture-affecting launch overrides.",
    )
    p.add_argument("--dataset-name", default="latency-sensitive-bench/benchmark-datasets")
    p.add_argument("--split", default="val")
    p.add_argument("--dataset-revision", default="5305590625c28a6922e73f46d18d52738762c906")
    p.add_argument("--env-name", default="flappy", choices=sorted(ENV_ACTION_LABELS))
    p.add_argument(
        "--shuffle",
        action="store_true",
        help="Randomly sample (seeded) instead of taking the first N per class in dataset order. "
        "Default is deterministic first-N so trained vs base runs use identical frames.",
    )
    p.add_argument("--sample-seed", type=int, default=0, help="Seed used only when --shuffle is set.")
    # --- balanced per-latency test set ---
    p.add_argument(
        "--latencies",
        default="0,1,2,3,4",
        help="Comma-separated latencies to sweep; each maps to a dataset subdir via --subdir-template.",
    )
    p.add_argument(
        "--subdir-template",
        default="flappy_200ep__flappy_fix_latency_{latency}_200ep",
        help="Dataset subdir per latency. {latency} is substituted.",
    )
    p.add_argument(
        "--classes",
        default="NOOP,FLAP",
        help="Comma-separated true-label classes to balance (matched against action_text).",
    )
    p.add_argument(
        "--per-class-samples",
        type=int,
        default=20,
        help="Number of samples per class per latency (e.g. 20 -> 20 NOOP + 20 FLAP for each latency).",
    )
    # --- upload results (CSV + frames + metadata) to HuggingFace ---
    p.add_argument("--push-to-hub", action="store_true", help="Upload the CSV + frames + metadata to HF after the run.")
    p.add_argument("--hf-output-repo", default="talha15032/reasoning_trace", help="HF dataset repo to upload results to.")
    p.add_argument(
        "--hf-output-subdir",
        default="",
        help="Subdir in the repo for this run (e.g. flappy_mixed / bridge). Default: output CSV's parent dir name.",
    )
    p.add_argument(
        "--hf-output-private",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Create/keep the output dataset repo private (default). Use --no-hf-output-private for public.",
    )
    p.add_argument("--frames-dir", default="", help="Dir to dump sampled frame PNGs. Default: <csv dir>/frames")
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--device", default="cuda")
    p.add_argument("--cache-dir", default=None)
    # --- live gameplay eval ---
    p.add_argument(
        "--live-eval-episodes",
        type=int,
        default=0,
        help="Number of live flappy episodes to play per latency before the reasoning trace. "
             "0 = skip live eval (default).",
    )
    p.add_argument(
        "--live-eval-latencies",
        default="",
        help="Comma-separated latencies for live eval (e.g. '0,1,2,3,4'). "
             "Defaults to --latencies when not set.",
    )
    p.add_argument(
        "--live-eval-max-steps",
        type=int,
        default=3600,
        help="Max env steps per live eval episode (default 3600).",
    )
    p.add_argument(
        "--action-condition",
        action="store_true",
        help="Deprecated alias for --two-forward-pass.",
    )
    p.add_argument(
        "--two-forward-pass",
        "--two_forward_pass",
        dest="two_forward_pass",
        action="store_true",
        help=(
            "Use two forward passes: first run the action head with framework.predict_action(), "
            "then generate a reasoning trace conditioned on that chosen action."
        ),
    )
    p.add_argument(
        "--reasoning-prompt-template",
        default=(
            "{instruction}\n\nLook carefully at the current game frame. Think step by step about "
            "what is happening, then state the single best next action and why."
        ),
        help="Template for the free (non action-conditioned) reasoning prompt. {instruction} is filled in. "
        "Used only in the single-call path.",
    )
    p.add_argument(
        "--reasoning-suffix",
        default=(
            "Now look at the current game frame and explain step by step what is happening "
            "and what the best next action is and why."
        ),
        help="Free-reasoning request appended after the action token in the single-call (default) path.",
    )
    p.add_argument(
        "--action-condition-template",
        default=(
            "{instruction}\n\nThe chosen next action is: {action}. Looking at the current game frame, "
            "explain step by step why this is a good action."
        ),
        help="Template used in --two-forward-pass mode. {instruction} and {action} are filled in.",
    )
    return p.parse_args()


def _softmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    e = np.exp(x - x.max())
    return e / e.sum()


def ensure_base_vlm(args, default_root: Path) -> str:
    """Return a local base-VLM dir, downloading --base-vlm-repo if needed."""
    if args.base_vlm:
        p = Path(args.base_vlm).expanduser().resolve()
        if not p.exists():
            raise SystemExit(f"--base-vlm path does not exist: {p}")
        return str(p)
    from huggingface_hub import snapshot_download

    name = args.base_vlm_repo.split("/")[-1]
    target = Path(args.base_vlm_dir).expanduser().resolve() if args.base_vlm_dir else default_root / "_base_vlm" / name
    target.mkdir(parents=True, exist_ok=True)
    print(f"Downloading base VLM {args.base_vlm_repo} -> {target} ...")
    local = snapshot_download(repo_id=args.base_vlm_repo, repo_type="model", local_dir=str(target), token=args.hf_token)
    return str(Path(local).resolve())


def reconstruct_run_config(run_dir: Path, base_vlm: str, args) -> None:
    """Recompose config.yaml (same Hydra config the run used) + write stats into run_dir."""
    from omegaconf import OmegaConf

    repo_root = Path(__file__).resolve().parents[3]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from hydra import compose, initialize_config_dir
    from starVLA.training.rl_games import apply_action_spec, apply_model_alias, validate_rl_games_config

    # The training config interpolates a few ${oc.env:...} vars (e.g. WANDB_ENTITY)
    # that aren't needed for inference; set harmless defaults so the config resolves
    # to a fully-concrete file (read_mode_config also resolves at load time).
    for _env_key in ("WANDB_ENTITY", "WANDB_PROJECT", "WANDB_API_KEY", "WANDB_MODE"):
        os.environ.setdefault(_env_key, "reconstructed")

    config_dir = repo_root / "examples" / "rl_games" / "config"
    overrides = [
        f"model={args.recon_model}",
        f"env={args.env_name}",
        f"init={args.recon_init}",
        f"mode={args.recon_mode}",
        f"run_id={args.recon_run_id}",
        f"workspace_dir={args.recon_workspace_dir}",
        f"paths.base_model_dir={base_vlm}",
        f"framework.qwenvl.base_vlm={base_vlm}",
        *args.recon_override,
    ]
    print(f"Reconstructing config.yaml via Hydra compose: model={args.recon_model} env={args.env_name} "
          f"init={args.recon_init} mode={args.recon_mode}")
    with initialize_config_dir(version_base="1.1", config_dir=str(config_dir)):
        cfg = compose(config_name=args.recon_config_name, overrides=overrides)
    try:
        validate_rl_games_config(cfg)
    except Exception as exc:  # noqa: BLE001 - validation may need run-only fields we don't use
        print(f"[warn] validate_rl_games_config raised (continuing): {exc}")
    apply_model_alias(cfg)
    apply_action_spec(cfg)

    config_path = run_dir / "config.yaml"
    try:
        container = OmegaConf.to_container(cfg, resolve=True)
        OmegaConf.save(config=OmegaConf.create(container), f=str(config_path))
    except Exception as exc:  # noqa: BLE001 - fall back to unresolved
        print(f"[warn] could not fully resolve config ({exc}); saving unresolved.")
        OmegaConf.save(config=cfg, f=str(config_path))

    stats_path = run_dir / "dataset_statistics.json"
    if args.dataset_statistics_json:
        shutil.copyfile(Path(args.dataset_statistics_json).expanduser(), stats_path)
    else:
        stats_path.write_text(json.dumps({}), encoding="utf-8")
    print(f"Wrote {config_path} and {stats_path}")


def _patch_base_vlm_in_config(config_yaml: Path, base_vlm: str) -> None:
    """Rewrite framework.qwenvl.base_vlm (and paths.base_model_dir) in a config.yaml.

    Self-contained checkpoints (e.g. the bridge) hard-code the backbone path from the
    machine they were trained on; repoint it at the local base VLM so from_pretrained
    can load the backbone here.
    """
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(str(config_yaml))
    OmegaConf.update(cfg, "framework.qwenvl.base_vlm", str(base_vlm), force_add=True)
    if OmegaConf.select(cfg, "paths.base_model_dir") is not None:
        OmegaConf.update(cfg, "paths.base_model_dir", str(base_vlm), force_add=True)
    OmegaConf.save(config=cfg, f=str(config_yaml))
    print(f"Patched base_vlm -> {base_vlm} in {config_yaml}")


def _find_run_config(ckpt_path: Path) -> Path | None:
    """Find the config.yaml read_mode_config will use, near the checkpoint."""
    for up in (ckpt_path.parents[1], ckpt_path.parents[0]):
        candidate = up / "config.yaml"
        if candidate.exists():
            return candidate
    return None


def _resolve_bridge(args, default_root: Path) -> str:
    """Download a self-contained checkpoint repo and return its weights path.

    Uses the repo's shipped config.yaml + dataset_statistics.json, finds the .pt /
    safetensors weights, and rewrites the stale base_vlm path to the local base VLM.
    """
    from huggingface_hub import snapshot_download

    repo_name = args.hf_repo_id.split("/")[-1]
    download_dir = Path(args.download_dir).expanduser().resolve() if args.download_dir else default_root / "_ckpt" / repo_name
    download_dir.mkdir(parents=True, exist_ok=True)
    # Grab the config, stats and weights (skip the many *.log files in the repo).
    allow_patterns = [args.hf_include] if args.hf_include else [
        "config.yaml",
        "dataset_statistics.json",
        "checkpoints/*.pt",
        "checkpoints/*.safetensors",
        "*.safetensors",
    ]
    print(f"Downloading bridge checkpoint {args.hf_repo_id} -> {download_dir} ...")
    local_root = Path(
        snapshot_download(
            repo_id=args.hf_repo_id,
            repo_type="model",
            local_dir=str(download_dir),
            allow_patterns=allow_patterns,
            revision=args.hf_revision,
            token=args.hf_token,
        )
    ).resolve()

    config_yaml = local_root / "config.yaml"
    if not config_yaml.exists():
        raise SystemExit(f"Bridge repo {args.hf_repo_id} has no config.yaml at root; not a self-contained checkpoint.")
    if not (local_root / "dataset_statistics.json").exists():
        raise SystemExit(f"Bridge repo {args.hf_repo_id} has no dataset_statistics.json at root.")

    # Locate the weights: prefer a .pt/.safetensors under checkpoints/, else a *_state dir.
    weights = None
    ckpt_dir = local_root / "checkpoints"
    if ckpt_dir.is_dir():
        cands = sorted(ckpt_dir.glob("*.pt")) + sorted(ckpt_dir.glob("*.safetensors"))
        if cands:
            weights = cands[0]
    if weights is None:
        for state in sorted(local_root.glob("**/*_state")):
            for n in ("model.safetensors", "pytorch_model.bin"):
                if (state / n).exists():
                    weights = state
                    break
            if weights is not None:
                break
    if weights is None:
        raise SystemExit(f"Could not find weights (.pt/.safetensors/_state) in {local_root}.")

    base_vlm = ensure_base_vlm(args, default_root)
    _patch_base_vlm_in_config(config_yaml, base_vlm)
    print(f"Bridge checkpoint weights: {weights}")
    return str(weights.resolve())


def resolve_ckpt_path(args, default_root: Path) -> str:
    """Return a local checkpoint path/dir, downloading from HuggingFace if requested.

    - --checkpoint-kind bridge: a self-contained repo (config + stats + weights).
    - --checkpoint-kind trained (default): a state-dir checkpoint whose config must be
      reconstructed (your flappy run).
    - local --ckpt-path: used as-is (base_vlm patched if --base-vlm/--base-vlm-repo given).
    """
    if not args.hf_repo_id:
        if not args.ckpt_path:
            raise SystemExit("Provide either --ckpt-path (local) or --hf-repo-id (to download).")
        ckpt = Path(args.ckpt_path).expanduser().resolve()
        # Only patch the local checkpoint's config when an explicit --base-vlm is given
        # (don't touch an already-valid local config just because --base-vlm-repo has a default).
        if args.base_vlm:
            config_yaml = _find_run_config(ckpt)
            if config_yaml is not None:
                _patch_base_vlm_in_config(config_yaml, ensure_base_vlm(args, default_root))
        return str(ckpt)

    if args.checkpoint_kind == "bridge":
        return _resolve_bridge(args, default_root)

    from huggingface_hub import snapshot_download

    repo_name = args.hf_repo_id.split("/")[-1]
    download_dir = Path(args.download_dir).expanduser().resolve() if args.download_dir else default_root / "_ckpt" / repo_name
    download_dir.mkdir(parents=True, exist_ok=True)

    # Target layout read_mode_config expects (run_dir == ckpt_dst.parents[1]):
    #   <loadable>/config.yaml, <loadable>/dataset_statistics.json,
    #   <loadable>/checkpoints/<state_name>/model.safetensors
    # NOTE: we *move* (not symlink) the state dir, because from_pretrained calls
    # Path(...).resolve(); a symlink would resolve back to the flat download path
    # and make run_dir wrong.
    loadable = (download_dir / "_loadable").resolve()
    state_name = Path(args.ckpt_subpath).name if args.ckpt_subpath else "checkpoint"
    ckpt_dst = loadable / "checkpoints" / state_name

    # A real (non-symlink) dir with weights means it's already arranged. A stale
    # symlink from an earlier (buggy) run must be re-done, not reused.
    have_weights = (not ckpt_dst.is_symlink()) and any(
        (ckpt_dst / n).exists() for n in ("model.safetensors", "pytorch_model.bin")
    )
    if have_weights:
        print(f"Reusing already-arranged checkpoint at {ckpt_dst}")
    else:
        allow_patterns = [args.hf_include] if args.hf_include else None
        print(f"Downloading {args.hf_repo_id} (include={allow_patterns or 'ALL'}) -> {download_dir} ...")
        local_root = snapshot_download(
            repo_id=args.hf_repo_id,
            repo_type="model",
            local_dir=str(download_dir),
            allow_patterns=allow_patterns,
            revision=args.hf_revision,
            token=args.hf_token,
        )
        state_src = (Path(local_root) / args.ckpt_subpath).resolve() if args.ckpt_subpath else Path(local_root).resolve()
        if not state_src.exists():
            raise SystemExit(f"Downloaded checkpoint path does not exist: {state_src} (check --ckpt-subpath).")
        if not any((state_src / n).exists() for n in ("model.safetensors", "pytorch_model.bin")):
            raise SystemExit(
                f"{state_src} has no model.safetensors/pytorch_model.bin. "
                "Point --ckpt-subpath at the state dir (e.g. 'steps_5000_state')."
            )
        ckpt_dst.parent.mkdir(parents=True, exist_ok=True)
        if ckpt_dst.is_symlink():
            ckpt_dst.unlink()
        elif ckpt_dst.exists():
            shutil.rmtree(ckpt_dst)
        shutil.move(str(state_src), str(ckpt_dst))
        print(f"Arranged checkpoint at {ckpt_dst}")

    # config.yaml + dataset_statistics.json at the run dir (== loadable).
    if args.config_yaml:
        shutil.copyfile(Path(args.config_yaml).expanduser(), loadable / "config.yaml")
        if args.dataset_statistics_json:
            shutil.copyfile(Path(args.dataset_statistics_json).expanduser(), loadable / "dataset_statistics.json")
        else:
            (loadable / "dataset_statistics.json").write_text(json.dumps({}), encoding="utf-8")
    else:
        print("config.yaml not provided; auto-reconstructing from the launch config ...")
        base_vlm = ensure_base_vlm(args, default_root)
        reconstruct_run_config(loadable, base_vlm, args)

    return str(ckpt_dst)


def load_framework(ckpt_path: str, device: str):
    import torch
    from starVLA.model.framework.base_framework import baseframework

    model = baseframework.from_pretrained(ckpt_path)
    model = model.to(device)
    model.eval()
    return model, torch


def pil_from_row(row) -> Image.Image:
    img = row["image"]
    if isinstance(img, Image.Image):
        return img.convert("RGB")
    if isinstance(img, dict):
        if img.get("bytes") is not None:
            return Image.open(io.BytesIO(img["bytes"])).convert("RGB")
        if img.get("path"):
            return Image.open(img["path"]).convert("RGB")
    return Image.fromarray(np.asarray(img)).convert("RGB")


def decode_action_head(framework, torch, pil: Image.Image, instruction: str, labels: list[str]):
    """Run the real policy decision (action head) and return (label, id, probs)."""
    example = {"image": [pil], "lang": instruction}
    with torch.inference_mode():
        out = framework.predict_action(examples=[example])
    vec = np.asarray(out["normalized_actions"])[0, 0]  # (D,)
    n = len(labels)
    head = np.asarray(vec[:n], dtype=np.float64)
    action_id = int(np.argmax(head))
    label = labels[action_id] if action_id < n else str(action_id)
    probs = _softmax(head)
    return label, action_id, probs


def generate_reasoning(framework, torch, pil: Image.Image, reasoning_prompt: str, max_new_tokens: int) -> str:
    """Free-form text from the VLM backbone (post-hoc narration)."""
    iface = framework.qwen_vl_interface
    pil_for_model = preprocess_image_for_action(framework, pil)
    inputs = iface.build_qwenvl_inputs(images=[[pil_for_model]], instructions=[reasoning_prompt])
    input_len = int(inputs["input_ids"].shape[1])
    with torch.inference_mode():
        generated = iface.generate(**inputs, max_new_tokens=int(max_new_tokens), do_sample=False)
    new_tokens = generated[:, input_len:]
    text = iface.processor.batch_decode(new_tokens, skip_special_tokens=True)[0]
    return " ".join(text.split()).strip()


def preprocess_image_for_action(framework, pil: Image.Image) -> Image.Image:
    """Mirror framework.predict_action image resizing for inspector-only paths."""
    train_obs_image_size = getattr(framework.config.datasets.vla_data, "obs_image_size", None)
    if not train_obs_image_size:
        return pil
    from starVLA.training.trainer_utils.trainer_tools import resize_images

    return resize_images(pil, target_size=train_obs_image_size)


def upload_results_to_hub(*, out_csv: Path, frames_dir: Path, metadata_path: Path, repo_id: str, subdir: str, private: bool, token) -> None:
    """Upload only the results (CSV + frames + metadata) to a HF dataset repo.

    Explicit per-artifact uploads (not the whole output dir) so the large
    downloaded checkpoints / base VLM under the same dir are never pushed.
    """
    from huggingface_hub import HfApi, upload_file, upload_folder

    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=bool(private), exist_ok=True)
    base = subdir.strip("/")
    print(f"Uploading results to dataset repo {repo_id} under '{base}/' ...")
    upload_file(
        path_or_fileobj=str(out_csv),
        path_in_repo=f"{base}/{out_csv.name}",
        repo_id=repo_id,
        repo_type="dataset",
        token=token,
    )
    upload_file(
        path_or_fileobj=str(metadata_path),
        path_in_repo=f"{base}/metadata.json",
        repo_id=repo_id,
        repo_type="dataset",
        token=token,
    )
    if frames_dir.exists():
        upload_folder(
            folder_path=str(frames_dir),
            path_in_repo=f"{base}/frames",
            repo_id=repo_id,
            repo_type="dataset",
            token=token,
        )
    print(f"Uploaded -> https://huggingface.co/datasets/{repo_id}/tree/main/{base}")


def sample_balanced_rows(ds, classes, per_class: int, *, shuffle: bool = False, seed: int = 0):
    """Return a list of (true_class, row) balanced across `classes`.

    Deterministic by default: takes the first `per_class` rows of each class in
    dataset order, so two runs (e.g. trained vs base model) hit the *same* frames
    with no seed needed. Pass shuffle=True for a seeded random sample instead.
    """
    if shuffle:
        ds = ds.shuffle(seed=int(seed))
    out = []
    for cls in classes:
        cls_ds = ds.filter(lambda r, c=cls: str(r.get("action_text", "")) == c)
        take = min(int(per_class), len(cls_ds))
        if take < int(per_class):
            print(f"  [warn] only {len(cls_ds)} '{cls}' rows available (< {per_class}); taking {take}.")
        for i in range(take):
            out.append((cls, cls_ds[i]))
    return out


def predict_action_and_reason_single_call(framework, torch, pil, instruction, reasoning_suffix, max_new_tokens, labels):
    """One generate() call: action head (from the prefill) + free reasoning text.

    Builds the exact training action prompt (so the `🔍` token's hidden state is
    faithful), appends a free-reasoning request, and runs a single generate() with
    output_hidden_states. The action head reads `🔍`'s prefill hidden state (the
    real policy decision); the same call produces the reasoning text.
    """
    iface = framework.qwen_vl_interface
    chunk_len = int(getattr(framework, "chunk_len", 1))
    action_token = getattr(framework, "action_token", "\U0001f50d")
    action_suffix = (
        f" Please predict the next {chunk_len} robot actions: "
        f"<action>{action_token * chunk_len}<action>."
    )
    full_instruction = f"{instruction}{action_suffix} {reasoning_suffix}"
    pil_for_model = preprocess_image_for_action(framework, pil)

    inputs = iface.build_qwenvl_inputs(images=[[pil_for_model]], instructions=[full_instruction])
    input_len = int(inputs["input_ids"].shape[1])
    # Run in bf16 (same precision as the deployed predict_action) so the action-head
    # decision matches the real model output, not the interface's fp16 generate().
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        out = iface.model.generate(
            **inputs,
            max_new_tokens=int(max_new_tokens),
            do_sample=False,
            output_hidden_states=True,
            return_dict_in_generate=True,
        )

    # Action head from the prefill step's last-layer hidden state at the 🔍 token.
    prefill_last_hidden = out.hidden_states[0][-1]  # (B, prompt_len, H)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float32):
        action_queries = framework._gather_action_token_embeddings(
            prefill_last_hidden, inputs["input_ids"], action_token_id=framework.action_token_id
        )
        pred_actions = framework.action_model.predict_action(action_queries)
    head = np.asarray(pred_actions[0, 0].float().cpu().numpy(), dtype=np.float64)[: len(labels)]
    action_id = int(np.argmax(head))
    action_label = labels[action_id] if action_id < len(labels) else str(action_id)
    probs = _softmax(head)

    new_tokens = out.sequences[:, input_len:]
    reasoning = iface.processor.batch_decode(new_tokens, skip_special_tokens=True)[0]
    reasoning = " ".join(reasoning.split()).strip()
    return action_label, action_id, probs, reasoning


def run_live_eval(framework, torch, args, labels: list[str], latencies: list[int]) -> dict:
    """Play args.live_eval_episodes flappy episodes per latency using seeds 0..N-1.

    Seeds are fixed and identical across models so results are directly comparable.
    Returns {latency: {mean_reward, std_reward, episode_rewards}}.
    """
    import collections

    try:
        import flappy_bird_gymnasium  # noqa: F401 — registers FlappyBird-v0
        import gymnasium as gym
    except ImportError:
        print("[live eval] flappy_bird_gymnasium not installed; skipping live eval.")
        return {}

    from datasets import load_dataset

    results: dict = {}
    for latency in latencies:
        subdir = args.subdir_template.format(latency=latency)
        ds = load_dataset(
            args.dataset_name,
            revision=args.dataset_revision,
            name=subdir,
            split=args.split,
            cache_dir=args.cache_dir,
            verification_mode="no_checks",
        )
        prompt = str(ds[0]["prompt"]) if len(ds) > 0 else ""

        episode_rewards: list[float] = []
        for ep_idx in range(args.live_eval_episodes):
            env = gym.make("FlappyBird-v0", render_mode="rgb_array", use_lidar=False)
            env.reset(seed=ep_idx)

            # Pre-fill queue with latency NOOPs so action from step 0 lands at step L.
            action_buffer: collections.deque[int] = collections.deque([0] * latency)
            total_reward = 0.0
            done = False
            step = 0

            while not done and step < args.live_eval_max_steps:
                frame = env.render()
                pil = Image.fromarray(np.asarray(frame, dtype=np.uint8))
                with torch.inference_mode():
                    out = framework.predict_action(examples=[{"image": [pil], "lang": prompt}])
                vec = np.asarray(out["normalized_actions"])[0, 0]
                new_action = int(np.argmax(vec[: len(labels)]))

                if latency == 0:
                    effective_action = new_action
                else:
                    effective_action = action_buffer.popleft()
                    action_buffer.append(new_action)

                _, reward, terminated, truncated, _ = env.step(effective_action)
                total_reward += float(reward)
                done = terminated or truncated
                step += 1

            env.close()
            episode_rewards.append(total_reward)
            print(f"  live ep {ep_idx + 1}/{args.live_eval_episodes} "
                  f"latency={latency} reward={total_reward:.1f} steps={step}")

        mean_r = float(np.mean(episode_rewards))
        std_r = float(np.std(episode_rewards))
        results[latency] = {"mean_reward": mean_r, "std_reward": std_r, "episode_rewards": episode_rewards}
        print(f"  >> latency={latency}: mean={mean_r:.2f} ± {std_r:.2f}  (n={args.live_eval_episodes})")

    return results


def main() -> None:
    args = parse_args()
    from datasets import load_dataset

    labels = ENV_ACTION_LABELS[args.env_name]
    two_forward_pass = bool(args.two_forward_pass or args.action_condition)
    reasoning_mode = "two_forward_pass" if two_forward_pass else "single_call"
    latencies = [int(x) for x in str(args.latencies).split(",") if str(x).strip() != ""]
    classes = [c.strip() for c in str(args.classes).split(",") if c.strip()]

    out_csv = Path(args.output_csv).expanduser().resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    frames_dir = Path(args.frames_dir).expanduser().resolve() if args.frames_dir else out_csv.parent / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = resolve_ckpt_path(args, default_root=out_csv.parent)
    print(f"Loading model from {ckpt_path} ...")
    framework, torch = load_framework(ckpt_path, args.device)

    # ── live gameplay eval ────────────────────────────────────────────────────
    live_latencies = (
        [int(x) for x in str(args.live_eval_latencies).split(",") if x.strip()]
        if args.live_eval_latencies
        else latencies
    )
    live_eval_results: dict = {}
    if args.live_eval_episodes > 0:
        print(f"\n=== Live gameplay eval ({args.live_eval_episodes} ep × {live_latencies} latencies) ===")
        live_eval_results = run_live_eval(framework, torch, args, labels, live_latencies)
        # Save standalone results file immediately so it's available even if reasoning crashes.
        live_results_path = out_csv.parent / "live_eval_results.json"
        import json as _json
        live_results_path.write_text(
            _json.dumps({str(lat): v for lat, v in live_eval_results.items()}, indent=2),
            encoding="utf-8",
        )
        print(f"Live eval results -> {live_results_path}\n")

    # ── reasoning trace ───────────────────────────────────────────────────────
    rows_out = []
    for latency in latencies:
        subdir = args.subdir_template.format(latency=latency)
        print(f"\n=== latency {latency}  [{subdir}] split={args.split} ===")
        ds = load_dataset(
            args.dataset_name,
            revision=args.dataset_revision,
            name=subdir,
            split=args.split,
            cache_dir=args.cache_dir,
            verification_mode="no_checks",
        )
        samples = sample_balanced_rows(
            ds, classes, args.per_class_samples, shuffle=args.shuffle, seed=args.sample_seed
        )
        lat_frames_dir = frames_dir / f"latency_{latency}"
        lat_frames_dir.mkdir(parents=True, exist_ok=True)

        for j, (true_label, row) in enumerate(samples):
            pil = pil_from_row(row)
            instruction = str(row.get("prompt", ""))

            frame_path = lat_frames_dir / f"{true_label}_{j:04d}.png"
            pil.save(frame_path)

            if two_forward_pass:
                # Two-step: decide first (action head), then explain that specific action.
                action_label, action_id, probs = decode_action_head(framework, torch, pil, instruction, labels)
                reasoning_prompt = args.action_condition_template.format(instruction=instruction, action=action_label)
                reasoning = generate_reasoning(framework, torch, pil, reasoning_prompt, args.max_new_tokens)
            else:
                # Single call: action head (from the prefill) + free reasoning text together.
                action_label, action_id, probs, reasoning = predict_action_and_reason_single_call(
                    framework, torch, pil, instruction, args.reasoning_suffix, args.max_new_tokens, labels
                )

            rows_out.append(
                {
                    "latency": latency,
                    "image_frame_path": str(frame_path),
                    "true_label": true_label,
                    "action_from_vlm": action_label,
                    "reasoning_trace": reasoning,
                    "action_id": action_id,
                    "action_probs": ",".join(f"{labels[k]}={probs[k]:.3f}" for k in range(len(labels))),
                    "latency_ms": row.get("latency_ms", ""),
                }
            )
            print(f"  lat={latency} [{j + 1}/{len(samples)}] true={true_label} pred={action_label} "
                  f"reasoning[:80]={reasoning[:80]!r}")

    fieldnames = [
        "latency",
        "image_frame_path",
        "true_label",
        "action_from_vlm",
        "reasoning_trace",
        "action_id",
        "action_probs",
        "latency_ms",
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)

    # Self-describing metadata for tracking.
    metadata = {
        "checkpoint_kind": args.checkpoint_kind,
        "hf_repo_id": args.hf_repo_id,
        "ckpt_path": args.ckpt_path,
        "base_vlm_repo": args.base_vlm_repo,
        "base_vlm": args.base_vlm,
        "dataset_name": args.dataset_name,
        "dataset_revision": args.dataset_revision,
        "split": args.split,
        "env_name": args.env_name,
        "latencies": latencies,
        "classes": classes,
        "per_class_samples": int(args.per_class_samples),
        "shuffle": bool(args.shuffle),
        "sample_seed": int(args.sample_seed),
        "reasoning_mode": reasoning_mode,
        "two_forward_pass": bool(two_forward_pass),
        "action_condition": bool(args.action_condition),
        "max_new_tokens": int(args.max_new_tokens),
        "num_rows": len(rows_out),
        "live_eval_episodes": args.live_eval_episodes,
        "live_eval_latencies": live_latencies,
        "live_eval_results": {
            str(lat): {k: v for k, v in res.items() if k != "episode_rewards"}
            for lat, res in live_eval_results.items()
        },
    }
    metadata_path = out_csv.parent / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"\nWrote {len(rows_out)} rows -> {out_csv}")
    print(f"Frames -> {frames_dir}")
    print(f"Metadata -> {metadata_path}")
    # Per-latency action-head agreement with dataset labels.
    print("Action-head agreement with dataset label:")
    for latency in latencies:
        lat_rows = [r for r in rows_out if r["latency"] == latency]
        if lat_rows:
            acc = np.mean([r["true_label"] == r["action_from_vlm"] for r in lat_rows])
            print(f"  latency {latency}: {acc:.1%}  (n={len(lat_rows)})")
    if rows_out:
        overall = np.mean([r["true_label"] == r["action_from_vlm"] for r in rows_out])
        print(f"  overall: {overall:.1%}  (n={len(rows_out)})")

    if args.push_to_hub:
        upload_results_to_hub(
            out_csv=out_csv,
            frames_dir=frames_dir,
            metadata_path=metadata_path,
            repo_id=args.hf_output_repo,
            subdir=args.hf_output_subdir or out_csv.parent.name,
            private=args.hf_output_private,
            token=args.hf_token,
        )


if __name__ == "__main__":
    main()
