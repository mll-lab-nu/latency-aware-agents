"""Publish MIKASA artifacts and record their repository revisions."""

import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import tempfile

from huggingface_hub import HfApi
import yaml

from latency_bench.utils.hf import resolve_token
from scripts.tasks.mikasa_intercept_grab_fast.fixed_latency import QWEN_BACKBONE_REVISION


MODEL_REPO = "MLL-Lab/LAGEN-models"
DATA_REPO = "MLL-Lab/LAGEN-datasets"


def publish_folder(source: Path, prefix: str, *, repo_id: str, repo_type: str) -> str:
    """Upload an artifact directory and return its publication revision."""
    api = HfApi(token=resolve_token(""))
    commit = api.upload_folder(repo_id=repo_id, repo_type=repo_type, folder_path=source,
                               path_in_repo=prefix, ignore_patterns=["**/.cache/**", "**/publication.json", "publication.json"],
                               commit_message=f"Publish {prefix}")
    print(f"HF_PUBLISHED={repo_id}/{prefix}@{commit.oid}", flush=True)
    return commit.oid


def publish_training_endpoint(directory: Path, model_prefix: str) -> None:
    """Publish the completed phase's inference weight and training provenance."""
    cfg = yaml.safe_load((directory / "config.full.yaml").read_text())
    step = cfg['trainer']['max_train_steps']
    model_path = directory / 'checkpoints' / f'steps_{step}_pytorch_model.pt'
    prefix = f"{model_prefix}/training/{cfg['run_id']}/step_{step:05d}"
    receipt = directory / f"publication_step_{step}.json"
    if receipt.is_file():
        publication = json.loads(receipt.read_text())
        if (publication["repo_id"], publication["path"], publication["step"]) != (MODEL_REPO, prefix, step):
            raise RuntimeError("Completed publication belongs to a different training endpoint")
        print(f"HF_ALREADY_PUBLISHED={MODEL_REPO}/{prefix}@{publication['revision']}", flush=True)
        return
    with tempfile.TemporaryDirectory(prefix="hf-checkpoint-", dir=directory) as staging:
        stage = Path(staging)
        (stage / "checkpoints").mkdir()
        os.link(model_path, stage / "checkpoints/model.pt")
        inference = {key: copy.deepcopy(cfg[key]) for key in ("framework", "version_id")}
        inference["framework"]["qwenvl"]["base_vlm"] = "Qwen/Qwen3-VL-4B-Instruct"
        inference["backbone"] = {"repo_id": "Qwen/Qwen3-VL-4B-Instruct",
                                 "revision": QWEN_BACKBONE_REVISION}
        inference["datasets"] = {"vla_data": {key: cfg["datasets"]["vla_data"][key] for key in (
            "include_state", "obs_image_size", "image_mode", "stitch_grid", "num_obs_frames")}}
        inference["trainer"] = {"pretrained_checkpoint": None}
        (stage / "config.yaml").write_text(yaml.safe_dump(inference, sort_keys=False))
        shutil.copyfile(directory / "config.full.yaml", stage / "trainer_config.yaml")
        shutil.copyfile(directory / "dataset_statistics.json", stage / "dataset_statistics.json")
        shutil.copyfile(cfg["datasets"]["vla_data"]["task_contract_path"], stage / "task_contract.json")
        (stage / "provenance.json").write_text(json.dumps({
            "step": step, "phase": cfg["run_id"],
            "condition": cfg["training_latency_condition"], "status": "training phase complete",
        }, indent=2) + "\n")
        revision = publish_folder(stage, prefix, repo_id=MODEL_REPO, repo_type="model")
    receipt.write_text(json.dumps({
        "repo_id": MODEL_REPO, "path": prefix, "revision": revision, "step": step,
    }, indent=2) + "\n")


def publish_diagnostics(sft: Path, dagger: Path, state: Path, prefix: str) -> str:
    """Retain the trainer exception as well as the outer controller's status."""
    with tempfile.TemporaryDirectory(prefix="hf-diagnostics-", dir=state) as temporary:
        stage = Path(temporary)
        for name, directory, files in (
            ("controller", state, ("phase", "controller.log", "sft.log", "dagger.log", "collection.log", "package.log")),
            ("sft", sft, ("bootstrap.log", "bootstrap.yaml", "run.json")),
            ("dagger", dagger, ("dagger.log", "dagger.yaml", "run.json", "best.json", "result.json")),
        ):
            for filename in files:
                source = directory / filename
                if source.is_file():
                    target = stage / name / filename
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(source, target)
        return publish_folder(stage, prefix + "/diagnostics", repo_id=MODEL_REPO, repo_type="model")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="stage", required=True)
    for name in ("folder", "training", "diagnostics", "selected"):
        command = commands.add_parser(name)
        command.add_argument("--prefix", required=True)
        if name in {"folder", "training", "selected"}:
            command.add_argument("--source", type=Path, required=True)
        if name == "folder":
            command.add_argument("--repo-type", choices=("model", "dataset"), required=True)
        if name in {"diagnostics", "selected"}:
            command.add_argument("--state", type=Path, required=True)
        if name == "diagnostics":
            command.add_argument("--sft", type=Path, required=True)
            command.add_argument("--dagger", type=Path, required=True)
    args = parser.parse_args()
    if args.stage == "training":
        publish_training_endpoint(args.source, args.prefix)
        return
    if args.stage == "diagnostics":
        publish_diagnostics(args.sft, args.dagger, args.state, args.prefix)
        return
    if args.stage == "selected":
        publish_training_endpoint(args.source / "training/dagger", args.prefix)
        best = json.loads((args.source / "best.json").read_text())
        publish_folder(Path(best["checkpoint"]).parent.parent, args.prefix, repo_id=MODEL_REPO, repo_type="model")
        publish_folder(args.state / "results", args.prefix + "/evaluation", repo_id=MODEL_REPO, repo_type="model")
        return
    repo = MODEL_REPO if args.repo_type == "model" else DATA_REPO
    revision = publish_folder(args.source, args.prefix, repo_id=repo, repo_type=args.repo_type)
    (args.source / "publication.json").write_text(json.dumps({
        "repo_id": repo, "path": args.prefix, "revision": revision,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
