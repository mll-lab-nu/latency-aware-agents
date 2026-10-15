"""Candidate filesystem and settings boundaries owned by the research evaluator."""

from copy import deepcopy
import difflib
import json
from pathlib import Path
import re
import shutil


SETTING_KEYS = {"image_offsets", "action_history", "plans_enabled", "memory_enabled", "tools",
                "request_function", "prompt_function"}


def read_candidate(root):
    """Validate external candidate files before executing any candidate code."""
    root = Path(root)
    files = {}
    for path in root.iterdir():
        if path.is_symlink() or not path.is_file() or not re.fullmatch(r"(?:[a-z][a-z0-9_]*\.py|prompt\.txt|settings\.json)", path.name):
            raise ValueError(f"Candidate path is outside the writable contract: {path.name}")
        files[path.name] = path.read_text()
        if path.suffix == ".py":
            compile(files[path.name], path.name, "exec")
    if not {"prompt.txt", "settings.json", "strategy.py", "diagnostic.py"} <= files.keys():
        raise ValueError("Candidate requires prompt.txt, settings.json, strategy.py and diagnostic.py")
    settings = json.loads(files["settings.json"])
    if set(settings) != SETTING_KEYS:
        raise ValueError("Candidate settings must contain only the fixed strategy settings")
    offsets = settings["image_offsets"]
    if not isinstance(offsets, list) or not 1 <= len(offsets) <= 3 or any(type(x) is not int or not 0 <= x <= 299 for x in offsets):
        raise ValueError("Choose one to three legal image offsets, each between 0 and 299 observations")
    if type(settings["action_history"]) is not int or not 1 <= settings["action_history"] <= 8:
        raise ValueError("Action history must contain one to eight commands")
    if any(type(settings[key]) is not bool for key in ("plans_enabled", "memory_enabled")):
        raise ValueError("Plan and memory switches must be booleans")
    if not isinstance(settings["tools"], dict) or any(not isinstance(name, str) for name in settings["tools"]):
        raise ValueError("tools must map names to candidate function references")
    references = [settings["prompt_function"], *settings["tools"].values(), "diagnostic:verify"]
    if settings["request_function"] is not None:
        references.append(settings["request_function"])
    for reference in references:
        if not isinstance(reference, str) or not re.fullmatch(r"[a-z][a-z0-9_]*:[a-z][a-z0-9_]*", reference):
            raise ValueError(f"Invalid candidate function reference: {reference!r}")
        if reference.split(":")[0] + ".py" not in files:
            raise ValueError(f"Candidate function must belong to a candidate file: {reference}")
    if not files["prompt.txt"].strip():
        raise ValueError("Candidate prompt must not be empty")
    return files, settings


def apply_proposal(parent, destination, proposal):
    """Apply model-returned complete file replacements within a fresh directory."""
    if set(proposal) != {"status", "diagnosis", "files"}:
        raise ValueError("Research response requires status, diagnosis and files")
    if proposal["status"] not in {"candidate", "no_candidate", "framework_issue"}:
        raise ValueError("Unknown research response status")
    diagnosis = proposal["diagnosis"]
    if set(diagnosis) != {"frame", "observation", "hypothesis", "expected_effect"}:
        raise ValueError("Diagnosis requires a frame, observation, hypothesis and expected_effect")
    if type(diagnosis["frame"]) is not int or diagnosis["frame"] < 0:
        raise ValueError("Diagnosis frame must be a nonnegative raw frame")
    if any(not isinstance(diagnosis[key], str) or not diagnosis[key].strip()
           for key in ("observation", "hypothesis", "expected_effect")):
        raise ValueError("Diagnosis explanations must be nonempty text")
    updates = proposal["files"]
    if not isinstance(updates, dict):
        raise ValueError("files must map candidate filenames to their complete text")
    if proposal["status"] != "candidate":
        if updates:
            raise ValueError("A stopped research proposal cannot modify files")
        return False
    if not updates:
        raise ValueError("A candidate must contain a modification")
    for name, contents in updates.items():
        if not re.fullmatch(r"(?:[a-z][a-z0-9_]*\.py|prompt\.txt|settings\.json)", name) or not isinstance(contents, str):
            raise ValueError(f"Research cannot modify this path: {name}")
    shutil.copytree(parent, destination)
    for name, contents in updates.items():
        (Path(destination) / name).write_text(contents)
    after, _ = read_candidate(destination)
    before, _ = read_candidate(parent)
    if after == before:
        raise ValueError("Candidate does not change the parent")
    return True


def freeze_candidate(source, destination):
    files, _ = read_candidate(source)
    Path(destination).mkdir()
    for name, contents in files.items():
        (Path(destination) / name).write_text(contents)
    return files


def candidate_diff(parent, candidate):
    before, _ = read_candidate(parent)
    after, _ = read_candidate(candidate)
    return "".join(line for name in sorted(before.keys() | after.keys()) for line in difflib.unified_diff(
        before.get(name, "").splitlines(keepends=True), after.get(name, "").splitlines(keepends=True),
        fromfile="parent/" + name, tofile="candidate/" + name))


def candidate_config(base, root, output, seed, frame_limit):
    _, settings = read_candidate(root)
    config = deepcopy(base)
    config["policy"].update(candidate_root=str(Path(root).resolve()), prompt_file=str(Path(root).resolve() / "prompt.txt"),
                            prompt_factory=settings["prompt_function"], stop_used_percent=None)
    config["harness"].update({key: settings[key] for key in SETTING_KEYS - {"prompt_function"}})
    config["logging"]["output_dir"] = str(output)
    config["latency"]["seed"] = 42 + seed
    config["experiment"]["seed"] = seed
    config["evaluation"].update(episodes=1, seeds=[seed], rounds=1, max_env_steps=frame_limit)
    return config


def promotion_decision(episodes, seeds, checks_passed):
    """Compare the complete paired cohort. Missing or failed runs never promote."""
    expected = {(version, seed) for version in ("parent", "candidate") for seed in seeds}
    actual = [(item["version"], item["seed"]) for item in episodes]
    if not checks_passed or len(actual) != len(expected) or set(actual) != expected or any(item["status"] != "completed" for item in episodes):
        return {"promoted": False, "reason": "validation_incomplete"}
    means = {version: sum(item["score"] for item in episodes if item["version"] == version) / len(seeds)
             for version in ("parent", "candidate")}
    improved = means["candidate"] > means["parent"]
    return {"promoted": improved, "reason": "mean_improved" if improved else "mean_not_improved", "means": means}
