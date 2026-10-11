"""Convert supported collection artifacts through their existing task converters."""
from copy import deepcopy
from pathlib import Path
import json
import shutil


def convert_dataset(config):
    """Convert a collection directory while retaining its task and label contract."""
    options = deepcopy(config["conversion"])
    if options["format"] != "lerobot":
        raise ValueError("conversion.format must be lerobot")
    source = Path(options["input_dir"]).expanduser().absolute()
    output = Path(options["output_dir"]).expanduser().absolute()
    validation = output.with_name(output.name + "__val")
    inputs = [source]
    inputs.extend(Path(options[key]).expanduser().absolute() for key in ("input_val", "normalization_source") if key in options)
    if any(item.resolve().is_relative_to(destination.resolve()) for item in inputs for destination in (output, validation)):
        raise ValueError("Conversion outputs must not contain input or normalization sources")
    force = options.get("force", False)
    for destination in (output, validation):
        if destination.exists() and not force:
            raise FileExistsError(destination)
    if options.get("task") == "mikasa_h8" and (source / "meta/info.json").is_file():
        from latency_bench.integrations.starvla.mikasa_history_export import prepare_mikasa_h8
        for destination in (output, validation):
            if destination.exists():
                shutil.rmtree(destination)
        return prepare_mikasa_h8(source, Path(options["input_val"]), output)
    if "task" in options and (source / "meta/info.json").is_file():
        from latency_bench.integrations.starvla.task_contracts import prepare_lerobot
        for destination in (output, validation):
            if destination.exists():
                shutil.rmtree(destination)
        return prepare_lerobot(options["task"], source,
            Path(options["input_val"]) if "input_val" in options else None, output,
            Path(options["normalization_source"]) if "normalization_source" in options else None)
    if options.get("task") == "mikasa" and not (source / "train.parquet").exists():
        from latency_bench.integrations.starvla.mikasa_h1_export import prepare
        for destination in (output, validation):
            if destination.exists():
                shutil.rmtree(destination)
        return prepare(source, output, train_episodes=options["train_episodes"], val_episodes=options["val_episodes"])
    # A task's native reader validates the source before its converter writes it.
    if (source / "collection.json").is_file():
        from latency_bench.integrations.starvla.mikasa_h8_export import prepare
        collection = json.loads((source / "collection.json").read_text())
        formats = {"mikasa_h8_naive_dagger_raw_v1": "mikasa_h8",
                   "mikasa_conditioned_h8_raw_v3": "mikasa_conditioned_h8"}
        task = formats[collection["format"]]
        if "task" in options and options["task"] != task:
            raise ValueError("conversion.task does not match the collected label contract")
        for destination in (output, validation):
            if destination.exists():
                shutil.rmtree(destination)
        return prepare(source, output, task=task,
            normalization_source=Path(options["normalization_source"]) if "normalization_source" in options else None)
    from latency_bench.integrations.starvla.lerobot_export import convert_canonical_rollout_to_lerobot
    if "task" in options:
        from tempfile import TemporaryDirectory
        from latency_bench.integrations.starvla.task_contracts import prepare_lerobot
        output.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix=".conversion-", dir=output.parent) as temporary:
            converted = Path(temporary) / "train"
            convert_canonical_rollout_to_lerobot(source, converted, action_carrier=options.get("action_carrier", "native"),
                                                write_starvla_handoff=False)
            for destination in (output, validation):
                if destination.exists():
                    shutil.rmtree(destination)
            result = prepare_lerobot(options["task"], converted, converted.with_name("train__val"), output,
                Path(options["normalization_source"]) if "normalization_source" in options else None)
            metadata = json.loads((source / "metadata.json").read_text())
            prefix = metadata.get("label_provenance", {}).get("action_prefix")
            if prefix is not None:
                contract_path = output / "task_contract.json"
                contract = json.loads(contract_path.read_text())
                contract["action_prefix"] = prefix
                contract_path.write_text(json.dumps(contract, indent=2) + "\n")
            return result
    return convert_canonical_rollout_to_lerobot(source, output, force=force,
        action_carrier=options.get("action_carrier", "native"),
        write_starvla_handoff=options.get("write_starvla_handoff", True))
