"""Command line inputs for collection and training-data conversion."""
import argparse
import json
from pathlib import Path

from latency_bench.core.config import load_config


def collect_main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="python -m latency_bench collect")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if args.output_dir is not None:
        config["logging"]["output_dir"] = str(args.output_dir)
    # Data runtimes are optional for other commands and argument help.
    from latency_bench.data.rollout import collect_rollouts

    result = collect_rollouts(config)
    print(f"OUTPUT_DIR={result['output_dir']}")
    print("COLLECTION_STATUS=" + ("probe_complete" if "probe_state" in result else "complete"))
    if "metadata" in result:
        print(json.dumps(result["metadata"], default=str))


def convert_main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="python -m latency_bench convert")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if args.output_dir is not None:
        config["conversion"]["output_dir"] = str(args.output_dir)
    if args.force:
        config["conversion"]["force"] = True
    from latency_bench.data.conversion import convert_dataset

    result = convert_dataset(config)
    print(f"OUTPUT_DIR={Path(config['conversion']['output_dir']).expanduser().absolute()}")
    print("CONVERSION_STATUS=complete")
    print(json.dumps(result, default=str))
