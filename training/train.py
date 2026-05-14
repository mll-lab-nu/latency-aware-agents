from __future__ import annotations

import argparse
from collections.abc import Sequence

from training.common import (
    apply_training_overrides,
    import_integration_module,
    load_training_config,
)


def run_from_config(config: dict) -> int:
    module = import_integration_module(config)
    return int(module.run_from_config(config))


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train config-driven Sample Factory backends.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-name", type=str)
    parser.add_argument("--train-dir", type=str)
    parser.add_argument("--mode", type=str)
    parser.add_argument("--checkpoint-kind", choices=["latest", "best"])
    parser.add_argument("--output-dir", type=str)
    args = parser.parse_args(argv)

    config = load_training_config(args.config)
    config = apply_training_overrides(
        config,
        run_name=args.run_name,
        train_dir=args.train_dir,
        mode=args.mode,
        checkpoint_kind=args.checkpoint_kind,
        output_dir=args.output_dir,
    )
    raise SystemExit(run_from_config(config))


if __name__ == "__main__":
    main()
