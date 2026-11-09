"""Measure deployment latency and fit completed local traces."""
import argparse
from pathlib import Path

from latency_bench.core.config import load_config
from latency_bench.profiling.spec import path_segment


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m latency_bench profile", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="measure and fit a deployment")
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--resume", metavar="RUN_ID", type=path_segment)
    fit = commands.add_parser("fit", help="fit completed local measurement sessions")
    fit.add_argument("--run-dir", type=Path, required=True)
    fit.add_argument("--method", choices=("temporal", "hmm", "block", "ar1", "arma"), default="temporal")
    fit.add_argument("--session-ids", type=int, nargs="+", help="Only these completed fitting sessions; exclude held-out runs")
    fit.add_argument("--output-dir", type=Path, help="Separate output for a selected fit or tuning candidate")
    fit.add_argument("--hmm-states", type=int, default=2)
    fit.add_argument("--hmm-max-iterations", type=int, default=1000)
    fit.add_argument("--block-mean-length", type=float, default=32)
    args = parser.parse_args(argv)
    # Hardware and numerical dependencies are unnecessary for argument help.
    from latency_bench.profiling.runner import fit_profile, run_profile

    if args.command == "run":
        result = run_profile(load_config(args.config), resume_run_id=args.resume)
    else:
        result = fit_profile(args.run_dir, method=args.method, session_ids=args.session_ids,
                             output_dir=args.output_dir, hmm_states=args.hmm_states,
                             hmm_max_iterations=args.hmm_max_iterations,
                             block_mean_length=args.block_mean_length)
    for name, value in result.items():
        print(f"{name}={value}")
