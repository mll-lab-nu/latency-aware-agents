"""Compose an agent evaluation from explicit reference, client and run settings."""

from copy import deepcopy


def adapt_reference_config(reference, agent_policy, *, provider, protocol):
    """Keep task/scheduler semantics; require the caller to select the run protocol."""
    config = deepcopy(reference)
    reference_policy = config["policy"]
    if reference_policy["type"] != "starvla":
        raise ValueError("The reference must be a StarVLA evaluation configuration")
    common = ("model", "effort", "bubblewrap_executable", "workspace_root", "runtime_root",
              "turn_timeout_seconds", "policy_id")
    native = {"codex": ("package_root", "codex_home"),
              "claude_code": ("executable", "credentials_file")}[provider]
    config["policy"] = {key: agent_policy[key] for key in common + native}
    config["policy"].update(type="coding_agent", provider=provider, reference_policy=reference_policy)
    for key in ("experiment", "executor", "latency", "evaluation", "logging"):
        config[key] = deepcopy(protocol[key])
    return config


def main():
    import argparse
    from pathlib import Path

    import yaml

    from latency_bench.core.config import load_config

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, help="VLA evaluation with resolved input sidecar paths")
    parser.add_argument("--agent", required=True, help="Native client config (existing pilot configs work)")
    parser.add_argument("--provider", required=True, choices=("codex", "claude_code"))
    parser.add_argument("--protocol", required=True, help="Explicit experiment/executor/latency/evaluation/logging YAML")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = adapt_reference_config(
        load_config(args.reference), load_config(args.agent)["policy"], provider=args.provider,
        protocol=yaml.safe_load(Path(args.protocol).read_text()),
    )
    Path(args.output).write_text(yaml.safe_dump(config, sort_keys=False))


if __name__ == "__main__":
    main()
