"""Plot Fixed-N prefix63 development and held-out results, with an optional old column."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.tasks.mikasa_intercept_grab_fast.fixed_latency import FIXED_LATENCIES


def summarize(run: Path, previous_column: Path | None, output: Path) -> dict:
    """Read saved metrics; development selects weights, held-out cells only compare them."""
    condition = json.loads((run / "run.json").read_text())["condition"]
    prefix_frames = FIXED_LATENCIES[condition]
    records = []
    for step in range(0, 5001, 500):
        for latency in range(5):
            path = run / "evaluations" / f"step_{step:04d}" / f"eval_{latency}/result.json"
            cell = json.loads(path.read_text())
            assert cell["seed"] == 4242424442 and cell["episodes"] == 100
            assert cell["action_prefix"]["frames"] == prefix_frames and cell["audit"]["passed"]
            records.append({"step": step, "eval_latency": latency, **cell,
                            "evidence_path": str(path.parent)})
    best = json.loads((run / "best.json").read_text())
    heldout = []
    for latency in range(5):
        path = run / "heldout" / f"eval_{latency}/result.json"
        new = json.loads(path.read_text())
        assert new["episodes"] == 200
        assert new["seed"] == 4242424242
        assert new["checkpoint"] == best["checkpoint"]
        assert new["action_prefix"]["frames"] == prefix_frames and new["audit"]["passed"]
        heldout.append({"eval_latency": latency, **new, "evidence_path": str(path.parent)})
    result = {"run": str(run), "condition": condition, "selection_latency": prefix_frames,
              "best": best, "periodic": records, "heldout": heldout}
    if previous_column is not None:
        previous = json.loads(previous_column.read_text())
        assert previous["train_latency"] == prefix_frames
        comparisons = []
        for old in previous["results"]:
            assert old["n_episodes"] == 200
            assert (old["seed_start"], old["seed_end"]) == (4242424242, 4242424441)
            latency = old["latency_frames"]
            comparisons.append({"eval_latency": latency, "old_71d": old,
                                "new_63d": heldout[latency]})
        result.update(comparison=comparisons, previous_column=str(previous_column))
    output.mkdir(parents=True, exist_ok=True)
    (output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def plot(result: dict, output: Path) -> None:
    colors = ("#77727E", "#355070", "#B56576", "#6D597A", "#E56B6F")
    fig, ax = plt.subplots(figsize=(6, 3.5))
    for latency, color in zip(range(5), colors):
        cells = [cell for cell in result["periodic"] if cell["eval_latency"] == latency]
        ax.plot([cell["step"] for cell in cells], [100 * cell["success_rate"] for cell in cells],
                marker="o", markersize=3, label=f"L{latency}", color=color)
    ax.set(xlabel="DAgger optimizer steps", ylabel="Success rate (%)", ylim=(0, 100))
    ax.legend(ncol=5, frameon=False)
    for extension in ("pdf", "png"):
        fig.savefig(output / f"periodic.{extension}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(5, 3.5))
    cells = result["heldout"]
    ax.plot([cell["eval_latency"] for cell in cells],
            [100 * cell["success_rate"] for cell in cells],
            marker="o", label=f"{result['condition']} (63D)", color=colors[3])
    if "comparison" in result:
        old = sorted(result["comparison"], key=lambda cell: cell["eval_latency"])
        ax.plot([cell["eval_latency"] for cell in old],
                [100 * cell["old_71d"]["success_rate"] for cell in old],
                marker="o", label="Old contract (71D)", color=colors[0])
    ax.set(xlabel="Eval latency (frames)", ylabel="Success rate (%)",
           ylim=(0, 100), xticks=range(5))
    ax.legend(frameon=False)
    for extension in ("pdf", "png"):
        name = "comparison" if "comparison" in result else "heldout"
        fig.savefig(output / f"{name}.{extension}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--previous-column", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plot(summarize(args.run, args.previous_column, args.output), args.output)


if __name__ == "__main__":
    main()
