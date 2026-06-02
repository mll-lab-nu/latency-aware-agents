# Human Realtime Evaluation

## Purpose

This branch provides a browser-based human evaluation interface for the realtime latency benchmark. It uses the same latency-bench environment timing and action scheduling logic as the model realtime eval, but replaces model actions with human keyboard input. It supports formal latency-sweep eval sessions and free-play debugging.

## Running And Outputs

Create or activate the runtime environment, then start the backend:

```bash
conda activate latency-human
bash run.sh
```

Optional flags:

```bash
bash run.sh --port 8003 --output-dir human_eval_outputs
```

The default port is `8002`, and the default output root is `outputs/human_realtime`. If you pass `--port`, use that port in the browser URL.

Then open:

```text
http://127.0.0.1:8002
```

Results are stored under:

```text
OUTPUT_DIR/<user>/<session>/
```

For eval sessions, each environment writes its own folder:

```text
OUTPUT_DIR/<user>/<session>/
  flappy/
  demon_attack/
  deadly/
  overall_summary.json
```

Each environment folder contains per-latency logs, summaries, and plots:

```text
summary.json
summary.csv
session_summary.json
latency_vs_mean_length.png
latency_vs_mean_reward.png
latency_<N>fr/
  episodes.jsonl
  interrupted_episodes.jsonl
  steps.jsonl
  actions.jsonl
```

`episodes.jsonl` contains completed valid episodes used for metrics. Interrupted partial episodes are stored separately and excluded from the plotted means.

## Participant Flow

Enter the participant name, select an environment, set the eval configuration, and press `Start Eval`. Before each environment starts, a controls popup appears; read it and press `Start` to begin. After an environment finishes, choose the next remaining environment from the popup until all three are complete.

Controls:

- Flappy Bird: `Space`, `Up`, or `W` to flap.
- Demon Attack: `Left`/`Right` or `A`/`D` to move, `Space`, `W`, or `Up` to fire.
- Deadly Corridor: `W`/`S` or `Up`/`Down` to move, `A`/`D` to strafe, `Left`/`Right` to turn, `Space` to attack.
