---
library_name: pytorch
tags:
- robotics
- vision-language-action
- latency-sensitive-bench
- multiple-checkpoints
---

# LAGEN models

Project resources: [LAGEN collection](https://huggingface.co/collections/MLL-Lab/lagen-6ac750f21139008b8114fff5).

Inference checkpoints, original configurations and model provenance.

| Experiment | Entry |
|---|---|
| Sim2Real calibration | [30-case held-out calibration](https://huggingface.co/datasets/MLL-Lab/LAGEN-datasets/tree/50d8f0aef64749a849760fc580ce3efb3c5c5e76/experiments/sim2real-calibration) |
| Visual history | [Visual history](https://huggingface.co/datasets/MLL-Lab/LAGEN-datasets/tree/main/experiments/visual-history) |
| Latency in prompt | [Latency in prompt](https://huggingface.co/datasets/MLL-Lab/LAGEN-datasets/tree/main/experiments/latency-in-prompt) |
| Latency transfer | [Latency transfer](https://huggingface.co/datasets/MLL-Lab/LAGEN-datasets/tree/main/experiments/latency-transfer) |
| Task transfer | [Task transfer](https://huggingface.co/datasets/MLL-Lab/LAGEN-datasets/tree/main/experiments/task-transfer) |
| VLA fine-tuning scope | [VLA fine-tuning scope](https://huggingface.co/datasets/MLL-Lab/LAGEN-datasets/tree/main/experiments/vla-finetuning-scope) |
| Mean vs. profile training | [Mean vs. profile training](https://huggingface.co/datasets/MLL-Lab/LAGEN-datasets/tree/main/experiments/mean-vs-profile) |
| Observation stride | [Observation stride](https://huggingface.co/datasets/MLL-Lab/LAGEN-datasets/tree/main/experiments/observation-stride) |
| Context window | [Context window](https://huggingface.co/datasets/MLL-Lab/LAGEN-datasets/tree/main/experiments/context-window) |

| Resource | Repository |
|---|---|
| benchmark-datasets | [MLL-Lab/LAGEN-datasets](https://huggingface.co/datasets/MLL-Lab/LAGEN-datasets) |
| profiles | [MLL-Lab/LAGEN-profiles](https://huggingface.co/datasets/MLL-Lab/LAGEN-profiles) |

[Benchmark release inventory](https://huggingface.co/MLL-Lab/LAGEN-models/blob/main/reports/benchmark-release/README.md)

## Reproduction

Select the exact task, checkpoint, dataset, and profile from the linked experiment manifest.
Each result identifies its episode count, clocks, latency settings, and seeds.
The manifest records any missing source bindings.
