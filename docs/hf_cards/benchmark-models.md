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

[Benchmark release inventory](reports/benchmark-release/README.md)

## Artifact retention

HAIC and Extreme Parkour releases are retired. Model bundles retain the published evaluation checkpoint, or the latest checkpoint when no evaluation selection exists. Optimizer and trainer recovery state are not release assets. Identical dataset copies use the canonical task paths. Original configurations and experiment evidence remain source records.

## Figure 2 latency-degradation sources

[Checkpoint source bindings](reports/latency-degradation-20261007/README.md) identify the six original game checkpoints and the six reused MIKASA checkpoints. All six game checkpoints are published with their original SHA256 and size. The report records the missing Flappy GR00T step 3000 and two unconfirmed OpenVLA bindings.

## Source consolidation

[Verified source index](https://huggingface.co/datasets/MLL-Lab/LAGEN-datasets/tree/a6e93d4a9a1277c5cd5a7145e0295e3e190374e1/reports/source-repository-retirement-20261008) records the former Standard-Pipeline, HumanoidBench, MIKASA and Figure 2 sources. The 65 selected Standard-Pipeline inference bundles reuse existing canonical model paths. All six confirmed Figure 2 game checkpoints are published with their original SHA256 and size. Available original configs and statistics remain with their canonical bundles. Unverified runtime capability and missing historical bindings remain explicit.
