<p align="center">
  <a href="https://ragen-ai.github.io/lagen/"><img src="docs/assets/lagen-logo.svg" alt="LAGEN logo" width="180"></a>
</p>

<h1 align="center">LAGEN: Are Vision-Language Agents Latency Aware?</h1>

<p align="center">
  Training agents for a world that keeps moving.
</p>

<p align="center">
  <a href="https://ragen-ai.github.io/lagen/"><img src="https://img.shields.io/badge/Website-6D597A?style=for-the-badge" alt="Website"></a>
  <a href="https://ragen-ai.github.io/lagen/assets/paper.pdf"><img src="https://img.shields.io/badge/Paper-355070?style=for-the-badge" alt="Paper"></a>
  <a href="https://huggingface.co/MLL-Lab/LAGEN-models"><img src="https://img.shields.io/badge/Models-B56576?style=for-the-badge&amp;logo=huggingface" alt="Hugging Face Models"></a>
  <a href="https://huggingface.co/datasets/MLL-Lab/LAGEN-datasets"><img src="https://img.shields.io/badge/Datasets-E56B6F?style=for-the-badge&amp;logo=huggingface" alt="Hugging Face Datasets"></a>
  <a href="https://huggingface.co/datasets/MLL-Lab/LAGEN-profiles"><img src="https://img.shields.io/badge/Profiles-EAAC8B?style=for-the-badge&amp;logo=huggingface" alt="Hugging Face Profiles"></a>
  <a href="#citation"><img src="https://img.shields.io/badge/Citation-77727E?style=for-the-badge" alt="Citation"></a>
</p>

## Overview

LAGEN measures how deployment latency affects vision-language agents.
It provides evaluation, demonstration collection, latency measurement, and profile replay.
Use these stages with your own environment, policy, and training framework.

The repository also provides reference recipes for games, Gymnasium tasks, MIKASA manipulation, and HumanoidBench balance.
Sample Factory supplies reference teacher training.
The pinned [StarVLA source](third_party/starVLA) supplies reference VLA training and inference.

## Released resources

| Resource | Contents |
| --- | --- |
| [Models](https://huggingface.co/MLL-Lab/LAGEN-models) | Teachers, VLA weights, model configurations, task contracts |
| [Datasets](https://huggingface.co/datasets/MLL-Lab/LAGEN-datasets) | Demonstrations, training data, evaluation records, experiment manifests |
| [Profiles](https://huggingface.co/datasets/MLL-Lab/LAGEN-profiles) | Measured traces, fitted latency profiles, calibration records |

The [artifact guide](docs/artifacts.md) identifies fixed revisions, paths, evaluation protocols, and publication status.
Browse the resources in the [LAGEN collection](https://huggingface.co/collections/MLL-Lab/lagen-6ac750f21139008b8114fff5).

## Choose a starting point

| Your input | Start here |
| --- | --- |
| A published or local model | [Select artifacts](docs/artifacts.md), then [evaluate](docs/workflow.md#evaluate-an-existing-model) |
| Your own environment or policy | [Run the CPU example](docs/guides/custom_components.md), then replace its factories |
| A teacher or training dataset | [Collect demonstrations](docs/guides/rollout_collection.md) or use a task cookbook |
| A model to measure | [Measure and replay latency](docs/cookbooks/profiling.md) |

The custom CPU example does not require an NVIDIA GPU.
Reference VLA and GPU simulator recipes require their selected model and simulator dependencies.
See the [installation guide](docs/installation.md) for each installation path.

Clone the source:

```bash
git lfs install
git clone --filter=blob:none https://github.com/ZihanWang314/latency-sensitive-bench.git
cd latency-sensitive-bench
```

Install the package in a Python 3.10 environment to run the CPU example:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
export PYTHONPATH="$PWD/configs/examples/custom_env${PYTHONPATH:+:$PYTHONPATH}"
python -m latency_bench eval --config configs/examples/custom_env/eval.yaml
```

The [installation guide](docs/installation.md) covers reference GPU environments and task dependencies.
The [workflow guide](docs/workflow.md) describes each stage's inputs and local outputs.

## Documentation

| Start here | Purpose |
| --- | --- |
| [Documentation index](docs/README.md) | Find guides, cookbooks, references, and experiment recipes |
| [Installation](docs/installation.md) | Install the selected task and model runtime |
| [Workflow](docs/workflow.md) | Start from an existing teacher, dataset, model, or profile |
| [Custom components](docs/guides/custom_components.md) | Run the complete CPU example with your own environment and policy |
| [Task cookbooks](docs/README.md#cookbooks) | Train, collect, evaluate, and measure reference tasks |
| [Artifacts](docs/artifacts.md) | Select published inputs and their fixed source protocols |

## Citation

If you use LAGEN in your research, cite:

```bibtex
@misc{chafekar2026lagen,
  title={LAGEN: Are Vision-Language Agents Latency Aware?},
  author={Talha Chafekar and Zihan Wang and Zeju Li and Xinyuan Li and Qineng Wang and Jiajun Wu and Yuke Zhu and Yi Dong and Zhiding Yu and Ruohan Zhang and Manling Li},
  year={2026}
}
```
