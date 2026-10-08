# Hugging Face cards

Maintained root cards for the three main LAGEN repositories:

| Repository | Card source |
|---|---|
| Models | [benchmark-models](benchmark-models.md) |
| Data and experiment manifests | [benchmark-datasets](benchmark-datasets.md) |
| Latency profiles | [profiles](profiles.md) |

The [paper experiment catalogue](https://huggingface.co/datasets/MLL-Lab/LAGEN-datasets/tree/e983a3a72e042c6fab5575b847cdf6f4432cddaa/experiments) expands into experiment, task and run pages. Root cards contain concise navigation. Detailed settings and source limits live in task pages and fixed manifests.

Task transfer includes the formal Demon Attack policies and a separate archive for joint training and evaluation reruns.

The memory, latency-information, latency-transfer and fine-tuning-scope cards identify retired source repositories. Factor-sweeps and fine-tuning-scope source cards remain in this directory. Their shared data references use the canonical data repository.

Dataset YAML retains explicit named selections and measured feature metadata. Original source snapshots remain under the experiments’ provenance directories.
