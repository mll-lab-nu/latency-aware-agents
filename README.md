# Latency Sensitive Bench

## Install

Install the repository by running:

```bash
bash install.sh
```

`install.sh` selects a PyTorch 2.10.0 CUDA wheel automatically. GPUs with compute capability `>= 10.0` use `cu130`; older CUDA GPUs use `cu126`.

Override the auto-selected wheel explicitly when needed:

```bash
LATENCY_BENCH_TORCH_PROFILE=cu130 bash install.sh
```
