#!/usr/bin/env bash
set -euo pipefail

uv sync --project third_party/MIKASA-Robo --python 3.10
uv pip install \
  --python third_party/MIKASA-Robo/.venv/bin/python \
  'websockets>=14,<16' 'msgpack>=1,<2'
