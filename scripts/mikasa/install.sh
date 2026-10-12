#!/usr/bin/env bash
set -euo pipefail

uv venv --python 3.10 third_party/MIKASA-Robo/.venv
uv pip install --python third_party/MIKASA-Robo/.venv/bin/python --prerelease allow -e third_party/MIKASA-Robo
