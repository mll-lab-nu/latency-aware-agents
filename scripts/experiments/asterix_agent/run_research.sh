#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
exec "${PYTHON:-python}" -m scripts.experiments.asterix_agent.research "$@"
