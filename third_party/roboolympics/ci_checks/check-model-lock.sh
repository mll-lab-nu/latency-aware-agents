#!/usr/bin/env bash
# Every model seat the run scripts (scripts/olympiad/run_{single,duel,race,team}.sh) start carries its explicit model
# lock THINK_MODEL_<seat> from players/players.toml, and the metering proxy refuses any other model under that lock
# (user decision 2026-10-08; players/README.md "Model lock"). Fake Harbor and fake upstream: no model is called.
set -uo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
python3 "$root/scripts/olympiad/test_model_lock.py" || { echo "FAIL [model lock] a model seat started by a run script lacks its THINK_MODEL_<seat> lock"; exit 1; }
