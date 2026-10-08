source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../../install/_host.sh"

python -m pip install --no-deps -e "${LATENCY_BENCH_ROOT}/third_party/flappy-bird-gymnasium" -e "${LATENCY_BENCH_ROOT}/third_party/sample-factory"
# regular pygame only — pygame-ce shadows the same `pygame` namespace and breaks
# the flappy fork's sprite loader (see install/env/flappy.sh).
python -m pip install pygame==2.6.1 gymnasium==0.29.1 stable-baselines3==2.8.0 ale-py==0.10.2 AutoROM==0.6.1 AutoROM.accept-rom-license==0.6.1
python -m pip install --force-reinstall --no-deps pygame==2.6.1
