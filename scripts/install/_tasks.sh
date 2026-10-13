#!/usr/bin/env bash
# Task dependencies shared by the benchmark and model installation entrypoints.

resolve_install_tasks() {
    INSTALL_ATARI=false
    INSTALL_FLAPPY=false
    INSTALL_VIZDOOM=false
    INSTALL_MUJOCO=false
    INSTALL_BALANCE=false
    INSTALL_BOX2D=false
    INSTALL_MIKASA=false
    local task
    local -a selected_tasks
    IFS=',' read -r -a selected_tasks <<< "$1"
    for task in "${selected_tasks[@]}"; do
        case "${task}" in
            all)
                INSTALL_ATARI=true
                INSTALL_FLAPPY=true
                INSTALL_VIZDOOM=true
                INSTALL_MUJOCO=true
                INSTALL_BALANCE=true
                INSTALL_BOX2D=true
                INSTALL_MIKASA=true
                ;;
            air_raid|demon_attack|asterix|atlantis) INSTALL_ATARI=true ;;
            flappy) INSTALL_FLAPPY=true ;;
            deadly_corridor) INSTALL_VIZDOOM=true ;;
            ant|half_cheetah|hopper|humanoid|inverted_pendulum|swimmer|walker2d|walker2d_continuous|walker2d_continuous_torque) INSTALL_MUJOCO=true ;;
            g1_sonic_balance_simple) INSTALL_BALANCE=true ;;
            lunar_lander_discrete) INSTALL_BOX2D=true ;;
            mikasa_intercept_grab_fast) INSTALL_MIKASA=true ;;
            dummy|pendulum_discrete|pendulum_continuous) ;;
            *)
                echo "[install] unknown task: ${task}" >&2
                return 1
                ;;
        esac
    done
}

install_selected_tasks() (
    cd "${LATENCY_BENCH_ROOT}"
    if [[ "${INSTALL_MUJOCO}" == true ]]; then
        pip_install -e ".[mujoco]"
    fi
    if [[ "${INSTALL_BALANCE}" == true ]]; then
        pip_install -e ".[sonic]" -e third_party/humanoid-bench
    fi
    if [[ "${INSTALL_BOX2D}" == true ]]; then
        pip_install swig==4.4.1
        pip_install -e ".[box2d]"
    fi
    if [[ "${INSTALL_MIKASA}" == true ]]; then
        PYTHON_BIN=python bash scripts/mikasa/install.sh
    fi
    if [[ "${INSTALL_ATARI}" == true ]]; then
        PYTHON_BIN=python bash third_party/starVLA/examples/rl_games/install/env/demon_attack.sh
        bash scripts/bash_scripts/build_cuda_extensions.sh --env atari
    fi
    if [[ "${INSTALL_VIZDOOM}" == true ]]; then
        PYTHON_BIN=python bash third_party/starVLA/examples/rl_games/install/env/deadly_corridor.sh
    fi
    # Flappy needs regular pygame to own the import after VizDoom installs pygame-ce.
    if [[ "${INSTALL_FLAPPY}" == true ]]; then
        PYTHON_BIN=python bash third_party/starVLA/examples/rl_games/install/env/flappy.sh
        bash scripts/bash_scripts/build_cuda_extensions.sh --env flappy
    fi
)
