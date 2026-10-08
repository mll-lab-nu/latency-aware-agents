#!/usr/bin/env bash
# Shared install helper. Source this file; do not run it.
#
#   source "${SCRIPT_DIR}/_pip.sh"
#   ensure_uv          # install uv into the active env once
#   pip_install <args> # install into PYTHON_BIN with uv

ensure_uv() {
    "${PYTHON_BIN:-python}" -m pip install uv==0.11.32
}

_pip_has_explicit_index_arg() {
    local arg
    for arg in "$@"; do
        case "${arg}" in
            --index-url|--index-url=*|-i)
                return 0
                ;;
        esac
    done
    return 1
}

pip_install() {
    local py
    py="$(command -v "${PYTHON_BIN:-python}")"
    local -a cmd
    if _pip_has_explicit_index_arg "$@"; then
        cmd=(
            env -u UV_DEFAULT_INDEX -u UV_INDEX_URL
            uv pip install
            --python "${py}"
            "$@"
        )
    else
        cmd=(
            uv pip install
            --python "${py}"
            "$@"
        )
    fi
    "${cmd[@]}"
}
