#!/usr/bin/env bash
# Shared install helper: prefer uv (parallel downloads + fast resolver) and
# fall back to plain pip when uv is unavailable. Source this file; do not run it.
#
#   source "${SCRIPT_DIR}/_pip.sh"
#   ensure_uv          # install uv into the active env once (no-op if present)
#   pip_install <args> # drop-in for `python -m pip install`
#
# To swap the installer in the future, edit only this file.

ensure_uv() {
    command -v uv >/dev/null 2>&1 && return 0
    "${PYTHON_BIN:-python}" -m pip install -q uv >/dev/null 2>&1 || true
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
    local attempts="${PIP_INSTALL_MAX_ATTEMPTS:-3}"
    local sleep_seconds="${PIP_INSTALL_RETRY_SLEEP_SECONDS:-5}"
    local attempt status
    local -a cmd
    if command -v uv >/dev/null 2>&1; then
        if _pip_has_explicit_index_arg "$@"; then
            cmd=(env -u UV_DEFAULT_INDEX -u UV_INDEX_URL uv pip install --python "${py}" "$@")
        else
            cmd=(uv pip install --python "${py}" "$@")
        fi
    else
        cmd=("${PYTHON_BIN:-python}" -m pip install "$@")
    fi

    for ((attempt = 1; attempt <= attempts; attempt++)); do
        if "${cmd[@]}"; then
            return 0
        fi
        status=$?
        if ((attempt == attempts)); then
            return "${status}"
        fi
        echo "[pip] pip install failed on attempt ${attempt}/${attempts}; retrying in ${sleep_seconds}s: ${cmd[*]}" >&2
        sleep "${sleep_seconds}"
    done
}
