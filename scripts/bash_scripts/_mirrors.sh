#!/usr/bin/env bash
# Mirror speed-test + provider selection for package downloads.
# Source this file; do not run it.
#
#   source "${SCRIPT_DIR}/_mirrors.sh"
#   torch_index_for cu128      # echo fastest torch index, e.g. https://.../whl/cu128
#   pypi_index_url cu128       # echo matching pypi simple index
#   export_mirror_env cu128    # export STARVLA_TORCH_INDEX_BASE / PIP_INDEX_URL / UV_*
#
# One bounded speed test probes the heavy torch wheels on each provider, picks the
# fastest, and routes both the torch index and the pypi index through that provider.
# The result is cached so install.sh, torch.sh and every parallel env build reuse one
# measurement. Falls back to the official PyTorch CDN if every probe fails.
#
# Knobs (env):
#   MIRROR_PROVIDERS         space-separated provider list (default: official aliyun sjtu)
#   MIRROR_PROVIDER_OVERRIDE force a provider, skip the probe (e.g. aliyun)
#   MIRROR_SPEED_FLOOR_BPS   min acceptable bytes/s before falling back (default: 1 MiB/s)
#   MIRROR_FORCE_REPROBE=1   ignore the cached result and re-probe

MIRROR_PROVIDERS="${MIRROR_PROVIDERS:-official aliyun sjtu}"
MIRROR_SPEED_FLOOR_BPS="${MIRROR_SPEED_FLOOR_BPS:-1048576}"

_mirror_torch_base() {
    case "$1" in
        official) echo "https://download.pytorch.org/whl" ;;
        aliyun)   echo "https://mirrors.aliyun.com/pytorch-wheels" ;;
        sjtu)     echo "https://mirror.sjtu.edu.cn/pytorch-wheels" ;;
    esac
}

_mirror_pypi_index() {
    case "$1" in
        official) echo "https://pypi.org/simple" ;;
        aliyun)   echo "https://mirrors.aliyun.com/pypi/simple" ;;
        # SJTU mirrors torch wheels but not a general pypi index; use the official one.
        sjtu)     echo "https://pypi.org/simple" ;;
    esac
}

_mirror_cache_file() {
    echo "${UV_CACHE_DIR:-${TMPDIR:-/tmp}}/.fastest_torch_provider_${1:-cu128}"
}

# Echo "<bytes_per_sec> <provider>" after range-downloading part of a large wheel.
# Discover a probe wheel by matching an actual torch wheel for this CUDA profile.
# Some mirrors expose a top-level CUDA index but not the torch package sub-index;
# those must score 0 or uv can fall through to PyPI and install a different CUDA
# wheel for the same plain torch version.
_mirror_probe_provider() {
    local provider="$1" profile="$2" base wheel speed
    base="$(_mirror_torch_base "$provider")"
    [ -n "${base}" ] || { echo "0 ${provider}"; return; }
    wheel="$(curl -sL --max-time 10 "${base}/${profile}/torch/" 2>/dev/null \
        | grep -oE "torch-[^\"#<> ]+(%2B|\\+)${profile}[^\"#<> ]+_x86_64\\.whl" | head -1)"
    [ -n "${wheel}" ] || { echo "0 ${provider}"; return; }
    speed="$(curl -sL --max-time 10 -r 0-26214400 -o /dev/null \
        -w '%{speed_download}' "${base}/${profile}/torch/${wheel}" 2>/dev/null || echo 0)"
    echo "${speed%.*} ${provider}"
}

# Echo the fastest provider name for a torch profile (cached after the first call).
pick_fastest_provider() {
    local profile="${1:-cu128}"

    if [ -n "${MIRROR_PROVIDER_OVERRIDE:-}" ]; then
        echo "${MIRROR_PROVIDER_OVERRIDE}"
        return 0
    fi

    local cache; cache="$(_mirror_cache_file "${profile}")"
    if [ "${MIRROR_FORCE_REPROBE:-0}" != "1" ] && [ -s "${cache}" ]; then
        cat "${cache}"
        return 0
    fi

    echo "[mirrors] probing providers (${MIRROR_PROVIDERS}) for ${profile} ..." >&2
    local tmp; tmp="$(mktemp)"
    local p
    for p in ${MIRROR_PROVIDERS}; do
        _mirror_probe_provider "${p}" "${profile}" >>"${tmp}" &
    done
    wait

    sort -rn "${tmp}" | while read -r bps prov; do
        printf '[mirrors]   %-9s %s MB/s\n' "${prov}" "$(( bps / 1048576 ))" >&2
    done

    local best_speed best_provider
    read -r best_speed best_provider < <(sort -rn "${tmp}" | head -1)
    rm -f "${tmp}"

    if [ -z "${best_provider:-}" ] || [ "${best_speed:-0}" -lt "${MIRROR_SPEED_FLOOR_BPS}" ]; then
        best_provider="official"
    fi
    echo "[mirrors] selected provider: ${best_provider}" >&2

    mkdir -p "$(dirname "${cache}")" 2>/dev/null || true
    printf '%s\n' "${best_provider}" >"${cache}" 2>/dev/null || true
    printf '%s\n' "${best_provider}"
}

torch_index_for() {
    local profile="${1:-cu128}"
    echo "$(_mirror_torch_base "$(pick_fastest_provider "${profile}")")/${profile}"
}

pypi_index_url() {
    _mirror_pypi_index "$(pick_fastest_provider "${1:-cu128}")"
}

# Export the chosen indexes so child processes (requirements install, starVLA bootstrap,
# parallel env builds) inherit them without re-probing.
export_mirror_env() {
    local profile="${1:-cu128}" provider pypi
    provider="$(pick_fastest_provider "${profile}")"
    export STARVLA_TORCH_INDEX_BASE="$(_mirror_torch_base "${provider}")"
    pypi="$(_mirror_pypi_index "${provider}")"
    export PIP_INDEX_URL="${pypi}"
    export UV_DEFAULT_INDEX="${pypi}"
    export UV_INDEX_URL="${pypi}"
    echo "[mirrors] provider=${provider} torch_base=${STARVLA_TORCH_INDEX_BASE} pypi=${pypi}" >&2
}
