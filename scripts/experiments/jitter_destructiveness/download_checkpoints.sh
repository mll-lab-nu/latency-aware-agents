#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
CHECKPOINT_ROOT="${REPO_ROOT}/results/checkpoints/jitter_destructiveness/released"

SMALL_REPO="latency-sensitive-bench/factor-sweeps-models"
SMALL_REVISION="229c17cbf697a239519433a575e988f27ffd50a8"
TEACHER_REPO="latency-sensitive-bench/latency-checkpoints"
TEACHER_REVISION="3590b0ee2d7e2c5a0b0ca3f854e85b88624eb8f5"
BASE_MODEL_REPO="Qwen/Qwen3-VL-4B-Instruct"
BASE_MODEL_REVISION="ebb281ec70b05090aa6165b016eac8ec08e71b17"

usage() {
  cat >&2 <<'EOF'
Usage: download_checkpoints.sh <small|vla|teachers> [flappy|demon_attack|deadly_corridor|all]
EOF
}

select_envs() {
  case "$1" in
    flappy|demon_attack|deadly_corridor) ENVS=("$1") ;;
    all) ENVS=(flappy demon_attack deadly_corridor) ;;
    *) usage; exit 2 ;;
  esac
}

download_file() {
  local repo="$1"
  local revision="$2"
  local source_path="$3"
  local destination="$4"
  local cached_path
  cached_path="$(conda run -n latency hf download "${repo}" "${source_path}" --revision "${revision}" --quiet)"
  install -D -m 0644 "${cached_path}" "${destination}"
}

download_small() {
  local env_name="$1"
  local seed source_path
  for seed in 10 11 12; do
    case "${env_name}:${seed}" in
      flappy:10) source_path='factor_sweeps/flappy/observation_stride/train/factor_sweep:flappy:observation_stride:fixed_l2:fs4:obs15:stride2:seed10/checkpoint_p0/checkpoint_000048760_25034752.pth' ;;
      flappy:11) source_path='factor_sweeps/flappy/observation_stride/train/factor_sweep:flappy:observation_stride:fixed_l2:fs4:obs15:stride2:seed11/checkpoint_p0/checkpoint_000048736_25034752.pth' ;;
      flappy:12) source_path='factor_sweeps/flappy/observation_stride/train/factor_sweep:flappy:observation_stride:fixed_l2:fs4:obs15:stride2:seed12/checkpoint_p0/checkpoint_000048832_25034752.pth' ;;
      demon_attack:10) source_path='factor_sweeps/demon_attack/observation_stride/train/factor_sweep:demon_attack:observation_stride:fixed_l4:fs4:obs15:stride4:seed10/checkpoint_p0/checkpoint_000048880_25026560.pth' ;;
      demon_attack:11) source_path='factor_sweeps/demon_attack/observation_stride/train/factor_sweep:demon_attack:observation_stride:fixed_l4:fs4:obs15:stride4:seed11/checkpoint_p0/checkpoint_000048880_25026560.pth' ;;
      demon_attack:12) source_path='factor_sweeps/demon_attack/observation_stride/train/factor_sweep:demon_attack:observation_stride:fixed_l4:fs4:obs15:stride4:seed12/checkpoint_p0/checkpoint_000048880_25026560.pth' ;;
      deadly_corridor:10) source_path='factor_sweeps/deadly_corridor/context_window/train/factor_sweep:deadly_corridor:frame_stack:fixed_l4:fs4:obs8p75:stride4:seed10/checkpoint_p0/checkpoint_000048836_25004032.pth' ;;
      deadly_corridor:11) source_path='factor_sweeps/deadly_corridor/context_window/train/factor_sweep:deadly_corridor:frame_stack:fixed_l4:fs4:obs8p75:stride4:seed11/checkpoint_p0/checkpoint_000048836_25004032.pth' ;;
      deadly_corridor:12) source_path='factor_sweeps/deadly_corridor/context_window/train/factor_sweep:deadly_corridor:frame_stack:fixed_l4:fs4:obs8p75:stride4:seed12/checkpoint_p0/checkpoint_000048836_25004032.pth' ;;
    esac
    download_file \
      "${SMALL_REPO}" \
      "${SMALL_REVISION}" \
      "${source_path}" \
      "${CHECKPOINT_ROOT}/small/${env_name}/seed_${seed}/checkpoint.pth"
  done
}

download_teacher() {
  local env_name="$1"
  local source_root run_name checkpoint_name
  case "${env_name}" in
    flappy)
      source_root="checkpoints/flappy_sf_same_latency_l2_seed0"
      run_name="flappy_sf_same_latency_l2_seed0"
      checkpoint_name="checkpoint_000029312_15040512.pth"
      ;;
    demon_attack)
      source_root="checkpoints_old/demon_attack_sf_same_latency_l2_seed0"
      run_name="demon_attack_sf_same_latency_l2_seed0"
      checkpoint_name="checkpoint_000097664_50003968.pth"
      ;;
    deadly_corridor)
      source_root="checkpoints_old/deadly_corridor_sf_same_latency_l2_seed0"
      run_name="deadly_corridor_sf_same_latency_l2_seed0"
      checkpoint_name="checkpoint_000024420_50012160.pth"
      ;;
  esac
  download_file \
    "${TEACHER_REPO}" \
    "${TEACHER_REVISION}" \
    "${source_root}/config.json" \
    "${CHECKPOINT_ROOT}/teachers/${run_name}/config.json"
  download_file \
    "${TEACHER_REPO}" \
    "${TEACHER_REVISION}" \
    "${source_root}/checkpoint_p0/${checkpoint_name}" \
    "${CHECKPOINT_ROOT}/teachers/${run_name}/checkpoint_p0/${checkpoint_name}"
}

patch_base_model_path() {
  local config_path="$1"
  local base_model_dir="$2"
  sed -i -E "s|^([[:space:]]*)base_vlm:.*|\1base_vlm: ${base_model_dir}|" "${config_path}"
}

download_vla_base_model() {
  local base_model_dir="${REPO_ROOT}/third_party/starVLA/playground/Pretrained_models/Qwen3-VL-4B-Instruct"
  conda run --no-capture-output -n latency hf download \
    "${BASE_MODEL_REPO}" \
    --revision "${BASE_MODEL_REVISION}" \
    --local-dir "${base_model_dir}"
}

download_vla() {
  local env_name="$1"
  local vla_root="${CHECKPOINT_ROOT}/vla"
  local config_path
  case "${env_name}" in
    flappy)
      conda run --no-capture-output -n latency hf download \
        latency-sensitive-bench/openvla_flappy_200ep \
        flappy_fix_latency_2_200ep/checkpoints/steps_5000_state/model.safetensors \
        flappy_fix_latency_2_200ep/config.yaml \
        flappy_fix_latency_2_200ep/config.full.yaml \
        flappy_fix_latency_2_200ep/dataset_statistics.json \
        flappy_fix_latency_2_200ep/dataset_statistics_eval.json \
        --revision 57d4263251043ab85f0d5f6fbdf23a4cb84d4a10 \
        --local-dir "${vla_root}"
      config_path="${vla_root}/flappy_fix_latency_2_200ep/config.yaml"
      ;;
    demon_attack)
      conda run --no-capture-output -n latency hf download \
        latency-sensitive-bench/openvla_demon_attack_200ep \
        demon_attack_fix_latency_2_200ep/checkpoints/steps_7000_state/model.safetensors \
        demon_attack_fix_latency_2_200ep/config.yaml \
        demon_attack_fix_latency_2_200ep/config.full.yaml \
        demon_attack_fix_latency_2_200ep/dataset_statistics.json \
        demon_attack_fix_latency_2_200ep/dataset_statistics_eval.json \
        --revision 55104205f32fea3b1ce38e0fc98f251dd824e9a8 \
        --local-dir "${vla_root}"
      config_path="${vla_root}/demon_attack_fix_latency_2_200ep/config.yaml"
      ;;
    deadly_corridor)
      local run_root="${vla_root}/deadly_corridor_fix_latency_2"
      conda run --no-capture-output -n latency hf download \
        latency-sensitive-bench/openvla_deadly_corridor_fix_latency_2 \
        checkpoints/best_state/model.safetensors \
        checkpoints/best_model_metadata.json \
        config.yaml \
        config.full.yaml \
        dataset_statistics.json \
        dataset_statistics_eval.json \
        --revision 16b55cb7d8185e0b58d45851a64ffa2c879cad72 \
        --local-dir "${run_root}"
      config_path="${run_root}/config.yaml"
      ;;
  esac
  patch_base_model_path \
    "${config_path}" \
    "${REPO_ROOT}/third_party/starVLA/playground/Pretrained_models/Qwen3-VL-4B-Instruct"
}

kind="${1:-}"
env_selector="${2:-all}"
select_envs "${env_selector}"

case "${kind}" in
  small)
    for env_name in "${ENVS[@]}"; do download_small "${env_name}"; done
    ;;
  teachers)
    for env_name in "${ENVS[@]}"; do download_teacher "${env_name}"; done
    ;;
  vla)
    download_vla_base_model
    for env_name in "${ENVS[@]}"; do download_vla "${env_name}"; done
    ;;
  *)
    usage
    exit 2
    ;;
esac
