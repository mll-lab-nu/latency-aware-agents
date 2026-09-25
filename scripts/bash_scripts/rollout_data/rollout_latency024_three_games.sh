#!/usr/bin/env bash
set -euo pipefail

# Generate/upload fixed-latency rollout datasets for the latency-0/2/4
# small-model teachers used to build mixed-latency data.
#
# New machine usage:
#   cd /path/to/placeholder
#   conda activate latency
#   export HF_TOKEN=hf_...
#   bash scripts/bash_scripts/rollout_data/rollout_latency024_three_games.sh all
#
# Or source and call individual functions:
#   source scripts/bash_scripts/rollout_data/rollout_latency024_three_games.sh
#   rollout_asterix_l0
#   rollout_asterix_l2
#   rollout_asterix_l4
#   rollout_atlantis_l0
#   rollout_atlantis_l2
#   rollout_atlantis_l4
#   rollout_defend_the_line_l0
#   rollout_defend_the_line_l2
#   rollout_defend_the_line_l4
#
# Common overrides:
#   MAX_EPISODES=200
#   MAX_STEPS=7200
#   NUM_ENVS=64
#   REPLAY_NUM_ENVS=64
#   DEVICE=cuda
#   FILTER_PRESET=none
#   EPISODE_RETURN_GT=<override per-task gate>
#   HF_MODEL_REPO=placeholder/paper-experiment-models
#   HF_ROLLOUT_REPO=<override one shared upload repo>
#   HF_ROLLOUT_REPO_PREFIX=placeholder
#
# Default teacher seeds are one selected small model per env/latency:
#   asterix:  l0=s11, l2=s14, l4=s14
#   atlantis: l0=s12, l2=s14, l4=s10
#   defend:   l2=s0,  l4=s0
# Defend-the-Line l0 is copied from the existing 1000ep memory-rollouts
# dataset because no l0 small-model checkpoint is uploaded in HF_MODEL_REPO.

export HF_MODEL_REPO="${HF_MODEL_REPO:-placeholder/paper-experiment-models}"
export HF_ROLLOUT_REPO="${HF_ROLLOUT_REPO:-}"
export HF_ROLLOUT_REPO_PREFIX="${HF_ROLLOUT_REPO_PREFIX:-placeholder}"
export SMALL_MODEL_CKPT_ROOT="${SMALL_MODEL_CKPT_ROOT:-/path/to/mount/placeholder/latency024_small_model_ckpts}"
export ROLLOUT_DATA_ROOT="${ROLLOUT_DATA_ROOT:-/path/to/mount/placeholder/latency024_rollout_datasets}"
export DEFEND_L0_SOURCE_REPO="${DEFEND_L0_SOURCE_REPO:-placeholder/memory-rollouts}"
export DEFEND_L0_SOURCE_CONFIG="${DEFEND_L0_SOURCE_CONFIG:-defend_the_line_fixed_latency_0_1000ep_7k2steps}"
export DEFEND_L0_SOURCE_REVISION="${DEFEND_L0_SOURCE_REVISION:-main}"

need_repo_root() {
  if [[ ! -f scripts/bash_scripts/run_sf_teacher_dataset_export.sh ]]; then
    echo "[error] run this from the placeholder repo root." >&2
    return 1
  fi
}

need_cmd() {
  local cmd="$1"
  if ! command -v "${cmd}" >/dev/null 2>&1; then
    echo "[error] missing command: ${cmd}" >&2
    return 1
  fi
}

setup_cuda_env() {
  if [[ -z "${CUDA_HOME:-}" ]]; then
    if [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/nvcc" ]]; then
      export CUDA_HOME="${CONDA_PREFIX}"
    elif [[ -d /usr/local/cuda ]]; then
      export CUDA_HOME=/usr/local/cuda
    fi
  fi

  if [[ -n "${CUDA_HOME:-}" ]]; then
    export PATH="${CUDA_HOME}/bin:${PATH}"
    export LD_LIBRARY_PATH="${CUDA_HOME}/lib:${CUDA_HOME}/lib64:${CUDA_HOME}/targets/x86_64-linux/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
  fi

  export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6;9.0;12.0}"
}

tag_for_env() {
  case "$1" in
    asterix|atlantis) printf '%s\n' "10m_fs4_1min_eval" ;;
    defend_the_line) printf '%s\n' "25m_fs4_1min_eval" ;;
    *)
      echo "[error] unknown env: $1" >&2
      return 2
      ;;
  esac
}

model_family_for_env_latency() {
  local env_name="$1"
  local latency="$2"
  local tag
  tag="$(tag_for_env "${env_name}")" || return 1

  if [[ "${latency}" == "0" ]]; then
    if [[ "${env_name}" == "defend_the_line" ]]; then
      echo "[error] no uploaded defend_the_line zero-latency small-model checkpoint was found in ${HF_MODEL_REPO}." >&2
      return 1
    fi
    printf '%s\n' "zero_latency_${tag}"
    return 0
  fi

  printf '%s\n' "latency_024_${tag}"
}

selected_seed_for_env_latency() {
  local env_name="$1"
  local latency="$2"
  local override_key

  override_key="$(printf '%s_L%s_SEED' "${env_name}" "${latency}" | tr '[:lower:]' '[:upper:]')"
  if [[ -n "${!override_key:-}" ]]; then
    printf '%s\n' "${!override_key}"
    return 0
  fi

  case "${env_name}:${latency}" in
    asterix:0) printf '%s\n' 11 ;;
    asterix:2) printf '%s\n' 14 ;;
    asterix:4) printf '%s\n' 14 ;;
    atlantis:0) printf '%s\n' 12 ;;
    atlantis:2) printf '%s\n' 14 ;;
    atlantis:4) printf '%s\n' 10 ;;
    defend_the_line:2) printf '%s\n' 0 ;;
    defend_the_line:4) printf '%s\n' 0 ;;
    defend_the_line:0)
      echo "[error] no selected defend_the_line l0 teacher; no uploaded l0 checkpoint was found." >&2
      return 1
      ;;
    *)
      echo "[error] no selected seed for ${env_name} latency ${latency}" >&2
      return 1
      ;;
  esac
}

default_episode_return_gt_for_env_latency() {
  local env_name="$1"
  local latency="$2"
  case "${env_name}:${latency}" in
    asterix:0) printf '%s\n' 6000 ;;
    asterix:2) printf '%s\n' 6000 ;;
    asterix:4) printf '%s\n' 6000 ;;
    atlantis:0) printf '%s\n' 30000 ;;
    atlantis:2) printf '%s\n' 30000 ;;
    atlantis:4) printf '%s\n' 30000 ;;
    defend_the_line:2) printf '%s\n' 29 ;;
    defend_the_line:4) printf '%s\n' 29 ;;
    defend_the_line:0)
      echo "[error] no default return gate for defend_the_line l0; no uploaded l0 teacher was found." >&2
      return 1
      ;;
    *)
      echo "[error] no default return gate for ${env_name} latency ${latency}" >&2
      return 1
      ;;
  esac
}

hf_rollout_repo_for_env() {
  local env_name="$1"
  if [[ -n "${HF_ROLLOUT_REPO:-}" ]]; then
    printf '%s\n' "${HF_ROLLOUT_REPO}"
    return 0
  fi
  printf '%s\n' "${HF_ROLLOUT_REPO_PREFIX}/${env_name}_200ep"
}

best_remote_ckpt() {
  local env_name="$1"
  local latency="$2"
  local family seed
  family="$(model_family_for_env_latency "${env_name}" "${latency}")" || return 1
  seed="$(selected_seed_for_env_latency "${env_name}" "${latency}")" || return 1

  local api_url="https://huggingface.co/api/models/${HF_MODEL_REPO}/tree/main/memory/small_model/${env_name}?recursive=true"

  curl -fsSL "${api_url}" |
    ENV_NAME="${env_name}" LATENCY="${latency}" FAMILY="${family}" SEED="${seed}" python3 -c '
import json
import os
import re
import sys

env_name = os.environ["ENV_NAME"]
latency = os.environ["LATENCY"]
family = os.environ["FAMILY"]
seed = os.environ["SEED"]

pattern = re.compile(
    r"^memory/small_model/" + re.escape(env_name) + r"/" + re.escape(family)
    + r"/train/memory_small_model:" + re.escape(env_name)
    + r":frame_stack:fixed_l" + re.escape(latency)
    + r":fs4:obs[^:]+:stride4:seed" + re.escape(seed) + r"/checkpoint_p0/"
    + r"best_.*reward_(-?\d+(?:\.\d+)?)\.pth$"
)

rows = []
for item in json.load(sys.stdin):
    path = item.get("path", "")
    match = pattern.match(path)
    if match:
        rows.append((float(match.group(1)), path))

if not rows:
    raise SystemExit(1)

rows.sort(key=lambda item: item[0])
print(rows[-1][1])
'
}

download_best_latency024_ckpt() {
  local env_name="$1"
  local latency="$2"

  need_cmd curl || return 1
  need_cmd hf || return 1

  local remote_ckpt remote_exp_dir remote_config local_root ckpt_name ckpt
  remote_ckpt="$(best_remote_ckpt "${env_name}" "${latency}")" || {
    echo "[error] no HF checkpoint found for ${env_name} fixed_l${latency} in ${HF_MODEL_REPO}" >&2
    return 1
  }
  remote_exp_dir="${remote_ckpt%/checkpoint_p0/*}"
  remote_config="${remote_exp_dir}/config.json"
  local_root="${SMALL_MODEL_CKPT_ROOT}/${env_name}/fixed_l${latency}"
  mkdir -p "${local_root}"

  echo "[download] ${HF_MODEL_REPO}:${remote_config}" >&2
  hf download "${HF_MODEL_REPO}" "${remote_config}" \
    --repo-type model \
    --local-dir "${local_root}" >/dev/null

  echo "[download] ${HF_MODEL_REPO}:${remote_ckpt}" >&2
  hf download "${HF_MODEL_REPO}" "${remote_ckpt}" \
    --repo-type model \
    --local-dir "${local_root}" >/dev/null

  ckpt_name="$(basename "${remote_ckpt}")"
  ckpt="$(find "${local_root}" -type f -path "*/checkpoint_p0/${ckpt_name}" -print -quit)"
  if [[ -z "${ckpt}" || ! -f "${ckpt}" ]]; then
    echo "[error] downloaded checkpoint missing under ${local_root}" >&2
    find "${local_root}" -type f -name "*.pth" -print >&2
    return 1
  fi

  printf '%s\n' "${ckpt}"
}

rollout_latency024_one() {
  local env_name="$1"
  local latency="$2"

  need_repo_root || return 1
  setup_cuda_env

  local ckpt config_name out_dir steps_label hf_rollout_repo
  ckpt="$(download_best_latency024_ckpt "${env_name}" "${latency}")" || return 1
  hf_rollout_repo="$(hf_rollout_repo_for_env "${env_name}")"

  case "${MAX_STEPS:-7200}" in
    7200) steps_label="7k2steps" ;;
    3600) steps_label="3k6steps" ;;
    *) steps_label="${MAX_STEPS:-7200}steps" ;;
  esac

  config_name="${env_name}_fixed_latency_${latency}_${MAX_EPISODES:-200}ep_${steps_label}"
  out_dir="${ROLLOUT_DATA_ROOT}/${config_name}"
  episode_return_gt="${EPISODE_RETURN_GT:-$(default_episode_return_gt_for_env_latency "${env_name}" "${latency}")}"

  echo "[run] ${env_name} fixed_l${latency}"
  echo "[ckpt] ${ckpt}"
  echo "[out]  ${out_dir}"
  echo "[hf]   ${hf_rollout_repo}/${config_name}"
  echo "[gate] FILTER_PRESET=${FILTER_PRESET:-none} EPISODE_RETURN_GT=${episode_return_gt}"

  MAX_EPISODES="${MAX_EPISODES:-200}" \
  MAX_STEPS="${MAX_STEPS:-7200}" \
  NUM_ENVS="${NUM_ENVS:-64}" \
  REPLAY_NUM_ENVS="${REPLAY_NUM_ENVS:-64}" \
  DEVICE="${DEVICE:-cuda}" \
  CONTEXT_WINDOW="${CONTEXT_WINDOW:-1}" \
  FILTER_PRESET="${FILTER_PRESET:-none}" \
  EPISODE_RETURN_GT="${episode_return_gt}" \
  IMAGE_WRITER_WORKERS="${IMAGE_WRITER_WORKERS:-16}" \
  HF_REPO_ID="${hf_rollout_repo}" \
  HF_CONFIG_NAME="${config_name}" \
  HF_PRIVATE="${HF_PRIVATE:-0}" \
    bash scripts/bash_scripts/run_sf_teacher_dataset_export.sh \
      "${ckpt}" \
      "${out_dir}"
}

rollout_defend_the_line_l0_from_memory_rollouts() {
  need_repo_root || return 1

  local env_name="defend_the_line"
  local latency="0"
  local steps_label config_name out_dir hf_rollout_repo max_episodes val_fraction train_episodes val_episodes

  case "${MAX_STEPS:-7200}" in
    7200) steps_label="7k2steps" ;;
    3600) steps_label="3k6steps" ;;
    *) steps_label="${MAX_STEPS:-7200}steps" ;;
  esac

  max_episodes="${MAX_EPISODES:-200}"
  val_fraction="${VAL_FRACTION:-0.01}"
  val_episodes="$(
    MAX_EPISODES="${max_episodes}" VAL_FRACTION="${val_fraction}" python3 - <<'PY'
import math
import os

max_episodes = int(os.environ["MAX_EPISODES"])
val_fraction = float(os.environ["VAL_FRACTION"])
print(min(max_episodes, max(1, int(math.floor(max_episodes * val_fraction)))))
PY
  )"
  train_episodes="$((max_episodes - val_episodes))"
  if [[ "${train_episodes}" -le 0 ]]; then
    echo "[error] MAX_EPISODES=${max_episodes} and VAL_FRACTION=${val_fraction} leave no train episodes." >&2
    return 1
  fi

  config_name="${env_name}_fixed_latency_${latency}_${max_episodes}ep_${steps_label}"
  out_dir="${ROLLOUT_DATA_ROOT}/${config_name}"
  hf_rollout_repo="$(hf_rollout_repo_for_env "${env_name}")"

  mkdir -p "${out_dir}"

  echo "[run] ${env_name} fixed_l${latency} from existing memory-rollouts dataset"
  echo "[source] ${DEFEND_L0_SOURCE_REPO}/${DEFEND_L0_SOURCE_CONFIG}@${DEFEND_L0_SOURCE_REVISION}"
  echo "[out]    ${out_dir}"
  echo "[hf]     ${hf_rollout_repo}/${config_name}"
  echo "[split]  train=${train_episodes} val=${val_episodes} VAL_FRACTION=${val_fraction}"

  SOURCE_REPO="${DEFEND_L0_SOURCE_REPO}" \
  SOURCE_CONFIG="${DEFEND_L0_SOURCE_CONFIG}" \
  SOURCE_REVISION="${DEFEND_L0_SOURCE_REVISION}" \
  DEST_REPO="${hf_rollout_repo}" \
  DEST_CONFIG="${config_name}" \
  OUTPUT_DIR="${out_dir}" \
  TRAIN_EPISODES="${train_episodes}" \
  VAL_EPISODES="${val_episodes}" \
  HF_PRIVATE="${HF_PRIVATE:-0}" \
  HF_TOKEN="${HF_TOKEN:-}" \
  HF_MAX_SHARD_SIZE="${HF_MAX_SHARD_SIZE:-500MB}" \
    python3 - <<'PY'
import json
import os
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download


def _token() -> str | None:
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN") or None


def _bool_env(name: str, default: str = "0") -> bool:
    return str(os.environ.get(name, default)).lower() in {"1", "true", "yes", "on"}


def _source_split_files(api: HfApi, *, repo: str, config: str, revision: str, split: str, token: str | None) -> list[str]:
    prefix = f"{config}/{split}-"
    files = api.list_repo_files(repo_id=repo, repo_type="dataset", revision=revision, token=token)
    matches = sorted(path for path in files if path.startswith(prefix) and path.endswith(".parquet"))
    if not matches:
        raise RuntimeError(f"no source parquet files found for {repo}/{config} split={split}")
    return matches


def _download(repo: str, revision: str, path: str, token: str | None) -> str:
    print(f"[download] {path}", flush=True)
    return hf_hub_download(
        repo_id=repo,
        repo_type="dataset",
        revision=revision,
        filename=path,
        token=token,
    )


def _select_episode_ids(
    *,
    repo: str,
    revision: str,
    files: list[str],
    budget: int,
    token: str | None,
) -> tuple[list[int], list[str]]:
    selected: list[int] = []
    seen: set[int] = set()
    downloaded: list[str] = []
    for path in files:
        local = _download(repo, revision, path, token)
        downloaded.append(local)
        table = pq.read_table(local, columns=["episode_idx"])
        for value in table.column("episode_idx").to_pylist():
            episode_idx = int(value)
            if episode_idx in seen:
                continue
            seen.add(episode_idx)
            selected.append(episode_idx)
            if len(selected) >= budget:
                print(
                    f"[select] selected {len(selected)} episodes "
                    f"first={selected[0]} last={selected[-1]}",
                    flush=True,
                )
                return selected, downloaded
    raise RuntimeError(f"source only exposed {len(selected)} unique episodes; requested {budget}")


def _write_filtered_split(
    *,
    split: str,
    selected: list[int],
    local_files: list[str],
    output_dir: Path,
) -> dict[str, int | list[str]]:
    selected_array = pa.array(selected, type=pa.int64())
    tmp_paths: list[Path] = []
    total_rows = 0
    for local in local_files:
        episode_idx_col = pq.read_table(local, columns=["episode_idx"]).column("episode_idx")
        mask = pc.is_in(episode_idx_col, value_set=selected_array)
        if not bool(pc.any(mask).as_py()):
            continue
        table = pq.read_table(local)
        filtered = table.filter(mask)
        if filtered.num_rows == 0:
            continue
        tmp_path = output_dir / f"{split}-tmp-{len(tmp_paths):05d}.parquet"
        pq.write_table(filtered, tmp_path)
        tmp_paths.append(tmp_path)
        total_rows += filtered.num_rows
        print(f"[write] {split}: {tmp_path.name} rows={filtered.num_rows}", flush=True)

    final_names: list[str] = []
    total_files = len(tmp_paths)
    if total_files == 0:
        raise RuntimeError(f"no rows written for split={split}")
    for idx, tmp_path in enumerate(tmp_paths):
        final_path = output_dir / f"{split}-{idx:05d}-of-{total_files:05d}.parquet"
        tmp_path.replace(final_path)
        final_names.append(final_path.name)
    print(f"[write] {split}: rows={total_rows} files={total_files}", flush=True)
    return {"rows": total_rows, "files": final_names}


repo = os.environ["SOURCE_REPO"]
config = os.environ["SOURCE_CONFIG"]
revision = os.environ["SOURCE_REVISION"]
dest_repo = os.environ["DEST_REPO"]
dest_config = os.environ["DEST_CONFIG"]
output_dir = Path(os.environ["OUTPUT_DIR"])
train_budget = int(os.environ["TRAIN_EPISODES"])
val_budget = int(os.environ["VAL_EPISODES"])
token = _token()
api = HfApi()

output_dir.mkdir(parents=True, exist_ok=True)
for stale in output_dir.glob("*.parquet"):
    stale.unlink()

print(f"[source] {repo}/{config}@{revision}", flush=True)
train_files = _source_split_files(api, repo=repo, config=config, revision=revision, split="train", token=token)
val_files = _source_split_files(api, repo=repo, config=config, revision=revision, split="val", token=token)

print(f"[select] train budget={train_budget}", flush=True)
train_ids, train_local_files = _select_episode_ids(
    repo=repo,
    revision=revision,
    files=train_files,
    budget=train_budget,
    token=token,
)
print(f"[select] val budget={val_budget}", flush=True)
val_ids, val_local_files = _select_episode_ids(
    repo=repo,
    revision=revision,
    files=val_files,
    budget=val_budget,
    token=token,
)

train_stats = _write_filtered_split(split="train", selected=train_ids, local_files=train_local_files, output_dir=output_dir)
val_stats = _write_filtered_split(split="val", selected=val_ids, local_files=val_local_files, output_dir=output_dir)

metadata = {
    "source_repo": repo,
    "source_config": config,
    "source_revision": revision,
    "derived_config": dest_config,
    "train_episodes": len(train_ids),
    "val_episodes": len(val_ids),
    "total_episodes": len(train_ids) + len(val_ids),
    "train_episode_idx": train_ids,
    "val_episode_idx": val_ids,
    "train_rows": int(train_stats["rows"]),
    "val_rows": int(val_stats["rows"]),
    "total_rows": int(train_stats["rows"]) + int(val_stats["rows"]),
    "train_files": train_stats["files"],
    "val_files": val_stats["files"],
}
(output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")

print(f"[upload] repo={dest_repo} path={dest_config}", flush=True)
api.create_repo(repo_id=dest_repo, repo_type="dataset", private=_bool_env("HF_PRIVATE"), exist_ok=True, token=token)
commit = api.upload_folder(
    folder_path=str(output_dir),
    path_in_repo=dest_config,
    repo_id=dest_repo,
    repo_type="dataset",
    allow_patterns=["*.parquet", "metadata.json"],
    commit_message=f"Upload {dest_config} from {config}",
    token=token,
)
metadata["commit_url"] = getattr(commit, "commit_url", None)
(output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
api.upload_file(
    path_or_fileobj=str(output_dir / "metadata.json"),
    path_in_repo=f"{dest_config}/metadata.json",
    repo_id=dest_repo,
    repo_type="dataset",
    token=token,
)
print("[upload] complete", flush=True)
print(json.dumps(metadata, indent=2, sort_keys=True))
PY
}

rollout_asterix_l0() { rollout_latency024_one asterix 0; }
rollout_asterix_l2() { rollout_latency024_one asterix 2; }
rollout_asterix_l4() { rollout_latency024_one asterix 4; }

rollout_atlantis_l0() { rollout_latency024_one atlantis 0; }
rollout_atlantis_l2() { rollout_latency024_one atlantis 2; }
rollout_atlantis_l4() { rollout_latency024_one atlantis 4; }

rollout_defend_the_line_l0() { rollout_defend_the_line_l0_from_memory_rollouts; }
rollout_defend_the_line_l2() { rollout_latency024_one defend_the_line 2; }
rollout_defend_the_line_l4() { rollout_latency024_one defend_the_line 4; }

rollout_asterix_all() {
  rollout_asterix_l0
  rollout_asterix_l2
  rollout_asterix_l4
}

rollout_atlantis_all() {
  rollout_atlantis_l0
  rollout_atlantis_l2
  rollout_atlantis_l4
}

rollout_defend_the_line_all() {
  rollout_defend_the_line_l0
  rollout_defend_the_line_l2
  rollout_defend_the_line_l4
}

rollout_all_latency024() {
  rollout_asterix_all
  rollout_atlantis_all
  rollout_defend_the_line_all
}

print_help() {
  cat <<'EOF'
Usage:
  bash scripts/bash_scripts/rollout_data/rollout_latency024_three_games.sh <target>

Targets:
  asterix-l0 | asterix-l2 | asterix-l4 | asterix | asterix-all
  atlantis-l0 | atlantis-l2 | atlantis-l4 | atlantis | atlantis-all
  defend-the-line-l0 | defend-the-line-l2 | defend-the-line-l4
  defend-the-line | defend-the-line-all
  all

Note:
  defend-the-line-l0 is derived from
  placeholder/memory-rollouts/defend_the_line_fixed_latency_0_1000ep_7k2steps
  by streaming the requested number of episodes into this repo's standard 200ep
  rollout destination.

Source mode:
  source scripts/bash_scripts/rollout_data/rollout_latency024_three_games.sh
  rollout_asterix_l0
EOF
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  case "${1:-help}" in
    asterix-l0) rollout_asterix_l0 ;;
    asterix-l2) rollout_asterix_l2 ;;
    asterix-l4) rollout_asterix_l4 ;;
    asterix|asterix-all) rollout_asterix_all ;;
    atlantis-l0) rollout_atlantis_l0 ;;
    atlantis-l2) rollout_atlantis_l2 ;;
    atlantis-l4) rollout_atlantis_l4 ;;
    atlantis|atlantis-all) rollout_atlantis_all ;;
    defend-the-line-l0|defend_the_line-l0) rollout_defend_the_line_l0 ;;
    defend-the-line-l2|defend_the_line-l2) rollout_defend_the_line_l2 ;;
    defend-the-line-l4|defend_the_line-l4) rollout_defend_the_line_l4 ;;
    defend-the-line|defend_the_line|defend-the-line-all|defend_the_line-all) rollout_defend_the_line_all ;;
    all) rollout_all_latency024 ;;
    help|-h|--help) print_help ;;
    *)
      echo "[error] unknown target: ${1}" >&2
      print_help >&2
      exit 2
      ;;
  esac
fi
