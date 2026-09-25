#!/usr/bin/env bash
set -euo pipefail

REMOTE_ROOT=/path/to/user/codex/mikasa-h8
RUN_ROOT="${REMOTE_ROOT}/outputs/starvla/runs/mikasa_intercept_grab_fast/h8-conditioned-dagger-20260923"
LAUNCHER="${REMOTE_ROOT}/scripts/tasks/mikasa_intercept_grab_fast/launch_h8_dagger_arm.sh"

launch_one() {
  local host="$1" model="$2" condition="$3" pair="$4" simulator="$5" port="$6" uv="$7"
  local output="${RUN_ROOT}/${condition}/${model}"
  ssh -o BatchMode=yes "ubuntu@${host}" bash -s -- \
    "${model}" "${condition}" "${pair}" "${simulator}" "${port}" \
    "${output}" "${uv}" "${LAUNCHER}" <<'REMOTE'
set -euo pipefail
model="$1"; condition="$2"; pair="$3"; simulator="$4"; port="$5"
output="$6"; uv="$7"; launcher="$8"
mkdir -p "$output"
if [[ -f "$output/controller.pid" ]]; then
  pid="$(cat "$output/controller.pid")"
  status="$(ps -o stat= -p "$pid" 2>/dev/null || true)"
  command="$(ps -o args= -p "$pid" 2>/dev/null || true)"
  if [[ -n "$status" && "$status" != Z* && "$command" == *"$output"* ]]; then
    printf '%s already running: %s\n' "$condition/$model" "$pid"
    exit 0
  fi
fi
nohup env UV="$uv" PYTHONUNBUFFERED=1 \
  bash "$launcher" "$model" "$condition" "$pair" "$simulator" "$port" "$output" formal \
  > "$output/controller.log" 2>&1 < /dev/null &
printf '%s\n' "$!" > "$output/controller.pid"
printf '%s started: %s\n' "$condition/$model" "$(cat "$output/controller.pid")"
REMOTE
}

launch_one 34.131.223.74 qwenoft profile-latency 0,1 6 29820 /path/to/user/.local/bin/uv
launch_one 34.131.223.74 qwenoft fixed-2 2,3 7 29822 /path/to/user/.local/bin/uv
launch_one 34.131.223.74 qwengr00t fixed-2 4,5 6 29824 /path/to/user/.local/bin/uv
launch_one 34.131.91.104 qwenpi_v3 profile-latency 0,1 6 29830 /path/to/user/miniconda3/envs/starvla_rl_games_openvla/bin/uv
launch_one 34.131.91.104 qwengr00t profile-latency 2,3 7 29832 /path/to/user/miniconda3/envs/starvla_rl_games_openvla/bin/uv
launch_one 34.131.91.104 qwenpi_v3 fixed-2 4,5 6 29834 /path/to/user/miniconda3/envs/starvla_rl_games_openvla/bin/uv
