#!/usr/bin/env bash
set -euo pipefail

calib_dir="/app/calibration"
calib_file="${calib_dir}/calibration.json"
model_dir="/tmp/models"

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"
mkdir -p "${MPLCONFIGDIR}"

mkdir -p "${calib_dir}"
mkdir -p "${model_dir}"

if [[ ! -w "${calib_dir}" ]]; then
  echo "[entrypoint] Error: ${calib_dir} is not writable by UID=$(id -u) GID=$(id -g)." >&2
  echo "[entrypoint] Fix on host: sudo chown -R $(id -u):$(id -g) calibration" >&2
  exit 1
fi

has_camera_flag=false
for arg in "$@"; do
  if [[ "${arg}" == "--camera" ]]; then
    has_camera_flag=true
    break
  fi
done

has_ui_flag=false
for arg in "$@"; do
  if [[ "${arg}" == "--show-ui" || "${arg}" == "--no-show-ui" ]]; then
    has_ui_flag=true
    break
  fi
done

ui_flag=""
if [[ "${has_ui_flag}" == "false" && -s "${calib_file}" ]]; then
  ui_flag="--no-show-ui"
fi

camera_arg=""
if [[ "${has_camera_flag}" == "false" ]]; then
  if [[ -n "${CAMERA_INDEX:-}" ]]; then
    camera_arg="--camera ${CAMERA_INDEX}"
  else
    first_camera="$(ls -1 /dev/video* 2>/dev/null | head -n 1 || true)"
    if [[ -n "${first_camera}" ]]; then
      camera_idx="${first_camera#/dev/video}"
      camera_arg="--camera ${camera_idx}"
    else
      echo "[entrypoint] Warning: no /dev/video* devices found; camera capture will fail." >&2
    fi
  fi
fi

exec python -m src.app ${ui_flag} ${camera_arg} --controller "${CONTROLLER:-off}" "$@"
