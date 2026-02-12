#!/usr/bin/env bash
set -euo pipefail

calib_dir="/app/calibration"
calib_file="${calib_dir}/calibration.json"

mkdir -p "${calib_dir}"

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

exec python -m src.app ${ui_flag} --controller "${CONTROLLER:-off}" "$@"
