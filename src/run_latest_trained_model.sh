#!/usr/bin/env bash
set -euo pipefail

# ---------------------------
# User settings (edit here)
# ---------------------------
PYTHON_BIN="python3"
CHECKPOINT_DIR="checkpoints"
CONFIG_PATH="models/track_config_eval_figure8.yaml"
OBS_SCALE=0.25
FRAME_STACK=4
ALGO="sac"
DETERMINISTIC=false
TERMINATE_OFF_TRACK=false
MAX_EPISODE_DURATION=""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

if [[ ! -d "${CHECKPOINT_DIR}" ]]; then
  echo "Checkpoint directory not found: ${CHECKPOINT_DIR}"
  exit 1
fi

# Find the most recently modified readable model folder.
LATEST_MODEL_DIR="$(
  find "${CHECKPOINT_DIR}" -type f -name metadata.json -print0 \
    | while IFS= read -r -d '' meta; do
        dir="$(dirname "${meta}")"
        if [[ -f "${dir}/policy_state.pt" ]]; then
          printf '%s|%s\n' "$(stat -f %m "${meta}")" "${dir}"
        fi
      done \
    | sort -t'|' -nr \
    | head -n1 \
    | cut -d'|' -f2-
)"

if [[ -z "${LATEST_MODEL_DIR}" ]]; then
  echo "No readable model found under ${CHECKPOINT_DIR} (expected metadata.json + policy_state.pt)."
  exit 1
fi

echo "Running latest model: ${LATEST_MODEL_DIR}"

CMD=(
  "${PYTHON_BIN}" run_trained_agent.py
  --model "${LATEST_MODEL_DIR}"
  --config "${CONFIG_PATH}"
  --obs-scale "${OBS_SCALE}"
  --frame-stack "${FRAME_STACK}"
  --algo "${ALGO}"
)

if [[ "${DETERMINISTIC}" == "true" ]]; then
  CMD+=(--deterministic)
fi

if [[ "${TERMINATE_OFF_TRACK}" == "true" ]]; then
  CMD+=(--terminate-off-track)
fi

if [[ -n "${MAX_EPISODE_DURATION}" ]]; then
  CMD+=(--max-episode-duration "${MAX_EPISODE_DURATION}")
fi

"${CMD[@]}"
