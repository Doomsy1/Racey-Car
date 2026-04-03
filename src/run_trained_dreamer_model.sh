#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="python3"
CHECKPOINT_DIR="dreamerv3-checkpoints"
CONFIG_PATH="models/track_config_eval_figure8.yaml"
OBS_SCALE=0.25
DETERMINISTIC=false
TERMINATE_OFF_TRACK=false
MAX_EPISODE_DURATION=""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

if [[ ! -d "${CHECKPOINT_DIR}" ]]; then
  echo "Checkpoint directory not found: ${CHECKPOINT_DIR}"
  exit 1
fi

# Find the most recently modified model folder containing metadata.json
# and either actor_only.pt or full_agent.pt.
LATEST_MODEL_DIR="$(
  find "${CHECKPOINT_DIR}" -type f -name metadata.json -print0 \
    | while IFS= read -r -d '' meta; do
        dir="$(dirname "${meta}")"
        if [[ -f "${dir}/actor_only.pt" || -f "${dir}/full_agent.pt" ]]; then
          printf '%s|%s\n' "$(stat -f %m "${meta}")" "${dir}"
        fi
      done \
    | sort -t'|' -nr \
    | head -n1 \
    | cut -d'|' -f2-
)"

if [[ -z "${LATEST_MODEL_DIR}" ]]; then
  echo "No DreamerV3 model found under ${CHECKPOINT_DIR}."
  echo "Expected metadata.json + (actor_only.pt or full_agent.pt)."
  exit 1
fi

echo "Running latest model: ${LATEST_MODEL_DIR}"

CMD=(
  "${PYTHON_BIN}" run_trained_dreamer.py
  --model "${LATEST_MODEL_DIR}"
  --config "${CONFIG_PATH}"
  --obs-scale "${OBS_SCALE}"
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
