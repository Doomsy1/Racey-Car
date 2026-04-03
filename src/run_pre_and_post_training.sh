#!/usr/bin/env bash
set -euo pipefail

# ---------------------------
# User settings (edit here)
# ---------------------------
PYTHON_BIN="python3"
CHECKPOINT_DIR="checkpoints"
TENSORBOARD_DIR="runs"
CONFIG_PATH="models/track_config.yaml"

# Shared training settings
NUM_ENVS=16
TOTAL_TIMESTEPS=1000000
# Occupancy-map resolution scale (0 < scale <= 1)
OBS_SCALE=0.25
FRAME_STACK=4
BUFFER_SIZE=300000
LEARNING_STARTS=50000
TRAIN_FREQ=1
GRADIENT_STEPS=1
BATCH_SIZE=128
THROTTLE_BIAS=0.0
THROTTLE_BIAS_ANNEAL_FRAC=0.0
ENT_START_MULT=4.0
ENT_ANNEAL_FRAC=0.9

# Pre-training settings
PRE_RUN_NAME="pre_sac_stage"
PRE_LEARNING_RATE=0.0003
SHARED_TRACK_SEED=1

# Post-training settings
POST_RUN_NAME="post_sac_stage"
POST_LEARNING_RATE=0.0001

# If true, post-training will load the model produced by pre-training.
# If false, set POST_MODEL_PATH manually.
USE_PRETRAIN_OUTPUT_FOR_POST=true
POST_MODEL_PATH="checkpoints/some_existing_run/sac_racey_final"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

echo "[1/2] Starting pre-training run: ${PRE_RUN_NAME}"
"${PYTHON_BIN}" train_sac_pre.py \
  --config "${CONFIG_PATH}" \
  --run-name "${PRE_RUN_NAME}" \
  --checkpoint-dir "${CHECKPOINT_DIR}" \
  --tensorboard-log "${TENSORBOARD_DIR}" \
  --num-envs "${NUM_ENVS}" \
  --total-timesteps "${TOTAL_TIMESTEPS}" \
  --learning-rate "${PRE_LEARNING_RATE}" \
  --obs-scale "${OBS_SCALE}" \
  --frame-stack "${FRAME_STACK}" \
  --buffer-size "${BUFFER_SIZE}" \
  --learning-starts "${LEARNING_STARTS}" \
  --train-freq "${TRAIN_FREQ}" \
  --gradient-steps "${GRADIENT_STEPS}" \
  --batch-size "${BATCH_SIZE}" \
  --throttle-bias "${THROTTLE_BIAS}" \
  --throttle-bias-anneal-frac "${THROTTLE_BIAS_ANNEAL_FRAC}" \
  --ent-start-mult "${ENT_START_MULT}" \
  --ent-anneal-frac "${ENT_ANNEAL_FRAC}" \
  --shared-track-seed "${SHARED_TRACK_SEED}"

if [[ "${USE_PRETRAIN_OUTPUT_FOR_POST}" == "true" ]]; then
  POST_MODEL_PATH="${CHECKPOINT_DIR}/${PRE_RUN_NAME}/sac_racey_pre_final"
fi

echo "[2/2] Starting post-training run: ${POST_RUN_NAME}"
echo "Using model: ${POST_MODEL_PATH}"
"${PYTHON_BIN}" train_sac_post.py \
  --model "${POST_MODEL_PATH}" \
  --config "${CONFIG_PATH}" \
  --run-name "${POST_RUN_NAME}" \
  --checkpoint-dir "${CHECKPOINT_DIR}" \
  --tensorboard-log "${TENSORBOARD_DIR}" \
  --num-envs "${NUM_ENVS}" \
  --total-timesteps "${TOTAL_TIMESTEPS}" \
  --learning-rate "${POST_LEARNING_RATE}" \
  --obs-scale "${OBS_SCALE}" \
  --frame-stack "${FRAME_STACK}" \
  --buffer-size "${BUFFER_SIZE}" \
  --learning-starts "${LEARNING_STARTS}" \
  --train-freq "${TRAIN_FREQ}" \
  --gradient-steps "${GRADIENT_STEPS}" \
  --batch-size "${BATCH_SIZE}" \
  --throttle-bias "${THROTTLE_BIAS}" \
  --throttle-bias-anneal-frac "${THROTTLE_BIAS_ANNEAL_FRAC}" \
  --ent-start-mult "${ENT_START_MULT}" \
  --ent-anneal-frac "${ENT_ANNEAL_FRAC}"

echo "Done."
