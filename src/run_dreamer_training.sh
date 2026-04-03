#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="python3"
CONFIG_PATH="models/track_config.yaml"
DREAMER_CONFIG_PATH="models/dreamer_medium.yaml"
CHECKPOINT_DIR="dreamerv3-checkpoints"
TENSORBOARD_DIR="dreamerv3-runs"

NUM_ENVS=8
TOTAL_TIMESTEPS=50000
OBS_SCALE=0.25
BUFFER_CAPACITY=1000
LEARNING_STARTS=5000
TRAIN_FREQ=2
BATCH_SIZE=16
SEQ_LEN=16
CHECKPOINT_FREQUENCY=0
ACTOR_CHECKPOINT_FREQUENCY=0
FULL_CHECKPOINT_FREQUENCY=0
THROTTLE_BIAS=0.3
THROTTLE_BIAS_ANNEAL_FRAC=0.3
CHECKPOINT_MIN_STEP=0
TRAIN_LOG_FREQUENCY=50
WORLD_MODEL_LR=1e-4
ACTOR_LR=3e-5
CRITIC_LR=3e-5
NO_COMPILE=0
EXPERT_PREFILL_STEPS=10000
EXPERT_KP=2.0
EXPERT_KD=0.5
EXPERT_THROTTLE=0.5
WM_PRETRAIN_UPDATES=500

RUN_NAME="dreamerv3_$(date +%Y%m%d_%H%M%S)"

while [[ $# -gt 0 ]]; do
  case "$1" in
  --num-envs)
    NUM_ENVS="$2"
    shift 2
    ;;
  --total-timesteps)
    TOTAL_TIMESTEPS="$2"
    shift 2
    ;;
  --obs-scale)
    OBS_SCALE="$2"
    shift 2
    ;;
  --dreamer-config)
    DREAMER_CONFIG_PATH="$2"
    shift 2
    ;;
  --run-name)
    RUN_NAME="$2"
    shift 2
    ;;
  --buffer-capacity)
    BUFFER_CAPACITY="$2"
    shift 2
    ;;
  --learning-starts)
    LEARNING_STARTS="$2"
    shift 2
    ;;
  --train-freq)
    TRAIN_FREQ="$2"
    shift 2
    ;;
  --batch-size)
    BATCH_SIZE="$2"
    shift 2
    ;;
  --seq-len)
    SEQ_LEN="$2"
    shift 2
    ;;
  --checkpoint-freq)
    CHECKPOINT_FREQUENCY="$2"
    shift 2
    ;;
  --actor-checkpoint-frequency)
    ACTOR_CHECKPOINT_FREQUENCY="$2"
    shift 2
    ;;
  --full-checkpoint-frequency)
    FULL_CHECKPOINT_FREQUENCY="$2"
    shift 2
    ;;
  --checkpoint-min-step)
    CHECKPOINT_MIN_STEP="$2"
    shift 2
    ;;
  --train-log-frequency)
    TRAIN_LOG_FREQUENCY="$2"
    shift 2
    ;;
  --world-model-lr)
    WORLD_MODEL_LR="$2"
    shift 2
    ;;
  --actor-lr)
    ACTOR_LR="$2"
    shift 2
    ;;
  --critic-lr)
    CRITIC_LR="$2"
    shift 2
    ;;
  --no-compile)
    NO_COMPILE=1
    shift 1
    ;;
  --expert-prefill-steps)
    EXPERT_PREFILL_STEPS="$2"
    shift 2
    ;;
  --wm-pretrain-updates)
    WM_PRETRAIN_UPDATES="$2"
    shift 2
    ;;
  *)
    echo "Unknown argument: $1" >&2
    exit 1
    ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

echo "Starting DreamerV3 training: ${RUN_NAME} (preset=${DREAMER_CONFIG_PATH})"
"${PYTHON_BIN}" train_dreamer.py \
  --config "${CONFIG_PATH}" \
  --dreamer-config "${DREAMER_CONFIG_PATH}" \
  --run-name "${RUN_NAME}" \
  --checkpoint-dir "${CHECKPOINT_DIR}" \
  --tensorboard-log "${TENSORBOARD_DIR}" \
  --checkpoint-frequency "${CHECKPOINT_FREQUENCY}" \
  --actor-checkpoint-frequency "${ACTOR_CHECKPOINT_FREQUENCY}" \
  --full-checkpoint-frequency "${FULL_CHECKPOINT_FREQUENCY}" \
  --checkpoint-min-step "${CHECKPOINT_MIN_STEP}" \
  --train-log-frequency "${TRAIN_LOG_FREQUENCY}" \
  --num-envs "${NUM_ENVS}" \
  --total-timesteps "${TOTAL_TIMESTEPS}" \
  --obs-scale "${OBS_SCALE}" \
  --buffer-capacity "${BUFFER_CAPACITY}" \
  --learning-starts "${LEARNING_STARTS}" \
  --train-freq "${TRAIN_FREQ}" \
  --batch-size "${BATCH_SIZE}" \
  --seq-len "${SEQ_LEN}" \
  --world-model-lr "${WORLD_MODEL_LR}" \
  --actor-lr "${ACTOR_LR}" \
  --critic-lr "${CRITIC_LR}" \
  --throttle-bias "${THROTTLE_BIAS}" \
  --throttle-bias-anneal-frac "${THROTTLE_BIAS_ANNEAL_FRAC}" \
  --expert-prefill-steps "${EXPERT_PREFILL_STEPS}" \
  --expert-kp "${EXPERT_KP}" \
  --expert-kd "${EXPERT_KD}" \
  --expert-throttle "${EXPERT_THROTTLE}" \
  --wm-pretrain-updates "${WM_PRETRAIN_UPDATES}" \
  $([[ "${NO_COMPILE}" -eq 1 ]] && printf '%s' "--no-compile")

echo "Done. Checkpoints in: ${CHECKPOINT_DIR}/${RUN_NAME}"
