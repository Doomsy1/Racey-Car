#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------
# Staged training pipeline:
# 1) Off-track control
# 2) Speed development
# 3) Full reward shaping ("the rest")
# ------------------------------------------

PYTHON_BIN="python3"
CONFIG_PATH="models/track_config.yaml"
CHECKPOINT_DIR="checkpoints"
TENSORBOARD_DIR="runs"

# Shared training settings (defaults — all overridable via CLI flags below)
NUM_ENVS=16
OBS_SCALE=0.25
FRAME_STACK=12
BUFFER_SIZE=100000   # reduced from 300K: ~3.2 GB vs ~9.8 GB, ample for 16-env training
LEARNING_STARTS=5000 # reduced from 50K: random warmup was wasting 12.5% of each stage
TRAIN_FREQ=1
GRADIENT_STEPS=1
BATCH_SIZE=128
CHECKPOINT_FREQUENCY=100000
ENT_START_MULT=2.0 # used as a divisor: target=-2/2=-1.0 (exploratory early, but not chaotically wide)
ENT_ANNEAL_FRAC=0.9

# Per-stage schedule
STAGE1_TIMESTEPS=1000000
STAGE2_TIMESTEPS=600000
STAGE3_TIMESTEPS=800000

RUN_VERSION="v2"

# ---------------------------------------------------------------------------
# Parse CLI overrides  (must come after defaults so flags win over defaults,
# and before stage-name derivation so --run-version takes effect)
# Usage: ./run_staged_training.sh [--frame-stack N] [--num-envs N] [...]
#   --frame-stack N     temporal memory window (default: 4)
#   --num-envs N
#   --obs-scale F
#   --buffer-size N
#   --run-version STR
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
  --frame-stack)
    FRAME_STACK="$2"
    shift 2
    ;;
  --num-envs)
    NUM_ENVS="$2"
    shift 2
    ;;
  --obs-scale)
    OBS_SCALE="$2"
    shift 2
    ;;
  --buffer-size)
    BUFFER_SIZE="$2"
    shift 2
    ;;
  --run-version)
    RUN_VERSION="$2"
    shift 2
    ;;
  *)
    echo "Unknown argument: $1" >&2
    exit 1
    ;;
  esac
done

STAGE1_RUN_NAME="${RUN_VERSION}_stage1_offtrack"
STAGE2_RUN_NAME="${RUN_VERSION}_stage2_speed"
STAGE3_RUN_NAME="${RUN_VERSION}_stage3_full"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# ---------------------------------------------------------------------------
# Temporary directories for inter-stage model hand-off.
# These hold only the model.zip + replay_buffer.pkl needed to start the next
# stage; they are cleaned up automatically when the script exits.
# ---------------------------------------------------------------------------
STAGE1_TMPDIR="$(mktemp -d)"
STAGE2_TMPDIR="$(mktemp -d)"
trap 'rm -rf "${STAGE1_TMPDIR}" "${STAGE2_TMPDIR}"' EXIT

echo "[1/3] Stage 1: Off-track control (${STAGE1_RUN_NAME})"
# throttle-bias=0.3 breaks the sit-still local optimum by forcing forward motion early.
# It anneals to 0 over the first 30% of stage 1 (120K steps) so the agent learns
# independent throttle control once it has discovered how to stay on track.
# --checkpoint-frequency 0 disables step-level checkpoints (only final model saved).
"${PYTHON_BIN}" train_sac.py \
  --config "${CONFIG_PATH}" \
  --run-name "${STAGE1_RUN_NAME}" \
  --checkpoint-dir "${CHECKPOINT_DIR}" \
  --tensorboard-log "${TENSORBOARD_DIR}" \
  --checkpoint-frequency 0 \
  --output-model "${STAGE1_TMPDIR}" \
  --num-envs "${NUM_ENVS}" \
  --total-timesteps "${STAGE1_TIMESTEPS}" \
  --learning-rate 0.0003 \
  --obs-scale "${OBS_SCALE}" \
  --frame-stack "${FRAME_STACK}" \
  --buffer-size "${BUFFER_SIZE}" \
  --learning-starts "${LEARNING_STARTS}" \
  --train-freq "${TRAIN_FREQ}" \
  --gradient-steps "${GRADIENT_STEPS}" \
  --batch-size "${BATCH_SIZE}" \
  --throttle-bias 0.3 \
  --throttle-bias-anneal-frac 0.3 \
  --ent-start-mult "${ENT_START_MULT}" \
  --ent-anneal-frac "${ENT_ANNEAL_FRAC}" \
  --reward-stage-fracs "1.0" \
  --stage-progress-weights "1.0" \
  --stage-speed-weights "0.05" \
  --stage-steer-penalties "0.02" \
  --stage-gate-rewards "0.0" \
  --stage-off-track-penalties "-2.0"

echo "[2/3] Stage 2: Speed focus (${STAGE2_RUN_NAME})"
# Progress weight restored to 1.0 (forward direction still matters).
# Speed weight raised to 0.8: combined gradient is 1.8x vs Stage 1's 1.05x — a 71% increase.
# Off-track penalty restored to -2.0: speed pursuit must not come at the cost of leaving the track.
# Throttle bias=0.15 prevents speed regression from Stage 1; anneals away over first 50% (200K steps)
# so the policy learns to self-generate high throttle rather than relying on the floor permanently.
# --checkpoint-frequency 0 disables step-level checkpoints (only final model saved to tempfile).
"${PYTHON_BIN}" train_sac_post.py \
  --model "${STAGE1_TMPDIR}" \
  --config "${CONFIG_PATH}" \
  --run-name "${STAGE2_RUN_NAME}" \
  --checkpoint-dir "${CHECKPOINT_DIR}" \
  --tensorboard-log "${TENSORBOARD_DIR}" \
  --checkpoint-frequency 0 \
  --output-model "${STAGE2_TMPDIR}" \
  --num-envs "${NUM_ENVS}" \
  --total-timesteps "${STAGE2_TIMESTEPS}" \
  --learning-rate 0.0001 \
  --obs-scale "${OBS_SCALE}" \
  --frame-stack "${FRAME_STACK}" \
  --buffer-size "${BUFFER_SIZE}" \
  --learning-starts "${LEARNING_STARTS}" \
  --train-freq "${TRAIN_FREQ}" \
  --gradient-steps "${GRADIENT_STEPS}" \
  --batch-size "${BATCH_SIZE}" \
  --throttle-bias 0.15 \
  --throttle-bias-anneal-frac 0.5 \
  --ent-start-mult "${ENT_START_MULT}" \
  --ent-anneal-frac "${ENT_ANNEAL_FRAC}" \
  --reward-stage-fracs "1.0" \
  --stage-progress-weights "1.0" \
  --stage-speed-weights "0.8" \
  --stage-steer-penalties "0.01" \
  --stage-gate-rewards "0.0" \
  --stage-off-track-penalties "-2.0"

echo "[3/3] Stage 3: Full reward terms (${STAGE3_RUN_NAME})"
# Stage 3 saves to permanent checkpoint_dir with step checkpoints enabled.
"${PYTHON_BIN}" train_sac_post.py \
  --model "${STAGE2_TMPDIR}" \
  --config "${CONFIG_PATH}" \
  --run-name "${STAGE3_RUN_NAME}" \
  --checkpoint-dir "${CHECKPOINT_DIR}" \
  --tensorboard-log "${TENSORBOARD_DIR}" \
  --checkpoint-frequency "${CHECKPOINT_FREQUENCY}" \
  --num-envs "${NUM_ENVS}" \
  --total-timesteps "${STAGE3_TIMESTEPS}" \
  --learning-rate 0.0001 \
  --obs-scale "${OBS_SCALE}" \
  --frame-stack "${FRAME_STACK}" \
  --buffer-size "${BUFFER_SIZE}" \
  --learning-starts "${LEARNING_STARTS}" \
  --train-freq "${TRAIN_FREQ}" \
  --gradient-steps "${GRADIENT_STEPS}" \
  --batch-size "${BATCH_SIZE}" \
  --throttle-bias 0.0 \
  --throttle-bias-anneal-frac 0.0 \
  --ent-start-mult "${ENT_START_MULT}" \
  --ent-anneal-frac "${ENT_ANNEAL_FRAC}" \
  --reward-stage-fracs "1.0" \
  --stage-progress-weights "1.0" \
  --stage-speed-weights "0.3" \
  --stage-steer-penalties "0.02" \
  --stage-gate-rewards "1.0" \
  --stage-off-track-penalties "-1.0"

FINAL_MODEL="${CHECKPOINT_DIR}/${STAGE3_RUN_NAME}/sac_racey_post_final"
echo "Done. Final model: ${FINAL_MODEL}"
