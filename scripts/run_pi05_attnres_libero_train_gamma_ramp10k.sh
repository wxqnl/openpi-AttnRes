#!/usr/bin/env bash
# Portable launcher for the best gamma_ramp10k AttnRes recipe (see docs/PI05_ATTNRES_LIBERO.md).
#
# Key choices vs. the in-config defaults:
#   - gamma ramp_steps = 10000 (not 30000): the 10k-ramp variant is empirically stronger
#     on LIBERO than the full 30k ramp.
#   - adapter_rank = 256, num_blocks = 9 (matches the default config).
#   - random init for AttnRes (not pretrained / sidecar load).
#
# Override knobs via env vars:
#   CUDA_VISIBLE_DEVICES  GPUs to use (default 0,1)
#   NPROC_PER_NODE        FSDP world size (default 2)
#   EXP_NAME              checkpoint subdir name
#   PYTORCH_WEIGHT_PATH   base pi0.5 ckpt prepared by scripts/prepare_pi05_base_pytorch.py
#   NUM_TRAIN_STEPS       total optimizer steps (default 30000)
#   BATCH_SIZE            global batch size (default 32)
#   GAMMA_RAMP_STEPS      AttnRes gamma 0->1 ramp length (default 10000)
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0,1 scripts/run_pi05_attnres_libero_train_gamma_ramp10k.sh \
#       --no-wandb-enabled --overwrite

set -euo pipefail

cd "$(dirname "$0")/.."

: "${CUDA_VISIBLE_DEVICES:=0,1}"
: "${NPROC_PER_NODE:=2}"
: "${EXP_NAME:=pi05_attnres_gamma_ramp10k_b32_fsdp_2gpu}"
: "${PYTORCH_WEIGHT_PATH:=./models/pi05_libero_pytorch}"
: "${NUM_TRAIN_STEPS:=30000}"
: "${BATCH_SIZE:=32}"
: "${GAMMA_RAMP_STEPS:=10000}"

LOCK_FILE="/tmp/openpi_${EXP_NAME}.lock"
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  echo "$EXP_NAME is already running; skip duplicate launch."
  exit 0
fi

export CUDA_VISIBLE_DEVICES
export PYTHONPATH="${PYTHONPATH:-$(pwd)/src}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# AttnRes knobs that mirror the dataclass defaults but are set explicitly here
# for reproducibility — env vars override the dataclass when set.
export OPENPI_ATTNRES_INIT=random
export OPENPI_ATTNRES_TRAINABLE=1
export OPENPI_ATTNRES_GAMMA_SCHEDULE=1
export OPENPI_ATTNRES_GAMMA_START=0
export OPENPI_ATTNRES_GAMMA_END=1
export OPENPI_ATTNRES_GAMMA_RAMP_STEPS="${GAMMA_RAMP_STEPS}"
export OPENPI_ATTNRES_ADAPTER_RANK=256
export OPENPI_ATTNRES_NUM_BLOCKS=9
unset OPENPI_ATTNRES_STATE_PATH
unset LEROBOT_HOME

exec ./.venv/bin/python -m torch.distributed.run \
  --standalone \
  --nnodes=1 \
  --nproc_per_node="${NPROC_PER_NODE}" \
  scripts/train_pytorch_pi05_attnres.py pi05_attnres_libero \
  --exp-name "${EXP_NAME}" \
  --pytorch-weight-path "${PYTORCH_WEIGHT_PATH}" \
  --batch-size "${BATCH_SIZE}" \
  --num-train-steps "${NUM_TRAIN_STEPS}" \
  --save-interval 1000 \
  --log-interval 100 \
  --num-workers 2 \
  --fsdp-devices "${NPROC_PER_NODE}" \
  --model.attnres-init random \
  --model.attnres-trainable \
  --model.attnres-gamma-schedule \
  --model.attnres-gamma-start 0 \
  --model.attnres-gamma-end 1 \
  --model.attnres-gamma-ramp-steps "${GAMMA_RAMP_STEPS}" \
  "$@"
