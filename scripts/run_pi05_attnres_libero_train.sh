#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

: "${CUDA_VISIBLE_DEVICES:=2,3}"
: "${NPROC_PER_NODE:=2}"
: "${EXP_NAME:=pi05_attnres_libero_ramp30k_b32}"
: "${PYTORCH_WEIGHT_PATH:=./models/pi05_libero_pytorch}"

export CUDA_VISIBLE_DEVICES
export PYTHONPATH="${PYTHONPATH:-$(pwd)/src}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

exec ./.venv/bin/python -m torch.distributed.run \
  --standalone \
  --nnodes=1 \
  --nproc_per_node="${NPROC_PER_NODE}" \
  scripts/train_pytorch_pi05_attnres.py pi05_attnres_libero \
  --exp-name "${EXP_NAME}" \
  --pytorch-weight-path "${PYTORCH_WEIGHT_PATH}" \
  --fsdp-devices "${NPROC_PER_NODE}" \
  "$@"
