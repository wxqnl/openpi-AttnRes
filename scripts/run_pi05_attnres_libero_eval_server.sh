#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <standalone-checkpoint-dir> [port]" >&2
  exit 2
fi

ckpt_dir="$1"
port="${2:-8000}"

export PYTHONPATH="${PYTHONPATH:-$(pwd)/src}"

exec ./.venv/bin/python scripts/serve_policy.py \
  --port "${port}" \
  policy:checkpoint \
  --policy.config pi05_attnres_libero \
  --policy.dir "${ckpt_dir}"
