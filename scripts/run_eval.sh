#!/usr/bin/env bash
set -e

if [ -z "$1" ]; then
  echo "Usage: $0 MODEL_PATH"
  exit 1
fi

MODEL_PATH="$1"
shift

python -m rl.evaluate --config configs/eval.yaml --model-path "$MODEL_PATH" "$@"

