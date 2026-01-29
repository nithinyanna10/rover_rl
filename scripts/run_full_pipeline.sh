#!/usr/bin/env bash
# Run the full RL Rover pipeline: train (full) → evaluate → inference.
# Usage: ./scripts/run_full_pipeline.sh   (from project root)

set -e
cd "$(dirname "$0")/.."

echo "=========================================="
echo "1/3 Training PPO (full run, 500k steps)"
echo "=========================================="
python -m rl.train --config configs/train_ppo.yaml

# Use the most recent run directory
LATEST_RUN=$(ls -td runs/202* 2>/dev/null | head -1)
if [ -z "$LATEST_RUN" ]; then
  echo "No run directory found under runs/. Aborting."
  exit 1
fi

MODEL_PATH="${LATEST_RUN}/checkpoints/final_model.zip"
if [ ! -f "$MODEL_PATH" ]; then
  MODEL_PATH="${LATEST_RUN}/checkpoints/best_model.zip"
fi
if [ ! -f "$MODEL_PATH" ]; then
  echo "No checkpoint found at ${LATEST_RUN}/checkpoints/. Aborting."
  exit 1
fi

echo ""
echo "Using model: $MODEL_PATH"
echo ""

echo "=========================================="
echo "2/3 Evaluating on 10 maps (20 episodes each)"
echo "=========================================="
python -m rl.evaluate --config configs/eval.yaml --model-path "$MODEL_PATH"

echo ""
echo "=========================================="
echo "3/3 Running inference with visualization"
echo "=========================================="
mkdir -p telemetry_logs
python -m rl.inference \
  --config configs/eval.yaml \
  --model-path "$MODEL_PATH" \
  --telemetry-path telemetry_logs/inference.jsonl

echo ""
echo "=========================================="
echo "Full pipeline complete."
echo "=========================================="
echo "To view telemetry live, run:"
echo "  streamlit run telemetry/streamlit_app.py -- --log-path telemetry_logs/inference.jsonl"
echo ""
