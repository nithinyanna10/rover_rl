#!/usr/bin/env bash
set -e

python -m rl.train --config configs/train_ppo.yaml "$@"

