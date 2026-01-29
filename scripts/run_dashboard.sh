#!/usr/bin/env bash
set -e

LOG_PATH="${1:-telemetry_logs/inference.jsonl}"

streamlit run telemetry/streamlit_app.py -- --log-path "$LOG_PATH"

