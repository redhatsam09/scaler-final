#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "[1/5] Syntax compile check"
python -m compileall src server training inference.py world_modeling_demo.py

echo "[2/5] Environment demo smoke test"
python world_modeling_demo.py > /tmp/world_modeling_demo.log

echo "[3/5] Inference smoke test (local backend)"
INFERENCE_BACKEND=local INFERENCE_SEED=2026 python inference.py > /tmp/inference.log

echo "[4/5] Regenerate policy-search artifacts"
python training/trl_sft_training.py

echo "[5/5] Regenerate evaluation artifacts"
python training/evaluate_reward_improvement.py

echo "Local checks completed successfully."
