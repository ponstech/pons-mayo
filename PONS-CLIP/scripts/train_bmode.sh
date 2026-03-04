#!/usr/bin/env bash
# Train PONS-CLIP with B-mode only.
# Usage: ./scripts/train_bmode.sh
# Folder-based: DATA_DIR=data/fold1 ./scripts/train_bmode.sh

set -e
cd "$(dirname "$0")/.."

python -m pons_clip.train \
  --config configs/bmode_only.yaml \
  --data_dir "${DATA_DIR:-data/fold1}" \
  --output_dir "${OUTPUT_DIR:-./outputs_bmode}"
