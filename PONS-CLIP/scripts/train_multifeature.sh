#!/usr/bin/env bash
# Train PONS-CLIP with multi-feature fusion (b-mode + enhanced + improved).
# Usage: DATA_DIR=data/fold1 ./scripts/train_multifeature.sh

set -e
cd "$(dirname "$0")/.."

python -m pons_clip.train \
  --config configs/multifeature.yaml \
  --data_dir "${DATA_DIR:-data/fold1}" \
  --output_dir "${OUTPUT_DIR:-./outputs_multifeature}"
