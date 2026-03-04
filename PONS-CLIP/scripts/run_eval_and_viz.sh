#!/usr/bin/env bash
# Run evaluation (embeddings + keyword distances) and visualization for a checkpoint.
# Usage:
#   EVAL_PREFIX=bmode_only CKPT=./outputs_bmode/pons_clip_final.pt ./scripts/run_eval_and_viz.sh
#   EVAL_PREFIX=multifeature CKPT=./outputs_multifeature/pons_clip_final.pt USE_MULTIFEATURE=1 ./scripts/run_eval_and_viz.sh

set -e
cd "$(dirname "$0")/.."

CKPT="${CKPT:?Set CKPT to checkpoint path}"
DATA_DIR="${DATA_DIR:-data/fold1}"
EVAL_PREFIX="${EVAL_PREFIX:-}"
USE_MULTIFEATURE="${USE_MULTIFEATURE:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-./eval_outputs}"
PLOTS_DIR="${PLOTS_DIR:-./plots}"

MF_FLAG=""
if [[ "$USE_MULTIFEATURE" == "1" ]]; then
  MF_FLAG="--use_multifeature"
fi

python -m pons_clip.evaluate \
  --checkpoint "$CKPT" \
  --data_dir "$DATA_DIR" \
  --output_dir "$OUTPUT_DIR" \
  $MF_FLAG

python -m pons_clip.visualize \
  --embeddings_dir "$OUTPUT_DIR" \
  --output_dir "$PLOTS_DIR" \
  --method both \
  --prefix "$EVAL_PREFIX"

echo "Eval outputs: $OUTPUT_DIR"
echo "Plots: $PLOTS_DIR"
