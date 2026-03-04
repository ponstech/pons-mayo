#!/usr/bin/env bash
# Run text-prompt retrieval with both B-mode and multi-feature checkpoints for comparison.
# Usage:
#   PROMPT="malignant ultrasound image with asymmetric lesion" \
#   CKPT_BMODE=./outputs_bmode/pons_clip_final.pt \
#   CKPT_MF=./outputs_multifeature/pons_clip_final.pt \
#   DATA_DIR=data/fold1 \
#   ./scripts/retrieve_compare.sh

set -e
cd "$(dirname "$0")/.."

PROMPT="${PROMPT:?Set PROMPT (e.g. 'malignant ultrasound image with asymmetric lesion')}"
CKPT_BMODE="${CKPT_BMODE:?Set CKPT_BMODE}"
CKPT_MF="${CKPT_MF:?Set CKPT_MF}"
DATA_DIR="${DATA_DIR:-data/fold1}"
TOP_K="${TOP_K:-1}"
OUTPUT_BASE="${OUTPUT_BASE:-./retrieval_comparison}"

echo "=== B-mode only ==="
python -m pons_clip.retrieve_by_prompt \
  --prompt "$PROMPT" \
  --data_dir "$DATA_DIR" \
  --checkpoint "$CKPT_BMODE" \
  --splits all \
  --top_k "$TOP_K" \
  --output_dir "$OUTPUT_BASE/bmode"

echo ""
echo "=== Multi-feature ==="
python -m pons_clip.retrieve_by_prompt \
  --prompt "$PROMPT" \
  --data_dir "$DATA_DIR" \
  --checkpoint "$CKPT_MF" \
  --use_multifeature \
  --splits all \
  --top_k "$TOP_K" \
  --output_dir "$OUTPUT_BASE/multifeature"

echo ""
echo "Results: $OUTPUT_BASE/bmode/retrieve_by_prompt.json and $OUTPUT_BASE/multifeature/retrieve_by_prompt.json"
