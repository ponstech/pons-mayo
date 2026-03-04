#!/usr/bin/env bash
# Compare B-mode-only vs multi-feature: run eval for both checkpoints, then visualize both.
# Usage:
#   CKPT_BMODE=./outputs_bmode/pons_clip_final.pt \
#   CKPT_MF=./outputs_multifeature/pons_clip_final.pt \
#   DATA_DIR=data/fold1 \
#   ./scripts/compare_bmode_vs_multifeature.sh

set -e
cd "$(dirname "$0")/.."

CKPT_BMODE="${CKPT_BMODE:?Set CKPT_BMODE}"
CKPT_MF="${CKPT_MF:?Set CKPT_MF}"
DATA_DIR="${DATA_DIR:-data/fold1}"
OUTPUT_BASE="${OUTPUT_BASE:-./comparison}"

echo "=== B-mode only ==="
python -m pons_clip.evaluate \
  --checkpoint "$CKPT_BMODE" \
  --data_dir "$DATA_DIR" \
  --output_dir "$OUTPUT_BASE/bmode_eval"

python -m pons_clip.visualize \
  --embeddings_dir "$OUTPUT_BASE/bmode_eval" \
  --output_dir "$OUTPUT_BASE/plots" \
  --method both \
  --prefix "bmode"

echo "=== Multi-feature ==="
python -m pons_clip.evaluate \
  --checkpoint "$CKPT_MF" \
  --data_dir "$DATA_DIR" \
  --output_dir "$OUTPUT_BASE/multifeature_eval" \
  --use_multifeature

python -m pons_clip.visualize \
  --embeddings_dir "$OUTPUT_BASE/multifeature_eval" \
  --output_dir "$OUTPUT_BASE/plots" \
  --method both \
  --prefix "multifeature"

echo ""
echo "=== Clustering comparison (test set) ==="
if [[ -f "$OUTPUT_BASE/bmode_eval/clustering_metrics.json" && -f "$OUTPUT_BASE/multifeature_eval/clustering_metrics.json" ]]; then
  python3 -c "
import json
b = json.load(open('$OUTPUT_BASE/bmode_eval/clustering_metrics.json'))
m = json.load(open('$OUTPUT_BASE/multifeature_eval/clustering_metrics.json'))
print('                      B-mode    Multi-feature')
print('Silhouette score      {:.4f}    {:.4f}  (higher = better separation)'.format(b.get('silhouette_score') or 0, m.get('silhouette_score') or 0))
print('Within-class dist     {:.4f}    {:.4f}  (lower = tighter clusters)'.format(b.get('mean_within_class_cosine_distance') or 0, m.get('mean_within_class_cosine_distance') or 0))
print('Between-class dist    {:.4f}    {:.4f}  (higher = more separated)'.format(b.get('mean_between_class_cosine_distance') or 0, m.get('mean_between_class_cosine_distance') or 0))
better = 'B-mode' if (b.get('silhouette_score') or 0) > (m.get('silhouette_score') or 0) else 'Multi-feature'
print('Better clustering:    {} (by silhouette)'.format(better))
"
else
  echo "  (Run visualize first to generate clustering_metrics.json)"
fi
echo ""
echo "Done. Compare:"
echo "  Keyword distances B-mode:    $OUTPUT_BASE/bmode_eval/keyword_distance_summary.json"
echo "  Keyword distances Multi-feat: $OUTPUT_BASE/multifeature_eval/keyword_distance_summary.json"
echo "  Clustering metrics:           $OUTPUT_BASE/bmode_eval/clustering_metrics.json, .../multifeature_eval/..."
echo "  Plots:                       $OUTPUT_BASE/plots/"
