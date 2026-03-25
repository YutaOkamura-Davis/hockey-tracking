#!/usr/bin/env bash
set -euo pipefail

SPLIT=${1:-train}
THRESHOLD=${2:-0.5}
MODEL_SIZE=${3:-base}
CHECKPOINT=${4:-}

DATA_DIR="data/vip_htd/${SPLIT}"
OUTPUT_BASE="eval/detections/rfdetr_coco/${SPLIT}"

if [[ ! -d "$DATA_DIR" ]]; then
  echo "Missing data dir: $DATA_DIR"
  exit 1
fi

mkdir -p "$OUTPUT_BASE"

echo "Split: $SPLIT"
echo "Threshold: $THRESHOLD"
echo "Model size: $MODEL_SIZE"
echo "Checkpoint: ${CHECKPOINT:-<none>}"

for SEQ_DIR in "$DATA_DIR"/*/; do
  SEQ_NAME=$(basename "$SEQ_DIR")
  echo "Processing: $SEQ_NAME"

  EXTRA_ARGS=()
  if [[ -n "$CHECKPOINT" ]]; then
    EXTRA_ARGS+=(--checkpoint "$CHECKPOINT")
  fi

  python src/detection/rfdetr_detector.py \
    --frame_dir "${SEQ_DIR}/img1" \
    --output "${OUTPUT_BASE}/${SEQ_NAME}.txt" \
    --threshold "$THRESHOLD" \
    --model_size "$MODEL_SIZE" \
    "${EXTRA_ARGS[@]}"
done

echo "Done. Outputs in $OUTPUT_BASE"
