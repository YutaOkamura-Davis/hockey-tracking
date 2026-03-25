#!/usr/bin/env bash
set -euo pipefail

GT_SPLIT_DIR="data/vip_htd/train"
TRACKER_PRED_DIR="eval/predictions/v0_baseline"
OUTPUT_DIR="eval/results/v0_baseline"

python eval/trackeval_wrapper.py \
  --gt_split_dir "$GT_SPLIT_DIR" \
  --tracker_pred_dir "$TRACKER_PRED_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --dataset_name VIPHTD \
  --split train \
  --tracker_name baseline

echo "Saved results to $OUTPUT_DIR"
