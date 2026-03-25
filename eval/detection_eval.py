#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.detection.data_loader import Detection, load_mot_txt, load_sequence  # noqa: E402


def compute_iou(box_a, box_b) -> float:
    ax1, ay1, aw, ah = box_a
    bx1, by1, bw, bh = box_b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def match_frame(gt_dets: List[Detection], pred_dets: List[Detection], iou_threshold: float):
    if not gt_dets and not pred_dets:
        return 0, 0, 0
    if not gt_dets:
        return 0, len(pred_dets), 0
    if not pred_dets:
        return 0, 0, len(gt_dets)

    ious = np.zeros((len(gt_dets), len(pred_dets)), dtype=np.float32)
    for i, g in enumerate(gt_dets):
        for j, p in enumerate(pred_dets):
            ious[i, j] = compute_iou((g.x, g.y, g.w, g.h), (p.x, p.y, p.w, p.h))

    row_ind, col_ind = linear_sum_assignment(1.0 - ious)

    matched_gt = set()
    matched_pred = set()

    for r, c in zip(row_ind, col_ind):
        if ious[r, c] >= iou_threshold:
            matched_gt.add(r)
            matched_pred.add(c)

    tp = len(matched_gt)
    fp = len(pred_dets) - len(matched_pred)
    fn = len(gt_dets) - len(matched_gt)
    return tp, fp, fn


def evaluate_detections(
    gt_dets: List[Detection],
    pred_dets: List[Detection],
    iou_threshold: float = 0.5,
    confidence_thresholds: List[float] | None = None,
):
    if confidence_thresholds is None:
        confidence_thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

    gt_by_frame: Dict[int, List[Detection]] = defaultdict(list)
    pred_by_frame: Dict[int, List[Detection]] = defaultdict(list)

    for d in gt_dets:
        if d.confidence <= 0:
            continue
        gt_by_frame[d.frame_id].append(d)

    for d in pred_dets:
        pred_by_frame[d.frame_id].append(d)

    all_frames = sorted(set(gt_by_frame.keys()) | set(pred_by_frame.keys()))
    results = []

    for conf_thr in confidence_thresholds:
        total_tp = total_fp = total_fn = 0

        for frame_id in all_frames:
            gt_frame = gt_by_frame.get(frame_id, [])
            pred_frame = [d for d in pred_by_frame.get(frame_id, []) if d.confidence >= conf_thr]

            tp, fp, fn = match_frame(gt_frame, pred_frame, iou_threshold)
            total_tp += tp
            total_fp += fp
            total_fn += fn

        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        f1 = 2 * recall * precision / (recall + precision) if (recall + precision) > 0 else 0.0

        results.append(
            {
                "threshold": conf_thr,
                "recall": round(recall, 4),
                "precision": round(precision, 4),
                "f1": round(f1, 4),
                "tp": int(total_tp),
                "fp": int(total_fp),
                "fn": int(total_fn),
            }
        )

    return results


def print_results_table(results: List[dict], sequence_name: str = ""):
    print(f"\n### {sequence_name}")
    print("| Conf | Recall | Precision | F1 | TP | FP | FN |")
    print("|------|--------|-----------|----|----|----|----|")
    for r in results:
        print(
            f"| {r['threshold']:.1f} | {r['recall']:.4f} | {r['precision']:.4f} | "
            f"{r['f1']:.4f} | {r['tp']} | {r['fp']} | {r['fn']} |"
        )

    best = max(results, key=lambda x: x["f1"])
    print(
        f"Best F1: thr={best['threshold']:.1f}, "
        f"R={best['recall']:.4f}, P={best['precision']:.4f}, F1={best['f1']:.4f}"
    )


def aggregate_results(all_sequence_results: Dict[str, List[dict]]) -> List[dict]:
    thresholds = [r["threshold"] for r in next(iter(all_sequence_results.values()))]
    out = []

    for thr in thresholds:
        rows = []
        for seq_name, seq_results in all_sequence_results.items():
            row = next(r for r in seq_results if r["threshold"] == thr)
            rows.append(row)

        tp = sum(r["tp"] for r in rows)
        fp = sum(r["fp"] for r in rows)
        fn = sum(r["fn"] for r in rows)

        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1 = 2 * recall * precision / (recall + precision) if (recall + precision) > 0 else 0.0

        out.append(
            {
                "threshold": thr,
                "recall": round(recall, 4),
                "precision": round(precision, 4),
                "f1": round(f1, 4),
                "tp": int(tp),
                "fp": int(fp),
                "fn": int(fn),
            }
        )

    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_dir", required=True, help="Dataset root, e.g. data/vip_htd")
    parser.add_argument("--det_dir", required=True, help="Detection root, e.g. eval/detections/rfdetr_coco")
    parser.add_argument("--split", default="train")
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--save_json", default="")
    args = parser.parse_args()

    gt_split = Path(args.gt_dir) / args.split
    det_split = Path(args.det_dir) / args.split

    if not gt_split.exists():
        raise FileNotFoundError(f"GT split not found: {gt_split}")
    if not det_split.exists():
        raise FileNotFoundError(f"Detection split not found: {det_split}")

    all_sequence_results = {}

    for seq_dir in sorted([p for p in gt_split.iterdir() if p.is_dir()]):
        seq_name = seq_dir.name
        det_file = det_split / f"{seq_name}.txt"
        if not det_file.exists():
            print(f"WARNING: missing detection file for {seq_name}: {det_file}")
            continue

        seq = load_sequence(str(seq_dir))
        preds = load_mot_txt(str(det_file))
        results = evaluate_detections(seq.gt, preds, iou_threshold=args.iou)
        all_sequence_results[seq_name] = results
        print_results_table(results, seq_name)

    if not all_sequence_results:
        raise RuntimeError("No sequence results computed.")

    overall = aggregate_results(all_sequence_results)
    print("\n## OVERALL")
    print_results_table(overall, "ALL")

    gate_rows = [r for r in overall if r["recall"] >= 0.85]
    if gate_rows:
        best_gate = max(gate_rows, key=lambda x: x["precision"])
        print(
            f"\nGATE PASS: recall >= 0.85 at threshold {best_gate['threshold']:.1f} "
            f"(R={best_gate['recall']:.4f}, P={best_gate['precision']:.4f})"
        )
    else:
        best_recall = max(overall, key=lambda x: x["recall"])
        print(
            f"\nGATE FAIL: best recall {best_recall['recall']:.4f} "
            f"at threshold {best_recall['threshold']:.1f}"
        )

    if args.save_json:
        payload = {
            "split": args.split,
            "iou": args.iou,
            "per_sequence": all_sequence_results,
            "overall": overall,
        }
        save_path = Path(args.save_json)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"Saved: {save_path}")


if __name__ == "__main__":
    main()
