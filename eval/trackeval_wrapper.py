#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import List


def _copy_if_exists(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.exists():
        shutil.copy2(src, dst)


def stage_for_trackeval(
    gt_split_dir: str,
    tracker_pred_dir: str,
    workspace_dir: str,
    dataset_name: str,
    split: str,
    tracker_name: str,
    sequences: List[str] | None = None,
):
    gt_split_dir = Path(gt_split_dir)
    tracker_pred_dir = Path(tracker_pred_dir)
    workspace_dir = Path(workspace_dir)

    dataset_key = f"{dataset_name}-{split}"
    gt_root = workspace_dir / "gt" / dataset_key
    tr_root = workspace_dir / "trackers" / dataset_key / tracker_name / "data"
    seqmap_root = workspace_dir / "seqmaps"
    seqmap_root.mkdir(parents=True, exist_ok=True)

    if sequences is None or len(sequences) == 0:
        sequences = sorted([p.name for p in gt_split_dir.iterdir() if p.is_dir()])

    for seq in sequences:
        src_seq = gt_split_dir / seq
        dst_seq = gt_root / seq
        (dst_seq / "gt").mkdir(parents=True, exist_ok=True)

        _copy_if_exists(src_seq / "gt" / "gt.txt", dst_seq / "gt" / "gt.txt")
        _copy_if_exists(src_seq / "seqinfo.ini", dst_seq / "seqinfo.ini")

        pred_src = tracker_pred_dir / f"{seq}.txt"
        pred_dst = tr_root / f"{seq}.txt"
        _copy_if_exists(pred_src, pred_dst)

    seqmap_file = seqmap_root / f"{dataset_key}.txt"
    with open(seqmap_file, "w", encoding="utf-8") as f:
        f.write("name\n")
        for seq in sequences:
            f.write(f"{seq}\n")

    return {
        "gt_root": str(workspace_dir / "gt"),
        "trackers_root": str(workspace_dir / "trackers"),
        "seqmap_root": str(seqmap_root),
        "dataset_name": dataset_name,
        "split": split,
        "tracker_name": tracker_name,
        "sequences": sequences,
    }


def _to_builtin(obj):
    if isinstance(obj, dict):
        return {str(k): _to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_builtin(v) for v in obj]
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:
            pass
    return obj


def format_results(results: dict, tracker_name: str) -> str:
    lines = []
    lines.append("| Scope | HOTA | IDF1 | MOTA | AssA | DetA | IDSW | Frag |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")

    mot_key = next(iter(results.keys())) if results else None
    if not mot_key:
        return "\n".join(lines)

    tracker_block = results.get(mot_key, {}).get(tracker_name, {})
    combined = tracker_block.get("COMBINED_SEQ", {})

    if isinstance(combined, dict) and combined:
        if "pedestrian" in combined:
            row = combined["pedestrian"]
        else:
            first_key = next(iter(combined.keys()))
            row = combined[first_key]

        lines.append(
            f"| COMBINED | "
            f"{row.get('HOTA', 0):.3f} | "
            f"{row.get('IDF1', 0):.3f} | "
            f"{row.get('MOTA', 0):.3f} | "
            f"{row.get('AssA', 0):.3f} | "
            f"{row.get('DetA', 0):.3f} | "
            f"{row.get('IDSW', 0)} | "
            f"{row.get('Frag', 0)} |"
        )

    return "\n".join(lines)


def run_eval(
    gt_split_dir: str,
    tracker_pred_dir: str,
    output_dir: str = "eval/results/v0_baseline",
    workspace_dir: str = "eval/trackeval_workspace",
    dataset_name: str = "VIPHTD",
    split: str = "train",
    tracker_name: str = "baseline",
    sequences: List[str] | None = None,
):
    import trackeval

    staging = stage_for_trackeval(
        gt_split_dir=gt_split_dir,
        tracker_pred_dir=tracker_pred_dir,
        workspace_dir=workspace_dir,
        dataset_name=dataset_name,
        split=split,
        tracker_name=tracker_name,
        sequences=sequences,
    )

    eval_config = trackeval.Evaluator.get_default_eval_config()
    eval_config["USE_PARALLEL"] = False
    eval_config["PRINT_RESULTS"] = True
    eval_config["OUTPUT_SUMMARY"] = True
    eval_config["OUTPUT_DETAILED"] = True
    eval_config["PLOT_CURVES"] = False

    dataset_config = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
    dataset_config["GT_FOLDER"] = staging["gt_root"]
    dataset_config["TRACKERS_FOLDER"] = staging["trackers_root"]
    dataset_config["SEQMAP_FOLDER"] = staging["seqmap_root"]
    dataset_config["TRACKERS_TO_EVAL"] = [tracker_name]
    dataset_config["BENCHMARK"] = dataset_name
    dataset_config["SPLIT_TO_EVAL"] = split
    dataset_config["OUTPUT_FOLDER"] = output_dir
    dataset_config["TRACKER_SUB_FOLDER"] = "data"
    dataset_config["OUTPUT_SUB_FOLDER"] = ""
    dataset_config["DO_PREPROC"] = False

    evaluator = trackeval.Evaluator(eval_config)
    dataset = trackeval.datasets.MotChallenge2DBox(dataset_config)
    metrics = [
        trackeval.metrics.HOTA(),
        trackeval.metrics.CLEAR(),
        trackeval.metrics.Identity(),
    ]

    results, messages = evaluator.evaluate([dataset], metrics)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(output_dir) / "results_raw.json", "w", encoding="utf-8") as f:
        json.dump(_to_builtin(results), f, indent=2)

    return results, messages


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_split_dir", required=True)
    parser.add_argument("--tracker_pred_dir", required=True)
    parser.add_argument("--output_dir", default="eval/results/v0_baseline")
    parser.add_argument("--workspace_dir", default="eval/trackeval_workspace")
    parser.add_argument("--dataset_name", default="VIPHTD")
    parser.add_argument("--split", default="train")
    parser.add_argument("--tracker_name", default="baseline")
    parser.add_argument("--sequences", nargs="*", default=None)
    args = parser.parse_args()

    results, _ = run_eval(
        gt_split_dir=args.gt_split_dir,
        tracker_pred_dir=args.tracker_pred_dir,
        output_dir=args.output_dir,
        workspace_dir=args.workspace_dir,
        dataset_name=args.dataset_name,
        split=args.split,
        tracker_name=args.tracker_name,
        sequences=args.sequences,
    )

    print(format_results(results, args.tracker_name))


if __name__ == "__main__":
    main()
