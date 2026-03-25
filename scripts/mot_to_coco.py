#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.detection.data_loader import load_sequence  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mot_dir", required=True, help="e.g. data/vip_htd/train")
    parser.add_argument("--output", required=True, help="e.g. data/vip_htd/train_coco.json")
    parser.add_argument("--class_name", default="player")
    args = parser.parse_args()

    mot_dir = Path(args.mot_dir)
    seq_dirs = sorted([p for p in mot_dir.iterdir() if p.is_dir()])

    images = []
    annotations = []
    categories = [{"id": 1, "name": args.class_name}]

    image_id = 1
    ann_id = 1

    for seq_dir in seq_dirs:
        seq = load_sequence(str(seq_dir))

        # Build frame map
        frame_files = sorted((seq_dir / "img1").glob("*.jpg")) + sorted((seq_dir / "img1").glob("*.png"))
        frame_map = {}
        for idx, fp in enumerate(frame_files, start=1):
            try:
                fid = int(fp.stem)
            except ValueError:
                fid = idx
            frame_map[fid] = fp

        frame_to_img_id = {}
        for fid, fp in sorted(frame_map.items()):
            frame_to_img_id[fid] = image_id
            images.append(
                {
                    "id": image_id,
                    "file_name": str(fp.relative_to(mot_dir.parent)),
                    "width": seq.width,
                    "height": seq.height,
                }
            )
            image_id += 1

        for det in seq.gt:
            if det.confidence <= 0:
                continue
            if det.frame_id not in frame_to_img_id:
                continue

            annotations.append(
                {
                    "id": ann_id,
                    "image_id": frame_to_img_id[det.frame_id],
                    "category_id": 1,
                    "bbox": [det.x, det.y, det.w, det.h],
                    "area": det.w * det.h,
                    "iscrowd": 0,
                    "track_id": det.track_id,
                }
            )
            ann_id += 1

    payload = {
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Saved COCO JSON: {out}")
    print(f"Images: {len(images)}")
    print(f"Annotations: {len(annotations)}")


if __name__ == "__main__":
    main()
