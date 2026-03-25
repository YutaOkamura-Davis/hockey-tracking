#!/usr/bin/env python3
import argparse
import glob
from pathlib import Path


def count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def inspect_split(base_dir: Path, split: str):
    split_dir = base_dir / split
    if not split_dir.exists():
        print(f"{split}: missing ({split_dir})")
        return

    seqs = sorted([p for p in split_dir.iterdir() if p.is_dir()])
    print(f"\n{split}: {len(seqs)} sequences")

    for seq in seqs:
        imgs = sorted(glob.glob(str(seq / "img1" / "*.jpg"))) + sorted(glob.glob(str(seq / "img1" / "*.png")))
        gt_path = seq / "gt" / "gt.txt"
        seqinfo = seq / "seqinfo.ini"

        print(
            f"  {seq.name}: "
            f"{len(imgs)} frames | "
            f"GT: {gt_path.exists()} ({count_lines(gt_path)} rows) | "
            f"seqinfo: {seqinfo.exists()}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", default="data/vip_htd")
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    if not base_dir.exists():
        raise FileNotFoundError(f"Base dir not found: {base_dir}")

    inspect_split(base_dir, "train")
    inspect_split(base_dir, "test")


if __name__ == "__main__":
    main()
