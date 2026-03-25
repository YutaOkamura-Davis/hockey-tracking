"""
VIP-HTD / MOT-format data loader utilities.
"""
from __future__ import annotations

import csv
import glob
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class Detection:
    frame_id: int
    track_id: int
    x: float
    y: float
    w: float
    h: float
    confidence: float = 1.0
    class_id: int = -1
    visibility: float = 1.0

    def to_mot_row(self) -> str:
        return (
            f"{self.frame_id},{self.track_id},"
            f"{self.x:.2f},{self.y:.2f},{self.w:.2f},{self.h:.2f},"
            f"{self.confidence:.4f},{self.class_id},{self.visibility:.4f},-1"
        )


@dataclass
class Sequence:
    name: str
    path: str
    fps: int
    frame_count: int
    width: int
    height: int
    frame_dir: str
    gt: List[Detection]


def parse_seqinfo(seq_dir: Path) -> Dict[str, str]:
    info = {}
    ini_path = seq_dir / "seqinfo.ini"
    if not ini_path.exists():
        return info

    with open(ini_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("[") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            info[k.strip()] = v.strip()
    return info


def load_mot_txt(txt_path: str) -> List[Detection]:
    path = Path(txt_path)
    if not path.exists():
        return []

    out: List[Detection] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 6:
                continue
            out.append(
                Detection(
                    frame_id=int(float(row[0])),
                    track_id=int(float(row[1])),
                    x=float(row[2]),
                    y=float(row[3]),
                    w=float(row[4]),
                    h=float(row[5]),
                    confidence=float(row[6]) if len(row) > 6 else 1.0,
                    class_id=int(float(row[7])) if len(row) > 7 and row[7].strip() != "" else -1,
                    visibility=float(row[8]) if len(row) > 8 and row[8].strip() != "" else 1.0,
                )
            )
    return out


def write_mot_txt(dets: List[Detection], txt_path: str) -> None:
    path = Path(txt_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for d in dets:
            f.write(d.to_mot_row() + "\n")


def load_sequence(seq_dir: str) -> Sequence:
    seq_dir = Path(seq_dir)
    info = parse_seqinfo(seq_dir)

    frame_dir = seq_dir / "img1"
    frames = sorted(glob.glob(str(frame_dir / "*.jpg"))) + sorted(glob.glob(str(frame_dir / "*.png")))

    gt = load_mot_txt(str(seq_dir / "gt" / "gt.txt"))

    return Sequence(
        name=seq_dir.name,
        path=str(seq_dir),
        fps=int(info.get("frameRate", 30)),
        frame_count=int(info.get("seqLength", len(frames))),
        width=int(info.get("imWidth", 1920)),
        height=int(info.get("imHeight", 1080)),
        frame_dir=str(frame_dir),
        gt=gt,
    )


def load_dataset(base_dir: str, split: str = "train") -> List[Sequence]:
    split_dir = Path(base_dir) / split
    if not split_dir.exists():
        return []
    seq_dirs = sorted([d for d in split_dir.iterdir() if d.is_dir()])
    return [load_sequence(str(d)) for d in seq_dirs]


if __name__ == "__main__":
    import sys

    base = sys.argv[1] if len(sys.argv) > 1 else "data/vip_htd"
    for split in ["train", "test"]:
        seqs = load_dataset(base, split)
        print(f"\n{split}: {len(seqs)} sequences")
        for s in seqs:
            unique_ids = len(set(d.track_id for d in s.gt))
            print(
                f"  {s.name}: {s.frame_count} frames | {s.width}x{s.height} | "
                f"{s.fps}fps | {len(s.gt)} GT rows | {unique_ids} IDs"
            )
