#!/usr/bin/env python3
"""
Unified multi-tracker benchmark framework.

Architecture:
  1. Frozen detections (from RF-DETR) are loaded once
  2. Each tracker adapter consumes detections and produces MOT-format tracks
  3. TrackEval evaluates all trackers against the same GT
  4. Results are collected into a single comparison table

To add a new tracker:
  1. Create an adapter class that inherits TrackerAdapter
  2. Implement reset() and update(frame_id, detections) -> tracks
  3. Register it in TRACKER_REGISTRY

Usage:
  python src/tracking/benchmark_trackers.py \
    --det_dir eval/detections/rfdetr_coco/train \
    --gt_dir data/vip_htd/train \
    --output_dir eval/tracker_benchmark \
    --trackers sort bytetrack ocsort botsort deep_eious
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.detection.data_loader import Detection, load_mot_txt, load_sequence, write_mot_txt


# ============================================================
# Base adapter
# ============================================================

class TrackerAdapter(ABC):
    """Base class for all tracker adapters."""

    name: str = "base"
    paper: str = ""
    year: int = 0
    url: str = ""
    requires_appearance: bool = False  # needs re-ID embeddings
    requires_masks: bool = False       # needs segmentation masks
    requires_training: bool = False    # needs training on target domain

    @abstractmethod
    def reset(self):
        """Reset tracker state for a new sequence."""
        pass

    @abstractmethod
    def update(self, frame_id: int, detections: List[Detection],
               frame_path: str = "") -> List[Detection]:
        """
        Process one frame of detections.
        Returns: List[Detection] with track_id assigned.
        """
        pass

    def get_config(self) -> dict:
        """Return current hyperparameters for logging."""
        return {}


# ============================================================
# Tracker implementations
# ============================================================

class SORTAdapter(TrackerAdapter):
    """SORT: Simple Online Realtime Tracking (Bewley et al., 2016)"""
    name = "sort"
    paper = "Simple Online and Realtime Tracking"
    year = 2016
    url = "https://github.com/abewley/sort"

    def __init__(self, max_age=30, min_hits=3, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.tracker = None

    def _load(self):
        try:
            # Try roboflow trackers first (cleanest API)
            from trackers import SORTTracker
            self.tracker = SORTTracker(
                max_age=self.max_age,
                min_hits=self.min_hits,
                iou_threshold=self.iou_threshold,
            )
            self._backend = "roboflow"
            return
        except ImportError:
            pass

        try:
            # Fallback: original SORT
            from sort.sort import Sort
            self.tracker = Sort(
                max_age=self.max_age,
                min_hits=self.min_hits,
                iou_threshold=self.iou_threshold,
            )
            self._backend = "original"
            return
        except ImportError:
            pass

        raise ImportError(
            "SORT not available. Install via:\n"
            "  pip install trackers  (roboflow)\n"
            "  OR pip install sort-tracker"
        )

    def reset(self):
        self._load()

    def update(self, frame_id: int, detections: List[Detection],
               frame_path: str = "") -> List[Detection]:
        if not detections:
            return []

        # Convert to numpy: [x1, y1, x2, y2, score]
        dets_np = np.array([
            [d.x, d.y, d.x + d.w, d.y + d.h, d.confidence]
            for d in detections
        ])

        if self._backend == "roboflow":
            import supervision as sv
            sv_dets = sv.Detections(
                xyxy=dets_np[:, :4],
                confidence=dets_np[:, 4],
            )
            tracked = self.tracker.update(sv_dets)
            out = []
            for i in range(len(tracked)):
                x1, y1, x2, y2 = tracked.xyxy[i]
                tid = tracked.tracker_id[i] if tracked.tracker_id is not None else -1
                conf = tracked.confidence[i] if tracked.confidence is not None else 1.0
                out.append(Detection(
                    frame_id=frame_id, track_id=int(tid),
                    x=float(x1), y=float(y1),
                    w=float(x2 - x1), h=float(y2 - y1),
                    confidence=float(conf),
                ))
            return out
        else:
            # Original SORT returns [x1, y1, x2, y2, track_id]
            tracks = self.tracker.update(dets_np)
            out = []
            for t in tracks:
                x1, y1, x2, y2, tid = t[:5]
                out.append(Detection(
                    frame_id=frame_id, track_id=int(tid),
                    x=float(x1), y=float(y1),
                    w=float(x2 - x1), h=float(y2 - y1),
                    confidence=1.0,
                ))
            return out

    def get_config(self):
        return {"max_age": self.max_age, "min_hits": self.min_hits,
                "iou_threshold": self.iou_threshold}


class ByteTrackAdapter(TrackerAdapter):
    """ByteTrack: Multi-Object Tracking by Associating Every Detection Box (Zhang et al., 2021)"""
    name = "bytetrack"
    paper = "ByteTrack: Multi-Object Tracking by Associating Every Detection Box"
    year = 2021
    url = "https://github.com/ifzhang/ByteTrack"

    def __init__(self, track_thresh=0.6, track_buffer=30, match_thresh=0.8):
        self.track_thresh = track_thresh
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh
        self.tracker = None

    def _load(self):
        try:
            from trackers import ByteTrackTracker
            self.tracker = ByteTrackTracker(
                track_thresh=self.track_thresh,
                track_buffer=self.track_buffer,
                match_thresh=self.match_thresh,
            )
            self._backend = "roboflow"
            return
        except ImportError:
            pass

        try:
            import supervision as sv
            self.tracker = sv.ByteTrack(
                track_activation_threshold=self.track_thresh,
                lost_track_buffer=self.track_buffer,
                minimum_matching_threshold=self.match_thresh,
            )
            self._backend = "supervision"
            return
        except (ImportError, AttributeError):
            pass

        raise ImportError(
            "ByteTrack not available. Install via:\n"
            "  pip install trackers  (roboflow)\n"
            "  OR pip install supervision"
        )

    def reset(self):
        self._load()

    def update(self, frame_id: int, detections: List[Detection],
               frame_path: str = "") -> List[Detection]:
        if not detections:
            return []

        dets_np = np.array([
            [d.x, d.y, d.x + d.w, d.y + d.h, d.confidence]
            for d in detections
        ])

        import supervision as sv
        sv_dets = sv.Detections(
            xyxy=dets_np[:, :4],
            confidence=dets_np[:, 4],
        )
        tracked = self.tracker.update(sv_dets)

        out = []
        for i in range(len(tracked)):
            x1, y1, x2, y2 = tracked.xyxy[i]
            tid = tracked.tracker_id[i] if tracked.tracker_id is not None else -1
            conf = tracked.confidence[i] if tracked.confidence is not None else 1.0
            out.append(Detection(
                frame_id=frame_id, track_id=int(tid),
                x=float(x1), y=float(y1),
                w=float(x2 - x1), h=float(y2 - y1),
                confidence=float(conf),
            ))
        return out

    def get_config(self):
        return {"track_thresh": self.track_thresh,
                "track_buffer": self.track_buffer,
                "match_thresh": self.match_thresh}


class OCSORTAdapter(TrackerAdapter):
    """OC-SORT: Observation-Centric SORT (Cao et al., 2022)"""
    name = "ocsort"
    paper = "Observation-Centric SORT: Rethinking SORT for Robust MOT"
    year = 2022
    url = "https://github.com/noahcao/OC_SORT"

    def __init__(self, det_thresh=0.5, max_age=30, min_hits=3,
                 iou_threshold=0.3, delta_t=3, asso_func="iou",
                 inertia=0.2, use_byte=False):
        self.params = {
            "det_thresh": det_thresh, "max_age": max_age,
            "min_hits": min_hits, "iou_threshold": iou_threshold,
            "delta_t": delta_t, "asso_func": asso_func,
            "inertia": inertia, "use_byte": use_byte,
        }
        self.tracker = None

    def _load(self):
        try:
            from ocsort.ocsort import OCSort
            self.tracker = OCSort(**self.params)
            self._backend = "official"
            return
        except ImportError:
            pass
        raise ImportError(
            "OC-SORT not available. Install via:\n"
            "  pip install ocsort\n"
            "  OR git clone https://github.com/noahcao/OC_SORT"
        )

    def reset(self):
        self._load()

    def update(self, frame_id: int, detections: List[Detection],
               frame_path: str = "") -> List[Detection]:
        if not detections:
            return []

        dets_np = np.array([
            [d.x, d.y, d.x + d.w, d.y + d.h, d.confidence]
            for d in detections
        ])
        tracks = self.tracker.update(dets_np, None)
        out = []
        for t in tracks:
            x1, y1, x2, y2, tid = t[0], t[1], t[2], t[3], t[4]
            out.append(Detection(
                frame_id=frame_id, track_id=int(tid),
                x=float(x1), y=float(y1),
                w=float(x2 - x1), h=float(y2 - y1),
                confidence=1.0,
            ))
        return out

    def get_config(self):
        return self.params


class BoTSORTAdapter(TrackerAdapter):
    """BoT-SORT: Robust Associations Multi-Pedestrian Tracking (Aharon et al., 2022)"""
    name = "botsort"
    paper = "BoT-SORT: Robust Associations Multi-Pedestrian Tracking"
    year = 2022
    url = "https://github.com/NirAharon/BoT-SORT"
    requires_appearance = True

    def __init__(self, track_high_thresh=0.6, track_low_thresh=0.1,
                 new_track_thresh=0.7, track_buffer=30, match_thresh=0.8,
                 proximity_thresh=0.5, appearance_thresh=0.25,
                 cmc_method="sparseOptFlow"):
        self.params = {
            "track_high_thresh": track_high_thresh,
            "track_low_thresh": track_low_thresh,
            "new_track_thresh": new_track_thresh,
            "track_buffer": track_buffer,
            "match_thresh": match_thresh,
            "proximity_thresh": proximity_thresh,
            "appearance_thresh": appearance_thresh,
            "cmc_method": cmc_method,
        }
        self.tracker = None

    def _load(self):
        # Try ultralytics BoT-SORT (easiest)
        try:
            from ultralytics.trackers.bot_sort import BOTSORT
            # Ultralytics requires an args namespace
            class Args:
                pass
            args = Args()
            for k, v in self.params.items():
                setattr(args, k, v)
            args.with_reid = False
            args.fuse_score = True
            self.tracker = BOTSORT(args, frame_rate=30)
            self._backend = "ultralytics"
            return
        except ImportError:
            pass
        raise ImportError(
            "BoT-SORT not available. Install via:\n"
            "  pip install ultralytics"
        )

    def reset(self):
        self._load()

    def update(self, frame_id: int, detections: List[Detection],
               frame_path: str = "") -> List[Detection]:
        if not detections:
            return []

        dets_np = np.array([
            [d.x, d.y, d.x + d.w, d.y + d.h, d.confidence, 0]
            for d in detections
        ])

        if self._backend == "ultralytics":
            import cv2
            img = None
            if frame_path and os.path.exists(frame_path):
                img = cv2.imread(frame_path)
            tracks = self.tracker.update(dets_np, img)
        else:
            tracks = self.tracker.update(dets_np, None)

        out = []
        for t in tracks:
            x1, y1, x2, y2 = float(t[0]), float(t[1]), float(t[2]), float(t[3])
            tid = int(t[4])
            conf = float(t[5]) if len(t) > 5 else 1.0
            out.append(Detection(
                frame_id=frame_id, track_id=tid,
                x=x1, y=y1, w=x2 - x1, h=y2 - y1,
                confidence=conf,
            ))
        return out

    def get_config(self):
        return self.params


class DeepEIoUAdapter(TrackerAdapter):
    """Deep-EIoU: Iterative Scale-Up ExpansionIoU + Deep Features (Huang et al., 2024)"""
    name = "deep_eiou"
    paper = "Iterative Scale-Up ExpansionIoU and Deep Features for Multi-Object Tracking in Sports"
    year = 2024
    url = "https://github.com/hsiangwei0903/Deep-EIoU"
    requires_appearance = True
    requires_training = False  # uses pretrained OSNet

    def __init__(self):
        self.tracker = None

    def _load(self):
        # Deep-EIoU must be cloned and available on sys.path
        try:
            from tracker.Deep_EIoU import Deep_EIoU
            self.tracker = Deep_EIoU()
            self._backend = "official"
            return
        except ImportError:
            pass
        raise ImportError(
            "Deep-EIoU not available. Clone from:\n"
            "  git clone https://github.com/hsiangwei0903/Deep-EIoU\n"
            "  Then add to PYTHONPATH"
        )

    def reset(self):
        self._load()

    def update(self, frame_id: int, detections: List[Detection],
               frame_path: str = "") -> List[Detection]:
        raise NotImplementedError(
            "Deep-EIoU requires custom integration — uses its own detection + "
            "embedding pipeline. See scripts/run_deep_eiou.py for standalone eval."
        )

    def get_config(self):
        return {"note": "requires standalone execution with own embeddings"}


class McByteAdapter(TrackerAdapter):
    """McByte: Mask-propagation + ByteTrack (Adžemović et al., 2025)"""
    name = "mcbyte"
    paper = "No Train Yet Gain: Towards Generic Multi-Object Tracking in Sports and Beyond"
    year = 2025
    url = "https://github.com/PetrPelaworker/McByte"  # check actual URL
    requires_masks = True

    def __init__(self):
        self.tracker = None

    def _load(self):
        try:
            # McByte is based on ByteTrack + mask propagation
            # The exact import depends on their repo structure
            from mcbyte.tracker import McByteTracker
            self.tracker = McByteTracker()
            self._backend = "official"
            return
        except ImportError:
            pass
        raise ImportError(
            "McByte not available. Clone from the official repo.\n"
            "Requires SAM2 for mask propagation."
        )

    def reset(self):
        self._load()

    def update(self, frame_id: int, detections: List[Detection],
               frame_path: str = "") -> List[Detection]:
        raise NotImplementedError(
            "McByte requires mask propagation pipeline. "
            "See scripts/run_mcbyte.py for standalone eval."
        )

    def get_config(self):
        return {"note": "requires mask propagation (SAM2)"}


class DiffMOTAdapter(TrackerAdapter):
    """DiffMOT: Diffusion-based motion prediction (Lv et al., 2024)"""
    name = "diffmot"
    paper = "DiffMOT: A Real-time Diffusion-based Multiple Object Tracker"
    year = 2024
    url = "https://github.com/Kroery/DiffMOT"

    def __init__(self):
        self.tracker = None

    def _load(self):
        try:
            from DiffMOT.tracker.DiffMOT import DiffMOTTracker
            self.tracker = DiffMOTTracker()
            self._backend = "official"
            return
        except ImportError:
            pass
        raise ImportError(
            "DiffMOT not available. Clone from:\n"
            "  git clone https://github.com/Kroery/DiffMOT"
        )

    def reset(self):
        self._load()

    def update(self, frame_id: int, detections: List[Detection],
               frame_path: str = "") -> List[Detection]:
        raise NotImplementedError(
            "DiffMOT requires its trained diffusion motion model. "
            "See scripts/run_diffmot.py for standalone eval."
        )

    def get_config(self):
        return {"note": "requires trained diffusion motion model"}


# ============================================================
# Registry
# ============================================================

TRACKER_REGISTRY: Dict[str, type] = {
    "sort": SORTAdapter,
    "bytetrack": ByteTrackAdapter,
    "ocsort": OCSORTAdapter,
    "botsort": BoTSORTAdapter,
    "deep_eiou": DeepEIoUAdapter,
    "mcbyte": McByteAdapter,
    "diffmot": DiffMOTAdapter,
}

# Metadata for trackers we know about but haven't integrated yet
KNOWN_TRACKERS: List[Dict[str, Any]] = [
    {"name": "sort", "year": 2016, "paper": "SORT", "sports_hota": "~65", "status": "integrated"},
    {"name": "deepsort", "year": 2017, "paper": "DeepSORT", "sports_hota": "~67", "status": "integrated_via_roboflow"},
    {"name": "bytetrack", "year": 2021, "paper": "ByteTrack", "sports_hota": "~72", "status": "integrated"},
    {"name": "ocsort", "year": 2022, "paper": "OC-SORT", "sports_hota": "~73", "status": "integrated"},
    {"name": "botsort", "year": 2022, "paper": "BoT-SORT", "sports_hota": "~74", "status": "integrated"},
    {"name": "hybrid_sort", "year": 2024, "paper": "Hybrid-SORT", "sports_hota": "~74", "status": "todo",
     "url": "https://github.com/ymzis69/HybirdSORT"},
    {"name": "deep_eiou", "year": 2024, "paper": "Deep-EIoU", "sports_hota": "~77", "status": "stub",
     "url": "https://github.com/hsiangwei0903/Deep-EIoU"},
    {"name": "deep_hm_sort", "year": 2024, "paper": "Deep HM-SORT", "sports_hota": "~78", "status": "todo",
     "url": "https://github.com/"},
    {"name": "diffmot", "year": 2024, "paper": "DiffMOT", "sports_hota": "~76", "status": "stub",
     "url": "https://github.com/Kroery/DiffMOT"},
    {"name": "sportmamba", "year": 2025, "paper": "SportMamba", "sports_hota": "~78", "status": "todo",
     "url": "https://github.com/"},
    {"name": "mcbyte", "year": 2025, "paper": "McByte", "sports_hota": "~79", "status": "stub",
     "url": "https://github.com/"},
    {"name": "cameltrack", "year": 2025, "paper": "CAMELTrack", "sports_hota": "~80", "status": "todo",
     "url": "https://github.com/"},
    {"name": "memosort", "year": 2025, "paper": "MeMoSORT", "sports_hota": "82.1", "status": "todo",
     "url": "https://github.com/"},
    {"name": "matr", "year": 2025, "paper": "MATR (E2E)", "sports_hota": "~72", "status": "todo",
     "url": "https://github.com/"},
    {"name": "plugtrack", "year": 2025, "paper": "PlugTrack", "sports_hota": "~77", "status": "todo",
     "url": "https://github.com/"},
]


def list_trackers():
    """Print all known trackers and their integration status."""
    print("\n" + "=" * 90)
    print("KNOWN TRACKERS — Integration Status")
    print("=" * 90)
    print(f"{'Name':<16} {'Year':<6} {'Paper':<25} {'SportsMOT HOTA':<16} {'Status':<20}")
    print("-" * 90)
    for t in sorted(KNOWN_TRACKERS, key=lambda x: x.get("sports_hota", "0"), reverse=True):
        print(f"{t['name']:<16} {t['year']:<6} {t['paper']:<25} {t.get('sports_hota', '?'):<16} {t['status']:<20}")
    print()
    print(f"Integrated and runnable: {sum(1 for t in KNOWN_TRACKERS if t['status'] == 'integrated')}")
    print(f"Stub (needs custom pipeline): {sum(1 for t in KNOWN_TRACKERS if t['status'] == 'stub')}")
    print(f"TODO (not yet started): {sum(1 for t in KNOWN_TRACKERS if t['status'] == 'todo')}")


# ============================================================
# Benchmark runner
# ============================================================

def run_tracker_on_sequence(
    tracker: TrackerAdapter,
    det_file: str,
    seq_dir: str,
    output_file: str,
) -> Tuple[int, float]:
    """
    Run a tracker on one sequence using precomputed detections.
    Returns: (num_tracks, elapsed_seconds)
    """
    dets = load_mot_txt(det_file)
    seq = load_sequence(seq_dir)

    # Group detections by frame
    dets_by_frame: Dict[int, List[Detection]] = defaultdict(list)
    for d in dets:
        dets_by_frame[d.frame_id].append(d)

    max_frame = seq.frame_count
    if dets:
        max_frame = max(max_frame, max(d.frame_id for d in dets))

    # Build frame path lookup
    frame_dir = Path(seq.frame_dir)
    frame_files = sorted(frame_dir.glob("*.jpg")) + sorted(frame_dir.glob("*.png"))
    frame_path_map = {}
    for fp in frame_files:
        try:
            fid = int(fp.stem)
        except ValueError:
            continue
        frame_path_map[fid] = str(fp)

    tracker.reset()
    all_tracks: List[Detection] = []

    start_time = time.time()
    for frame_id in range(1, max_frame + 1):
        frame_dets = dets_by_frame.get(frame_id, [])
        frame_path = frame_path_map.get(frame_id, "")

        try:
            tracked = tracker.update(frame_id, frame_dets, frame_path)
            all_tracks.extend(tracked)
        except NotImplementedError as e:
            print(f"  SKIP: {tracker.name} — {e}")
            return 0, 0.0

    elapsed = time.time() - start_time

    write_mot_txt(all_tracks, output_file)
    return len(all_tracks), elapsed


def run_benchmark(
    det_dir: str,
    gt_dir: str,
    output_dir: str,
    tracker_names: List[str],
    sequences: List[str] | None = None,
):
    """
    Run all specified trackers on all sequences and collect results.
    """
    det_dir = Path(det_dir)
    gt_dir = Path(gt_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Discover sequences
    if sequences is None:
        sequences = sorted([
            p.name for p in gt_dir.iterdir()
            if p.is_dir() and (p / "gt" / "gt.txt").exists()
        ])

    print(f"\nSequences: {sequences}")
    print(f"Trackers: {tracker_names}")
    print(f"Detection dir: {det_dir}")
    print(f"GT dir: {gt_dir}")
    print(f"Output dir: {output_dir}\n")

    results = {}

    for tracker_name in tracker_names:
        if tracker_name not in TRACKER_REGISTRY:
            print(f"WARNING: unknown tracker '{tracker_name}', skipping")
            continue

        TrackerClass = TRACKER_REGISTRY[tracker_name]

        print(f"{'=' * 60}")
        print(f"TRACKER: {tracker_name} ({TrackerClass.paper}, {TrackerClass.year})")
        print(f"{'=' * 60}")

        try:
            tracker = TrackerClass()
        except Exception as e:
            print(f"  FAILED to create tracker: {e}")
            results[tracker_name] = {"error": str(e)}
            continue

        tracker_output_dir = output_dir / tracker_name
        tracker_results = {"sequences": {}, "config": tracker.get_config()}
        all_ok = True

        for seq_name in sequences:
            det_file = det_dir / f"{seq_name}.txt"
            seq_dir_path = gt_dir / seq_name

            if not det_file.exists():
                print(f"  {seq_name}: no detection file, skipping")
                continue
            if not seq_dir_path.exists():
                print(f"  {seq_name}: no GT directory, skipping")
                continue

            out_file = str(tracker_output_dir / f"{seq_name}.txt")

            try:
                n_tracks, elapsed = run_tracker_on_sequence(
                    tracker, str(det_file), str(seq_dir_path), out_file
                )
                if n_tracks == 0 and elapsed == 0:
                    # NotImplementedError was caught
                    all_ok = False
                    tracker_results["sequences"][seq_name] = {"status": "not_implemented"}
                    break

                fps = (load_sequence(str(seq_dir_path)).frame_count / elapsed) if elapsed > 0 else 0
                tracker_results["sequences"][seq_name] = {
                    "tracks": n_tracks,
                    "time_s": round(elapsed, 2),
                    "fps": round(fps, 1),
                    "status": "ok",
                }
                print(f"  {seq_name}: {n_tracks} track rows, {elapsed:.2f}s ({fps:.1f} FPS)")
            except Exception as e:
                print(f"  {seq_name}: FAILED — {e}")
                tracker_results["sequences"][seq_name] = {"status": "error", "error": str(e)}
                all_ok = False

        results[tracker_name] = tracker_results
        print()

    # Save results
    results_path = output_dir / "benchmark_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved benchmark results to {results_path}")

    # Print summary table
    print_summary(results, sequences)

    return results


def print_summary(results: dict, sequences: list):
    """Print a markdown summary table."""
    print("\n" + "=" * 80)
    print("TRACKER BENCHMARK SUMMARY")
    print("=" * 80)
    print(f"{'Tracker':<16} {'Status':<12} {'Total Tracks':<14} {'Avg FPS':<10} {'Note':<30}")
    print("-" * 80)

    for name, data in results.items():
        if "error" in data:
            print(f"{name:<16} {'ERROR':<12} {'—':<14} {'—':<10} {data['error'][:30]}")
            continue

        seqs = data.get("sequences", {})
        ok_seqs = [s for s in seqs.values() if s.get("status") == "ok"]

        if not ok_seqs:
            status = seqs.get(next(iter(seqs), ""), {}).get("status", "unknown")
            print(f"{name:<16} {status:<12} {'—':<14} {'—':<10} {'needs custom pipeline':<30}")
            continue

        total_tracks = sum(s["tracks"] for s in ok_seqs)
        avg_fps = np.mean([s["fps"] for s in ok_seqs]) if ok_seqs else 0

        print(f"{name:<16} {'OK':<12} {total_tracks:<14} {avg_fps:<10.1f} {'':<30}")


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Multi-tracker benchmark")
    parser.add_argument("--det_dir", required=True,
                        help="Detection files dir (e.g. eval/detections/rfdetr_coco/train)")
    parser.add_argument("--gt_dir", required=True,
                        help="GT sequence dir (e.g. data/vip_htd/train)")
    parser.add_argument("--output_dir", default="eval/tracker_benchmark",
                        help="Output directory for tracker results")
    parser.add_argument("--trackers", nargs="+",
                        default=["sort", "bytetrack"],
                        help="Tracker names to benchmark")
    parser.add_argument("--sequences", nargs="*", default=None,
                        help="Specific sequences to run (default: all)")
    parser.add_argument("--list", action="store_true",
                        help="List all known trackers and exit")

    args = parser.parse_args()

    if args.list:
        list_trackers()
        return

    run_benchmark(
        det_dir=args.det_dir,
        gt_dir=args.gt_dir,
        output_dir=args.output_dir,
        tracker_names=args.trackers,
        sequences=args.sequences,
    )


if __name__ == "__main__":
    main()
