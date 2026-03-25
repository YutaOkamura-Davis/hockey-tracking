# hockey-tracking

Broadcast hockey player tracking + identity pipeline.

## Quickstart (Google Colab)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/FirebladeFN/hockey-tracking/blob/main/notebooks/01_v0_benchmark.ipynb)

The Colab notebook clones the repo, installs everything, creates synthetic data, runs RF-DETR detection, benchmarks all available trackers, and produces a comparison table. One click.

## Quickstart (Local)

```bash
git clone https://github.com/FirebladeFN/hockey-tracking.git
cd hockey-tracking
bash scripts/setup_colab.sh   # works on any Linux/Mac, not just Colab
```

## Tracker Benchmark

15 trackers tracked, 7 integrated. Run them all against the same frozen detections:

```bash
python src/tracking/benchmark_trackers.py --list
```

| Tracker | Year | SportsMOT HOTA | Status |
|---------|------|----------------|--------|
| MeMoSORT | 2025 | 82.1 | todo |
| CAMELTrack | 2025 | ~80 | todo |
| McByte | 2025 | ~79 | stub |
| Deep HM-SORT | 2024 | ~78 | todo |
| SportMamba | 2025 | ~78 | todo |
| Deep-EIoU | 2024 | ~77 | stub |
| DiffMOT | 2024 | ~76 | stub |
| BoT-SORT | 2022 | ~74 | ✅ integrated |
| OC-SORT | 2022 | ~73 | ✅ integrated |
| ByteTrack | 2021 | ~72 | ✅ integrated |
| DeepSORT | 2017 | ~67 | ✅ integrated |
| SORT | 2016 | ~65 | ✅ integrated |

```bash
# Run all available trackers
python src/tracking/benchmark_trackers.py \
  --det_dir eval/detections/rfdetr_coco/train \
  --gt_dir data/vip_htd/train \
  --output_dir eval/tracker_benchmark \
  --trackers sort bytetrack ocsort botsort
```

## Repo Layout

```
configs/     experiment configs
data/        datasets (gitignored)
src/         pipeline code (detection, tracking, homography, jersey, identity)
eval/        evaluation scripts (detection recall, TrackEval wrapper)
scripts/     CLI utilities (setup, data verification, format conversion)
notebooks/   Colab notebooks
docs/        benchmark tables and summaries
tests/       test suite (50+ assertions, synthetic data)
```

## Pipeline Commands

```bash
# 1. Verify dataset
python scripts/verify_vip_htd.py --base_dir data/vip_htd

# 2. Run RF-DETR detections
bash scripts/run_rfdetr_vip_htd.sh train 0.5 base

# 3. Evaluate detection recall
python eval/detection_eval.py \
  --gt_dir data/vip_htd \
  --det_dir eval/detections/rfdetr_coco \
  --split train

# 4. Benchmark trackers
python src/tracking/benchmark_trackers.py \
  --det_dir eval/detections/rfdetr_coco/train \
  --gt_dir data/vip_htd/train \
  --trackers sort bytetrack botsort

# 5. Run tests
python tests/test_phase0.py
```

## Adding a New Tracker

```python
from src.tracking.benchmark_trackers import TrackerAdapter, TRACKER_REGISTRY

class MyTracker(TrackerAdapter):
    name = "my_tracker"

    def reset(self):
        # init your tracker
        pass

    def update(self, frame_id, detections, frame_path=""):
        # detections in, tracks out (same Detection dataclass, with track_id set)
        pass

TRACKER_REGISTRY["my_tracker"] = MyTracker
```

Then run: `python src/tracking/benchmark_trackers.py --trackers my_tracker`
