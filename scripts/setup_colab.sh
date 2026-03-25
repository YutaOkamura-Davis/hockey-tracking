#!/usr/bin/env bash
# Hockey Tracking — Colab Setup
# Usage (from Colab): !bash scripts/setup_colab.sh
set -euo pipefail

echo "=== Hockey Tracking — Colab Setup ==="

# Core deps
pip install -q numpy scipy opencv-python-headless tqdm matplotlib pandas Pillow PyYAML

# Tracker backends
pip install -q supervision
pip install -q trackers 2>/dev/null || echo "WARN: roboflow trackers not installed (optional)"
pip install -q ultralytics 2>/dev/null || echo "WARN: ultralytics not installed (optional)"
pip install -q lapx 2>/dev/null || true

# Detection
pip install -q rfdetr

# Evaluation
pip install -q git+https://github.com/JonathonLuiten/TrackEval.git 2>/dev/null \
  || echo "WARN: TrackEval install failed — eval will be skipped"

echo ""
echo "=== Checking available trackers ==="
python -c "
backends = []
try:
    import trackers; backends.append('roboflow/trackers (SORT, ByteTrack)')
except: pass
try:
    import supervision; backends.append('supervision (ByteTrack)')
except: pass
try:
    from ultralytics.trackers.bot_sort import BOTSORT; backends.append('ultralytics (BoT-SORT)')
except: pass
try:
    from ultralytics.trackers.byte_tracker import BYTETracker; backends.append('ultralytics (ByteTrack)')
except: pass
try:
    from rfdetr import RFDETRBase; backends.append('RF-DETR (detection)')
except: pass
try:
    import trackeval; backends.append('TrackEval (evaluation)')
except: pass
print('Available:')
for b in backends:
    print(f'  ✓ {b}')
if not backends:
    print('  (none)')
"
echo ""
echo "=== Setup complete ==="
