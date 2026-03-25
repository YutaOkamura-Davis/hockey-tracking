#!/usr/bin/env python3
"""
Comprehensive test suite for hockey-tracking Phase 0 codebase.
Creates synthetic MOT-format data and validates every module.
"""
import json
import os
import sys
import tempfile
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def create_synthetic_dataset(base_dir: str, num_sequences: int = 2, num_frames: int = 20, num_objects: int = 4):
    """
    Create a synthetic MOT-format dataset with known ground truth.
    Objects move linearly with slight noise for realistic testing.
    """
    import random
    random.seed(42)

    base = Path(base_dir)
    for split in ["train", "test"]:
        for seq_idx in range(1, num_sequences + 1):
            seq_name = f"seq_{seq_idx:03d}"
            seq_dir = base / split / seq_name
            img_dir = seq_dir / "img1"
            gt_dir = seq_dir / "gt"
            img_dir.mkdir(parents=True, exist_ok=True)
            gt_dir.mkdir(parents=True, exist_ok=True)

            # Create dummy frame files (1-indexed, zero-padded)
            for f in range(1, num_frames + 1):
                # Create tiny valid PNG (1x1 pixel)
                (img_dir / f"{f:06d}.jpg").touch()

            # Create seqinfo.ini
            with open(seq_dir / "seqinfo.ini", "w") as f:
                f.write(f"[Sequence]\n")
                f.write(f"name={seq_name}\n")
                f.write(f"imDir=img1\n")
                f.write(f"frameRate=30\n")
                f.write(f"seqLength={num_frames}\n")
                f.write(f"imWidth=1920\n")
                f.write(f"imHeight=1080\n")
                f.write(f"imExt=.jpg\n")

            # Create GT: objects moving linearly
            gt_lines = []
            for obj_id in range(1, num_objects + 1):
                start_x = 100 + obj_id * 200
                start_y = 200 + obj_id * 100
                for frame in range(1, num_frames + 1):
                    x = start_x + frame * 5 + random.uniform(-2, 2)
                    y = start_y + frame * 2 + random.uniform(-1, 1)
                    w = 50 + random.uniform(-3, 3)
                    h = 120 + random.uniform(-5, 5)
                    conf = 1.0
                    # MOT format: frame,id,x,y,w,h,conf,class,visibility
                    gt_lines.append(f"{frame},{obj_id},{x:.2f},{y:.2f},{w:.2f},{h:.2f},{conf:.4f},1,1.0,-1")

            with open(gt_dir / "gt.txt", "w") as f:
                f.write("\n".join(gt_lines))

    print(f"Created synthetic dataset at {base_dir}")
    print(f"  Splits: train, test")
    print(f"  Sequences per split: {num_sequences}")
    print(f"  Frames per sequence: {num_frames}")
    print(f"  Objects per sequence: {num_objects}")
    return base_dir


def create_synthetic_detections(gt_dir: str, det_dir: str, split: str = "train",
                                noise_px: float = 5.0, miss_rate: float = 0.05,
                                fp_rate: float = 0.02, conf_noise: float = 0.1):
    """
    Create synthetic detections by perturbing GT.
    Simulates: slight box jitter, occasional misses, occasional false positives.
    """
    import random
    random.seed(123)

    gt_split = Path(gt_dir) / split
    det_split = Path(det_dir) / split
    det_split.mkdir(parents=True, exist_ok=True)

    for seq_dir in sorted([p for p in gt_split.iterdir() if p.is_dir()]):
        gt_file = seq_dir / "gt" / "gt.txt"
        if not gt_file.exists():
            continue

        det_lines = []
        with open(gt_file) as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) < 6:
                    continue

                # Randomly miss some detections
                if random.random() < miss_rate:
                    continue

                frame = int(float(parts[0]))
                x = float(parts[2]) + random.uniform(-noise_px, noise_px)
                y = float(parts[3]) + random.uniform(-noise_px, noise_px)
                w = float(parts[4]) + random.uniform(-noise_px/2, noise_px/2)
                h = float(parts[5]) + random.uniform(-noise_px/2, noise_px/2)
                conf = max(0.1, min(1.0, 0.9 + random.uniform(-conf_noise, conf_noise)))

                det_lines.append(f"{frame},-1,{x:.2f},{y:.2f},{w:.2f},{h:.2f},{conf:.4f},-1,-1,-1")

            # Add some false positives
            max_frame = max(int(float(l.split(",")[0])) for l in open(gt_file).readlines() if l.strip())
            num_fp = int(max_frame * fp_rate)
            for _ in range(num_fp):
                frame = random.randint(1, max_frame)
                x = random.uniform(0, 1800)
                y = random.uniform(0, 900)
                w = random.uniform(30, 80)
                h = random.uniform(80, 150)
                conf = random.uniform(0.3, 0.7)
                det_lines.append(f"{frame},-1,{x:.2f},{y:.2f},{w:.2f},{h:.2f},{conf:.4f},-1,-1,-1")

        out_file = det_split / f"{seq_dir.name}.txt"
        with open(out_file, "w") as f:
            f.write("\n".join(det_lines))

    print(f"Created synthetic detections at {det_dir}/{split}/")


# ============================================================
# TESTS
# ============================================================

def test_data_loader(data_dir):
    """Test: data_loader.py loads synthetic dataset correctly."""
    print("\n" + "=" * 60)
    print("TEST: data_loader")
    print("=" * 60)

    from src.detection.data_loader import load_dataset, load_sequence, load_mot_txt, write_mot_txt, Detection

    # Test load_dataset
    seqs = load_dataset(data_dir, "train")
    assert len(seqs) == 2, f"Expected 2 sequences, got {len(seqs)}"
    print(f"  ✓ load_dataset: found {len(seqs)} train sequences")

    # Test sequence properties
    seq = seqs[0]
    assert seq.fps == 30, f"Expected fps=30, got {seq.fps}"
    assert seq.width == 1920
    assert seq.height == 1080
    assert seq.frame_count == 20
    assert len(seq.gt) == 80, f"Expected 80 GT detections (4 objects × 20 frames), got {len(seq.gt)}"
    print(f"  ✓ Sequence metadata correct: {seq.name}, {seq.frame_count} frames, {len(seq.gt)} GT")

    # Test unique track IDs
    unique_ids = set(d.track_id for d in seq.gt)
    assert len(unique_ids) == 4, f"Expected 4 unique IDs, got {len(unique_ids)}"
    print(f"  ✓ Unique track IDs: {sorted(unique_ids)}")

    # Test Detection.to_mot_row roundtrip
    det = Detection(frame_id=1, track_id=5, x=100.0, y=200.0, w=50.0, h=120.0, confidence=0.95)
    row = det.to_mot_row()
    assert "1,5,100.00,200.00,50.00,120.00,0.9500" in row
    print(f"  ✓ Detection.to_mot_row: {row}")

    # Test write_mot_txt + load_mot_txt roundtrip
    tmp_path = os.path.join(data_dir, "_test_roundtrip.txt")
    test_dets = [
        Detection(1, 1, 10.0, 20.0, 30.0, 40.0, 0.9),
        Detection(1, 2, 50.0, 60.0, 70.0, 80.0, 0.8),
        Detection(2, 1, 15.0, 25.0, 30.0, 40.0, 0.85),
    ]
    write_mot_txt(test_dets, tmp_path)
    loaded = load_mot_txt(tmp_path)
    assert len(loaded) == 3, f"Roundtrip: expected 3, got {len(loaded)}"
    assert loaded[0].frame_id == 1
    assert loaded[0].track_id == 1
    assert abs(loaded[0].x - 10.0) < 0.1
    assert abs(loaded[0].confidence - 0.9) < 0.01
    os.remove(tmp_path)
    print(f"  ✓ write_mot_txt → load_mot_txt roundtrip OK")

    # Test missing file
    empty = load_mot_txt("/nonexistent/file.txt")
    assert empty == [], f"Expected empty list for missing file, got {empty}"
    print(f"  ✓ load_mot_txt on missing file returns []")

    # Test empty split
    empty_seqs = load_dataset(data_dir, "nonexistent_split")
    assert empty_seqs == []
    print(f"  ✓ load_dataset on missing split returns []")

    print("  ✅ data_loader: ALL PASSED")
    return True


def test_detection_eval(data_dir, det_dir):
    """Test: detection_eval.py computes correct metrics on synthetic data."""
    print("\n" + "=" * 60)
    print("TEST: detection_eval")
    print("=" * 60)

    from eval.detection_eval import compute_iou, match_frame, evaluate_detections, aggregate_results
    from src.detection.data_loader import Detection, load_sequence, load_mot_txt

    # Test IoU: identical boxes
    iou = compute_iou((0, 0, 10, 10), (0, 0, 10, 10))
    assert abs(iou - 1.0) < 1e-6, f"Identical boxes should have IoU=1.0, got {iou}"
    print(f"  ✓ IoU identical boxes: {iou:.4f}")

    # Test IoU: no overlap
    iou = compute_iou((0, 0, 10, 10), (20, 20, 10, 10))
    assert iou == 0.0, f"Non-overlapping boxes should have IoU=0.0, got {iou}"
    print(f"  ✓ IoU no overlap: {iou:.4f}")

    # Test IoU: partial overlap
    iou = compute_iou((0, 0, 10, 10), (5, 5, 10, 10))
    expected = 25.0 / 175.0  # intersection=5*5=25, union=100+100-25=175
    assert abs(iou - expected) < 1e-4, f"Expected IoU={expected:.4f}, got {iou:.4f}"
    print(f"  ✓ IoU partial overlap: {iou:.4f} (expected {expected:.4f})")

    # Test IoU: zero-area box
    iou = compute_iou((0, 0, 0, 0), (0, 0, 10, 10))
    assert iou == 0.0
    print(f"  ✓ IoU zero-area box: {iou:.4f}")

    # Test IoU: contained box
    iou = compute_iou((0, 0, 20, 20), (5, 5, 10, 10))
    expected = 100.0 / 400.0  # inner is 10x10=100, outer is 20x20=400, union=400
    assert abs(iou - expected) < 1e-4
    print(f"  ✓ IoU contained box: {iou:.4f}")

    # Test match_frame: empty cases
    tp, fp, fn = match_frame([], [], 0.5)
    assert (tp, fp, fn) == (0, 0, 0)
    print(f"  ✓ match_frame empty: tp={tp}, fp={fp}, fn={fn}")

    tp, fp, fn = match_frame([], [Detection(1, -1, 0, 0, 10, 10, 0.9)], 0.5)
    assert (tp, fp, fn) == (0, 1, 0)
    print(f"  ✓ match_frame no GT: tp={tp}, fp={fp}, fn={fn}")

    tp, fp, fn = match_frame([Detection(1, 1, 0, 0, 10, 10)], [], 0.5)
    assert (tp, fp, fn) == (0, 0, 1)
    print(f"  ✓ match_frame no pred: tp={tp}, fp={fp}, fn={fn}")

    # Test match_frame: perfect match
    gt = [Detection(1, 1, 0, 0, 10, 10), Detection(1, 2, 50, 50, 10, 10)]
    pred = [Detection(1, -1, 0, 0, 10, 10, 0.9), Detection(1, -1, 50, 50, 10, 10, 0.8)]
    tp, fp, fn = match_frame(gt, pred, 0.5)
    assert (tp, fp, fn) == (2, 0, 0), f"Expected (2,0,0), got ({tp},{fp},{fn})"
    print(f"  ✓ match_frame perfect: tp={tp}, fp={fp}, fn={fn}")

    # Test evaluate_detections on synthetic data
    seq_dir = Path(data_dir) / "train" / "seq_001"
    seq = load_sequence(str(seq_dir))
    det_file = Path(det_dir) / "train" / "seq_001.txt"
    preds = load_mot_txt(str(det_file))

    assert len(seq.gt) > 0, "No GT loaded"
    assert len(preds) > 0, "No predictions loaded"
    print(f"  ✓ Loaded {len(seq.gt)} GT, {len(preds)} predictions for seq_001")

    results = evaluate_detections(seq.gt, preds, iou_threshold=0.5)
    assert len(results) == 6, f"Expected 6 threshold results, got {len(results)}"

    # At low threshold, recall should be high (synthetic detections are close to GT)
    low_thresh = results[0]  # threshold=0.3
    assert low_thresh["recall"] > 0.8, f"Expected high recall at low threshold, got {low_thresh['recall']}"
    print(f"  ✓ Recall at conf=0.3: {low_thresh['recall']:.4f} (expected >0.8)")

    # At high threshold, some detections filtered out → lower recall or fewer preds
    high_thresh = results[-1]  # threshold=0.8
    print(f"  ✓ Recall at conf=0.8: {high_thresh['recall']:.4f}")

    # Test aggregate_results
    all_res = {"seq_001": results}
    agg = aggregate_results(all_res)
    assert len(agg) == 6
    assert agg[0]["threshold"] == 0.3
    print(f"  ✓ aggregate_results: {len(agg)} threshold levels")

    # Test with multiple sequences
    seq2_dir = Path(data_dir) / "train" / "seq_002"
    seq2 = load_sequence(str(seq2_dir))
    det2_file = Path(det_dir) / "train" / "seq_002.txt"
    preds2 = load_mot_txt(str(det2_file))
    results2 = evaluate_detections(seq2.gt, preds2)
    all_res["seq_002"] = results2
    agg2 = aggregate_results(all_res)
    # Aggregated TP should be sum of individual TPs
    for i in range(len(agg2)):
        expected_tp = results[i]["tp"] + results2[i]["tp"]
        assert agg2[i]["tp"] == expected_tp, f"Aggregated TP mismatch at threshold {agg2[i]['threshold']}"
    print(f"  ✓ aggregate_results multi-sequence: TP sums correct")

    print("  ✅ detection_eval: ALL PASSED")
    return True


def test_verify_vip_htd(data_dir):
    """Test: verify_vip_htd.py runs without error on synthetic data."""
    print("\n" + "=" * 60)
    print("TEST: verify_vip_htd")
    print("=" * 60)

    import subprocess
    result = subprocess.run(
        [sys.executable, "scripts/verify_vip_htd.py", "--base_dir", data_dir],
        capture_output=True, text=True, cwd=str(ROOT)
    )
    assert result.returncode == 0, f"verify_vip_htd failed:\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}"
    assert "seq_001" in result.stdout
    assert "seq_002" in result.stdout
    assert "2 sequences" in result.stdout
    print(f"  ✓ verify_vip_htd ran successfully")
    print(f"  Output preview:\n{result.stdout[:500]}")
    print("  ✅ verify_vip_htd: PASSED")
    return True


def test_mot_to_coco(data_dir):
    """Test: mot_to_coco.py produces valid COCO JSON."""
    print("\n" + "=" * 60)
    print("TEST: mot_to_coco")
    print("=" * 60)

    import subprocess
    out_path = os.path.join(data_dir, "train_coco.json")

    result = subprocess.run(
        [sys.executable, "scripts/mot_to_coco.py",
         "--mot_dir", os.path.join(data_dir, "train"),
         "--output", out_path,
         "--class_name", "player"],
        capture_output=True, text=True, cwd=str(ROOT)
    )
    assert result.returncode == 0, f"mot_to_coco failed:\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}"
    print(f"  ✓ mot_to_coco ran successfully")

    # Validate COCO JSON structure
    with open(out_path) as f:
        coco = json.load(f)

    assert "images" in coco
    assert "annotations" in coco
    assert "categories" in coco
    assert len(coco["categories"]) == 1
    assert coco["categories"][0]["name"] == "player"
    print(f"  ✓ COCO JSON structure valid")

    # Should have 2 sequences × 20 frames = 40 images
    assert len(coco["images"]) == 40, f"Expected 40 images, got {len(coco['images'])}"
    print(f"  ✓ Images: {len(coco['images'])}")

    # Should have 2 sequences × 4 objects × 20 frames = 160 annotations
    assert len(coco["annotations"]) == 160, f"Expected 160 annotations, got {len(coco['annotations'])}"
    print(f"  ✓ Annotations: {len(coco['annotations'])}")

    # Validate annotation fields
    ann = coco["annotations"][0]
    assert "id" in ann
    assert "image_id" in ann
    assert "category_id" in ann
    assert "bbox" in ann
    assert len(ann["bbox"]) == 4
    assert "area" in ann
    assert ann["area"] > 0
    print(f"  ✓ Annotation fields valid: {list(ann.keys())}")

    # Validate image IDs are unique
    img_ids = [img["id"] for img in coco["images"]]
    assert len(img_ids) == len(set(img_ids)), "Duplicate image IDs found"
    print(f"  ✓ Image IDs unique")

    # Validate annotation IDs are unique
    ann_ids = [a["id"] for a in coco["annotations"]]
    assert len(ann_ids) == len(set(ann_ids)), "Duplicate annotation IDs found"
    print(f"  ✓ Annotation IDs unique")

    # Validate all annotation image_ids reference existing images
    img_id_set = set(img_ids)
    for a in coco["annotations"]:
        assert a["image_id"] in img_id_set, f"Annotation references missing image_id {a['image_id']}"
    print(f"  ✓ All annotation image_ids valid")

    os.remove(out_path)
    print("  ✅ mot_to_coco: ALL PASSED")
    return True


def test_detection_eval_cli(data_dir, det_dir):
    """Test: detection_eval.py CLI runs end-to-end."""
    print("\n" + "=" * 60)
    print("TEST: detection_eval CLI")
    print("=" * 60)

    import subprocess
    json_out = os.path.join(det_dir, "eval_results.json")

    result = subprocess.run(
        [sys.executable, "eval/detection_eval.py",
         "--gt_dir", data_dir,
         "--det_dir", det_dir,
         "--split", "train",
         "--save_json", json_out],
        capture_output=True, text=True, cwd=str(ROOT)
    )
    assert result.returncode == 0, f"detection_eval CLI failed:\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}"
    print(f"  ✓ detection_eval CLI ran successfully")

    # Check output contains expected structure
    assert "OVERALL" in result.stdout
    assert "Recall" in result.stdout
    assert "GATE" in result.stdout
    print(f"  ✓ Output contains OVERALL, Recall, and GATE sections")

    # Check JSON output
    assert os.path.exists(json_out), f"JSON output not created: {json_out}"
    with open(json_out) as f:
        payload = json.load(f)
    assert "per_sequence" in payload
    assert "overall" in payload
    assert "seq_001" in payload["per_sequence"]
    assert len(payload["overall"]) == 6
    print(f"  ✓ JSON output valid: {len(payload['overall'])} threshold levels")

    # Check gate result is in output
    assert "GATE PASS" in result.stdout or "GATE FAIL" in result.stdout
    print(f"  ✓ Gate decision present in output")

    # Print actual output for inspection
    print(f"\n  CLI output:\n{result.stdout}")

    os.remove(json_out)
    print("  ✅ detection_eval CLI: PASSED")
    return True


def test_data_loader_edge_cases():
    """Test: data_loader handles edge cases."""
    print("\n" + "=" * 60)
    print("TEST: data_loader edge cases")
    print("=" * 60)

    from src.detection.data_loader import load_mot_txt, Detection, parse_seqinfo

    # Test parsing malformed MOT rows
    tmp = Path(ROOT / "_test_malformed.txt")
    with open(tmp, "w") as f:
        f.write("1,1,100,200,50,120,0.9,1,1.0,-1\n")  # valid
        f.write("bad line\n")  # invalid — should skip
        f.write("2,2,150\n")  # too few fields — should skip
        f.write("3,3,200,300,60,130,0.8,-1,0.5,-1\n")  # valid
        f.write("\n")  # empty line
    dets = load_mot_txt(str(tmp))
    assert len(dets) == 2, f"Expected 2 valid detections from malformed file, got {len(dets)}"
    assert dets[0].frame_id == 1
    assert dets[1].frame_id == 3
    tmp.unlink()
    print(f"  ✓ Malformed MOT file: correctly parsed 2 of 5 lines")

    # Test seqinfo with missing keys
    tmp_dir = Path(ROOT / "_test_seq")
    tmp_dir.mkdir(exist_ok=True)
    with open(tmp_dir / "seqinfo.ini", "w") as f:
        f.write("[Sequence]\nframeRate=25\n")  # missing most fields
    info = parse_seqinfo(tmp_dir)
    assert info.get("frameRate") == "25"
    assert "imWidth" not in info  # missing, should not crash
    (tmp_dir / "seqinfo.ini").unlink()
    tmp_dir.rmdir()
    print(f"  ✓ Partial seqinfo: parsed without error")

    # Test parse_seqinfo with no file
    tmp_dir2 = Path(ROOT / "_test_seq2")
    tmp_dir2.mkdir(exist_ok=True)
    info2 = parse_seqinfo(tmp_dir2)
    assert info2 == {}
    tmp_dir2.rmdir()
    print(f"  ✓ Missing seqinfo: returns empty dict")

    print("  ✅ data_loader edge cases: ALL PASSED")
    return True


def test_detection_eval_edge_cases():
    """Test: detection_eval handles degenerate inputs."""
    print("\n" + "=" * 60)
    print("TEST: detection_eval edge cases")
    print("=" * 60)

    from eval.detection_eval import compute_iou, evaluate_detections
    from src.detection.data_loader import Detection

    # All GT with confidence=0 (should be filtered)
    gt = [Detection(1, 1, 0, 0, 10, 10, confidence=0)]
    pred = [Detection(1, -1, 0, 0, 10, 10, 0.9)]
    results = evaluate_detections(gt, pred, confidence_thresholds=[0.5])
    assert results[0]["fn"] == 0, "GT with conf=0 should be filtered out"
    assert results[0]["fp"] == 1, "Pred matching filtered GT should be FP"
    print(f"  ✓ GT with confidence=0 filtered correctly")

    # Empty GT and predictions
    results = evaluate_detections([], [], confidence_thresholds=[0.5])
    assert results[0]["tp"] == 0 and results[0]["fp"] == 0 and results[0]["fn"] == 0
    print(f"  ✓ Empty inputs: all zeros")

    # Many objects, one frame
    gt = [Detection(1, i, i * 100, 0, 50, 50) for i in range(10)]
    pred = [Detection(1, -1, i * 100, 0, 50, 50, 0.9) for i in range(10)]
    results = evaluate_detections(gt, pred, confidence_thresholds=[0.5])
    assert results[0]["tp"] == 10
    assert results[0]["fp"] == 0
    assert results[0]["fn"] == 0
    print(f"  ✓ 10 objects perfect match: tp=10, fp=0, fn=0")

    # Negative IoU edge case (negative coordinates)
    iou = compute_iou((-10, -10, 5, 5), (0, 0, 5, 5))
    assert iou == 0.0, f"Non-overlapping negative coords should have IoU=0, got {iou}"
    print(f"  ✓ Negative coordinates handled")

    print("  ✅ detection_eval edge cases: ALL PASSED")
    return True


def main():
    print("=" * 60)
    print("HOCKEY-TRACKING PHASE 0 — CODE VALIDATION")
    print("=" * 60)

    # Create temp dataset
    data_dir = str(ROOT / "data" / "_test_synthetic")
    det_dir = str(ROOT / "eval" / "_test_detections")

    try:
        # Setup
        create_synthetic_dataset(data_dir, num_sequences=2, num_frames=20, num_objects=4)
        create_synthetic_detections(data_dir, det_dir, split="train")

        # Run all tests
        results = {}
        results["data_loader"] = test_data_loader(data_dir)
        results["data_loader_edge"] = test_data_loader_edge_cases()
        results["detection_eval"] = test_detection_eval(data_dir, det_dir)
        results["detection_eval_edge"] = test_detection_eval_edge_cases()
        results["detection_eval_cli"] = test_detection_eval_cli(data_dir, det_dir)
        results["verify_vip_htd"] = test_verify_vip_htd(data_dir)
        results["mot_to_coco"] = test_mot_to_coco(data_dir)

        # Summary
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        all_passed = True
        for name, passed in results.items():
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"  {status}: {name}")
            if not passed:
                all_passed = False

        if all_passed:
            print(f"\n🎉 ALL {len(results)} TEST SUITES PASSED")
        else:
            failed = [n for n, p in results.items() if not p]
            print(f"\n💥 {len(failed)} TEST SUITE(S) FAILED: {failed}")
            sys.exit(1)

    finally:
        # Cleanup
        import shutil
        if os.path.exists(data_dir):
            shutil.rmtree(data_dir)
        if os.path.exists(det_dir):
            shutil.rmtree(det_dir)
        print(f"\nCleaned up test artifacts")


if __name__ == "__main__":
    main()
