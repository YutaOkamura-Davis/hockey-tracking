#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np
from tqdm import tqdm


def load_detector(model_size: str = "base", checkpoint: str = "", device: str = "cuda"):
    """
    Best-effort RF-DETR loader.
    This may need a small patch depending on the installed rfdetr version.
    """
    if model_size == "base":
        from rfdetr import RFDETRBase
        model = RFDETRBase()
    elif model_size == "large":
        from rfdetr import RFDETRLarge
        model = RFDETRLarge()
    else:
        raise ValueError(f"Unsupported model_size: {model_size}")

    if checkpoint:
        if hasattr(model, "load_weights"):
            model.load_weights(checkpoint)
        elif hasattr(model, "load_state_dict"):
            import torch
            state = torch.load(checkpoint, map_location=device)
            model.load_state_dict(state)
        else:
            print(f"WARNING: checkpoint provided but no known load method found: {checkpoint}")

    if hasattr(model, "to"):
        model = model.to(device)

    return model


def _to_numpy(x):
    if x is None:
        return None
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu()
    if hasattr(x, "numpy"):
        return x.numpy()
    return np.asarray(x)


def _normalize_prediction_output(pred: Any) -> List[Dict[str, Any]]:
    """
    Normalize different possible RF-DETR prediction outputs into:
    [{"bbox": [x1,y1,x2,y2], "score": float, "label": int|str}, ...]
    """
    out = []

    # Case 1: dict-like batch result
    if isinstance(pred, dict):
        boxes = _to_numpy(pred.get("boxes"))
        scores = _to_numpy(pred.get("scores"))
        labels = _to_numpy(pred.get("labels"))
        if boxes is not None:
            for i in range(len(boxes)):
                out.append(
                    {
                        "bbox": boxes[i].tolist(),
                        "score": float(scores[i]) if scores is not None else 1.0,
                        "label": labels[i].item() if labels is not None and hasattr(labels[i], "item") else (labels[i] if labels is not None else -1),
                    }
                )
            return out

    # Case 2: object with boxes/scores/labels attributes
    if hasattr(pred, "boxes"):
        boxes = _to_numpy(getattr(pred, "boxes"))
        scores = _to_numpy(getattr(pred, "scores", None))
        labels = _to_numpy(getattr(pred, "labels", None))
        if boxes is not None:
            for i in range(len(boxes)):
                out.append(
                    {
                        "bbox": boxes[i].tolist(),
                        "score": float(scores[i]) if scores is not None else 1.0,
                        "label": labels[i].item() if labels is not None and hasattr(labels[i], "item") else (labels[i] if labels is not None else -1),
                    }
                )
            return out

    # Case 3: list of det dicts or objects
    if isinstance(pred, list):
        # Batched result: [single_image_result]
        if len(pred) == 1 and (
            (isinstance(pred[0], dict) and "boxes" in pred[0])
            or hasattr(pred[0], "boxes")
            or isinstance(pred[0], list)
        ):
            nested = _normalize_prediction_output(pred[0])
            if nested:
                return nested

        for item in pred:
            if isinstance(item, dict):
                bbox = item.get("bbox") or item.get("xyxy") or item.get("box")
                score = item.get("score", item.get("confidence", 1.0))
                label = item.get("label", item.get("class_id", -1))
                if bbox is not None:
                    out.append({"bbox": list(map(float, bbox)), "score": float(score), "label": label})
            else:
                bbox = getattr(item, "bbox", None) or getattr(item, "xyxy", None)
                score = getattr(item, "score", getattr(item, "confidence", 1.0))
                label = getattr(item, "label", getattr(item, "class_id", -1))
                if bbox is not None:
                    out.append({"bbox": list(map(float, bbox)), "score": float(score), "label": label})
        if out:
            return out

    raise TypeError(f"Unsupported RF-DETR prediction format: {type(pred)}")


def _label_is_person(label, target_class_ids=(0, 1), target_class_names=("person",)):
    if isinstance(label, str):
        return label.lower() in {n.lower() for n in target_class_names}
    try:
        return int(label) in set(target_class_ids)
    except Exception:
        return False


def frame_id_from_name(frame_path: Path, default_idx: int) -> int:
    stem = frame_path.stem
    try:
        return int(stem)
    except ValueError:
        return default_idx


def predict_image(model, image_path: str, threshold: float):
    try:
        return model.predict(image_path, threshold=threshold)
    except TypeError:
        return model.predict(image_path)


def detect_sequence(
    model,
    frame_dir: str,
    output_path: str,
    confidence_threshold: float = 0.5,
    target_class_ids=(0, 1),
    target_class_names=("person",),
):
    frame_dir = Path(frame_dir)
    frames = sorted(frame_dir.glob("*.jpg")) + sorted(frame_dir.glob("*.png"))
    results = []

    for idx, frame_path in enumerate(tqdm(frames, desc=f"Detecting {frame_dir.parent.name}"), start=1):
        img = cv2.imread(str(frame_path))
        if img is None:
            continue

        raw_pred = predict_image(model, str(frame_path), confidence_threshold)
        detections = _normalize_prediction_output(raw_pred)

        frame_id = frame_id_from_name(frame_path, idx)

        for det in detections:
            bbox = det["bbox"]
            if len(bbox) != 4:
                continue
            x1, y1, x2, y2 = map(float, bbox)
            score = float(det["score"])
            label = det["label"]

            if score < confidence_threshold:
                continue
            if not _label_is_person(label, target_class_ids, target_class_names):
                continue

            w = max(0.0, x2 - x1)
            h = max(0.0, y2 - y1)
            if w <= 0 or h <= 0:
                continue

            results.append(
                f"{frame_id},-1,{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},{score:.4f},-1,-1,-1"
            )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(results))

    print(f"Wrote {len(results)} detections -> {output_path}")
    return len(results), len(frames)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame_dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--model_size", default="base", choices=["base", "large"])
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    model = load_detector(args.model_size, args.checkpoint, args.device)
    detect_sequence(
        model=model,
        frame_dir=args.frame_dir,
        output_path=args.output,
        confidence_threshold=args.threshold,
    )


if __name__ == "__main__":
    main()
