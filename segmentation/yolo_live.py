# project/cr6se_pipeline/yolo_live.py

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from ultralytics import YOLO

from .labels import normalize_label


_MODEL = None


def load_yolo_model(weights_path: str | Path):
    global _MODEL
    if _MODEL is None:
        _MODEL = YOLO(str(weights_path))
    return _MODEL


def yolo_to_detections(
    image_path: str | Path,
    weights_path: str | Path,
    conf: float = 0.35,
    imgsz: int = 640,
) -> List[Dict[str, Any]]:
    model = load_yolo_model(weights_path)

    result = model(
        str(image_path),
        conf=conf,
        imgsz=imgsz,
        verbose=False,
    )[0]

    detections: List[Dict[str, Any]] = []

    if result.boxes is None:
        return detections

    names = model.names
    boxes = result.boxes
    masks = result.masks

    mask_polygons = []
    if masks is not None and masks.xy is not None:
        mask_polygons = masks.xy

    for i, box in enumerate(boxes):
        cls_id = int(box.cls.item())
        raw_label = names[cls_id]
        label = normalize_label(raw_label)

        confidence = float(box.conf.item())
        x1, y1, x2, y2 = box.xyxy[0].tolist()

        segmentation = []
        if i < len(mask_polygons):
            poly = mask_polygons[i]
            if poly is not None and len(poly) >= 3:
                flat_poly = poly.reshape(-1).tolist()
                segmentation = [flat_poly]

        detections.append({
            "label": label,
            "raw_label": raw_label,
            "confidence": confidence,
            "bbox": [float(x1), float(y1), float(x2), float(y2)],
            "bbox_format": "xyxy",
            "segmentation": segmentation,
            "source": "yolo_live",
        })

    return detections