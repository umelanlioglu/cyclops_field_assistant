"""Vision utilities: COCO -> mock detections, scene JSON, drawing."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from .data_loading import coco_category_map
from .labels import normalize_label


def coco_to_detections(
    coco: Dict[str, Any],
    aliases: Optional[Dict[str, str]] = None,
    confidence: float = 1.0,
) -> List[Dict[str, Any]]:
    """Convert a single-image COCO annotation into mock YOLO-like detections."""
    categories = coco_category_map(coco)
    image_id = coco["images"][0]["id"]
    detections: List[Dict[str, Any]] = []

    for ann in coco.get("annotations", []):
        if ann.get("image_id") != image_id:
            continue

        raw_label = categories[ann["category_id"]]
        label = normalize_label(raw_label, aliases)
        x, y, w, h = ann["bbox"]

        detections.append({
            "label": label,
            "raw_label": raw_label,
            "confidence": confidence,
            "bbox": [float(x), float(y), float(x + w), float(y + h)],
            "bbox_format": "xyxy",
            "segmentation": ann.get("segmentation", []),
            "area": ann.get("area"),
        })

    return detections


def build_scene_json(
    detections: List[Dict[str, Any]],
    conf_threshold: float = 0.5,
    source: str = "mock_yolo_from_coco",
) -> Dict[str, Any]:
    visible_objects = sorted({
        d["label"]
        for d in detections
        if d.get("confidence", 0.0) >= conf_threshold
    })

    return {
        "visible_objects": visible_objects,
        "detections": detections,
        "source": source,
    }


def select_target_by_center(
    detections: List[Dict[str, Any]],
    image_width: int,
    image_height: int,
) -> Optional[Dict[str, Any]]:
    """For questions like 'what is this?', choose object closest to image center."""
    if not detections:
        return None

    center_x = image_width / 2
    center_y = image_height / 2
    best_det = None
    best_dist = float("inf")

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        obj_x = (x1 + x2) / 2
        obj_y = (y1 + y2) / 2
        dist = ((obj_x - center_x) ** 2 + (obj_y - center_y) ** 2) ** 0.5
        if dist < best_dist:
            best_dist = dist
            best_det = det

    return best_det


def color_for_label(label: str) -> Tuple[int, int, int]:
    h = abs(hash(label)) % 255
    return int(h), int((h * 2) % 255), int((h * 3) % 255)


def draw_referenced_objects(
    image_path: str | Path,
    detections: List[Dict[str, Any]],
    referenced_objects: List[str],
    output_path: str | Path,
    use_masks: bool = True,
) -> Path:
    image_path = Path(image_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    overlay = img.copy()
    refs = set(referenced_objects)

    for det in detections:
        label = det["label"]
        if label not in refs:
            continue

        color = color_for_label(label)

        if use_masks:
            for poly in det.get("segmentation", []):
                if not poly:
                    continue
                pts = np.asarray(poly, dtype=np.int32).reshape(-1, 2)
                cv2.fillPoly(overlay, [pts], color)
                cv2.polylines(img, [pts], isClosed=True, color=color, thickness=3)

        x1, y1, x2, y2 = map(int, det["bbox"])
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        cv2.putText(
            img,
            label,
            (x1, max(y1 - 10, 25)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2,
        )

    img = cv2.addWeighted(overlay, 0.25, img, 0.75, 0)
    out_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(output_path), out_bgr)
    return output_path
