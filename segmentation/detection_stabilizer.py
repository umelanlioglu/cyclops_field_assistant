from __future__ import annotations

import time
from typing import Any, Dict, List, Optional


def _now_ms() -> int:
    return int(time.time() * 1000)


def _center(box: List[float]) -> List[float]:
    x1, y1, x2, y2 = box
    return [(x1 + x2) / 2.0, (y1 + y2) / 2.0]


def _ema(old: List[float], new: List[float], alpha: float) -> List[float]:
    return [
        (1.0 - alpha) * float(o) + alpha * float(n)
        for o, n in zip(old, new)
    ]


class DetectionStabilizer:
    """
    Simple label-level stabilizer.

    This is enough for CR-10 Smart parts because most classes appear once:
    print_bed, display_screen, nozzle_kit, filament_holder, etc.

    If later you have multiple same-label instances, we can upgrade this to IoU tracking.
    """

    def __init__(
        self,
        conf_threshold: float = 0.35,
        min_hits: int = 2,
        max_missing_ms: int = 700,
        bbox_alpha: float = 0.35,
        conf_alpha: float = 0.45,
    ):
        self.conf_threshold = conf_threshold
        self.min_hits = min_hits
        self.max_missing_ms = max_missing_ms
        self.bbox_alpha = bbox_alpha
        self.conf_alpha = conf_alpha
        self.tracks: Dict[str, Dict[str, Any]] = {}

    def reset(self) -> None:
        self.tracks.clear()

    def update(
        self,
        detections: List[Dict[str, Any]],
        now_ms: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        now_ms = _now_ms() if now_ms is None else now_ms

        # Keep best detection per label.
        best_by_label: Dict[str, Dict[str, Any]] = {}

        for det in detections:
            label = det.get("label")
            conf = float(det.get("confidence", det.get("score", 0.0)))

            if not label or conf < self.conf_threshold:
                continue

            if label not in best_by_label:
                best_by_label[label] = det
            else:
                old_conf = float(best_by_label[label].get("confidence", 0.0))
                if conf > old_conf:
                    best_by_label[label] = det

        # Update matched tracks.
        for label, det in best_by_label.items():
            box = [float(x) for x in det["bbox"]]
            center = _center(box)

            if label not in self.tracks:
                self.tracks[label] = {
                    "label": label,
                    "bbox": box,
                    "stable_center": center,
                    "confidence": float(det.get("confidence", 1.0)),
                    "raw_label": det.get("raw_label", label),
                    "first_seen_ms": now_ms,
                    "last_seen_ms": now_ms,
                    "hits": 1,
                    "last_detection": dict(det),
                }
            else:
                tr = self.tracks[label]
                tr["bbox"] = _ema(tr["bbox"], box, self.bbox_alpha)
                tr["stable_center"] = _ema(tr["stable_center"], center, self.bbox_alpha)
                tr["confidence"] = (
                    (1.0 - self.conf_alpha) * float(tr["confidence"])
                    + self.conf_alpha * float(det.get("confidence", 1.0))
                )
                tr["last_seen_ms"] = now_ms
                tr["hits"] += 1
                tr["last_detection"] = dict(det)

        # Build stable output.
        output: List[Dict[str, Any]] = []

        for label in list(self.tracks.keys()):
            tr = self.tracks[label]
            age_missing = now_ms - int(tr["last_seen_ms"])

            if age_missing > self.max_missing_ms:
                del self.tracks[label]
                continue

            stable = int(tr["hits"]) >= self.min_hits

            # Keep latest raw segmentation/mask when available, but use smoothed bbox/center.
            base = dict(tr.get("last_detection", {}))
            base["label"] = label
            base["raw_label"] = tr.get("raw_label", label)
            base["confidence"] = float(tr["confidence"])
            base["bbox"] = [float(x) for x in tr["bbox"]]
            base["bbox_format"] = "xyxy"
            base["stable_center"] = [float(x) for x in tr["stable_center"]]
            base["stable_for_ms"] = max(0, now_ms - int(tr["first_seen_ms"]))
            base["missing_for_ms"] = max(0, age_missing)
            base["is_stable"] = stable
            base["track_id"] = label

            if stable:
                output.append(base)

        output.sort(key=lambda d: d["label"])
        return output