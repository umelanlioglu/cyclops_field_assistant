"""
Visual guidance drawing for CR-10 Smart assistant.

Goal:
- Decide which detected objects should be visually highlighted.
- Use predicted segmentation masks when available.
- Draw arrows + short captions.
- Keep the decision grounded in llm_json["referenced_objects"],
  verification["referenced_objects"], and chunk highlight objects.

This file does NOT decide the answer.
It only visualizes the objects that the generated answer decided are relevant.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
import math
import hashlib

import numpy as np
from PIL import Image, ImageDraw, ImageFont

PRINT_BED_FULL_AREA_BOX = True
PRINT_BED_FULL_AREA_FILL_ALPHA = 18      # lower = less washed out
PRINT_BED_FULL_AREA_OUTLINE_WIDTH = 9    # higher = stronger border
GUIDANCE_MASK_ALPHA = 115                # higher = stronger object masks
GUIDANCE_BBOX_WIDTH = 12
GUIDANCE_MASK_BBOX_WIDTH = 9
GUIDANCE_ARROW_WIDTH = 20
GUIDANCE_ARROW_HEAD_SIZE = 48
GUIDANCE_LABEL_OUTLINE_WIDTH = 10

# Guidance UI behavior
# - Prefer arrow targets on actual segmentation pixels/polygon area.
# - Hide duplicate object-name sublabels by default; captions should carry the instruction.
DRAW_OBJECT_SUBLABEL = False
ARROW_ANCHOR_MAX_RASTER_AREA = 250_000


# YOLO does not decide colors; this renderer does.
LABEL_COLOR_OVERRIDES = {
    "bed_clamp": (255, 215, 0),
    "bowden_tube": (245, 245, 245),
    "button": (255, 105, 180),
    "display_screen": (0, 190, 255),
    "extruder_motor": (180, 80, 255),
    "filament_detector": (255, 220, 0),
    "filament_holder": (90, 120, 255),
    "filament_spool": (0, 210, 120),
    "gantry_frame": (255, 140, 0),
    "network_interface": (0, 255, 200),
    "nozzle_kit": (255, 170, 0),
    "power_switch": (255, 60, 60),
    "print_bed": (0, 230, 80),
    "qr_code": (220, 220, 220),
    "sd_card_port": (160, 255, 80),
    "side_ports": (80, 180, 255),
    "toolbox": (255, 100, 0),
    "usb_port": (0, 150, 255),
    "x_axis_gantry": (255, 120, 220),
    "x_axis_motor": (200, 100, 255),
    "y_axis_belt_adjuster": (255, 90, 90),
    "y_axis_motor": (120, 255, 180),
}

# ---------------------------------------------------------------------
# Display/caption helpers
# ---------------------------------------------------------------------

PRETTY_LABELS = {
    "bed_clamp": "Bed clamp",
    "bowden_tube": "Bowden / Teflon tube",
    "button": "Side button",
    "display_screen": "Display screen",
    "extruder_motor": "Extruder motor",
    "filament_detector": "Filament detector",
    "filament_holder": "Filament holder",
    "filament_spool": "Filament spool",
    "gantry_frame": "Gantry frame",
    "network_interface": "Network / LAN interface",
    "nozzle_kit": "Nozzle kit",
    "power_switch": "Power switch",
    "print_bed": "Print bed",
    "qr_code": "QR code",
    "sd_card_port": "SD/TF card port",
    "side_ports": "Side ports area",
    "toolbox": "Toolbox knob",
    "usb_port": "USB port",
    "x_axis_gantry": "X-axis gantry",
    "x_axis_motor": "X-axis motor",
    "y_axis_belt_adjuster": "Y-axis belt adjuster",
    "y_axis_motor": "Y-axis motor",
}


# Legacy names are normalized here so older chunks/LLM outputs do not break
# visualization. New chunks should still use only the canonical YOLO labels.
VISUAL_LABEL_ALIASES = {
    "bowen_tube": "bowden_tube",
    "filament": "filament_spool",
    "nozzle": "nozzle_kit",
    "nozzle_assembly": "nozzle_kit",
    "hotend": "nozzle_kit",
    "hot_end": "nozzle_kit",
    "extruder": "extruder_motor",
    "extruder_area": "extruder_motor",
    "sd_card": "sd_card_port",
    "sd_card_slot": "sd_card_port",
    "tf_card_slot": "sd_card_port",
    "storage_card_slot": "sd_card_port",
    "tool_box": "toolbox",
    "x_axis_belt_tension_knob": "x_axis_gantry",
    "y_axis_belt_tension_knob": "y_axis_belt_adjuster",
    "switch_control": "power_switch",
    "power_outlet": "power_switch",
    "power_port_switch": "power_switch",
    "lcd_screen": "display_screen",
    "screen": "display_screen",
    "display": "display_screen",
    "teflon_tube": "bowden_tube",
    "printing_platform": "print_bed",
    "platform": "print_bed",
    "lan_port": "network_interface",
    "usb": "usb_port",
    "gantry": "gantry_frame",
    "x_axis_bar": "x_axis_gantry",
    "pull_rod": "gantry_frame",
    "printer_base": "print_bed",
    "cable_connection_area": "side_ports",
    "z_axis_photoelectric_switch": "gantry_frame",
    "power_cord": "power_switch",
    "glass_handle_plate": "bed_clamp",
}


def normalize_visual_label(label: str) -> str:
    clean = str(label or "").strip().lower().replace(" ", "_").replace("-", "_")
    return VISUAL_LABEL_ALIASES.get(clean, clean)


ACTION_CAPTION_HINTS = {
    "identify_part": {
        "bed_clamp": "Bed clamp",
        "bowden_tube": "Bowden tube",
        "button": "Side button",
        "display_screen": "Display screen",
        "extruder_motor": "Extruder motor",
        "filament_detector": "Filament detector",
        "filament_holder": "Filament holder",
        "filament_spool": "Filament spool",
        "gantry_frame": "Gantry frame",
        "network_interface": "Network/LAN port",
        "nozzle_kit": "Nozzle kit",
        "power_switch": "Power switch",
        "print_bed": "Print bed",
        "qr_code": "QR code",
        "sd_card_port": "SD/TF card port",
        "side_ports": "Side ports area",
        "toolbox": "Toolbox area",
        "usb_port": "USB port",
        "x_axis_gantry": "X-axis gantry",
        "x_axis_motor": "X-axis motor",
        "y_axis_belt_adjuster": "Y-axis belt adjuster",
        "y_axis_motor": "Y-axis motor",
    },
    "load_filament": {
        "filament_spool": "Place spool here",
        "filament_holder": "Hang spool here",
        "filament_detector": "Pass through detector",
        "extruder_motor": "Feed into extruder",
        "nozzle_kit": "Wait until heated",
        "bowden_tube": "Filament path",
        "display_screen": "Check temperature",
    },
    "replace_filament": {
        "filament_spool": "New spool",
        "filament_holder": "Spool holder",
        "filament_detector": "Detector path",
        "extruder_motor": "Cut/withdraw here",
        "nozzle_kit": "Preheat first",
        "bowden_tube": "Filament path",
        "display_screen": "Check temperature",
    },
    "preheating": {
        "display_screen": "Select preheat",
        "nozzle_kit": "Nozzle heats here",
        "print_bed": "Bed also heats",
    },
    "bed_leveling": {
        "display_screen": "Select Level",
        "print_bed": "Leveling area",
        "nozzle_kit": "Do not touch",
        "bowden_tube": "Do not touch",
    },
    "cable_connection": {
        "nozzle_kit": "Nozzle adapter area",
        "gantry_frame": "Cable routing area",
        "side_ports": "Cable/port area",
        "power_switch": "Power off first",
        "display_screen": "Check power state",
    },
    "power_connection": {
        "power_switch": "Power switch",
        "display_screen": "Check power state",
        "side_ports": "Side port area",
    },
    "power_safety": {
        "power_switch": "Turn off first",
        "nozzle_kit": "May be hot",
        "print_bed": "May be hot",
        "gantry_frame": "Moving frame",
    },
    "cleaning_maintenance": {
        "print_bed": "Clean surface",
        "nozzle_kit": "Avoid hot nozzle",
        "gantry_frame": "Wipe rails/frame",
        "power_switch": "Power off first",
    },
    "start_printing_storage_card": {
        "sd_card_port": "Insert card here",
        "display_screen": "Select print file",
    },
    "storage_card_safety": {
        "sd_card_port": "Do not remove while printing",
        "display_screen": "Check print state",
    },
    "print_file_troubleshooting": {
        "sd_card_port": "Check card port",
        "display_screen": "Check file list",
    },
    "usb_online_printing": {
        "usb_port": "USB connection",
        "side_ports": "Side ports area",
        "display_screen": "Check printer state",
    },
    "wifi_printing": {
        "display_screen": "Wi-Fi setup",
        "network_interface": "Network/LAN interface",
        "side_ports": "Side ports area",
        "button": "Side button",
        "qr_code": "Scan QR code",
        "usb_port": "Nearby USB port",
    },
    "belt_platform_adjustment": {
        "y_axis_belt_adjuster": "Adjust carefully",
        "print_bed": "Platform area",
        "bed_clamp": "Bed clamp",
        "x_axis_gantry": "X-axis area",
    },
    "pull_rod_installation": {
        "gantry_frame": "Attach to frame",
        "x_axis_gantry": "Upper frame area",
        "print_bed": "Base area",
    },
    "gantry_frame_installation": {
        "gantry_frame": "Install frame here",
        "x_axis_gantry": "Raise X-axis",
        "print_bed": "Base/frame area",
    },
    "filament_holder_installation": {
        "filament_holder": "Mount holder here",
        "filament_spool": "Spool goes here",
        "gantry_frame": "Z-axis frame area",
    },
    "tools_and_parts": {
        "gantry_frame": "Gantry frame",
        "display_screen": "Display unit",
        "filament_holder": "Rack/holder",
        "filament_spool": "Spool/filament",
        "nozzle_kit": "Nozzle/spares area",
    },
    "circuit_wiring": {
        "side_ports": "External port area",
        "usb_port": "USB port",
        "sd_card_port": "SD/TF card port",
        "network_interface": "Network interface",
        "display_screen": "Screen interface",
        "button": "Wi-Fi reset/side button",
        "power_switch": "Power area",
        "y_axis_motor": "Y-axis motor",
    },
    "extruder_troubleshooting": {
        "filament_detector": "Check detector path",
        "extruder_motor": "Check extruder",
        "nozzle_kit": "Check nozzle flow",
        "filament_spool": "Check spool feed",
        "bowden_tube": "Check filament path",
        "display_screen": "Check temperature",
    },
    "heating_troubleshooting": {
        "display_screen": "Check temperature",
        "nozzle_kit": "Nozzle heating",
        "print_bed": "Bed heating",
    },
    "motion_troubleshooting": {
        "x_axis_gantry": "X-axis movement",
        "x_axis_motor": "X-axis motor",
        "y_axis_motor": "Y-axis motor",
        "print_bed": "Bed movement",
        "gantry_frame": "Frame movement",
    },
    "firmware_support": {
        "display_screen": "Check printer info",
        "sd_card_port": "TF/SD card instructions",
        "qr_code": "Support/app code",
    },
    "start_printing_software": {
        "sd_card_port": "Save G-code card",
        "usb_port": "USB transfer option",
        "display_screen": "Select print file",
        "side_ports": "Port area",
    },
    "specifications": {
        "print_bed": "Build volume/bed",
        "nozzle_kit": "Nozzle specs",
        "filament_spool": "Filament type",
        "display_screen": "Printer interface",
        "sd_card_port": "Storage card transfer",
        "usb_port": "USB transfer",
    },
}


def pretty_label(label: str) -> str:
    label = normalize_visual_label(label)
    return PRETTY_LABELS.get(label, label.replace("_", " ").title())


def short_caption(
    label: str,
    action_id: Optional[str] = None,
    llm_json: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Caption priority:
    1. Optional llm_json["visual_captions"][label] if you later add it.
    2. Action-specific hardcoded caption.
    3. Pretty object label.
    """
    llm_json = llm_json or {}
    label = normalize_visual_label(label)

    visual_captions = llm_json.get("visual_captions") or {}
    if isinstance(visual_captions, dict) and label in visual_captions:
        return str(visual_captions[label])[:42]

    if action_id in ACTION_CAPTION_HINTS:
        if label in ACTION_CAPTION_HINTS[action_id]:
            return ACTION_CAPTION_HINTS[action_id][label]

    return pretty_label(label)

def stable_color_for_label(label: str) -> Tuple[int, int, int]:
    label = normalize_visual_label(label)

    if label in LABEL_COLOR_OVERRIDES:
        return LABEL_COLOR_OVERRIDES[label]

    digest = hashlib.md5(label.encode("utf-8")).hexdigest()

    r = 60 + int(digest[0:2], 16) % 170
    g = 60 + int(digest[2:4], 16) % 170
    b = 60 + int(digest[4:6], 16) % 170

    return r, g, b

# ---------------------------------------------------------------------
# Target selection
# ---------------------------------------------------------------------

def _dedupe_keep_order(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for item in items:
        if item and item not in seen:
            out.append(item)
            seen.add(item)
    return out


def detection_label(det: Dict[str, Any]) -> str:
    for key in ["label", "class_name", "category_name", "name", "class", "object"]:
        if key in det and det[key] is not None:
            return normalize_visual_label(str(det[key]))
    return ""


def visible_labels_from_detections(detections: List[Dict[str, Any]]) -> set:
    return {detection_label(d) for d in detections if detection_label(d)}


def should_make_visual(llm_json: Optional[Dict[str, Any]], referenced_objects: List[str]) -> bool:
    """
    Use the LLM's answer-level visual decision.
    If the model explicitly says no visual annotation, do not draw.
    """
    llm_json = llm_json or {}

    if not referenced_objects:
        return False

    if llm_json.get("needs_visual_annotation") is False:
        return False

    return True


def build_visual_targets(
    llm_json: Optional[Dict[str, Any]],
    verification: Dict[str, Any],
    selected_chunk: Optional[Dict[str, Any]],
    detections: List[Dict[str, Any]],
    max_targets: int = 4,
) -> List[Dict[str, str]]:
    """
    Decide which object labels to highlight.

    Priority:
    1. llm_json["referenced_objects"] from the generated response.
    2. verification["referenced_objects"].
    3. selected_chunk["highlight_objects"].

    Then:
    - Only keep objects actually detected.
    - Do not highlight objects listed as missing/uncertain.
    - Limit to max_targets to avoid clutter.
    """
    llm_json = llm_json or {}
    selected_chunk = selected_chunk or {}

    visible = visible_labels_from_detections(detections)

    labels = []
    labels.extend(llm_json.get("referenced_objects") or [])
    labels.extend(verification.get("referenced_objects") or [])
    labels.extend(selected_chunk.get("highlight_objects") or [])

    labels = _dedupe_keep_order([normalize_visual_label(str(x)) for x in labels])

    missing_or_uncertain = {
        normalize_visual_label(str(x))
        for x in (llm_json.get("missing_or_uncertain_objects") or [])
    }
    missing_or_uncertain.update(
        normalize_visual_label(str(x))
        for x in (verification.get("missing_required") or [])
    )

    labels = [
        label for label in labels
        if label in visible and label not in missing_or_uncertain
    ]

    if not should_make_visual(llm_json, labels):
        return []

    action_id = selected_chunk.get("action_id", selected_chunk.get("id"))

    targets = []
    for label in labels[:max_targets]:
        targets.append({
            "label": label,
            "caption": short_caption(label, action_id=action_id, llm_json=llm_json),
        })

    return targets


# ---------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------

def bbox_xyxy(det: Dict[str, Any], image_size: Tuple[int, int]) -> Tuple[int, int, int, int]:
    """
    Supports common detection formats:
    - det["xyxy"] = [x1, y1, x2, y2]
    - det["bbox_xyxy"] = [x1, y1, x2, y2]
    - det["box"] = [x1, y1, x2, y2]
    - det["bbox"] = COCO-style [x, y, w, h] by default
    - normalized values in [0, 1] are scaled to image size
    """
    w, h = image_size

    if "xyxy" in det:
        box = det["xyxy"]
        mode = "xyxy"
    elif "bbox_xyxy" in det:
        box = det["bbox_xyxy"]
        mode = "xyxy"
    elif "box" in det:
        box = det["box"]
        mode = det.get("box_format", "xyxy")
    elif "bbox" in det:
        box = det["bbox"]
        mode = det.get("bbox_format", det.get("bbox_mode", "xywh"))
    else:
        return (0, 0, w - 1, h - 1)

    if box is None or len(box) != 4:
        return (0, 0, w - 1, h - 1)

    x1, y1, a, b = [float(v) for v in box]

    # scale normalized coordinates
    if max(abs(x1), abs(y1), abs(a), abs(b)) <= 1.5:
        x1 *= w
        a *= w
        y1 *= h
        b *= h

    if str(mode).lower() in {"xyxy", "x1y1x2y2"}:
        x2, y2 = a, b
    else:
        # COCO/default xywh
        x2, y2 = x1 + a, y1 + b

    x1 = int(max(0, min(w - 1, round(x1))))
    y1 = int(max(0, min(h - 1, round(y1))))
    x2 = int(max(0, min(w - 1, round(x2))))
    y2 = int(max(0, min(h - 1, round(y2))))

    if x2 <= x1:
        x2 = min(w - 1, x1 + 1)
    if y2 <= y1:
        y2 = min(h - 1, y1 + 1)

    return x1, y1, x2, y2


def segmentation_polygons(det: Dict[str, Any]) -> List[List[Tuple[int, int]]]:
    """
    Supports COCO polygon segmentation:
    - det["segmentation"] = [[x1,y1,x2,y2,...], ...]
    - det["segmentation"] = [x1,y1,x2,y2,...]
    - det["polygon"] = same idea
    """
    seg = det.get("segmentation", det.get("polygon"))
    if seg is None:
        return []

    if isinstance(seg, dict):
        # RLE is intentionally not decoded here.
        return []

    polygons = []

    if isinstance(seg, list):
        # Flat polygon
        if seg and all(isinstance(x, (int, float)) for x in seg):
            pts = [(int(seg[i]), int(seg[i + 1])) for i in range(0, len(seg) - 1, 2)]
            if len(pts) >= 3:
                polygons.append(pts)

        # List of polygons
        else:
            for poly in seg:
                if not poly:
                    continue

                if isinstance(poly, list) and all(isinstance(x, (int, float)) for x in poly):
                    pts = [(int(poly[i]), int(poly[i + 1])) for i in range(0, len(poly) - 1, 2)]
                elif isinstance(poly, list) and all(isinstance(x, (list, tuple)) and len(x) >= 2 for x in poly):
                    pts = [(int(x), int(y)) for x, y, *_ in poly]
                else:
                    pts = []

                if len(pts) >= 3:
                    polygons.append(pts)

    return polygons


def mask_array(det: Dict[str, Any]) -> Optional[np.ndarray]:
    """
    Supports det["mask"] as a boolean/0-1 nested list or numpy array.
    RLE masks are not decoded here.
    """
    mask = det.get("mask")
    if mask is None:
        return None

    if isinstance(mask, dict):
        return None

    arr = np.asarray(mask)
    if arr.ndim != 2:
        return None

    return arr.astype(bool)


def _nearest_sample_to_mean(xs: np.ndarray, ys: np.ndarray) -> Tuple[int, int]:
    """
    Return an actual sampled pixel/point closest to the mean.
    This keeps arrow targets on the segmented object instead of empty bbox space.
    """
    if xs.size == 0 or ys.size == 0:
        return 0, 0

    # Avoid huge distance calculations on large masks.
    max_samples = 4000
    if xs.size > max_samples:
        step = max(1, xs.size // max_samples)
        xs_s = xs[::step]
        ys_s = ys[::step]
    else:
        xs_s = xs
        ys_s = ys

    mx = float(xs_s.mean())
    my = float(ys_s.mean())
    idx = int(np.argmin((xs_s - mx) ** 2 + (ys_s - my) ** 2))
    return int(xs_s[idx]), int(ys_s[idx])


def segmentation_anchor_point(
    det: Dict[str, Any],
    image_size: Tuple[int, int],
) -> Optional[Tuple[int, int]]:
    """
    Find an arrow anchor on actual segmentation content.

    Priority:
    1. Real mask pixels, if det["mask"] exists.
    2. Rasterized polygon pixels for normal-sized polygon regions.
    3. Nearest polygon vertex to polygon center for very large regions.

    This is intentionally stateless: temporal smoothing still comes from the
    detection stabilizer, but the final arrow point is constrained to the
    current visible segmentation instead of the bbox center.
    """
    image_w, image_h = image_size

    arr = mask_array(det)
    if arr is not None and arr.any():
        ys, xs = np.where(arr)
        return _nearest_sample_to_mean(xs.astype(float), ys.astype(float))

    polys = segmentation_polygons(det)
    if not polys:
        return None

    # Flatten polygon vertices and clip to image.
    all_points = []
    for poly in polys:
        for x, y in poly:
            x = max(0, min(image_w - 1, int(x)))
            y = max(0, min(image_h - 1, int(y)))
            all_points.append((x, y))

    if not all_points:
        return None

    xs_np = np.asarray([p[0] for p in all_points], dtype=float)
    ys_np = np.asarray([p[1] for p in all_points], dtype=float)

    x1 = int(max(0, np.floor(xs_np.min())))
    y1 = int(max(0, np.floor(ys_np.min())))
    x2 = int(min(image_w - 1, np.ceil(xs_np.max())))
    y2 = int(min(image_h - 1, np.ceil(ys_np.max())))

    crop_w = max(1, x2 - x1 + 1)
    crop_h = max(1, y2 - y1 + 1)
    crop_area = crop_w * crop_h

    # For normal regions, rasterize polygons in a local crop and select an
    # actual filled pixel near the center. This is accurate and still cheap.
    if crop_area <= ARROW_ANCHOR_MAX_RASTER_AREA:
        poly_mask = Image.new("L", (crop_w, crop_h), 0)
        poly_draw = ImageDraw.Draw(poly_mask)

        for poly in polys:
            shifted = [
                (
                    max(0, min(image_w - 1, int(x))) - x1,
                    max(0, min(image_h - 1, int(y))) - y1,
                )
                for x, y in poly
            ]
            if len(shifted) >= 3:
                poly_draw.polygon(shifted, fill=1)

        mask_np = np.asarray(poly_mask, dtype=bool)
        if mask_np.any():
            ys, xs = np.where(mask_np)
            ax, ay = _nearest_sample_to_mean(xs.astype(float), ys.astype(float))
            return int(ax + x1), int(ay + y1)

    # Large-object fallback: choose an actual polygon vertex near the polygon center.
    # This is fast and avoids pointing into empty space for non-convex shapes.
    return _nearest_sample_to_mean(xs_np, ys_np)


def target_point(det: Dict[str, Any], image_size: Tuple[int, int]) -> Tuple[int, int]:
    """
    Pick a visually stable arrow target.

    Most objects:
    - Prefer an anchor on actual segmentation content.
    - Fall back to stabilized bbox center, then bbox center.

    print_bed:
    - The mask/polygon is usually large and can be clipped/occluded.
    - Polygon anchors can land on an edge, which looks wrong.
    - Use a semantic point inside the visible bed bbox instead.
    """
    label = detection_label(det)
    x1, y1, x2, y2 = bbox_xyxy(det, image_size)

    if label == "print_bed":
        cx = int(x1 + 0.50 * (x2 - x1))
        cy = int(y1 + 0.55 * (y2 - y1))
        return cx, cy

    anchor = segmentation_anchor_point(det, image_size)
    if anchor is not None:
        return anchor

    if "stable_center" in det and det["stable_center"] is not None:
        cx, cy = det["stable_center"]
        return int(cx), int(cy)

    return (x1 + x2) // 2, (y1 + y2) // 2


def choose_best_detection_for_label(
    detections: List[Dict[str, Any]],
    label: str,
) -> Optional[Dict[str, Any]]:
    """
    If multiple instances exist, choose highest confidence/score.
    """
    matches = [det for det in detections if detection_label(det) == label]
    if not matches:
        return None

    def score(det: Dict[str, Any]) -> float:
        return float(det.get("score", det.get("confidence", 1.0)))

    matches.sort(key=score, reverse=True)
    return matches[0]


# ---------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------

def _get_font(size: int = 18):
    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf", size)
    except Exception:
        return ImageFont.load_default()


def _text_size(draw: ImageDraw.ImageDraw, text: str, font) -> Tuple[int, int]:
    try:
        box = draw.textbbox((0, 0), text, font=font)
        return box[2] - box[0], box[3] - box[1]
    except Exception:
        return draw.textsize(text, font=font)


def _arrow_head(p_from: Tuple[int, int], p_to: Tuple[int, int], size: int = 12) -> List[Tuple[int, int]]:
    x1, y1 = p_from
    x2, y2 = p_to
    angle = math.atan2(y2 - y1, x2 - x1)

    a1 = angle + math.pi * 0.82
    a2 = angle - math.pi * 0.82

    return [
        (x2, y2),
        (int(x2 + size * math.cos(a1)), int(y2 + size * math.sin(a1))),
        (int(x2 + size * math.cos(a2)), int(y2 + size * math.sin(a2))),
    ]


def _label_position(
    bbox: Tuple[int, int, int, int],
    image_size: Tuple[int, int],
    text_size: Tuple[int, int],
    index: int,
) -> Tuple[int, int]:
    """
    Place caption near object but keep it inside image.
    """
    w, h = image_size
    x1, y1, x2, y2 = bbox
    tw, th = text_size

    # Alternate positions to reduce overlap.
    offsets = [
        (0, -th - 18),
        (0, 10),
        (-tw - 18, 0),
        (18, 0),
    ]
    dx, dy = offsets[index % len(offsets)]

    lx = x1 + dx
    ly = y1 + dy

    lx = max(6, min(w - tw - 18, lx))
    ly = max(6, min(h - th - 18, ly))

    return int(lx), int(ly)


def draw_guided_annotations(
    image_path: str | Path,
    detections: List[Dict[str, Any]],
    visual_targets: List[Dict[str, str]],
    output_path: str | Path = "outputs/guided_result.jpg",
    mask_alpha: int = GUIDANCE_MASK_ALPHA,
    draw_boxes_when_no_mask: bool = True,
) -> Path:
    """
    Draw segmentation highlights + arrows + captions.

    visual_targets format:
    [
        {"label": "display_screen", "caption": "Select Level"},
        {"label": "nozzle_kit", "caption": "Do not touch"},
    ]
    """
    image_path = Path(image_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    image = Image.open(image_path).convert("RGBA")
    w, h = image.size

    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw_overlay = ImageDraw.Draw(overlay)
    draw = ImageDraw.Draw(image)

    font = _get_font(42)
    small_font = _get_font(30)

    palette = [
        (0, 220, 255),
        (255, 120, 0),
        (120, 255, 90),
        (255, 80, 200),
        (255, 230, 0),
        (120, 160, 255),
    ]

    resolved = []

    for i, target in enumerate(visual_targets):
        label = normalize_visual_label(target["label"])
        caption = target.get("caption") or pretty_label(label)

        det = choose_best_detection_for_label(detections, label)
        if det is None:
            continue

        color = stable_color_for_label(label)
        fill = (*color, mask_alpha)
        stroke = (*color, 255)

        bbox = bbox_xyxy(det, image.size)
        center = target_point(det, image.size)

        arr = mask_array(det)
        polys = segmentation_polygons(det)

        drew_mask = False

        if arr is not None and arr.any():
            mask_img = Image.fromarray((arr.astype(np.uint8) * mask_alpha), mode="L")
            color_img = Image.new("RGBA", image.size, fill)
            overlay.alpha_composite(Image.composite(color_img, Image.new("RGBA", image.size, (0, 0, 0, 0)), mask_img))
            drew_mask = True

        elif polys:
            for poly in polys:
                draw_overlay.polygon(poly, fill=fill, outline=stroke)
            drew_mask = True

        # Segmentation-first drawing:
        # - If YOLO provides a mask/polygon, do not cover it with a bbox/fill.
        # - If there is no usable mask, fall back to a bbox so the target is still visible.
        if not drew_mask and draw_boxes_when_no_mask:
            draw_overlay.rectangle(bbox, outline=stroke, width=GUIDANCE_BBOX_WIDTH)

        resolved.append({
            "label": label,
            "caption": caption,
            "bbox": bbox,
            "center": center,
            "color": color,
            "target_index": i,
        })

    image = Image.alpha_composite(image, overlay)
    draw = ImageDraw.Draw(image)

    for i, item in enumerate(resolved):
        label = item["label"]
        caption = item["caption"]
        bbox = item["bbox"]
        center = item["center"]
        color = item["color"]

        text = caption
        tw, th = _text_size(draw, text, font)
        layout_index = int(item.get("target_index", i))

        lx, ly = _label_position(bbox, image.size, (tw, th), layout_index)

        # caption box
        pad_x = 22
        pad_y = 14
        box = [
            lx,
            ly,
            lx + tw + 2 * pad_x,
            ly + th + 2 * pad_y,
        ]

        draw.rounded_rectangle(
            box,
            radius=18,
            fill=(20, 20, 20, 220),
            outline=(*color, 255),
            width=GUIDANCE_LABEL_OUTLINE_WIDTH,
        )
        draw.text(
            (lx + pad_x, ly + pad_y),
            text,
            font=font,
            fill=(255, 255, 255, 255),
        )

        # Optional small object label under caption.
        # Disabled by default to avoid duplicate labels like
        # "Filament spool" + "Filament spool". The main caption should carry
        # the response-specific guidance.
        obj_text = pretty_label(label)
        obj_y = box[3] + 2
        if DRAW_OBJECT_SUBLABEL and obj_text.strip().lower() != text.strip().lower():
            ow, oh = _text_size(draw, obj_text, small_font)
            if obj_y + oh + 6 < image.size[1]:
                draw.rounded_rectangle(
                    [lx, obj_y, lx + ow + 2 * pad_x, obj_y + oh + 2 * pad_y],
                    radius=6,
                    fill=(20, 20, 20, 175),
                )
                draw.text(
                    (lx + pad_x, obj_y + pad_y),
                    obj_text,
                    font=small_font,
                    fill=(*color, 255),
                )

        # arrow from label box to object center
        start = (box[0] + (box[2] - box[0]) // 2, box[3])
        end = center

        draw.line([start, end], fill=(*color, 255), width=GUIDANCE_ARROW_WIDTH)
        draw.polygon(_arrow_head(start, end, size=GUIDANCE_ARROW_HEAD_SIZE), fill=(*color, 255))

    image.convert("RGB").save(output_path, quality=95)
    return output_path


def draw_from_pipeline_result(
    image_path: str | Path,
    detections: List[Dict[str, Any]],
    result: Dict[str, Any],
    output_path: str | Path = "outputs/guided_result.jpg",
    max_targets: int = 4,
) -> Path:
    """
    Convenience wrapper after run_pipeline(...).

    It expects the usual pipeline result:
    - result["llm_json"]
    - result["verification"]
    - result["selected_chunk"]
    """
    targets = build_visual_targets(
        llm_json=result.get("llm_json") or {},
        verification=result.get("verification") or {},
        selected_chunk=result.get("selected_chunk") or {},
        detections=detections,
        max_targets=max_targets,
    )

    return draw_guided_annotations(
        image_path=image_path,
        detections=detections,
        visual_targets=targets,
        output_path=output_path,
    )