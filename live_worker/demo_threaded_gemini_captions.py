#!/usr/bin/env python3
"""
combined_live_rag_yolo_demo_threaded.py

Threaded live demo:
  - One thread continuously processes the latest camera frame with YOLO and writes latest_yolo.jpg.
  - Main thread accepts typed questions, runs RAG/Gemini, and updates live visual targets.

This avoids the old problem where Gemini/RAG blocked the live YOLO stream, and it avoids the
separate-process problem where RAG targets were not reliably applied to the YOLO preview.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2

DEMO_COLOR_NAMES = {
    "bed_clamp": "Yellow",
    "bowden_tube": "White",
    "button": "Pink",
    "display_screen": "Cyan",
    "extruder_motor": "Purple",
    "filament_detector": "Yellow",
    "filament_holder": "Blue",
    "filament_spool": "Green",
    "gantry_frame": "Orange",
    "network_interface": "Aqua",
    "nozzle_kit": "Amber",
    "power_switch": "Red",
    "print_bed": "Green",
    "qr_code": "Light gray",
    "sd_card_port": "Lime green",
    "side_ports": "Sky blue",
    "toolbox": "Orange",
    "usb_port": "Blue",
    "x_axis_gantry": "Pink",
    "x_axis_motor": "Lavender",
    "y_axis_belt_adjuster": "Coral",
    "y_axis_motor": "Mint green",
}


def now() -> str:
    return time.strftime("%H:%M:%S")


def log(msg: str) -> None:
    print(f"[{now()}] {msg}", flush=True)


def atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)


def atomic_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    shutil.copyfile(src, tmp)
    os.replace(tmp, dst)


def atomic_write_image(path: Path, frame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp.jpg")
    ok = cv2.imwrite(str(tmp), frame)
    if not ok:
        raise RuntimeError(f"cv2.imwrite failed for {tmp}")
    os.replace(tmp, path)


def import_package(package_dir: Path, package_name: str = "cr10_threaded_demo_pkg") -> str:
    init_file = package_dir / "__init__.py"
    spec = importlib.util.spec_from_file_location(
        package_name,
        init_file,
        submodule_search_locations=[str(package_dir)],
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import package from {package_dir}")

    pkg = importlib.util.module_from_spec(spec)
    sys.modules[package_name] = pkg
    spec.loader.exec_module(pkg)
    return package_name


# Common aliases seen in manual/Gemini/user wording -> YOLO labels.
LOCAL_LABEL_ALIASES = {
    "sd": "sd_card_port",
    "sd_card": "sd_card_port",
    "sd_card_slot": "sd_card_port",
    "tf_card": "sd_card_port",
    "tf_card_slot": "sd_card_port",
    "card_slot": "sd_card_port",
    "screen": "display_screen",
    "display": "display_screen",
    "display": "display_screen",
    "touch_screen": "display_screen",
    "bed": "print_bed",
    "plate": "print_bed",
    "printing_platform": "print_bed",
    "platform": "print_bed",
    "filament_sensor": "filament_detector",
    "filament_runout_sensor": "filament_detector",
    "filament_tube": "bowden_tube",
    "teflon_tube": "bowden_tube",
    "tube": "bowden_tube",
    "extruder": "extruder_motor",
    "hotend": "nozzle_kit",
    "nozzle": "nozzle_kit",
    "spool": "filament_spool",
    "spool_holder": "filament_holder",
    "holder": "filament_holder",
    "usb": "usb_port",
    "lan": "network_interface",
    "ethernet": "network_interface",
    "network": "network_interface",
    "power": "power_switch",
}

QUESTION_HINTS = [
    (("sd",), "sd_card_port"),
    (("card", "slot"), "sd_card_port"),
    (("tf",), "sd_card_port"),
    (("filament", "load"), "filament_detector"),
    (("filament", "fill"), "filament_detector"),
    (("filament",), "filament_spool"),
    (("spool",), "filament_spool"),
    (("extruder",), "extruder_motor"),
    (("tube",), "bowden_tube"),
    (("bed",), "print_bed"),
    (("level",), "print_bed"),
    (("screen",), "display_screen"),
    (("display",), "display_screen"),
    (("usb",), "usb_port"),
    (("power",), "power_switch"),
    (("nozzle",), "nozzle_kit"),
]


def normalize_label(label: Any, visual_guidance_module=None) -> Optional[str]:
    if label is None:
        return None
    clean = str(label).strip().lower().replace(" ", "_").replace("-", "_")
    if not clean:
        return None

    if visual_guidance_module is not None and hasattr(visual_guidance_module, "normalize_visual_label"):
        try:
            clean = str(visual_guidance_module.normalize_visual_label(clean))
        except Exception:
            pass

    return LOCAL_LABEL_ALIASES.get(clean, clean)


def labels_from_question(question: str) -> List[str]:
    q = question.lower()
    labels: List[str] = []
    for tokens, label in QUESTION_HINTS:
        if all(tok in q for tok in tokens):
            labels.append(label)
    return list(dict.fromkeys(labels))


def visible_labels_from_detections(detections: Any, visual_guidance_module=None) -> List[str]:
    if isinstance(detections, dict):
        detections = detections.get("detections", [])
    if not isinstance(detections, list):
        return []

    out: List[str] = []
    for det in detections:
        if not isinstance(det, dict):
            continue
        raw = det.get("label") or det.get("class_name") or det.get("name") or det.get("class") or det.get("object")
        label = normalize_label(raw, visual_guidance_module)
        if label:
            out.append(label)
    return list(dict.fromkeys(out))


def compact_detection_summary(detections: Any, max_items: int = 30) -> List[str]:
    if isinstance(detections, dict):
        detections = detections.get("detections", [])
    if not isinstance(detections, list):
        return []

    rows: List[str] = []
    for det in detections[:max_items]:
        if not isinstance(det, dict):
            rows.append(str(det))
            continue
        label = det.get("label") or det.get("class_name") or det.get("name") or "unknown"
        conf = det.get("confidence") or det.get("conf") or det.get("score")
        if conf is not None:
            try:
                rows.append(f"{label} ({float(conf):.2f})")
            except Exception:
                rows.append(f"{label} ({conf})")
        else:
            rows.append(str(label))
    if isinstance(detections, list) and len(detections) > max_items:
        rows.append(f"... +{len(detections) - max_items} more")
    return rows


# Short, stable fallbacks are only used when Gemini does not provide a caption.
DEMO_CAPTION_FALLBACKS = {
    "sd_card_port": "SD card slot",
    "usb_port": "USB port",
    "side_ports": "Side ports",
    "filament_detector": "Filament sensor",
    "extruder_motor": "Extruder feed",
    "filament_spool": "Filament spool",
    "filament_holder": "Spool holder",
    "bowden_tube": "Filament path",
    "nozzle_kit": "Nozzle / hotend",
    "print_bed": "Print bed",
    "display_screen": "Display screen",
    "power_switch": "Power switch",
    "x_axis_gantry": "X-axis gantry",
    "y_axis_belt_adjuster": "Belt adjuster",
    "gantry_frame": "Gantry frame",
    "network_interface": "Network port",
    "toolbox": "Toolbox",
}

# Phrases used to decide whether a YOLO label is actually relevant to the final answer.
# This prevents random chunk-required objects from being highlighted just because the
# selected manual chunk mentions them.
ANSWER_LABEL_SYNONYMS = {
    "sd_card_port": ["sd card", "memory card", "tf card", "card slot", "sd/tf"],
    "usb_port": ["usb", "usb port"],
    "side_ports": ["side port", "side ports", "port area"],
    "filament_detector": ["filament detector", "filament sensor", "runout sensor", "filament runout"],
    "extruder_motor": ["extruder", "extruder motor", "feeder", "feed gear", "tension lever", "release tension", "feed mechanism"],
    "filament_spool": ["spool", "filament spool", "filament roll"],
    "filament_holder": ["holder", "spool holder", "filament holder"],
    "bowden_tube": ["bowden", "teflon tube", "tube", "filament path"],
    "nozzle_kit": ["nozzle", "hotend", "hot end", "melt", "heated nozzle"],
    "print_bed": ["print bed", "bed", "platform", "build plate", "level"],
    "display_screen": ["screen", "display", "touch screen"],
    "power_switch": ["power switch", "power button", "switch", "turn on", "power on"],
    "x_axis_gantry": ["x axis", "x-axis", "gantry"],
    "y_axis_belt_adjuster": ["belt adjuster", "belt tension", "y axis"],
    "gantry_frame": ["frame", "gantry frame"],
    "network_interface": ["network", "ethernet", "lan"],
    "toolbox": ["toolbox", "tool box"],
}


def norm_text_for_match(text: str) -> str:
    return (text or "").lower().replace("_", " ").replace("-", " ")


def gemini_caption_for_label(label: str, llm_json: Dict[str, Any], visual_guidance_module=None, max_chars: int = 110) -> Optional[str]:
    """
    Return Gemini's caption for this YOLO label, accepting exact or normalized keys.
    This is intentionally preferred over fallback captions.
    """
    captions = llm_json.get("visual_captions") or {}
    if not isinstance(captions, dict):
        return None

    if label in captions and captions[label]:
        return str(captions[label])[:max_chars]

    for k, v in captions.items():
        if normalize_label(k, visual_guidance_module) == label and v:
            return str(v)[:max_chars]

    return None


def label_is_answer_aligned(label: str, question: str, answer: str) -> bool:
    """
    True only when the label is explicitly connected to the question/answer text.
    This is the main anti-noise filter.
    """
    text = norm_text_for_match(f"{question} {answer}")
    label_text = norm_text_for_match(label)

    if label_text in text:
        return True

    for phrase in ANSWER_LABEL_SYNONYMS.get(label, []):
        if norm_text_for_match(phrase) in text:
            return True

    return False


def caption_for_label(
    label: str,
    llm_json: Dict[str, Any],
    selected_chunk: Dict[str, Any],
    visual_guidance_module=None,
    fallback_caption: Optional[str] = None,
) -> str:
    """
    Caption policy:
      1. Gemini visual_captions for the selected object.
      2. Existing pipeline caption if available.
      3. visual_guidance.short_caption if available.
      4. Short stable demo fallback.
    """
    gemini_caption = gemini_caption_for_label(label, llm_json, visual_guidance_module)
    if gemini_caption:
        return gemini_caption

    if fallback_caption:
        return str(fallback_caption)[:110]

    action_id = selected_chunk.get("action_id") or selected_chunk.get("id")
    if visual_guidance_module is not None and hasattr(visual_guidance_module, "short_caption"):
        try:
            return str(visual_guidance_module.short_caption(label, action_id=action_id, llm_json=llm_json))[:110]
        except Exception:
            pass

    return DEMO_CAPTION_FALLBACKS.get(label, label.replace("_", " ").title())


def build_robust_visual_targets(
    question: str,
    result: Dict[str, Any],
    detections: List[Dict[str, Any]],
    max_targets: int,
    visual_guidance_module=None,
) -> List[Dict[str, str]]:
    """
    Demo-safe annotation decision.

    Object selection is strict:
      - Prefer Gemini/pipeline objects.
      - Keep only objects that are actually related to the final question/answer.
      - Avoid noisy chunk-required/highlight objects unless answer-aligned.
      - Clear targets if nothing relevant is found.

    Caption selection is flexible:
      - For selected objects, prefer Gemini visual_captions.
      - Fall back only when Gemini has no caption for that object.
    """
    answer = result.get("answer") or ""
    llm_json = result.get("llm_json") or {}
    verification = result.get("verification") or {}
    selected_chunk = result.get("selected_chunk") or {}
    visible = set(visible_labels_from_detections(detections, visual_guidance_module))

    candidates: List[Dict[str, Any]] = []

    def push(label: Any, source: str, caption: Optional[str] = None, priority: int = 50) -> None:
        norm = normalize_label(label, visual_guidance_module)
        if not norm:
            return
        candidates.append({
            "label": norm,
            "caption": caption,
            "source": source,
            "priority": priority,
        })

    # Strongest signals first.
    for raw in llm_json.get("referenced_objects") or []:
        push(raw, "gemini_referenced_objects", priority=5)

    for target in result.get("visual_targets") or []:
        if isinstance(target, dict):
            push(target.get("label"), "pipeline_visual_targets", caption=target.get("caption"), priority=10)

    # Direct user intent is usually reliable, e.g. "highlight SD card slot".
    for raw in labels_from_question(question):
        push(raw, "question_hint", priority=15)

    # Lower-priority traces. These are often too broad, so answer alignment filters them.
    for raw in verification.get("referenced_objects") or []:
        push(raw, "verification_referenced_objects", priority=40)

    for raw in selected_chunk.get("highlight_objects") or []:
        push(raw, "chunk_highlight_objects", priority=70)

    for raw in selected_chunk.get("required_objects") or []:
        push(raw, "chunk_required_objects", priority=90)

    # Sort deterministic: priority first, then visible objects before non-visible.
    candidates.sort(key=lambda t: (int(t.get("priority", 50)), 0 if t.get("label") in visible else 1))

    final: List[Dict[str, str]] = []
    seen = set()

    for target in candidates:
        label = normalize_label(target.get("label"), visual_guidance_module)
        if not label or label in seen:
            continue

        # Main noise guard: do not highlight objects unrelated to the generated answer.
        if not label_is_answer_aligned(label, question, answer):
            continue

        seen.add(label)
        final.append({
            "label": label,
            "caption": caption_for_label(
                label,
                llm_json,
                selected_chunk,
                visual_guidance_module=visual_guidance_module,
                fallback_caption=target.get("caption"),
            ),
            "source": str(target.get("source", "answer_aligned")),
        })

        if len(final) >= max_targets:
            break

    # Intentional fallback: if the user explicitly asked to highlight an object,
    # and answer text failed to mention it, allow one question-hint target.
    # This prevents empty overlays for commands like "highlight the sd card slot".
    if not final:
        for label in labels_from_question(question):
            norm = normalize_label(label, visual_guidance_module)
            if not norm:
                continue
            final.append({
                "label": norm,
                "caption": caption_for_label(norm, llm_json, selected_chunk, visual_guidance_module=visual_guidance_module),
                "source": "question_hint_fallback_with_gemini_caption",
            })
            break

    return final[:max_targets]


def format_ref(ref: Any) -> str:
    if not isinstance(ref, dict):
        return str(ref)
    if ref.get("display_text"):
        return str(ref["display_text"])
    source = ref.get("source_title") or ref.get("source_id") or "Unknown source"
    section = ref.get("section") or ref.get("chunk_title") or "Unknown section"
    pages = ref.get("pages")
    page = ref.get("page")
    if isinstance(pages, list) and pages:
        page_text = f"pp. {pages[0]}-{pages[-1]}" if len(pages) > 1 else f"p. {pages[0]}"
        return f"{source}, {page_text}, {section}"
    if page is not None:
        return f"{source}, p. {page}, {section}"
    return f"{source}, {section}"


def draw_debug_overlay(frame, visible_objects: List[str], active_targets: List[Dict[str, Any]]):
    """
    Keep fallback frame clean for the demo.

    The previous version drew a large black status banner on top of the image,
    which occluded the live view. Status/debug information is already shown
    in the notebook UI, so the image itself should stay unobstructed.
    """
    return frame.copy()


def force_pipeline_to_use_rag(pipeline_module) -> None:
    def always_use_rag(question, scene=None, session_state=None, turn_type="task_question", use_gemini=False):
        return {
            "scope": "forced_printer_rag",
            "confidence": 1.0,
            "should_use_rag": True,
            "should_answer_general": False,
            "should_use_yolo_annotations": True,
            "reason": "Demo mode: forced every question through manual RAG.",
        }
    pipeline_module.classify_request_scope = always_use_rag


def write_answer_json(
    out_dir: Path,
    question: str,
    result: Dict[str, Any],
    visible_objects: List[str],
    detections: List[Dict[str, Any]],
    visual_targets: List[Dict[str, Any]],
    gemini_requested: bool,
    gemini_key_present: bool,
    elapsed_sec: float,
) -> None:
    llm_json = result.get("llm_json") or {}
    verification = result.get("verification") or {}
    selected_chunk = result.get("selected_chunk") or {}
    gemini_config = result.get("gemini_config") or {}

    refs = result.get("used_references") or result.get("references") or []

    answer_text = result.get("answer") or ""

    if visual_targets:
        color_notes = []
        for target in visual_targets:
            label = target.get("label")
            caption = target.get("caption") or label
            color = DEMO_COLOR_NAMES.get(label)
    
            if color:
                color_notes.append(f"{caption} ({color})")
            else:
                color_notes.append(str(caption))
    
        answer_text = answer_text + "\n\nVisual guide: " + "; ".join(color_notes)

    payload = {
        "question": question,
        "answer": answer_text,
        "references": refs,
        "reference_texts": [format_ref(r) for r in refs],
        "visible_objects": visible_objects,
        "detections": compact_detection_summary(detections),
        "selected_action": result.get("selected_action"),
        "turn_type": result.get("turn_type"),
        "selected_chunk_id": selected_chunk.get("id"),
        "selected_chunk_action_id": selected_chunk.get("action_id"),
        "gemini_requested": gemini_requested,
        "gemini_api_key_present": gemini_key_present,
        "gemini_config": gemini_config,
        "gemini_used": bool(gemini_config.get("use_gemini_answer") or gemini_config.get("use_gemini_rerank") or gemini_config.get("use_gemini")),
        "gemini_referenced_objects": llm_json.get("referenced_objects"),
        "gemini_visual_captions": llm_json.get("visual_captions"),
        "gemini_missing_or_uncertain_objects": llm_json.get("missing_or_uncertain_objects"),
        "verification_referenced_objects": verification.get("referenced_objects"),
        "verification_found_required": verification.get("found_required"),
        "verification_missing_required": verification.get("missing_required"),
        "chunk_required_objects": selected_chunk.get("required_objects"),
        "chunk_highlight_objects": selected_chunk.get("highlight_objects"),
        "visual_targets": visual_targets,
        "annotated_image_path": result.get("annotated_image_path"),
        "elapsed_sec": round(elapsed_sec, 3),
        "created_at": now(),
    }

    atomic_write_json(out_dir / "latest_answer.json", payload)
    atomic_write_json(out_dir / "visual_targets.json", {
        "question": question,
        "answer": result.get("answer") or "",
        "visual_targets": visual_targets,
        "created_at": now(),
    })


def write_scene_json(out_dir: Path, frame_path: Path, visible_objects: List[str], active_targets: List[Dict[str, Any]], detections: List[Dict[str, Any]], processed: int) -> None:
    atomic_write_json(out_dir / "latest_scene.json", {
        "frame_path": str(frame_path),
        "visible_objects": visible_objects,
        "active_visual_targets": active_targets,
        "detections": compact_detection_summary(detections, max_items=40),
        "processed_frame_count": processed,
        "created_at": now(),
    })


def frame_loop(
    args,
    controller,
    out_dir: Path,
    frame_path: Path,
    out_image: Path,
    state: Dict[str, Any],
    state_lock: threading.Lock,
    stop_event: threading.Event,
) -> None:
    last_mtime: Optional[float] = None
    processed = 0
    delay = 1.0 / max(args.fps, 0.01)
    last_processed_t = 0.0

    while not stop_event.is_set():
        try:
            if not frame_path.exists():
                time.sleep(0.05)
                continue

            try:
                mtime = frame_path.stat().st_mtime
            except FileNotFoundError:
                time.sleep(0.02)
                continue

            now_perf = time.perf_counter()
            if mtime == last_mtime or (now_perf - last_processed_t) < delay:
                time.sleep(0.005)
                continue

            last_mtime = mtime
            last_processed_t = now_perf

            frame = cv2.imread(str(frame_path))
            if frame is None:
                time.sleep(0.02)
                continue

            with state_lock:
                active_targets = list(state.get("active_targets", []))
                controller.active_visual_targets = active_targets

            info = controller.process_frame(frame)
            detections = getattr(controller, "latest_stable_detections", []) or []
            visible_objects = info.get("visible_objects", []) or visible_labels_from_detections(detections)
            annotated_path = info.get("annotated_image_path")

            if annotated_path and Path(annotated_path).exists():
                atomic_copy(Path(annotated_path), out_image)
            else:
                atomic_write_image(out_image, draw_debug_overlay(frame, visible_objects, active_targets))

            processed += 1
            with state_lock:
                state["latest_detections"] = detections
                state["latest_visible_objects"] = visible_objects
                state["latest_controller_frame_path"] = controller.latest_frame_path
                state["processed"] = processed

            write_scene_json(out_dir, frame_path, visible_objects, active_targets, detections, processed)

            if args.log_every > 0 and processed % args.log_every == 0:
                log(f"frame={processed} visible={visible_objects} targets={len(active_targets)}")

        except Exception as e:
            log(f"Frame loop error: {repr(e)}")
            traceback.print_exc()
            time.sleep(0.2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rag-package-dir", required=True)
    parser.add_argument("--frame-path", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--out-image", default=None)
    parser.add_argument("--yolo-weights", required=True)
    parser.add_argument("--manual-chunks", required=True)

    parser.add_argument("--semantic-index", action="store_true")
    parser.add_argument("--yolo-conf", type=float, default=0.35)
    parser.add_argument("--yolo-imgsz", type=int, default=640)
    parser.add_argument("--fps", type=float, default=6.0)
    parser.add_argument("--log-every", type=int, default=60)

    parser.add_argument("--use-gemini", action="store_true")
    parser.add_argument("--use-gemini-retrieval", action="store_true")
    parser.add_argument("--use-gemini-rerank", action="store_true")
    parser.add_argument("--use-gemini-answer", action="store_true")

    parser.add_argument("--force-rag", action="store_true", default=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-targets", type=int, default=2)
    parser.add_argument("--clear-targets-on-empty", action="store_true", help="Clear old live annotations when a question produces no targets.")

    args = parser.parse_args()

    package_dir = Path(args.rag_package_dir).resolve()
    package_name = import_package(package_dir)

    LiveStreamController = __import__(f"{package_name}.live_stream_controller", fromlist=["LiveStreamController"]).LiveStreamController
    data_loading = __import__(f"{package_name}.data_loading", fromlist=[""])
    retrieval = __import__(f"{package_name}.retrieval", fromlist=[""])
    pipeline = __import__(f"{package_name}.pipeline", fromlist=[""])
    visual_guidance = __import__(f"{package_name}.visual_guidance", fromlist=[""])

    if args.force_rag:
        force_pipeline_to_use_rag(pipeline)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frame_path = Path(args.frame_path)
    out_image = Path(args.out_image) if args.out_image else out_dir / "latest_yolo.jpg"
    latest_answer_image = out_dir / "latest_answer.jpg"

    log("Loading manual chunks...")
    manual_chunks = data_loading.load_manual_chunks(Path(args.manual_chunks))

    rag_index = None
    if args.semantic_index:
        log("Building semantic RAG index...")
        rag_index = retrieval.build_rag_index(manual_chunks)
        log("Semantic RAG index ready.")

    log("Creating shared LiveStreamController...")
    controller = LiveStreamController(
        yolo_weights_path=Path(args.yolo_weights),
        manual_chunks=manual_chunks,
        rag_index=rag_index,
        output_dir=out_dir / "threaded_controller",
        yolo_conf=args.yolo_conf,
        yolo_imgsz=args.yolo_imgsz,
    )

    gemini_key_present = bool(os.getenv("GEMINI_API_KEY"))
    gemini_requested = bool(args.use_gemini or args.use_gemini_answer or args.use_gemini_rerank or args.use_gemini_retrieval)

    log("Threaded live RAG + YOLO demo started.")
    log(f"Input frame       : {frame_path}")
    log(f"Live YOLO output  : {out_image}")
    log(f"Answer JSON       : {out_dir / 'latest_answer.json'}")
    log(f"Gemini requested  : {gemini_requested}")
    log(f"Gemini key present: {gemini_key_present}")
    log("Type questions here. Type q to quit.")

    state: Dict[str, Any] = {
        "active_targets": [],
        "latest_detections": [],
        "latest_visible_objects": [],
        "latest_controller_frame_path": None,
        "processed": 0,
    }
    state_lock = threading.Lock()
    stop_event = threading.Event()

    t = threading.Thread(
        target=frame_loop,
        args=(args, controller, out_dir, frame_path, out_image, state, state_lock, stop_event),
        daemon=True,
    )
    t.start()

    try:
        while not stop_event.is_set():
            question = input("Question> ").strip()
            if question.lower() in {"q", "quit", "exit"}:
                break
            if not question:
                continue

            with state_lock:
                latest_detections = list(state.get("latest_detections", []))
                latest_visible_objects = list(state.get("latest_visible_objects", []))
                latest_frame_for_rag = state.get("latest_controller_frame_path")

            if latest_frame_for_rag is None:
                log("No processed frame yet; wait until the camera stream appears.")
                continue

            use_gemini_base = bool(os.getenv("GEMINI_API_KEY")) and bool(args.use_gemini)
            use_gemini_answer = bool(os.getenv("GEMINI_API_KEY")) and bool(args.use_gemini or args.use_gemini_answer)
            use_gemini_rerank = bool(os.getenv("GEMINI_API_KEY")) and bool(args.use_gemini_rerank)
            use_gemini_retrieval = bool(os.getenv("GEMINI_API_KEY")) and bool(args.use_gemini_retrieval)

            log(f"Running RAG/Gemini for: {question!r}")
            t0 = time.perf_counter()

            result = pipeline.run_pipeline(
                question=question,
                image_path=latest_frame_for_rag,
                detections=latest_detections,
                manual_chunks=manual_chunks,
                rag_index=rag_index,
                output_path=latest_answer_image,
                session_state=controller.session_state,
                top_k=args.top_k,

                use_gemini=use_gemini_base,
                use_gemini_request_guard=False,
                use_gemini_retrieval=use_gemini_retrieval,
                use_gemini_rerank=use_gemini_rerank,
                use_gemini_answer=use_gemini_answer,
                use_gemini_fallback_router=False,
            )

            controller.session_state = result.get("updated_session_state")

            targets = build_robust_visual_targets(
                question=question,
                result=result,
                detections=latest_detections,
                max_targets=args.max_targets,
                visual_guidance_module=visual_guidance,
            )
            result["visual_targets"] = targets

            # Always overwrite targets. If this turn has no relevant visual target,
            # clear old highlights instead of leaving stale annotations from a previous question.
            with state_lock:
                state["active_targets"] = targets
                controller.active_visual_targets = targets

            elapsed = time.perf_counter() - t0

            write_answer_json(
                out_dir=out_dir,
                question=question,
                result=result,
                visible_objects=latest_visible_objects,
                detections=latest_detections,
                visual_targets=targets,
                gemini_requested=gemini_requested,
                gemini_key_present=bool(os.getenv("GEMINI_API_KEY")),
                elapsed_sec=elapsed,
            )

            llm_json = result.get("llm_json") or {}
            gemini_config = result.get("gemini_config") or {}
            selected_chunk = result.get("selected_chunk") or {}

            print("\n" + "=" * 100, flush=True)
            print(f"Question: {question}", flush=True)
            print(f"Answer:\n{result.get('answer') or ''}", flush=True)
            print("\nReferences:", flush=True)
            for ref in result.get("used_references") or result.get("references") or []:
                print(f"- {format_ref(ref)}", flush=True)
            print("\n--- Annotation decision trace ---", flush=True)
            print("Gemini flags actually passed:", json.dumps({
                "use_gemini": use_gemini_base,
                "use_gemini_retrieval": use_gemini_retrieval,
                "use_gemini_rerank": use_gemini_rerank,
                "use_gemini_answer": use_gemini_answer,
            }, indent=2), flush=True)
            print("Pipeline gemini_config:", json.dumps(gemini_config, indent=2, ensure_ascii=False), flush=True)
            print("Gemini referenced_objects:", llm_json.get("referenced_objects"), flush=True)
            print("Gemini visual_captions:", json.dumps(llm_json.get("visual_captions"), indent=2, ensure_ascii=False), flush=True)
            print("Verification referenced_objects:", (result.get("verification") or {}).get("referenced_objects"), flush=True)
            print("Chunk highlight_objects:", selected_chunk.get("highlight_objects"), flush=True)
            print("Chunk required_objects:", selected_chunk.get("required_objects"), flush=True)
            print("Visible YOLO objects:", latest_visible_objects, flush=True)
            print("FINAL live visual_targets:", json.dumps(targets, indent=2, ensure_ascii=False), flush=True)
            print(f"latest_yolo.jpg will update on next frame: {out_image}", flush=True)
            print("=" * 100 + "\n", flush=True)

    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        log("Stopping threaded demo...")


if __name__ == "__main__":
    main()
