#!/usr/bin/env python3
"""
live_ai_worker.py

Long-running file-bridge worker for the headset AI assistant.

Watches:
    data/gelen_json/<conv_id>/

Writes:
    data/giden_json/<conv_id>/

Frame behavior:
    incoming f_000001.jpg
      -> YOLO + current active visual targets
      -> r_frame_000001.jpg + r_frame_000001.json

Audio behavior:
    incoming a_xxx.m4a + a_xxx.json
      -> faster-whisper STT
      -> RAG
      -> active visual targets
      -> r_a_xxx.json + optional r_a_xxx.jpg

This version includes optional profiling:
    --profile

Profiling adds timings_ms to output JSON files and prints timing logs.
"""

from __future__ import annotations

import argparse
import copy
import importlib
import importlib.util
import json
import os
import re
import signal
import sys
import time
import traceback
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------
# Logging / timing
# ---------------------------------------------------------------------

def now() -> str:
    return time.strftime("%H:%M:%S")


def log(msg: str) -> None:
    print(f"[{now()}] {msg}", flush=True)


def warn(msg: str) -> None:
    print(f"[{now()}][WARN] {msg}", flush=True)


def err(msg: str) -> None:
    print(f"[{now()}][ERROR] {msg}", file=sys.stderr, flush=True)


def elapsed_ms(t0: float) -> float:
    return (time.perf_counter() - t0) * 1000.0


def safe_mtime(path: Path) -> Optional[float]:
    try:
        return path.stat().st_mtime
    except FileNotFoundError:
        return None


def fmt_ms(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"{value:.1f}ms"


# ---------------------------------------------------------------------
# Atomic IO helpers
# ---------------------------------------------------------------------

def atomic_write_text(path: Path, text: str, encoding: str = "utf-8") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding=encoding)
    os.replace(tmp, path)


def atomic_write_json(path: Path, obj: Dict[str, Any]) -> None:
    atomic_write_text(path, json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def atomic_copy_file(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(src)
    atomic_write_bytes(dst, src.read_bytes())


def wait_until_file_stable(path: Path, checks: int = 2, sleep_sec: float = 0.05) -> bool:
    """
    Extra protection for non-atomic writers.
    If backend already writes tmp + os.replace, this is just a safety layer.
    """
    if not path.exists():
        return False

    last_size = -1
    for _ in range(checks):
        try:
            size = path.stat().st_size
        except FileNotFoundError:
            return False

        if size <= 0:
            time.sleep(sleep_sec)
            continue

        if size == last_size:
            return True

        last_size = size
        time.sleep(sleep_sec)

    try:
        return path.stat().st_size == last_size and last_size > 0
    except FileNotFoundError:
        return False




# ---------------------------------------------------------------------
# Gemini-driven visual target selection
# ---------------------------------------------------------------------

# Keep YOLO labels stable even if Gemini/manual text uses aliases.
LOCAL_LABEL_ALIASES = {
    "sd": "sd_card_port",
    "sd_card": "sd_card_port",
    "sd_card_slot": "sd_card_port",
    "tf_card": "sd_card_port",
    "tf_card_slot": "sd_card_port",
    "card_slot": "sd_card_port",
    "storage_card_slot": "sd_card_port",
    "screen": "display_screen",
    "display": "display_screen",
    "lcd_screen": "display_screen",
    "touch_screen": "display_screen",
    "bed": "print_bed",
    "platform": "print_bed",
    "printing_platform": "print_bed",
    "hotbed": "print_bed",
    "hot_bed": "print_bed",
    "filament_sensor": "filament_detector",
    "filament_runout_sensor": "filament_detector",
    "material_breakage_detection": "filament_detector",
    "extruder": "extruder_motor",
    "feeder": "extruder_motor",
    "feed_motor": "extruder_motor",
    "filament_tube": "bowden_tube",
    "teflon_tube": "bowden_tube",
    "tube": "bowden_tube",
    "hotend": "nozzle_kit",
    "hot_end": "nozzle_kit",
    "nozzle": "nozzle_kit",
    "spool": "filament_spool",
    "rack": "filament_holder",
    "spool_holder": "filament_holder",
    "holder": "filament_holder",
    "feeding_holder": "filament_holder",
    "feeding_holder_components": "filament_holder",
    "usb": "usb_port",
    "usb_port": "usb_port",
    "lan": "network_interface",
    "ethernet": "network_interface",
    "network": "network_interface",
    "wifi_port": "network_interface",
    "power": "power_switch",
    "switch": "power_switch",
    "switch_control": "power_switch",
    "power_button": "power_switch",
    "tool_box": "toolbox",
}

# Text phrases used only for deciding whether a candidate is actually related
# to the question/answer. Captions still come from Gemini when available.
ANSWER_LABEL_ALIASES = {
    "sd_card_port": ["sd card", "tf card", "storage card", "memory card", "card slot", "sd/tf"],
    "usb_port": ["usb", "usb port"],
    "side_ports": ["side port", "side ports", "port area"],
    "network_interface": ["network interface", "wi-fi port", "wifi port", "ethernet", "lan"],
    "display_screen": ["screen", "display", "lcd", "touch screen"],
    "power_switch": ["power switch", "switch control", "power button", "powered off", "power off"],
    "filament_detector": ["filament detector", "filament sensor", "runout sensor", "material breakage detection"],
    "extruder_motor": ["extruder", "extruder motor", "feeder", "feed motor", "extrusion spring"],
    "bowden_tube": ["bowden tube", "teflon tube", "tube"],
    "filament_spool": ["filament spool", "spool", "filament roll"],
    "filament_holder": ["filament holder", "spool holder", "feeding holder", "rack"],
    "nozzle_kit": ["nozzle", "nozzle kit", "hotend", "hot end"],
    "print_bed": ["print bed", "printing platform", "platform", "bed", "hot bed", "glass"],
    "gantry_frame": ["gantry frame", "frame", "z-axis profile"],
    "x_axis_gantry": ["x axis", "x-axis", "gantry"],
    "x_axis_motor": ["x-axis motor", "x axis motor"],
    "y_axis_motor": ["y-axis motor", "y axis motor"],
    "y_axis_belt_adjuster": ["belt adjuster", "belt adjusting knob", "y-axis belt", "belt tension"],
    "bed_clamp": ["bed clamp", "glass handle", "platform replacement"],
    "toolbox": ["tool box", "toolbox"],
    "qr_code": ["qr code", "scan qr"],
}

QUESTION_HINTS = [
    (("sd",), "sd_card_port"),
    (("tf",), "sd_card_port"),
    (("card", "slot"), "sd_card_port"),
    (("storage", "card"), "sd_card_port"),
    (("power",), "power_switch"),
    (("switch",), "power_switch"),
    (("screen",), "display_screen"),
    (("display",), "display_screen"),
    (("filament", "sensor"), "filament_detector"),
    (("filament", "detector"), "filament_detector"),
    (("filament", "load"), "filament_detector"),
    (("filament", "stuck"), "extruder_motor"),
    (("extruder",), "extruder_motor"),
    (("spool",), "filament_spool"),
    (("holder",), "filament_holder"),
    (("tube",), "bowden_tube"),
    (("nozzle",), "nozzle_kit"),
    (("bed",), "print_bed"),
    (("level",), "print_bed"),
    (("usb",), "usb_port"),
    (("network",), "network_interface"),
]


def _norm_text(value: Any) -> str:
    return str(value or "").lower().replace("_", " ").replace("-", " ")


def normalize_visual_label(label: Any, visual_guidance_module=None) -> Optional[str]:
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
    q = _norm_text(question)
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
        label = normalize_visual_label(raw, visual_guidance_module)
        if label:
            out.append(label)

    return list(dict.fromkeys(out))


def label_is_answer_aligned(label: str, question: str, answer: str, llm_json: Dict[str, Any], visual_guidance_module=None) -> bool:
    """
    Prevent noisy chunk-level targets from appearing when the answer does not
    actually talk about that object. Gemini captions still come through later.
    """
    text = _norm_text(f"{question} {answer}")
    label_text = _norm_text(label)

    if label_text and label_text in text:
        return True

    for phrase in ANSWER_LABEL_ALIASES.get(label, []):
        if _norm_text(phrase) in text:
            return True

    captions = llm_json.get("visual_captions") or {}
    if isinstance(captions, dict):
        for k, v in captions.items():
            key_label = normalize_visual_label(k, visual_guidance_module)
            if key_label == label and v:
                # Caption is allowed to align the object, but only if it is not
                # a generic object name. This helps with concise Gemini captions
                # like "Insert SD Card here".
                cap = _norm_text(v)
                if any(_norm_text(p) in cap for p in ANSWER_LABEL_ALIASES.get(label, [])):
                    return True

    return False


def gemini_caption_for_label(
    label: str,
    llm_json: Dict[str, Any],
    selected_chunk: Dict[str, Any],
    visual_guidance_module=None,
) -> str:
    captions = llm_json.get("visual_captions") or {}

    if isinstance(captions, dict):
        if label in captions and captions[label]:
            return str(captions[label])[:80]

        for k, v in captions.items():
            if normalize_visual_label(k, visual_guidance_module) == label and v:
                return str(v)[:80]

    action_id = selected_chunk.get("action_id") or selected_chunk.get("id")
    if visual_guidance_module is not None and hasattr(visual_guidance_module, "short_caption"):
        try:
            return str(visual_guidance_module.short_caption(label, action_id=action_id, llm_json=llm_json))[:80]
        except Exception:
            pass

    return label.replace("_", " ").title()


def build_gemini_visual_targets(
    question: str,
    result: Dict[str, Any],
    detections: Any,
    max_targets: int = 2,
    visual_guidance_module=None,
) -> List[Dict[str, str]]:
    """
    Demo-style Gemini annotation builder.

    Selection is strict:
      1. Prefer Gemini referenced_objects.
      2. Accept pipeline visual_targets only if they align with the final answer.
      3. Use question hints only as a last resort.
      4. Avoid broad chunk required/highlight object spam.

    Captioning is Gemini-first:
      - If llm_json.visual_captions contains a caption for the selected label, use it.
      - Otherwise fall back to the package short_caption or label title.
    """
    answer = result.get("answer") or ""
    llm_json = result.get("llm_json") or {}
    selected_chunk = result.get("selected_chunk") or {}

    visible = set(visible_labels_from_detections(detections, visual_guidance_module))

    candidates: List[Dict[str, Any]] = []

    # Strongest signal: Gemini explicitly selected the object.
    ref_objects = llm_json.get("referenced_objects")
    if isinstance(ref_objects, list):
        for raw in ref_objects:
            label = normalize_visual_label(raw, visual_guidance_module)
            if label:
                candidates.append({"label": label, "source": "gemini_referenced_objects"})

    # Pipeline-generated targets can be good, but filter them against final text.
    for target in result.get("visual_targets") or []:
        if not isinstance(target, dict):
            continue
        label = normalize_visual_label(target.get("label"), visual_guidance_module)
        if label:
            candidates.append({
                "label": label,
                "caption": target.get("caption"),
                "source": target.get("source", "pipeline_visual_targets"),
            })

    # Last-resort question hint for cases where Gemini does not expose llm_json.
    for label in labels_from_question(question):
        candidates.append({"label": label, "source": "question_hint"})

    # If Gemini was silent, allow verification/chunk targets only when they appear
    # in the final answer. This prevents unrelated required_objects from showing.
    fallback_groups = [
        (result.get("verification") or {}).get("referenced_objects"),
        selected_chunk.get("highlight_objects"),
    ]
    for group in fallback_groups:
        if not isinstance(group, list):
            continue
        for raw in group:
            label = normalize_visual_label(raw, visual_guidance_module)
            if label:
                candidates.append({"label": label, "source": "answer_aligned_fallback"})

    final: List[Dict[str, str]] = []
    seen = set()

    def try_add(candidate: Dict[str, Any], require_visible: bool) -> None:
        label = normalize_visual_label(candidate.get("label"), visual_guidance_module)
        if not label or label in seen or len(final) >= max_targets:
            return

        if require_visible and visible and label not in visible:
            return

        source = str(candidate.get("source", "gemini_visual_target"))

        # Gemini references are allowed if either answer-aligned or no clear answer
        # mention exists but the question itself points to the same object.
        is_gemini = source == "gemini_referenced_objects"
        is_question_hint = source == "question_hint"

        if not (is_gemini or is_question_hint):
            if not label_is_answer_aligned(label, question, answer, llm_json, visual_guidance_module):
                return
        else:
            # Even Gemini should not produce visual clutter. Keep it if it aligns
            # with answer/question. If not, skip unless it is the only candidate later.
            if not label_is_answer_aligned(label, question, answer, llm_json, visual_guidance_module):
                return

        seen.add(label)
        final.append({
            "label": label,
            "caption": str(candidate.get("caption") or gemini_caption_for_label(label, llm_json, selected_chunk, visual_guidance_module)),
            "source": source,
        })

    # Pass 1: visible and answer-aligned objects.
    for candidate in candidates:
        try_add(candidate, require_visible=True)

    # Pass 2: answer-aligned objects even if not currently visible; they will draw
    # when the matching YOLO object appears.
    if not final:
        for candidate in candidates:
            try_add(candidate, require_visible=False)

    # Last-resort: if everything was filtered but Gemini clearly provided one object,
    # keep only one to avoid stale/noisy annotations.
    if not final and isinstance(ref_objects, list):
        for raw in ref_objects:
            label = normalize_visual_label(raw, visual_guidance_module)
            if label:
                final.append({
                    "label": label,
                    "caption": gemini_caption_for_label(label, llm_json, selected_chunk, visual_guidance_module),
                    "source": "gemini_last_resort",
                })
                break

    return final[:max_targets]

# ---------------------------------------------------------------------
# IDs / paths
# ---------------------------------------------------------------------

def frame_id_from_path(path: Path) -> Optional[int]:
    """
    Supports:
      f_000130.jpg
      frame_00069.jpg
      names containing f_123 or frame_123
    """
    m = re.search(r"(?:^|_)(?:f|frame)_(\d+)", path.stem)
    if not m:
        m = re.search(r"(?:f|frame)_(\d+)", path.stem)
    if not m:
        nums = re.findall(r"\d+", path.stem)
        return int(nums[-1]) if nums else None
    return int(m.group(1))


def audio_id_from_path(path: Path) -> str:
    return path.stem


def load_optional_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        warn(f"Could not read JSON {path}: {e!r}")
        return {}


def sorted_frame_paths(conv_dir: Path) -> List[Path]:
    paths: List[Path] = []
    for pat in ["f_*.jpg", "frame_*.jpg", "f_*.jpeg", "frame_*.jpeg"]:
        paths.extend(conv_dir.glob(pat))

    return sorted(
        set(paths),
        key=lambda p: (
            frame_id_from_path(p) if frame_id_from_path(p) is not None else 10**12,
            p.name,
        ),
    )


def sorted_audio_paths(conv_dir: Path) -> List[Path]:
    paths: List[Path] = []
    for pat in ["a_*.m4a", "a_*.wav", "a_*.mp3"]:
        paths.extend(conv_dir.glob(pat))

    return sorted(
        set(paths),
        key=lambda p: (p.stat().st_mtime if p.exists() else 0.0, p.name),
    )


# ---------------------------------------------------------------------
# Whisper
# ---------------------------------------------------------------------

FAST_WHISPER_REPOS = {
    "tiny": "Systran/faster-whisper-tiny",
    "tiny.en": "Systran/faster-whisper-tiny.en",
    "base": "Systran/faster-whisper-base",
    "base.en": "Systran/faster-whisper-base.en",
    "small": "Systran/faster-whisper-small",
    "small.en": "Systran/faster-whisper-small.en",
    "medium": "Systran/faster-whisper-medium",
    "medium.en": "Systran/faster-whisper-medium.en",
    "large-v1": "Systran/faster-whisper-large-v1",
    "large-v2": "Systran/faster-whisper-large-v2",
    "large-v3": "Systran/faster-whisper-large-v3",
}


def resolve_whisper_repo(model: str) -> str:
    p = Path(model).expanduser()
    if p.exists():
        return str(p.resolve())
    if model in FAST_WHISPER_REPOS:
        return FAST_WHISPER_REPOS[model]
    if "/" in model:
        return model
    return f"Systran/faster-whisper-{model}"


def setup_hf_cache(cache_dir: str) -> str:
    cache_path = str(Path(cache_dir).expanduser().resolve())
    os.environ["HF_HOME"] = cache_path
    os.environ["HF_HUB_CACHE"] = str(Path(cache_path) / "hub")
    os.environ["TRANSFORMERS_CACHE"] = str(Path(cache_path) / "transformers")
    return cache_path


@dataclass
class WhisperConfig:
    model: str = "small.en"
    cache_dir: str = "~/.cache/huggingface"
    local_files_only: bool = True
    device: str = "cpu"
    compute_type: str = "int8"
    language: str = "en"
    beam_size: int = 5
    vad_filter: bool = False
    word_timestamps: bool = True


class WhisperTranscriber:
    """
    Raw faster-whisper wrapper.
    No domain correction, no hotwords, no initial prompt.
    """
    def __init__(self, cfg: WhisperConfig):
        self.cfg = cfg
        self.model = None
        self.model_path: Optional[str] = None

    def load(self) -> None:
        if self.model is not None:
            return

        from huggingface_hub import snapshot_download
        from faster_whisper import WhisperModel

        cache_dir = setup_hf_cache(self.cfg.cache_dir)
        repo_or_path = resolve_whisper_repo(self.cfg.model)

        log(f"Loading STT model: {self.cfg.model}")
        log(f"Resolved STT model: {repo_or_path}")
        log(f"HF cache: {cache_dir}")
        log(f"local_files_only={self.cfg.local_files_only}")

        p = Path(repo_or_path).expanduser()
        if p.exists():
            self.model_path = str(p.resolve())
        else:
            t0 = time.time()
            self.model_path = snapshot_download(
                repo_id=repo_or_path,
                cache_dir=cache_dir,
                local_files_only=self.cfg.local_files_only,
            )
            log(f"STT model path resolved in {time.time() - t0:.2f}s")

        t1 = time.time()
        self.model = WhisperModel(
            self.model_path,
            device=self.cfg.device,
            compute_type=self.cfg.compute_type,
        )
        log(f"STT model loaded in {time.time() - t1:.2f}s")

    def transcribe(self, audio_path: Path) -> Dict[str, Any]:
        self.load()

        if not audio_path.exists():
            raise FileNotFoundError(audio_path)

        log(f"Transcribing audio: {audio_path}")
        t0 = time.time()

        segments_iter, info = self.model.transcribe(
            str(audio_path),
            language=self.cfg.language if self.cfg.language else None,
            beam_size=self.cfg.beam_size,
            vad_filter=self.cfg.vad_filter,
            word_timestamps=self.cfg.word_timestamps,
        )

        segments = []
        parts = []

        for i, seg in enumerate(segments_iter, start=1):
            text = (seg.text or "").strip()
            parts.append(text)

            row: Dict[str, Any] = {
                "segment_id": i,
                "start": float(seg.start),
                "end": float(seg.end),
                "text": text,
                "words": [],
            }

            if getattr(seg, "words", None):
                for w in seg.words:
                    row["words"].append({
                        "start": float(w.start),
                        "end": float(w.end),
                        "word": w.word,
                        "probability": float(w.probability) if w.probability is not None else None,
                    })

            # faster-whisper segments may expose these depending on version.
            for attr in ["avg_logprob", "no_speech_prob", "compression_ratio"]:
                if hasattr(seg, attr):
                    value = getattr(seg, attr)
                    if isinstance(value, (int, float)):
                        row[attr] = float(value)

            segments.append(row)
            log(f"  [{seg.start:6.2f} - {seg.end:6.2f}] {text}")

        transcript = " ".join(x for x in parts if x).strip()

        result = {
            "audio_path": str(audio_path),
            "audio_size_bytes": audio_path.stat().st_size,
            "model": self.cfg.model,
            "model_path": self.model_path,
            "device": self.cfg.device,
            "compute_type": self.cfg.compute_type,
            "language": getattr(info, "language", None),
            "language_probability": getattr(info, "language_probability", None),
            "transcript": transcript,
            "segments": segments,
            "duration_sec": time.time() - t0,
        }

        log(f"STT done in {result['duration_sec']:.2f}s")
        log(f"Transcript: {transcript!r}")
        return result


# ---------------------------------------------------------------------
# RAG package import
# ---------------------------------------------------------------------

def find_rag_package_dir(user_dir: Optional[str]) -> Path:
    if user_dir:
        p = Path(user_dir).expanduser().resolve()
        if not (p / "__init__.py").exists():
            raise FileNotFoundError(f"RAG package dir must contain __init__.py: {p}")
        if not (p / "pipeline.py").exists():
            raise FileNotFoundError(f"RAG package dir must contain pipeline.py: {p}")
        return p

    candidates = [
        Path.cwd(),
        Path.cwd() / "cr6se_pipeline",
        Path.cwd() / "src",
        Path("/home/imelanlioglu21/project/cr6se_pipeline"),
    ]

    for p in candidates:
        if (p / "__init__.py").exists() and (p / "pipeline.py").exists():
            return p.resolve()

    raise FileNotFoundError(
        "Could not find RAG package directory. Pass --rag-package-dir /path/to/cr6se_pipeline"
    )


def import_rag_package(package_dir: Path, package_name: str = "cr10_live_worker_pkg") -> str:
    init_file = package_dir / "__init__.py"

    spec = importlib.util.spec_from_file_location(
        package_name,
        init_file,
        submodule_search_locations=[str(package_dir)],
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not create import spec for {package_dir}")

    pkg = importlib.util.module_from_spec(spec)
    sys.modules[package_name] = pkg
    spec.loader.exec_module(pkg)

    return package_name


# ---------------------------------------------------------------------
# State
# ---------------------------------------------------------------------

@dataclass
class FrameRecord:
    frame_id: Optional[int]
    input_path: str
    output_image_file: Optional[str]
    visible_objects: List[str]
    detections: Any
    timestamp: float
    num_visible_objects: int = 0


@dataclass
class ConversationState:
    conv_id: str
    conv_in_dir: Path
    conv_out_dir: Path
    controller: Any

    seen_frames: set = field(default_factory=set)
    seen_audios: set = field(default_factory=set)

    frame_counter: int = 0
    audio_counter: int = 0

    recent_frames: "OrderedDict[str, FrameRecord]" = field(default_factory=OrderedDict)

    last_text_response: Optional[Dict[str, Any]] = None


# ---------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------

class LiveAIWorker:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.gelen_dir = Path(args.gelen_dir).expanduser()
        self.giden_dir = Path(args.giden_dir).expanduser()
        self.runtime_dir = Path(args.runtime_dir).expanduser()

        self.gelen_dir.mkdir(parents=True, exist_ok=True)
        self.giden_dir.mkdir(parents=True, exist_ok=True)
        self.runtime_dir.mkdir(parents=True, exist_ok=True)

        self.running = True
        self.conversations: Dict[str, ConversationState] = {}

        self._load_rag_components()
        self._load_stt()

    def _load_rag_components(self) -> None:
        package_dir = find_rag_package_dir(self.args.rag_package_dir)
        package_name = import_rag_package(package_dir)

        self.LiveStreamController = importlib.import_module(
            f"{package_name}.live_stream_controller"
        ).LiveStreamController
        self.data_loading = importlib.import_module(f"{package_name}.data_loading")
        self.retrieval = importlib.import_module(f"{package_name}.retrieval")
        self.pipeline = importlib.import_module(f"{package_name}.pipeline")
        try:
            self.visual_guidance = importlib.import_module(f"{package_name}.visual_guidance")
        except Exception:
            self.visual_guidance = None

        log(f"RAG package loaded from: {package_dir}")

        self.yolo_weights = Path(self.args.yolo_weights).expanduser()
        self.manual_chunks_path = Path(self.args.manual_chunks).expanduser()

        if not self.yolo_weights.exists():
            raise FileNotFoundError(f"YOLO weights not found: {self.yolo_weights}")
        if not self.manual_chunks_path.exists():
            raise FileNotFoundError(f"Manual chunks not found: {self.manual_chunks_path}")

        self.manual_chunks = self.data_loading.load_manual_chunks(self.manual_chunks_path)
        log(f"Loaded {len(self.manual_chunks)} manual chunks from: {self.manual_chunks_path}")

        self.rag_index = None
        if self.args.semantic_index:
            try:
                log("Building semantic RAG index...")
                t0 = time.time()
                self.rag_index = self.retrieval.build_rag_index(self.manual_chunks)
                log(f"Semantic RAG index built in {time.time() - t0:.2f}s")
            except Exception as e:
                warn(f"Semantic RAG index failed; falling back to keyword/manual retrieval. Error: {e!r}")
                self.rag_index = None
        else:
            log("Semantic RAG index disabled.")

    def _load_stt(self) -> None:
        cfg = WhisperConfig(
            model=self.args.whisper_model,
            cache_dir=self.args.hf_cache,
            local_files_only=not self.args.allow_download,
            device=self.args.whisper_device,
            compute_type=self.args.whisper_compute_type,
            language=self.args.whisper_language,
            beam_size=self.args.whisper_beam_size,
            vad_filter=self.args.whisper_vad_filter,
            word_timestamps=not self.args.no_word_timestamps,
        )

        self.stt = WhisperTranscriber(cfg)

        if self.args.load_stt_at_startup:
            self.stt.load()
        else:
            log("STT will be loaded lazily on first audio.")

    def stop(self, *_args) -> None:
        log("Stopping worker...")
        self.running = False

    def list_conversation_dirs(self) -> List[Path]:
        """
        Only process real conversation folders.
        This avoids Jupyter folders like .ipynb_checkpoints.
        """
        if not self.gelen_dir.exists():
            return []

        conv_dirs = []
        for p in self.gelen_dir.iterdir():
            if not p.is_dir():
                continue
            if p.name.startswith("."):
                continue
            if p.name in {"__pycache__", "processed", "archive"}:
                continue
            if not p.name.startswith("conv_"):
                continue
            conv_dirs.append(p)

        return sorted(conv_dirs, key=lambda p: p.name)

    def get_or_create_conversation(self, conv_dir: Path) -> ConversationState:
        conv_id = conv_dir.name

        if conv_id in self.conversations:
            return self.conversations[conv_id]

        conv_out_dir = self.giden_dir / conv_id
        conv_out_dir.mkdir(parents=True, exist_ok=True)

        controller_output_dir = self.runtime_dir / conv_id / "controller"
        controller_output_dir.mkdir(parents=True, exist_ok=True)

        log(f"Creating conversation state: {conv_id}")

        controller = self.LiveStreamController(
            yolo_weights_path=self.yolo_weights,
            manual_chunks=self.manual_chunks,
            rag_index=self.rag_index,
            output_dir=controller_output_dir,
            yolo_conf=self.args.yolo_conf,
            yolo_imgsz=self.args.yolo_imgsz,
        )

        state = ConversationState(
            conv_id=conv_id,
            conv_in_dir=conv_dir,
            conv_out_dir=conv_out_dir,
            controller=controller,
        )

        self.conversations[conv_id] = state
        return state

    def trim_frame_buffer(self, state: ConversationState) -> None:
        while len(state.recent_frames) > self.args.max_frame_buffer:
            state.recent_frames.popitem(last=False)

    def output_frame_stem(self, frame_path: Path, fallback_counter: int) -> str:
        fid = frame_id_from_path(frame_path)
        if fid is not None:
            return f"r_frame_{fid:06d}"
        return f"r_frame_unknown_{fallback_counter:06d}"

    def _process_one_frame(
        self,
        state: ConversationState,
        frame_path: Path,
        dropped_input_frames: int = 0,
        ignore_frame_stride: bool = False,
    ) -> None:
        import cv2

        state.frame_counter += 1

        profile: Dict[str, Any] = {}
        frame_total_t0 = time.perf_counter()

        input_mtime = safe_mtime(frame_path)
        profile["queue_delay_ms"] = ((time.time() - input_mtime) * 1000.0) if input_mtime else None
        profile["dropped_input_frames"] = dropped_input_frames

        if (
            not ignore_frame_stride
            and self.args.frame_stride > 1
            and (state.frame_counter % self.args.frame_stride) != 0
        ):
            return

        imread_t0 = time.perf_counter()
        frame_bgr = cv2.imread(str(frame_path))
        profile["cv2_imread_ms"] = elapsed_ms(imread_t0)

        if frame_bgr is None:
            warn(f"Could not read frame: {frame_path}")
            return

        try:
            process_t0 = time.perf_counter()
            info = state.controller.process_frame(frame_bgr)
            profile["process_frame_ms"] = elapsed_ms(process_t0)
        except Exception as e:
            warn(f"YOLO/frame processing failed for {frame_path}: {e!r}")
            traceback.print_exc()
            return

        visible_objects = info.get("visible_objects", []) or []
        detections = copy.deepcopy(getattr(state.controller, "latest_stable_detections", None))

        out_stem = self.output_frame_stem(frame_path, state.frame_counter)
        out_img_name = f"{out_stem}.jpg"
        out_json_name = f"{out_stem}.json"
        out_img_path = state.conv_out_dir / out_img_name
        out_json_path = state.conv_out_dir / out_json_name

        annotated_path = info.get("annotated_image_path")

        try:
            write_jpg_t0 = time.perf_counter()
            if annotated_path and Path(annotated_path).exists():
                atomic_copy_file(Path(annotated_path), out_img_path)
            else:
                tmp_img = out_img_path.with_suffix(".tmp.jpg")
                ok = cv2.imwrite(str(tmp_img), frame_bgr)
                if not ok:
                    raise RuntimeError("cv2.imwrite returned False")
                os.replace(tmp_img, out_img_path)
            profile["write_jpg_ms"] = elapsed_ms(write_jpg_t0)
        except Exception as e:
            warn(f"Could not write output frame for {frame_path}: {e!r}")
            return

        profile["total_before_json_ms"] = elapsed_ms(frame_total_t0)

        meta = {
            "v": 1,
            "type": "ai.image.meta",
            "conv_id": state.conv_id,
            "frame_id": frame_id_from_path(frame_path),
            "source_frame_file": frame_path.name,
            "image_file": out_img_name,
            "has_annotations": bool(getattr(state.controller, "active_visual_targets", []) or []),
            "active_visual_targets": getattr(state.controller, "active_visual_targets", []) or [],
            "visible_objects": visible_objects,
            "dropped_input_frames": dropped_input_frames,
            "latest_frame_only": bool(self.args.latest_frame_only),
            "created_at_unix": time.time(),
            "timings_ms": profile if self.args.profile else None,
        }

        write_json_t0 = time.perf_counter()
        atomic_write_json(out_json_path, meta)
        profile["write_json_ms"] = elapsed_ms(write_json_t0)
        profile["total_frame_ms"] = elapsed_ms(frame_total_t0)

        if self.args.profile:
            # Re-write with complete timing.
            meta["timings_ms"] = profile
            atomic_write_json(out_json_path, meta)
            log(
                f"[PROFILE][{state.conv_id}] frame {frame_path.name}: "
                f"queue={fmt_ms(profile.get('queue_delay_ms'))} "
                f"read={fmt_ms(profile.get('cv2_imread_ms'))} "
                f"process={fmt_ms(profile.get('process_frame_ms'))} "
                f"write_jpg={fmt_ms(profile.get('write_jpg_ms'))} "
                f"write_json={fmt_ms(profile.get('write_json_ms'))} "
                f"total={fmt_ms(profile.get('total_frame_ms'))} "
                f"dropped={dropped_input_frames}"
            )

        # Keep a stable runtime copy for later audio/RAG turns.
        # The incoming gelen_json frame may be deleted/rotated by the producer,
        # but audio processing can happen seconds later.
        snapshot_dir = self.runtime_dir / state.conv_id / "frame_snapshots"
        snapshot_dir.mkdir(parents=True, exist_ok=True)

        snapshot_name = f"{out_stem}_input.jpg"
        snapshot_path = snapshot_dir / snapshot_name

        try:
            tmp_snapshot = snapshot_path.with_suffix(".tmp.jpg")
            ok = cv2.imwrite(str(tmp_snapshot), frame_bgr)
            if ok:
                os.replace(tmp_snapshot, snapshot_path)
            else:
                snapshot_path = frame_path
        except Exception as e:
            warn(f"Could not write frame snapshot for {frame_path}: {e!r}")
            snapshot_path = frame_path

        rec_key = str(frame_id_from_path(frame_path)) if frame_id_from_path(frame_path) is not None else frame_path.name
        state.recent_frames[rec_key] = FrameRecord(
            frame_id=frame_id_from_path(frame_path),
            input_path=str(snapshot_path),
            output_image_file=out_img_name,
            visible_objects=visible_objects,
            detections=detections,
            timestamp=time.time(),
            num_visible_objects=len(visible_objects),
        )
        self.trim_frame_buffer(state)

        if self.args.verbose_frames:
            log(f"[{state.conv_id}] frame {frame_path.name} -> {out_json_name}, objects={visible_objects}")

    def process_new_frames(self, state: ConversationState) -> None:
        frame_paths = sorted_frame_paths(state.conv_in_dir)

        if not frame_paths:
            return

        if self.args.latest_frame_only:
            # Live-video mode:
            #   - collect unseen stable frames
            #   - mark all stable unseen frames as seen
            #   - process only the newest stable frame
            #
            # This prevents the worker from processing old frames in order when the
            # camera stream is faster than the output consumer.
            stable_unseen: List[Path] = []

            for frame_path in frame_paths:
                key = str(frame_path.resolve())
                if key in state.seen_frames:
                    continue

                if not wait_until_file_stable(
                    frame_path,
                    checks=self.args.stability_checks,
                    sleep_sec=self.args.stability_sleep,
                ):
                    continue

                stable_unseen.append(frame_path)

            if not stable_unseen:
                return

            # Mark all stable unseen frames as seen. Older ones are intentionally dropped.
            for p in stable_unseen:
                state.seen_frames.add(str(p.resolve()))

            # In live mode, "latest" must mean newest file written, not highest frame id.
            # This is important when a conversation folder contains mixed frame ranges
            # from reconnects/restarts, e.g. f_000170 and f_000300 interleaved.
            stable_unseen.sort(
                key=lambda p: (
                    safe_mtime(p) or 0.0,
                    frame_id_from_path(p) if frame_id_from_path(p) is not None else -1,
                    p.name,
                )
            )

            frame_to_process = stable_unseen[-1]
            dropped = max(0, len(stable_unseen) - 1)

            self._process_one_frame(
                state=state,
                frame_path=frame_to_process,
                dropped_input_frames=dropped,
                ignore_frame_stride=True,
            )
            return

        # Original ordered mode.
        for frame_path in frame_paths:
            key = str(frame_path.resolve())
            if key in state.seen_frames:
                continue

            if not wait_until_file_stable(
                frame_path,
                checks=self.args.stability_checks,
                sleep_sec=self.args.stability_sleep,
            ):
                continue

            state.seen_frames.add(key)

            self._process_one_frame(
                state=state,
                frame_path=frame_path,
                dropped_input_frames=0,
                ignore_frame_stride=False,
            )

    def select_frame_for_audio(self, state: ConversationState, audio_meta: Dict[str, Any]) -> Optional[FrameRecord]:
        if not state.recent_frames:
            return None

        target = audio_meta.get("ended_at_frame")
        if target is None:
            target = audio_meta.get("started_at_frame")

        records = list(state.recent_frames.values())

        if target is not None:
            try:
                target_int = int(target)
            except Exception:
                target_int = None

            if target_int is not None:
                near = []
                for rec in records:
                    if rec.frame_id is None:
                        continue
                    dist = abs(rec.frame_id - target_int)
                    if dist <= self.args.audio_frame_search_radius:
                        near.append((dist, rec))

                useful_near = [(dist, rec) for dist, rec in near if rec.num_visible_objects > 0]
                if useful_near:
                    useful_near.sort(key=lambda x: (x[0], -x[1].num_visible_objects))
                    return useful_near[0][1]

                if near:
                    near.sort(key=lambda x: x[0])
                    return near[0][1]

        for rec in reversed(records):
            if rec.num_visible_objects > 0:
                return rec

        return records[-1]

    def write_audio_response(
        self,
        state: ConversationState,
        audio_path: Path,
        audio_meta: Dict[str, Any],
        stt_result: Dict[str, Any],
        rag_result: Dict[str, Any],
        selected_frame: Optional[FrameRecord],
        response_image_name: Optional[str],
    ) -> None:
        audio_id = audio_meta.get("audio_id") or audio_id_from_path(audio_path)
        out_json_path = state.conv_out_dir / f"r_{audio_id}.json"

        response = {
            "v": 1,
            "type": "ai.text.done",
            "status": rag_result.get("status", "ok"),
            "reason": rag_result.get("reason"),
            "conv_id": state.conv_id,
            "audio_id": audio_id,
            "audio_file": audio_path.name,
            "transcript": stt_result.get("transcript", ""),
            "answer": rag_result.get("answer"),
            "selected_action": rag_result.get("selected_action"),
            "visual_targets": rag_result.get("visual_targets", []),
            "gemini_used": bool((rag_result.get("gemini_config") or {}).get("use_gemini_answer") or (rag_result.get("gemini_config") or {}).get("use_gemini_rerank") or (rag_result.get("llm_json") or {})),
            "gemini_config": rag_result.get("gemini_config"),
            "gemini_referenced_objects": (rag_result.get("llm_json") or {}).get("referenced_objects"),
            "gemini_visual_captions": (rag_result.get("llm_json") or {}).get("visual_captions"),
            "gemini_annotation_debug": rag_result.get("gemini_annotation_debug"),
            "used_references": rag_result.get("used_references", []),
            "image_file": response_image_name,
            "selected_frame": {
                "frame_id": selected_frame.frame_id if selected_frame else None,
                "input_path": selected_frame.input_path if selected_frame else None,
                "visible_objects": selected_frame.visible_objects if selected_frame else [],
            },
            "stt": stt_result,
            "timings_ms": rag_result.get("timings_ms"),
            "created_at_unix": time.time(),
        }

        atomic_write_json(out_json_path, response)
        state.last_text_response = response

        log(f"[{state.conv_id}] wrote text response: {out_json_path.name}")

    def process_new_audios(self, state: ConversationState) -> None:
        audio_paths = sorted_audio_paths(state.conv_in_dir)

        for audio_path in audio_paths:
            key = str(audio_path.resolve())
            if key in state.seen_audios:
                continue

            if not wait_until_file_stable(
                audio_path,
                checks=self.args.stability_checks,
                sleep_sec=self.args.stability_sleep,
            ):
                continue

            audio_meta_path = audio_path.with_suffix(".json")
            audio_meta = load_optional_json(audio_meta_path)
            audio_id = audio_meta.get("audio_id") or audio_id_from_path(audio_path)

            log(f"[{state.conv_id}] new audio: {audio_path.name}")

            try:
                audio_profile: Dict[str, Any] = {}
                audio_total_t0 = time.perf_counter()

                audio_mtime = safe_mtime(audio_path)
                audio_profile["audio_queue_delay_ms"] = ((time.time() - audio_mtime) * 1000.0) if audio_mtime else None

                stt_t0 = time.perf_counter()
                stt_result = self.stt.transcribe(audio_path)
                audio_profile["stt_ms"] = elapsed_ms(stt_t0)

                question = (stt_result.get("transcript") or "").strip()

                if not question:
                    raise RuntimeError("Empty transcript from Whisper.")

                select_t0 = time.perf_counter()
                selected_frame = self.select_frame_for_audio(state, audio_meta)
                audio_profile["select_frame_ms"] = elapsed_ms(select_t0)

                if selected_frame is None:
                    warn(f"[{state.conv_id}] no processed frame available for audio; answer will be text-only.")
                    selected_image_path = None
                    selected_detections = None
                else:
                    selected_image_path = Path(selected_frame.input_path)
                    selected_detections = selected_frame.detections

                    if not selected_image_path.exists():
                        warn(
                            f"[{state.conv_id}] selected frame no longer exists: "
                            f"{selected_image_path}; falling back to text-only RAG."
                        )
                        selected_image_path = None
                        selected_detections = None
                    else:
                        log(
                            f"[{state.conv_id}] selected visual frame: "
                            f"{selected_image_path.name}, objects={selected_frame.visible_objects}"
                        )

                response_image_name = f"r_{audio_id}.jpg"
                response_image_path = state.conv_out_dir / response_image_name

                log(f"[{state.conv_id}] running RAG for: {question!r}")

                rag_t0 = time.perf_counter()
                rag_result = self.pipeline.run_pipeline(
                    question=question,
                    image_path=selected_image_path if selected_image_path is not None else None,
                    detections=selected_detections,
                    manual_chunks=self.manual_chunks,
                    rag_index=self.rag_index,
                    output_path=response_image_path,
                    session_state=state.controller.session_state,

                    use_gemini=bool(os.getenv("GEMINI_API_KEY")) and self.args.use_gemini,
                    use_gemini_request_guard=False,
                    use_gemini_retrieval=False,
                    use_gemini_rerank=bool(os.getenv("GEMINI_API_KEY")) and self.args.use_gemini_rerank,
                    use_gemini_answer=bool(os.getenv("GEMINI_API_KEY")) and self.args.use_gemini_answer,
                    use_gemini_fallback_router=False,
                )
                audio_profile["rag_ms"] = elapsed_ms(rag_t0)
                audio_profile["total_before_write_ms"] = elapsed_ms(audio_total_t0)

                state.controller.session_state = rag_result.get("updated_session_state")

                original_pipeline_visual_targets = copy.deepcopy(rag_result.get("visual_targets") or [])

                visual_targets = build_gemini_visual_targets(
                    question=question,
                    result=rag_result,
                    detections=selected_detections,
                    max_targets=self.args.max_visual_targets,
                    visual_guidance_module=getattr(self, "visual_guidance", None),
                )

                rag_result["visual_targets"] = visual_targets
                rag_result["gemini_annotation_debug"] = {
                    "llm_referenced_objects": (rag_result.get("llm_json") or {}).get("referenced_objects"),
                    "llm_visual_captions": (rag_result.get("llm_json") or {}).get("visual_captions"),
                    "pipeline_visual_targets_original": original_pipeline_visual_targets,
                    "final_visual_targets": visual_targets,
                }

                if visual_targets or self.args.clear_targets_on_empty:
                    state.controller.active_visual_targets = visual_targets

                log(f"[{state.conv_id}] active visual targets: {visual_targets}")

                # Force the per-audio response image to use the final Gemini-filtered
                # targets. pipeline.run_pipeline may have drawn an image before this
                # stricter target selection ran.
                if selected_image_path is not None:
                    try:
                        import cv2
                        redraw_t0 = time.perf_counter()
                        frame_bgr = cv2.imread(str(selected_image_path))
                        if frame_bgr is not None:
                            redraw_info = state.controller.process_frame(frame_bgr)
                            annotated_path = redraw_info.get("annotated_image_path")
                            if annotated_path and Path(annotated_path).exists():
                                atomic_copy_file(Path(annotated_path), response_image_path)
                            else:
                                tmp_img = response_image_path.with_suffix(".tmp.jpg")
                                ok = cv2.imwrite(str(tmp_img), frame_bgr)
                                if ok:
                                    os.replace(tmp_img, response_image_path)
                        audio_profile["visual_redraw_ms"] = elapsed_ms(redraw_t0)
                    except Exception as e:
                        warn(f"[{state.conv_id}] could not redraw Gemini annotation image: {e!r}")

                if not response_image_path.exists():
                    response_image_name = None

                rag_result["timings_ms"] = audio_profile if self.args.profile else None

                write_resp_t0 = time.perf_counter()
                self.write_audio_response(
                    state=state,
                    audio_path=audio_path,
                    audio_meta=audio_meta,
                    stt_result=stt_result,
                    rag_result=rag_result,
                    selected_frame=selected_frame,
                    response_image_name=response_image_name,
                )
                audio_profile["write_response_json_ms"] = elapsed_ms(write_resp_t0)
                audio_profile["total_audio_ms"] = elapsed_ms(audio_total_t0)

                if self.args.profile:
                    # Re-write response once more so it contains complete timing.
                    rag_result["timings_ms"] = audio_profile
                    self.write_audio_response(
                        state=state,
                        audio_path=audio_path,
                        audio_meta=audio_meta,
                        stt_result=stt_result,
                        rag_result=rag_result,
                        selected_frame=selected_frame,
                        response_image_name=response_image_name,
                    )
                    log(
                        f"[PROFILE][{state.conv_id}] audio {audio_path.name}: "
                        f"queue={fmt_ms(audio_profile.get('audio_queue_delay_ms'))} "
                        f"stt={fmt_ms(audio_profile.get('stt_ms'))} "
                        f"select={fmt_ms(audio_profile.get('select_frame_ms'))} "
                        f"rag={fmt_ms(audio_profile.get('rag_ms'))} "
                        f"write={fmt_ms(audio_profile.get('write_response_json_ms'))} "
                        f"total={fmt_ms(audio_profile.get('total_audio_ms'))}"
                    )

                state.seen_audios.add(key)

            except Exception as e:
                warn(f"[{state.conv_id}] audio processing failed for {audio_path.name}: {e!r}")
                traceback.print_exc()

                error_json = state.conv_out_dir / f"r_{audio_id}_error.json"
                atomic_write_json(
                    error_json,
                    {
                        "v": 1,
                        "type": "ai.error",
                        "conv_id": state.conv_id,
                        "audio_id": audio_id,
                        "audio_file": audio_path.name,
                        "error": repr(e),
                        "created_at_unix": time.time(),
                    },
                )

                state.seen_audios.add(key)

    def run_once(self) -> None:
        for conv_dir in self.list_conversation_dirs():
            state = self.get_or_create_conversation(conv_dir)
            self.process_new_frames(state)
            self.process_new_audios(state)

    def run_forever(self) -> None:
        log("Live AI worker started.")
        log(f"Watching gelen dir: {self.gelen_dir}")
        log(f"Writing giden dir: {self.giden_dir}")

        while self.running:
            try:
                self.run_once()
            except Exception as e:
                warn(f"Top-level loop error: {e!r}")
                traceback.print_exc()

            time.sleep(self.args.poll_interval)

        log("Live AI worker stopped.")


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()

    p.add_argument("--gelen-dir", default="data/gelen_json")
    p.add_argument("--giden-dir", default="data/giden_json")
    p.add_argument("--runtime-dir", default="outputs/live_ai_worker_runtime")

    p.add_argument("--rag-package-dir", default=None)
    p.add_argument("--yolo-weights", required=True)
    p.add_argument("--manual-chunks", required=True)

    p.add_argument("--semantic-index", action="store_true")
    p.add_argument("--yolo-conf", type=float, default=0.25)
    p.add_argument("--yolo-imgsz", type=int, default=640)

    p.add_argument("--whisper-model", default="small.en")
    p.add_argument("--hf-cache", default="~/.cache/huggingface")
    p.add_argument("--allow-download", action="store_true")
    p.add_argument("--whisper-device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--whisper-compute-type", default="int8")
    p.add_argument("--whisper-language", default="en")
    p.add_argument("--whisper-beam-size", type=int, default=5)
    p.add_argument("--whisper-vad-filter", action="store_true")
    p.add_argument("--no-word-timestamps", action="store_true")
    p.add_argument("--load-stt-at-startup", action="store_true")

    p.add_argument("--poll-interval", type=float, default=0.2)
    p.add_argument("--stability-checks", type=int, default=2)
    p.add_argument("--stability-sleep", type=float, default=0.05)

    p.add_argument("--frame-stride", type=int, default=1, help="Process every Nth frame. 1 = every frame.")
    p.add_argument("--latest-frame-only", action="store_true", help="Live mode: drop stale unseen frames and process only the newest stable frame each loop. Overrides frame-stride for frames.")
    p.add_argument("--max-frame-buffer", type=int, default=300)
    p.add_argument("--audio-frame-search-radius", type=int, default=60)

    p.add_argument("--use-gemini", action="store_true")
    p.add_argument("--use-gemini-rerank", action="store_true")
    p.add_argument("--use-gemini-answer", action="store_true")

    p.add_argument("--max-visual-targets", type=int, default=2, help="Maximum Gemini/RAG visual annotations to keep active.")
    p.add_argument("--clear-targets-on-empty", action="store_true", default=True, help="Clear old visual targets when a new audio answer produces no targets.")

    p.add_argument("--verbose-frames", action="store_true")
    p.add_argument("--profile", action="store_true", help="Log and write detailed timing for frames and audio.")
    p.add_argument("--run-once", action="store_true", help="Process current files once, then exit.")

    return p


def main() -> None:
    args = build_argparser().parse_args()

    worker = LiveAIWorker(args)

    signal.signal(signal.SIGINT, worker.stop)
    signal.signal(signal.SIGTERM, worker.stop)

    if args.run_once:
        worker.run_once()
    else:
        worker.run_forever()


if __name__ == "__main__":
    main()
