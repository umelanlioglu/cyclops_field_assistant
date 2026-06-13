"""Prompt builders for the CR-10 Smart assistant."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List


HUMAN_READABLE_LABELS = {
    "bed_clamp": "bed clamp",
    "bowden_tube": "Bowden / Teflon tube",
    "button": "side button",
    "display_screen": "display screen",
    "extruder_motor": "extruder motor",
    "filament_detector": "filament detector / runout sensor",
    "filament_holder": "filament holder / spool holder",
    "filament_spool": "filament spool",
    "gantry_frame": "gantry frame",
    "network_interface": "network / LAN interface",
    "nozzle_kit": "nozzle kit / hotend area",
    "power_switch": "power switch",
    "print_bed": "print bed / glass bed",
    "qr_code": "QR code",
    "sd_card_port": "SD/TF card port",
    "side_ports": "side ports area",
    "toolbox": "toolbox knob / toolbox area",
    "usb_port": "USB port",
    "x_axis_gantry": "X-axis gantry",
    "x_axis_motor": "X-axis motor",
    "y_axis_belt_adjuster": "Y-axis belt adjuster",
    "y_axis_motor": "Y-axis motor",
}

LABEL_COLOR_NAMES = {
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


def _action_id(chunk: Dict[str, Any]) -> str:
    return str(chunk.get("action_id", chunk.get("id", "")))


def _stable_ref_id(chunk: Dict[str, Any], index: int = 0) -> str:
    """Match the fallback style used by references.py when chunks have no refs."""
    action_id = _action_id(chunk) or "chunk"
    title = str(chunk.get("title", ""))
    raw = f"{action_id}:{title}:{index}"
    digest = hashlib.md5(raw.encode("utf-8")).hexdigest()[:8]
    return f"{action_id}_{digest}"


def _normalize_chunk_references_for_prompt(chunk: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return compact reference metadata for the LLM prompt.

    The pipeline should still return deterministic references separately. This
    prompt metadata only helps Gemini fill used_reference_ids without inventing
    source IDs, page numbers, URLs, or titles.
    """
    refs = chunk.get("references") or []

    if isinstance(refs, dict):
        refs = [refs]

    normalized: List[Dict[str, Any]] = []

    for i, ref in enumerate(refs):
        if not isinstance(ref, dict):
            continue

        normalized.append({
            "ref_id": ref.get("ref_id") or _stable_ref_id(chunk, i),
            "source_id": ref.get("source_id", chunk.get("source_id", "unknown")),
            "source_title": ref.get("source_title", chunk.get("source_title", "Unknown source")),
            "source_type": ref.get("source_type", chunk.get("source_type", "unknown")),
            "source_trust": ref.get("source_trust", chunk.get("source_trust", "unknown")),
            "section": ref.get("section") or chunk.get("title"),
            "page": ref.get("page"),
            "figure": ref.get("figure"),
            "locator": ref.get("locator"),
            "url": ref.get("url"),
            "action_id": _action_id(chunk),
            "chunk_title": chunk.get("title"),
        })

    if normalized:
        return normalized

    # Fallback for the current chunk JSON, which may not yet contain references.
    return [{
        "ref_id": _stable_ref_id(chunk),
        "source_id": chunk.get("source_id", "cr10smart_manual"),
        "source_title": chunk.get("source_title", "Creality CR-10 Smart User Manual"),
        "source_type": chunk.get("source_type", "official_manual"),
        "source_trust": chunk.get("source_trust", "high"),
        "section": chunk.get("title"),
        "page": chunk.get("page"),
        "figure": chunk.get("figure"),
        "locator": chunk.get("locator") or f"Official manual / {chunk.get('title')}",
        "url": chunk.get("url"),
        "action_id": _action_id(chunk),
        "chunk_title": chunk.get("title"),
    }]


def _build_reference_metadata(retrieved_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    metadata: List[Dict[str, Any]] = []
    seen = set()

    for chunk_rank, chunk in enumerate(retrieved_chunks, start=1):
        for ref in _normalize_chunk_references_for_prompt(chunk):
            ref_id = ref.get("ref_id")
            if not ref_id or ref_id in seen:
                continue

            compact = dict(ref)
            compact["chunk_rank"] = chunk_rank
            compact["chunk_score"] = chunk.get("score")
            metadata.append(compact)
            seen.add(ref_id)

    return metadata


def build_gemini_prompt(
    question: str,
    scene: Dict[str, Any],
    retrieved_chunks: List[Dict[str, Any]],
    verification: Dict[str, Any],
    session_state: Dict[str, Any] | None = None,
    turn_type: str = "task_question",
) -> str:
    session_state = session_state or {}
    reference_metadata = _build_reference_metadata(retrieved_chunks)
    valid_reference_ids = [ref["ref_id"] for ref in reference_metadata if ref.get("ref_id")]

    chunks_text = "\n\n".join([
        f"""
[Chunk {i + 1}]
Title: {chunk.get('title')}
Action ID: {chunk.get('action_id')}
Source type: {chunk.get('source_type', 'unknown')}
Source trust: {chunk.get('source_trust', 'unknown')}
Device ID: {chunk.get('device_id', 'unknown')}
Score: {chunk.get('score')}
Reference IDs: {[ref.get('ref_id') for ref in _normalize_chunk_references_for_prompt(chunk)]}
Language: {chunk.get('manual_language') or chunk.get('language') or 'English'}
Canonical chunk ID: {chunk.get('canonical_id') or chunk.get('id')}
Text: {chunk.get('text')}
Warnings: {chunk.get('warnings', [])}
""".strip()
        for i, chunk in enumerate(retrieved_chunks)
    ])

    visible_objects = scene.get("visible_objects", [])
    has_visual_context = bool(visible_objects)

    return f"""
You are a manual-grounded 3D printer installation assistant for the Creality CR-10 Smart.

User question:
{question}

Previous session memory:
{json.dumps(session_state, indent=2)}

User turn type:
{turn_type}

Detected visual context from the image:
{json.dumps(visible_objects, indent=2)}

Human-readable names for detected labels:
{json.dumps(HUMAN_READABLE_LABELS, indent=2)}

Visual annotation color names:
{json.dumps(LABEL_COLOR_NAMES, indent=2)}

Expectation check:
{json.dumps(verification, indent=2)}

Retrieved knowledge context:
{chunks_text}

Retrieved reference metadata:
{json.dumps(reference_metadata, indent=2)}

Valid reference IDs:
{json.dumps(valid_reference_ids, indent=2)}

Return ONLY valid JSON with this exact structure:
{{
  "answer": "short practical answer",
  "needs_visual_annotation": true,
  "referenced_objects": ["only labels from detected visual context"],
  "visual_captions": {{
    "detected_label": "2-5 word caption that matches the answer"
  }},
  "used_reference_ids": ["only ref_id values from Valid reference IDs"],
  "missing_or_uncertain_objects": [],
  "safety_warning": "",
  "memory_note": "short note about how this answer continues or updates the session",
  "output_language": "detected language name"
}}

Rules:
- Detect the user question language. Answer in the same language when it is English, Czech, Slovak, Hungarian, or German. If the language is mixed or unclear, answer in English.
- Keep official CR-10 Smart terminology exactly as it appears in the retrieved manual context. Preserve product/UI terms such as CR-10 Smart, Creality Slicer, G-Code, TF Card, SD Card, USB, PLA, ABS, TPU, PETG, and Wood.
- For visual_captions, use the same language as the answer when possible, but keep the detected label keys unchanged. Captions should be 2-5 words and should point to the object named in the answer.
- Treat source_type="official_manual" as official CR-10 Smart manual information.
- If a chunk has any other source_type, such as secondary_guide, manufacturer_support, annotated_video, community_forum, general_fdm_guide, or slicer_docs, explicitly label that information as non-manual guidance in the answer.
- Never present secondary-source guidance as if it came from the official manual.
- Prefer official_manual sources when they directly answer the question.
- If the answer uses non-manual sources, mention that the source is not the official CR-10 Smart manual.
- If the retrieved official manual context does not specify a detail, say that the official CR-10 Smart manual context does not specify it.
- Do not provide exact slicer settings, material temperatures, causes, consequences, or troubleshooting explanations unless they appear in the retrieved knowledge context.
- If the manual gives a warning but not the reason, state the warning only and say the manual context does not explain the reason.
- Do not invent consequences such as data corruption, clogging, gear damage, filament snapping, bacteria buildup, or failed prints unless the retrieved context says so.
- Use only retrieved knowledge context, retrieved reference metadata, detected visual context, and previous session memory.
- used_reference_ids must contain only ref_id values listed in Valid reference IDs.
- Include the ref_id values for the retrieved chunks that actually support your answer. Prefer the selected/top chunk reference when it answers the question.
- Do not invent page numbers, URLs, source titles, source types, or reference IDs.
- Do not write raw ref_id values in the user-facing answer text unless the user explicitly asks for reference IDs; the JSON field used_reference_ids is enough.
- If previous session memory has an active_action_id and the user asks a continuation question like "okay now?", "what next?", "done", or "continue", continue that task instead of restarting.
- Do not repeat the whole manual every turn.
- Give the next practical step or the next 1-2 actions.
- For text-only/manual questions, answer normally from the retrieved context. Do not over-emphasize missing visual verification when no image objects are detected.
- If the answer depends on seeing a specific part and that required object is missing, pause and mention what is missing or not visible.
- Do not claim an object is visible unless it appears in detected visual context.
- referenced_objects must only contain labels from detected visual context.
- If there is no detected visual context, referenced_objects must be an empty list, visual_captions must be an empty object, and needs_visual_annotation should be false.
- If needs_visual_annotation is true, visual_captions must be an object whose keys are detected labels from referenced_objects and whose values are short 2-5 word captions.
- visual_captions must parallel the answer text: use action phrases like "Insert filament", "Check card slot", or "Watch for flow", not generic object names like "Extruder motor" unless the user is asking to identify a part.
- When the answer refers to an object that is included in referenced_objects, append its visual color name in parentheses on first mention, for example: filament detector (Yellow). Use only the provided color names. Do not invent colors.
- Do not include visual_captions for missing_or_uncertain_objects or for labels not in detected visual context.
- Keep each visual caption concise, practical, and grounded in the retrieved context; do not add new instructions that are not also reflected in the answer.
- If something cannot be verified from the image and it matters for the user's task, say so briefly.
- Never expose raw detection labels such as display_screen, sd_card_port, nozzle_kit, extruder_motor, or filament_detector in the answer text. Convert them to natural names like display screen, SD/TF card port, nozzle kit, extruder motor, or filament detector. Raw labels are allowed only inside referenced_objects.
- Keep the answer concise and practical.
- If user turn type is "thanks_or_closing", only acknowledge politely and do not continue to the next installation step.
- If user turn type is "progress_update", treat the user's message as a progress update or correction. Do not go backward to an earlier step unless the user asks to repeat.
- If user turn type is "completion_confirmation", do not repeat the same instruction. Treat the message as confirmation that the active task/step was completed, state the success condition or what cannot be visually verified, then stop unless the user explicitly asks for the next step.
- If user turn type is "continuation", continue the active task from memory.
- Do not add UI paths, menu names, locations, or steps unless they are explicitly present in the retrieved manual context.
- If the retrieved chunk gives only a high-level sequence, answer with that high-level sequence instead of inventing exact submenus.
- If the user question is vague, such as "it does not work", "it failed", or "what should I check", ask a short clarification question instead of committing to one specific troubleshooting path.
""".strip()


def build_simple_answer(
    question: str,
    retrieved_chunks: List[Dict[str, Any]],
    scene: Dict[str, Any],
    verification: Dict[str, Any],
) -> str:
    top = retrieved_chunks[0]
    visible = scene.get("visible_objects", [])

    lines = []
    lines.append(f"Question: {question}")
    lines.append("")
    lines.append(f"Selected manual task: {top.get('title', top.get('id'))}")
    lines.append(top.get("text", ""))
    lines.append("")

    if visible:
        lines.append("Visible detected parts: " + ", ".join(visible) + ".")
    else:
        lines.append("No relevant printer parts were detected in the current image.")

    if verification["is_visually_ready"]:
        lines.append("Visual check: the required visible parts for this task are present.")
    else:
        lines.append("Visual check: the setup is not fully verifiable from this image.")
        if verification["found_required"]:
            lines.append("Found required parts: " + ", ".join(verification["found_required"]) + ".")
        if verification["missing_required"]:
            lines.append("Missing or not visible required parts: " + ", ".join(verification["missing_required"]) + ".")

    if verification["cannot_verify_from_image"]:
        lines.append("Cannot verify from image: " + ", ".join(verification["cannot_verify_from_image"]) + ".")

    if verification["warnings"]:
        lines.append("")
        lines.append("Safety / caution:")
        for warning in verification["warnings"]:
            lines.append(f"- {warning}")

    return "\n".join(lines)
