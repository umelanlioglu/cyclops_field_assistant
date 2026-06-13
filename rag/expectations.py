"""Expectation checking: compare detected parts against the selected task."""

from __future__ import annotations

from typing import Any, Dict, List

from .labels import normalize_label


DEFAULT_REQUIRED_OBJECTS_BY_ACTION = {
    # Fallbacks only. Chunk JSON should normally define required_objects.
    "load_filament": ["filament_spool", "filament_holder", "filament_detector", "extruder_motor", "nozzle_kit"],
    "replace_filament": ["extruder_motor", "nozzle_kit"],
    "bed_leveling": ["print_bed", "display_screen", "nozzle_kit"],
    "preheating": ["display_screen", "nozzle_kit", "print_bed"],
    "cable_connection": ["nozzle_kit", "power_switch"],
    "power_connection": ["power_switch"],
    "power_safety": ["power_switch"],
    "pull_rod_installation": ["gantry_frame"],
    "start_printing_storage_card": ["sd_card_port"],
    "storage_card_safety": ["sd_card_port"],
    "print_file_troubleshooting": ["sd_card_port", "display_screen"],
    "extruder_troubleshooting": ["filament_detector", "extruder_motor", "nozzle_kit"],
    "heating_troubleshooting": ["display_screen", "nozzle_kit"],
    "wifi_printing": ["display_screen", "qr_code"],
    "usb_online_printing": ["usb_port"],
    "motion_troubleshooting": ["x_axis_gantry", "x_axis_motor", "y_axis_motor", "print_bed"],
    "cleaning_maintenance": ["print_bed"],
    "belt_platform_adjustment": ["print_bed", "y_axis_belt_adjuster"],
    "identify_part": [],
    "specifications": [],
    "tools_and_parts": [],
    "firmware_support": [],
    "circuit_wiring": [],
    "start_printing_software": [],
}


def _normalize_list(items: List[str]) -> List[str]:
    return list(dict.fromkeys(normalize_label(str(x)) for x in items if str(x).strip()))


def required_objects_for_chunk(chunk: Dict[str, Any]) -> List[str]:
    action_id = chunk.get("action_id", chunk.get("id", ""))

    # Chunk JSON must win first.
    if "required_objects" in chunk and chunk["required_objects"] is not None:
        return _normalize_list(chunk["required_objects"])

    if "expected_visible_objects" in chunk and chunk["expected_visible_objects"] is not None:
        return _normalize_list(chunk["expected_visible_objects"])

    return _normalize_list(DEFAULT_REQUIRED_OBJECTS_BY_ACTION.get(action_id, []))


def highlight_objects_for_chunk(chunk: Dict[str, Any]) -> List[str]:
    if chunk.get("highlight_objects") is not None:
        return _normalize_list(chunk.get("highlight_objects", []))

    if chunk.get("expected_visible_objects") is not None:
        return _normalize_list(chunk.get("expected_visible_objects", []))

    return required_objects_for_chunk(chunk)


def verify_expectations(chunk: Dict[str, Any], scene: Dict[str, Any]) -> Dict[str, Any]:
    visible = {normalize_label(x) for x in scene.get("visible_objects", [])}
    required = set(required_objects_for_chunk(chunk))
    highlight = set(highlight_objects_for_chunk(chunk))
    action_id = chunk.get("action_id", chunk.get("id", "unknown"))

    if action_id == "identify_part":
        referenced = sorted(visible)
    else:
        referenced = sorted(highlight & visible)

    missing_required = sorted(required - visible)
    found_required = sorted(required & visible)

    return {
        "action_id": action_id,
        "found_required": found_required,
        "missing_required": missing_required,
        "referenced_objects": referenced,
        "is_visually_ready": len(missing_required) == 0,
        "cannot_verify_from_image": chunk.get("cannot_verify_from_image", []),
        "warnings": chunk.get("warnings", []),
    }