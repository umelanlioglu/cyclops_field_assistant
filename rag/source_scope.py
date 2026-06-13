"""Source-aware manual scope guards."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional


SECONDARY_SOURCE_TYPES = {
    "secondary_guide",
    "manufacturer_support",
    "annotated_video",
    "community_forum",
    "forum",
    "general_fdm_guide",
    "slicer_docs",
}


def has_secondary_context(chunks: List[Dict[str, Any]]) -> bool:
    for chunk in chunks:
        source_type = str(chunk.get("source_type", "official_manual"))
        if source_type in SECONDARY_SOURCE_TYPES:
            return True
    return False


def detect_official_manual_scope_gap(question: str) -> Optional[str]:
    q = question.lower()

    if any(x in q for x in ["infill", "fill density", "fill percentage"]):
        return "recommended infill settings"

    if any(x in q for x in ["layer height", "smooth print", "resolution setting"]):
        return "recommended layer height settings"

    if any(x in q for x in ["retraction", "stringing", "oozing"]):
        return "retraction/stringing settings"

    # Support material / overhang settings.
    # Do not trigger on "supports resume printing" or "supports filament detection".
    support_material_query = (
        re.search(r"\bsupports?\b", q) is not None
        and any(x in q for x in [
            "overhang",
            "angle",
            "slicer",
            "slicing",
            "model",
            "when do i need",
            "do i need",
            "need supports",
            "support material",
        ])
    )

    if support_material_query:
        return "support/overhang rules"

    if any(x in q for x in ["food safe", "food-safe", "food contact", "cup"]):
        return "food-safety guidance"

    if any(x in q for x in ["glue stick", "glue", "hairspray", "adhesive"]):
        return "bed-adhesion glue guidance"

    if (
        any(x in q for x in ["petg", "tpu", "wood"])
        and any(x in q for x in ["temperature", "temp", "heat", "hot", "exact", "profile"])
    ):
        return "exact material-specific temperature profile"

    if (
        any(x in q for x in ["filament detector", "filament sensor", "material breakage"])
        and any(x in q for x in ["pause", "resume", "what does", "function", "does it"])
    ):
        return "filament detector runtime behavior"

    if (
        any(x in q for x in ["why", "reason", "what happens"])
        and any(x in q for x in ["force filament", "cold nozzle", "before the nozzle is hot", "before heating"])
    ):
        return "mechanical reason behind the filament-loading warning"

    return None


def build_scope_gap_answer(topic: str, retrieved: List[Dict[str, Any]]) -> str:
    if topic == "mechanical reason behind the filament-loading warning":
        return (
            "The official CR-10 Smart manual warns not to force filament before the nozzle is heated, "
            "and says to wait until the current temperature reaches the target temperature. "
            "The official manual context does not explain the mechanical reason."
        )

    if topic == "exact material-specific temperature profile":
        return (
            "The official CR-10 Smart manual lists PETG as a supported filament and gives printer limits "
            "such as nozzle temperature up to 260°C and bed temperature up to 100°C. "
            "However, the official manual context does not specify an exact PETG nozzle or bed temperature profile."
        )

    if topic == "filament detector runtime behavior":
        return (
            "The official CR-10 Smart manual shows the filament detector/material breakage path as part of "
            "the filament loading path. The official manual context does not specify whether it pauses the "
            "printer when filament runs out."
        )

    return (
        f"The official CR-10 Smart manual context I have does not specify {topic}. "
        "I can only answer from the retrieved official manual chunks right now. "
        "If we add secondary FDM/printer-guide chunks later, I can answer this as general guidance and clearly label it as non-manual information."
    )


def is_vague_failure_question(question: str) -> bool:
    q = question.lower()

    vague_failure = any(x in q for x in [
        "failed halfway",
        "failed during",
        "it failed",
        "doesn't work",
        "does not work",
        "not working",
        "bad print",
        "messy",
        "ugly",
        "what should i check",
    ])

    specific_symptom = any(x in q for x in [
        "heating",
        "temperature",
        "filament not coming out",
        "clicking",
        "sd card",
        "tf card",
        "file not showing",
        "axis",
        "motor",
        "wifi",
        "wi-fi",
        "screen says",
        "error code",
    ])

    return vague_failure and not specific_symptom


def build_vague_failure_answer() -> str:
    return (
        "What exactly failed: heating/temperature, filament extrusion, SD card/file reading, "
        "axis movement, Wi-Fi/app connection, or print quality? If there is an error message on the screen, "
        "send the exact text or a close-up image."
    )