"""Manual chunk retrieval: keyword + semantic hybrid retrieval, intent boosts, and optional LLM reranking.

This file is a drop-in replacement for the earlier retrieval.py.

Main ideas:
- Keyword retrieval catches exact terms such as "SD card", "TF card", "nozzle".
- Semantic retrieval catches paraphrases such as "plastic roll" -> filament/spool.
- Lightweight intent detection separates setup questions from troubleshooting questions.
- Optional Gemini query understanding / reranking can be enabled for more flexible demos.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


# -----------------------------------------------------------------------------
# Basic chunk helpers
# -----------------------------------------------------------------------------


def get_action_id(chunk: Dict[str, Any]) -> str:
    return str(chunk.get("action_id", chunk.get("id", "")))



def chunk_to_embedding_text(chunk: Dict[str, Any]) -> str:
    """Text representation embedded into FAISS.

    Keep this rich: semantic retrieval works better when each chunk has action,
    objects, warnings, aliases/query terms, and the actual manual instruction.
    """
    if chunk.get("embedding_text"):
        return chunk["embedding_text"]

    return f"""
Title: {chunk.get('title', '')}
Language: {chunk.get('manual_language') or chunk.get('language') or 'English'}
Canonical chunk ID: {chunk.get('canonical_id') or chunk.get('id', '')}
Action ID: {get_action_id(chunk)}
Source: {chunk.get('source_type', '')}
Expected visible objects: {', '.join(chunk.get('expected_visible_objects', []))}
Highlight objects: {', '.join(chunk.get('highlight_objects', []))}
Warnings: {', '.join(chunk.get('warnings', []))}
Query terms and aliases: {', '.join(chunk.get('query_terms', []))}
Instruction: {chunk.get('text', '')}
""".strip()



def chunk_task_type(chunk: Dict[str, Any]) -> str:
    """Coarse type used for intent-aware ranking."""
    action_id = get_action_id(chunk).lower()
    title = str(chunk.get("title", "")).lower()

    if action_id == "identify_part" or "component overview" in title:
        return "identify"
    if "troubleshooting" in action_id or "troubleshooting" in title:
        return "troubleshooting"
    if "safety" in action_id or "safety" in title:
        return "safety"
    if "spec" in action_id or "spec" in title:
        return "specifications"
    return "setup"


# -----------------------------------------------------------------------------
# Query understanding
# -----------------------------------------------------------------------------


IDENTIFY_TRIGGERS = [
    "what part",
    "what is this",
    "what am i looking at",
    "identify",
    "which part",
    "what is this thing",
    "what is that",
]

TROUBLE_WORDS = [
    "can't",
    "cannot",
    "won't",
    "doesn't",
    "dont",
    "don't",
    "stuck",
    "jam",
    "jamming",
    "clicking",
    "problem",
    "issue",
    "error",
    "fail",
    "failed",
    "broken",
    "troubleshoot",
    "troubleshooting",
    "not coming out",
    "not heating",
    "not moving",
    "not detected",
    "not recognized",
    "fluctuation",
    "blocked",
    "stops",
    "stopped",
]

SETUP_WORDS = [
    "how",
    "can i",
    "where",
    "set",
    "setup",
    "install",
    "load",
    "insert",
    "feed",
    "put",
    "place",
    "mount",
    "attach",
    "connect",
    "start",
    "prepare",
    "use",
    "level",
    "print",
    "print from",
    "scan",
    "register",
    "log in",
    "login",
    "adjust",
    "change",
    "generate",
    "download",
]

SAFETY_WORDS = [
    "safe",
    "safety",
    "danger",
    "electric",
    "shock",
    "power",
    "unplug",
    "fire",
    "warning",
    "warn",
    "avoid",
    "must i avoid",
    "why must",
    "why does the manual warn",
]

SPEC_WORDS = [
    "spec",
    "specification",
    "parameter",
    "parameters",
    "dimension",
    "size",
    "weight",
    "voltage",
    "diameter",
    "temperature range",
    "build volume",
    "print size",
    "printing size",
]

SPEC_PATTERNS = [
    r"\bmaximum\b",
    r"\bminimum\b",
    r"\bstandard\b",
    r"\brecommended\b.*\bspeed\b",
    r"\bbuild volume\b",
    r"\bprint size\b",
    r"\bprinting size\b",
    r"\bhow hot\b",
    r"\bhow much\b.*\bpower\b",
    r"\btotal power\b",
    r"\bcompatible\b",
    r"\bwhat types?\b.*\bfilament\b",
    r"\bfile formats?\b",
    r"\bdoes .* support\b",
    r"\bsupports?\b.*\bresume\b",
    r"\bresume printing\b",
    r"\bdual z\b",
    r"\bdual z-axis\b",
    r"\bhow many\b.*\bmotors?\b",
]

LOCATION_PATTERNS = [
    r"\bwhere is\b",
    r"\bwhere can i find\b",
    r"\bwhere .* located\b",
    r"\bwhere .* situated\b",
]

MAINTENANCE_TERMS = [
    "clean",
    "wipe",
    "dust",
    "guide rail",
    "guide rails",
    "wheels",
    "glass print surface",
    "print surface",
]

SAFETY_WARNING_TERMS = [
    "gloves",
    "cotton gloves",
    "hot",
    "remove it",
    "remove print",
    "print finishes",
    "booting",
    "manually",
    "moved the print head",
    "moved the nozzle",
    "moved the bed",
]

# Canonical object labels should follow the YOLO/RAG label names.
OBJECT_ALIASES = {
    "filament_spool": ["filament spool", "spool", "roll", "filament roll", "plastic roll"],
    "filament_holder": ["filament holder", "spool holder", "material rack", "rack", "material rack folded", "spool rack"],
    "filament_detector": ["filament detector", "filament sensor", "runout sensor", "material breakage detector"],
    "extruder_motor": ["extruder", "extruder motor", "feeder", "feed gear", "extruder spring", "spring knob", "extrusion spring"],
    "nozzle_kit": ["nozzle", "nozzle kit", "hotend", "hot end", "print head"],
    "bowden_tube": ["bowden tube", "teflon tube", "filament tube", "white tube"],
    "display_screen": ["screen", "display", "touchscreen", "panel", "lcd", "control menu"],
    "print_bed": ["bed", "print bed", "platform", "build plate", "printing platform", "glass bed", "glass print surface", "print surface"],
    "sd_card_port": ["sd card", "tf card", "storage card", "memory card", "card slot", "card port", "sd/tf"],
    "usb_port": ["usb", "usb port", "usb drive", "usb connection", "computer connection"],
    "network_interface": ["network interface", "wlan", "lan", "ethernet", "network port", "wi-fi port", "wifi port", "wireless", "app", "cloud"],
    "side_ports": ["side ports", "port area", "usb and lan area"],
    "button": ["button", "side button", "wifi reset", "wi-fi reset"],
    "qr_code": ["qr code", "scan code", "scan qr"],
    "power_switch": ["power switch", "red switch", "switch control", "power outlet", "power cord", "power cable", "turn it on", "turn it off"],
    "gantry_frame": ["gantry", "gantry frame", "vertical frame", "z-axis frame", "z axis profile", "base frame"],
    "x_axis_gantry": ["x-axis gantry", "x axis gantry", "x-axis bar", "horizontal bar"],
    "x_axis_motor": ["x-axis motor", "x motor"],
    "y_axis_motor": ["y-axis motor", "y motor"],
    "y_axis_belt_adjuster": ["y-axis belt adjuster", "y belt knob", "belt adjuster", "belt tension knob"],
    "toolbox": ["toolbox", "tool box", "toolbox knob"],
    "bed_clamp": ["bed clamp", "bed clip", "glass clamp"],
}

# -----------------------------------------------------------------------------
# Multilingual manual support
# -----------------------------------------------------------------------------
# The bundled CR-10 Smart PDF contains English, Czech, Slovak, Hungarian, and German
# manual sections. These terms keep retrieval aligned with the official wording while
# preserving canonical YOLO labels and product/UI terms.

MULTILINGUAL_SETUP_WORDS = [
    # Czech / Slovak
    "jak", "kde", "nastavit", "instalovat", "vložit", "připojit", "pripojiť", "tisk", "tlač", "vyrovnat", "vyrovnám", "vyrovnávanie", "vyrovnanie", "předehřev", "predohrev",
    # Hungarian
    "hogyan", "hol", "beállítás", "telepítés", "betöltés", "betölteni", "csatlakoztatás", "nyomtatás", "szintezés", "szintezni", "előmelegítés",
    # German
    "wie", "wo", "einstellen", "installieren", "laden", "lade", "lädt", "einlegen", "anschließen", "drucken", "nivellierung", "nivellieren", "vorheizen",
]

MULTILINGUAL_TROUBLE_WORDS = [
    "problém", "chyba", "nefunguje", "zaseknut", "uvízl", "upchat", "nevyteká", "nevychází",
    "problém", "chyba", "nefunguje", "zaseknut", "zablokovan", "nevychádza",
    "probléma", "hiba", "nem működik", "elakadt", "beragadt", "nem jön ki",
    "problem", "fehler", "funktioniert nicht", "steckt fest", "blockiert", "kommt nicht heraus",
]

MULTILINGUAL_SAFETY_WORDS = [
    "bezpečnost", "upozornění", "varování", "vypnout", "napájení", "rukavice",
    "bezpečnosť", "upozornenie", "varovanie", "vypnúť", "napájanie", "rukavice",
    "biztonság", "figyelmeztetés", "kikapcsol", "áram", "kesztyű",
    "sicherheit", "hinweis", "warnung", "ausschalten", "strom", "handschuhe",
]

MULTILINGUAL_SPEC_WORDS = [
    "parametry", "specifikace", "velikost tisku", "teplota", "tryska", "vlákno",
    "parametre", "špecifikácie", "veľkosť tlače", "teplota", "dýza", "vlákno",
    "paraméterek", "specifikáció", "nyomtatási méret", "hőmérséklet", "fúvóka", "szálak",
    "parameter", "spezifikation", "druckgröße", "temperatur", "düse",
]

SETUP_WORDS.extend(MULTILINGUAL_SETUP_WORDS)
TROUBLE_WORDS.extend(MULTILINGUAL_TROUBLE_WORDS)
SAFETY_WORDS.extend(MULTILINGUAL_SAFETY_WORDS)
SPEC_WORDS.extend(MULTILINGUAL_SPEC_WORDS)

OBJECT_ALIASES.update({
    "filament_spool": [*OBJECT_ALIASES["filament_spool"], "filament", "vlákno", "vlákna", "szál", "szálak", "cívka", "cievka", "spol", "orsó", "spule"],
    "filament_holder": [*OBJECT_ALIASES["filament_holder"], "držák filamentu", "držák cívky", "držiak filamentu", "držiak cievky", "filament tartó", "spulenhalter", "filamenthalter", "materialhalter"],
    "filament_detector": [*OBJECT_ALIASES["filament_detector"], "detektor vláken", "detektor vlákien", "szálas érzékelő", "faser-detektor", "filamentsensor"],
    "extruder_motor": [*OBJECT_ALIASES["extruder_motor"], "extrudér", "motor extrudéru", "motor extrudéra", "extruder motor", "extrudermotor"],
    "nozzle_kit": [*OBJECT_ALIASES["nozzle_kit"], "tryska", "sada trysek", "dýza", "súprava trysiek", "fúvóka", "fúvókakészlet", "düse", "düsensatz"],
    "bowden_tube": [*OBJECT_ALIASES["bowden_tube"], "teflonová trubice", "teflónová trubica", "teflon cső", "teflonrohr"],
    "display_screen": [*OBJECT_ALIASES["display_screen"], "lcd obrazovka", "lcd displej", "lcd képernyő", "lcd-bildschirm", "bildschirm"],
    "print_bed": [*OBJECT_ALIASES["print_bed"], "lůžko", "lôžko", "ložisko", "ágy", "tisková platforma", "tlačová platforma", "nyomtatási platform", "druckplattform", "bett", "heizbett"],
    "sd_card_port": [*OBJECT_ALIASES["sd_card_port"], "slot pro kartu sd", "zásuvka na kartu sd", "sd-kártya foglalat", "sd-kartensteckplatz", "tf karta", "tf-kártya", "tf-karte", "úložná karta", "tárolókártya", "speicherkarte"],
    "usb_port": [*OBJECT_ALIASES["usb_port"], "usb port", "usb-port", "usb připojení", "usb pripojenie", "usb kapcsolat", "usb-verbindung"],
    "network_interface": [*OBJECT_ALIASES["network_interface"], "síťové rozhraní", "sieťové rozhranie", "hálózati interfész", "netzwerkschnittstelle"],
    "power_switch": [*OBJECT_ALIASES["power_switch"], "vypínač", "napájecí zásuvka", "elektrická zásuvka", "kapcsoló", "elektromos aljzat", "netzschalter", "steckdose"],
    "gantry_frame": [*OBJECT_ALIASES["gantry_frame"], "rám portálu", "portálový rám", "portálkeret", "portalrahmen"],
    "y_axis_belt_adjuster": [*OBJECT_ALIASES["y_axis_belt_adjuster"], "knoflík nastavení pásu", "gombík na nastavenie pásu", "szíjbeállító gomb", "riemeneinstellung"],
    "toolbox": [*OBJECT_ALIASES["toolbox"], "skříňka na nářadí", "skrinka na náradie", "szerszámos szekrény", "werkzeugschrank"],
})


def _phrase_in_text(text: str, phrase: str) -> bool:
    phrase = phrase.lower().strip()
    if not phrase:
        return False

    # Unicode-aware word boundaries. This supports terms like Düsentemperatur,
    # úložná karta, kártya, etc.
    pattern = r"(?<!\w)" + re.escape(phrase) + r"(?!\w)"
    return re.search(pattern, text.lower(), flags=re.UNICODE) is not None


def _contains_any(text: str, phrases: Iterable[str]) -> bool:
    return any(_phrase_in_text(text, p) for p in phrases)


def _mentioned_objects(q: str) -> List[str]:
    mentioned: List[str] = []
    for canonical, aliases in OBJECT_ALIASES.items():
        if any(_phrase_in_text(q, alias) for alias in aliases):
            mentioned.append(canonical)
    return mentioned


def is_identification_question(question: str) -> bool:
    q = question.lower()
    return _contains_any(q, IDENTIFY_TRIGGERS)


def is_specification_question(q: str) -> bool:
    if _contains_any(q, SPEC_WORDS):
        return True
    return any(re.search(pattern, q) for pattern in SPEC_PATTERNS)


def is_part_location_question(q: str) -> bool:
    return any(re.search(pattern, q) for pattern in LOCATION_PATTERNS)


def understand_query_heuristic(question: str, scene: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Cheap query understanding without any LLM call.

    This stays general: classify the question type, detect mentioned canonical
    printer objects, then produce action-level boosts/downranks. It should not
    depend on one exact evaluation question.
    """
    q = question.lower().strip()
    mentioned = _mentioned_objects(q)

    has_trouble = _contains_any(q, TROUBLE_WORDS)
    has_setup = _contains_any(q, SETUP_WORDS)
    has_maintenance = _contains_any(q, MAINTENANCE_TERMS)

    if is_identification_question(question):
        question_type = "identify"
    elif is_specification_question(q):
        question_type = "specifications"
    elif has_trouble:
        question_type = "troubleshooting"
    elif _contains_any(q, SAFETY_WORDS) and not has_setup:
        question_type = "safety"
    elif has_maintenance:
        question_type = "setup"
    elif is_part_location_question(q):
        question_type = "identify"
    elif has_setup:
        question_type = "setup"
    else:
        question_type = "unknown"

    boost_action_ids: List[str] = []
    downrank_action_ids: List[str] = []

    # Specification/capability questions should not be stolen by operational chunks.
    if question_type == "specifications":
        boost_action_ids.append("specifications")
        downrank_action_ids.extend([
            "load_filament",
            "bed_leveling",
            "heating_troubleshooting",
            "motion_troubleshooting",
            "power_connection",
            "cable_connection",
            "preheating",
        ])

    # General location questions about parts/components.
    if question_type == "identify":
        boost_action_ids.append("identify_part")

    # Maintenance/cleaning questions.
    if has_maintenance:
        boost_action_ids.append("cleaning_maintenance")
        downrank_action_ids.extend(["specifications", "start_printing_software"])

    # Safety/warning questions.
    if any(x in q for x in ["why", "warn", "warning", "recommended to wait", "safe", "safety", "avoid"]):
        if _contains_any(q, SAFETY_WARNING_TERMS):
            question_type = "safety"
            boost_action_ids.append("power_safety")

    # Storage-card questions: distinguish insertion from warnings/removal/file names.
    if "sd_card_port" in mentioned:
        if any(x in q for x in ["safe", "remove", "during printing", "while printing", "file name", "filename", "chinese", "special symbol"]):
            boost_action_ids.append("storage_card_safety")
            downrank_action_ids.append("start_printing_storage_card")
        elif question_type == "troubleshooting" or any(x in q for x in ["file", "read", "recognized", "identified", "showing"]):
            boost_action_ids.append("print_file_troubleshooting")
        else:
            boost_action_ids.append("start_printing_storage_card")

    # Filament questions: setup unless symptom/failure language is present.
    if any(obj in mentioned for obj in ["filament_spool", "filament_holder", "filament_detector", "extruder_motor", "nozzle_kit"]):
        if question_type == "troubleshooting" or any(x in q for x in ["stops", "stopped", "not coming out", "flow stops", "mid-print"]):
            boost_action_ids.append("extruder_troubleshooting")
            downrank_action_ids.append("load_filament")
        elif any(x in q for x in ["replace", "changing colors", "change colors", "remove and replace"]):
            boost_action_ids.append("replace_filament")
        elif any(x in q for x in ["compatible", "materials", "types of filament"]):
            boost_action_ids.append("specifications")
            downrank_action_ids.append("load_filament")
        else:
            boost_action_ids.append("load_filament")
            downrank_action_ids.append("extruder_troubleshooting")

    # Bed/platform questions. Keep max-temperature questions under specs.
    if "print_bed" in mentioned or "level" in q or "platform" in q:
        if question_type != "specifications":
            boost_action_ids.append("bed_leveling")

    # Heat/preheat questions. Max temperature belongs to specifications.
    if any(x in q for x in ["heat", "heating", "temperature", "hot", "nozzle"]):
        if question_type == "specifications":
            boost_action_ids.append("specifications")
            downrank_action_ids.extend(["heating_troubleshooting", "preheating"])
        elif question_type == "troubleshooting":
            boost_action_ids.append("heating_troubleshooting")
        elif "preheat" in q or "pre heating" in q or "heat up" in q:
            boost_action_ids.append("preheating")

    # Network/Wi-Fi/app questions.
    if "network_interface" in mentioned or any(x in q for x in ["wifi", "wi-fi", "wireless", "app", "cloud", "qr code", "network", "wlan"]):
        if is_part_location_question(q):
            boost_action_ids.append("identify_part")
        else:
            boost_action_ids.append("wifi_printing")

    # USB questions.
    if "usb_port" in mentioned:
        boost_action_ids.append("usb_online_printing")

    # Pull rod / rack / gantry setup.
    if "gantry_frame" in mentioned and any(x in q for x in ["install", "attach", "base frame", "gantry"]):
        boost_action_ids.append("gantry_frame_installation")

    if any(x in q for x in ["pull rod", "support rod"]):
        boost_action_ids.append("pull_rod_installation")

    if "filament_holder" in mentioned and any(x in q for x in ["rack", "spool holder", "mount", "fold", "folded", "different positions"]):
        boost_action_ids.append("filament_holder_installation")

    # Cable/board/internal port questions.
    if any(x in q for x in ["15-pin", "15 pin", "15pin", "adapter board", "thermistor", "photoelectric switch", "circuit", "board", "wire", "connector"]):
        if any(x in q for x in ["thermistor", "photoelectric switch", "circuit", "board"]):
            boost_action_ids.append("circuit_wiring")
        else:
            boost_action_ids.append("cable_connection")

    # Axis/belt/motion questions.
    if any(x in q for x in ["axis", "motor", "belt", "limit switch", "not moving", "print head by hand", "moved the print head", "moved the nozzle"]):
        if "belt" in q and any(x in q for x in ["adjust", "tension", "tightness"]):
            boost_action_ids.append("belt_platform_adjustment")
        elif question_type == "troubleshooting" or any(x in q for x in ["power off", "moved", "by hand", "manually"]):
            boost_action_ids.append("motion_troubleshooting")
        elif question_type == "identify":
            boost_action_ids.append("identify_part")

    # Power questions. Specs own power consumption/resume support; safety owns warnings.
    if "power_switch" in mentioned or "power" in q:
        if question_type == "specifications":
            boost_action_ids.append("specifications")
            downrank_action_ids.extend(["power_connection", "power_safety"])
        elif question_type == "safety" or any(x in q for x in ["booting", "manually", "power off", "moved"]):
            boost_action_ids.append("power_safety")
        else:
            boost_action_ids.append("power_connection")

    # Software/slicing questions.
    if any(x in q for x in ["slicer", "slicing", "g-code", "gcode", "generate", "model file"]):
        if any(x in q for x in ["compatible", "what slicing software"]):
            boost_action_ids.append("specifications")
        elif any(x in q for x in ["file name", "filename", "chinese", "special symbol"]):
            boost_action_ids.append("storage_card_safety")
        else:
            boost_action_ids.append("start_printing_software")

    # Firmware/support questions.
    if any(x in q for x in ["firmware", "upgrade", "download", "latest firmware", "support source"]):
        boost_action_ids.append("firmware_support")

    extra_terms = []
    for obj in mentioned:
        extra_terms.extend(OBJECT_ALIASES.get(obj, []))

    search_query = " ".join([
        question,
        f"intent: {question_type}",
        "mentioned objects: " + ", ".join(mentioned),
        "aliases: " + ", ".join(extra_terms),
    ]).strip()

    boost_action_ids = list(dict.fromkeys(boost_action_ids))
    downrank_action_ids = list(dict.fromkeys(downrank_action_ids))

    return {
        "original_question": question,
        "question_type": question_type,
        "search_query": search_query,
        "mentioned_objects": mentioned,
        "boost_action_ids": boost_action_ids,
        "downrank_action_ids": downrank_action_ids,
        "source": "heuristic",
    }


def understand_query_with_gemini(
    question: str,
    scene: Optional[Dict[str, Any]] = None,
    action_ids: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Optional LLM query understanding.

    Falls back to heuristic if Gemini is unavailable or returns malformed JSON.
    """
    fallback = understand_query_heuristic(question, scene)

    try:
        from .gemini_client import generate_json_answer
    except Exception:
        return fallback

    scene_objects = [] if scene is None else scene.get("visible_objects", [])
    action_ids = list(action_ids or [])

    prompt = f"""
You are a query understanding module for a CR-10 Smart 3D printer manual RAG system.

User question:
{question}

Detected visible objects:
{json.dumps(scene_objects, indent=2)}

Available action IDs:
{json.dumps(action_ids, indent=2)}

Return ONLY valid JSON:
{{
  "question_type": "setup | troubleshooting | identify | safety | specifications | unknown",
  "search_query": "expanded retrieval query with synonyms",
  "mentioned_objects": ["canonical object labels if any"],
  "boost_action_ids": ["action ids that should be favored"],
  "downrank_action_ids": ["action ids that should be penalized"],
  "reason": "one short reason"
}}

Rules:
- If the user asks how to put/load/install/set filament, prefer setup/load_filament.
- If the filament is stuck, not coming out, or the extruder clicks, prefer troubleshooting/extruder_troubleshooting.
- If the user asks about Wi-Fi, app setup, cloud printing, QR scanning, or wireless connection, prefer wifi_printing.
- If the user asks about pull rods or support rods, prefer pull_rod_installation.
- Do not invent action IDs outside Available action IDs.
""".strip()

    try:
        parsed = generate_json_answer(prompt)
        if not isinstance(parsed, dict):
            return fallback

        allowed = set(action_ids)
        boost = [a for a in parsed.get("boost_action_ids", []) if not allowed or a in allowed]
        downrank = [a for a in parsed.get("downrank_action_ids", []) if not allowed or a in allowed]

        return {
            "original_question": question,
            "question_type": parsed.get("question_type") or fallback["question_type"],
            "search_query": parsed.get("search_query") or fallback["search_query"],
            "mentioned_objects": parsed.get("mentioned_objects") or fallback["mentioned_objects"],
            "boost_action_ids": list(dict.fromkeys(boost or fallback["boost_action_ids"])),
            "downrank_action_ids": list(dict.fromkeys(downrank or fallback["downrank_action_ids"])),
            "reason": parsed.get("reason", ""),
            "source": "gemini",
        }
    except Exception:
        return fallback


# -----------------------------------------------------------------------------
# Query building
# -----------------------------------------------------------------------------



def build_rag_query(
    question: str,
    scene: Dict[str, Any],
    include_visible_objects: bool = False,
    query_understanding: Optional[Dict[str, Any]] = None,
) -> str:
    """Build retrieval query.

    Default keeps retrieval mostly question-driven. Visible objects are optional
    because generic component chunks can otherwise dominate retrieval.
    """
    base = query_understanding.get("search_query") if query_understanding else question

    if not include_visible_objects:
        return f"""
User question: {base}
Retrieve the most relevant CR-10 Smart manual task, expected objects, warnings, and troubleshooting guidance.
""".strip()

    visible = ", ".join(scene.get("visible_objects", []))
    return f"""
User question: {base}
Visible printer parts: {visible}
Retrieve the most relevant CR-10 Smart manual task, expected objects, warnings, and troubleshooting guidance.
""".strip()


# -----------------------------------------------------------------------------
# Indexing and base retrievers
# -----------------------------------------------------------------------------



def build_rag_index(manual_chunks: List[Dict[str, Any]], model_name: str = "paraphrase-multilingual-MiniLM-L12-v2") -> Dict[str, Any]:
    from sentence_transformers import SentenceTransformer
    import faiss

    # Multilingual manual chunks need a multilingual embedding model.
    # RAG_EMBEDDING_MODEL lets the live demo override this without code changes.
    model_name = os.getenv("RAG_EMBEDDING_MODEL", model_name)
    embedder = SentenceTransformer(model_name)
    texts = [chunk_to_embedding_text(c) for c in manual_chunks]
    embeddings = embedder.encode(texts, normalize_embeddings=True, show_progress_bar=True)
    embeddings = np.asarray(embeddings, dtype="float32")

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    return {"embedder": embedder, "index": index, "chunks": manual_chunks, "texts": texts}



def retrieve_chunks_semantic(query: str, rag_index: Dict[str, Any], k: int = 8) -> List[Dict[str, Any]]:
    embedder = rag_index["embedder"]
    index = rag_index["index"]
    chunks = rag_index["chunks"]

    q_emb = embedder.encode([query], normalize_embeddings=True)
    q_emb = np.asarray(q_emb, dtype="float32")
    scores, ids = index.search(q_emb, min(k, len(chunks)))

    results: List[Dict[str, Any]] = []
    for rank, (score, idx) in enumerate(zip(scores[0], ids[0]), start=1):
        if idx < 0:
            continue
        chunk = chunks[int(idx)].copy()
        chunk["_semantic_score"] = float(score)
        chunk["_semantic_rank"] = rank
        results.append(chunk)
    return results



def _tokenize(text: str) -> List[str]:
    # Unicode-aware tokenization for Czech, Slovak, Hungarian, German, etc.
    return [t for t in re.split(r"[^\w]+", text.lower(), flags=re.UNICODE) if len(t) >= 2]



def retrieve_chunks_keyword(query: str, manual_chunks: List[Dict[str, Any]], k: int = 8) -> List[Dict[str, Any]]:
    q = query.lower()
    q_words = _tokenize(q)
    scored = []

    for chunk in manual_chunks:
        score = 0.0

        # Strong match: curated query terms / aliases.
        for term in chunk.get("query_terms", []):
            term_l = term.lower().strip()
            if not term_l:
                continue
            if term_l in q:
                score += 3.0

        # Medium match: action/title terms.
        title = chunk.get("title", "").lower()
        action_id = get_action_id(chunk).lower().replace("_", " ")
        text = chunk.get("text", "").lower()
        expected = " ".join(chunk.get("expected_visible_objects", [])).lower().replace("_", " ")
        highlights = " ".join(chunk.get("highlight_objects", [])).lower().replace("_", " ")

        for word in q_words:
            if word in action_id:
                score += 1.25
            if word in title:
                score += 1.0
            if word in expected:
                score += 0.4
            if word in highlights:
                score += 0.4
            if word in text:
                score += 0.25

        chunk_copy = chunk.copy()
        chunk_copy["_keyword_score"] = float(score)
        scored.append(chunk_copy)

    scored.sort(key=lambda c: c["_keyword_score"], reverse=True)

    results = []
    for rank, chunk in enumerate(scored[:k], start=1):
        chunk["_keyword_rank"] = rank
        results.append(chunk)
    return results


# -----------------------------------------------------------------------------
# Fusion, boosting, reranking
# -----------------------------------------------------------------------------



def _normalize_scores(values: List[float]) -> List[float]:
    if not values:
        return []
    mn, mx = min(values), max(values)
    if mx == mn:
        return [1.0 if mx > 0 else 0.0 for _ in values]
    return [(v - mn) / (mx - mn) for v in values]



def _merge_candidates(*result_lists: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_id: Dict[str, Dict[str, Any]] = {}

    for results in result_lists:
        for chunk in results:
            action_id = get_action_id(chunk)
            if action_id not in by_id:
                by_id[action_id] = chunk.copy()
            else:
                by_id[action_id].update({k: v for k, v in chunk.items() if k.startswith("_")})

    return list(by_id.values())



def _intent_score(chunk: Dict[str, Any], query_understanding: Dict[str, Any]) -> float:
    action_id = get_action_id(chunk)
    question_type = query_understanding.get("question_type", "unknown")
    boost_ids = set(query_understanding.get("boost_action_ids", []))
    downrank_ids = set(query_understanding.get("downrank_action_ids", []))
    task_type = chunk_task_type(chunk)

    score = 0.0

    if action_id in boost_ids:
        score += 0.70
    if action_id in downrank_ids:
        score -= 0.45

    if question_type == "identify":
        score += 0.50 if task_type == "identify" else -0.18
    elif question_type == "troubleshooting":
        score += 0.35 if task_type == "troubleshooting" else -0.10
    elif question_type == "setup":
        score += 0.25 if task_type == "setup" else -0.08
    elif question_type == "safety":
        score += 0.35 if task_type == "safety" else -0.05
    elif question_type == "specifications":
        score += 0.55 if task_type == "specifications" else -0.18

    return score



def fuse_and_rank_candidates(
    candidates: List[Dict[str, Any]],
    query_understanding: Dict[str, Any],
    semantic_weight: float = 0.50,
    keyword_weight: float = 0.35,
    intent_weight: float = 0.15,
) -> List[Dict[str, Any]]:
    """Combine semantic, keyword, and intent scores into one ranking."""
    if not candidates:
        return []

    semantic_raw = [float(c.get("_semantic_score", 0.0)) for c in candidates]
    keyword_raw = [float(c.get("_keyword_score", 0.0)) for c in candidates]
    intent_raw = [_intent_score(c, query_understanding) for c in candidates]

    semantic_norm = _normalize_scores(semantic_raw)
    keyword_norm = _normalize_scores(keyword_raw)

    # Intent can be negative; map roughly into [0, 1].
    intent_norm = [(s + 0.5) / 1.2 for s in intent_raw]
    intent_norm = [max(0.0, min(1.0, s)) for s in intent_norm]

    ranked = []
    for c, s_sem, s_key, s_int, raw_int in zip(candidates, semantic_norm, keyword_norm, intent_norm, intent_raw):
        chunk = c.copy()
        hybrid = semantic_weight * s_sem + keyword_weight * s_key + intent_weight * s_int
        chunk["_intent_score"] = float(raw_int)
        chunk["score"] = float(hybrid)
        ranked.append(chunk)

    ranked.sort(key=lambda c: c.get("score", 0.0), reverse=True)
    return ranked



def rerank_candidates_with_gemini(
    question: str,
    scene: Dict[str, Any],
    candidates: List[Dict[str, Any]],
    query_understanding: Dict[str, Any],
    k: int = 5,
) -> List[Dict[str, Any]]:
    """Optional LLM reranking over already-retrieved candidates.

    Use this for demo robustness. If anything fails, return candidates unchanged.
    """
    try:
        from .gemini_client import generate_json_answer
    except Exception:
        return candidates[:k]

    compact_candidates = []
    for c in candidates[: min(len(candidates), 10)]:
        compact_candidates.append({
            "action_id": get_action_id(c),
            "title": c.get("title", ""),
            "task_type": chunk_task_type(c),
            "score_before_rerank": round(float(c.get("score", 0.0)), 4),
            "expected_visible_objects": c.get("expected_visible_objects", []),
            "query_terms": c.get("query_terms", []),
            "text": str(c.get("text", ""))[:600],
        })

    prompt = f"""
You are reranking retrieved CR-10 Smart 3D printer manual chunks.

User question:
{question}

Query understanding:
{json.dumps(query_understanding, indent=2)}

Detected visible objects:
{json.dumps(scene.get('visible_objects', []), indent=2)}

Candidate chunks:
{json.dumps(compact_candidates, indent=2)}

Return ONLY valid JSON:
{{
  "ranked_action_ids": ["best action_id first"],
  "confidence": 0.0,
  "reason": "one short reason"
}}

Rules:
- Prefer the chunk that directly answers the user's intent.
- Distinguish setup from troubleshooting.
- Do not choose troubleshooting just because the same object is mentioned.
- Only use action IDs from Candidate chunks.
""".strip()

    try:
        parsed = generate_json_answer(prompt)
        ranked_ids = parsed.get("ranked_action_ids", [])
        if not isinstance(ranked_ids, list):
            return candidates[:k]

        by_id = {get_action_id(c): c for c in candidates}
        reranked: List[Dict[str, Any]] = []
        seen = set()

        for i, action_id in enumerate(ranked_ids):
            if action_id in by_id and action_id not in seen:
                chunk = by_id[action_id].copy()
                chunk["_llm_rerank_rank"] = i + 1
                chunk["_llm_rerank_confidence"] = parsed.get("confidence")
                chunk["_llm_rerank_reason"] = parsed.get("reason", "")
                # Keep old score but add a tiny rank marker for display.
                chunk["score"] = float(chunk.get("score", 0.0)) + max(0.0, 0.01 * (10 - i))
                reranked.append(chunk)
                seen.add(action_id)

        for chunk in candidates:
            action_id = get_action_id(chunk)
            if action_id not in seen:
                reranked.append(chunk)

        return reranked[:k]
    except Exception:
        return candidates[:k]


# -----------------------------------------------------------------------------
# Public retrieval API
# -----------------------------------------------------------------------------



def retrieve_best_chunks(
    question: str,
    scene: Dict[str, Any],
    manual_chunks: List[Dict[str, Any]],
    rag_index: Optional[Dict[str, Any]] = None,
    k: int = 5,
    candidate_k: int = 10,
    use_hybrid: bool = True,
    use_llm_query_understanding: bool = False,
    use_llm_rerank: bool = False,
    include_visible_objects_in_query: bool = False,
) -> List[Dict[str, Any]]:
    """Retrieve the best manual chunks for a user question.

    Backward-compatible defaults:
    - If rag_index is None, this still works with keyword retrieval only.
    - If rag_index is provided, this uses hybrid retrieval by default.
    - LLM query understanding and reranking are opt-in.
    """
    action_ids = [get_action_id(c) for c in manual_chunks]

    if use_llm_query_understanding:
        query_understanding = understand_query_with_gemini(question, scene, action_ids)
    else:
        query_understanding = understand_query_heuristic(question, scene)

    query = build_rag_query(
        question=question,
        scene=scene,
        include_visible_objects=include_visible_objects_in_query,
        query_understanding=query_understanding,
    )

    keyword_results = retrieve_chunks_keyword(query, manual_chunks, k=candidate_k)

    if rag_index is not None and use_hybrid:
        semantic_results = retrieve_chunks_semantic(query, rag_index, k=candidate_k)
        candidates = _merge_candidates(keyword_results, semantic_results)
    elif rag_index is not None:
        candidates = retrieve_chunks_semantic(query, rag_index, k=candidate_k)
    else:
        candidates = keyword_results

    ranked = fuse_and_rank_candidates(candidates, query_understanding)

    # Prevent component overview from hijacking non-identification questions.
    if query_understanding.get("question_type") != "identify":
        filtered = [c for c in ranked if get_action_id(c) != "identify_part"]
        if filtered:
            ranked = filtered

    if use_llm_rerank:
        ranked = rerank_candidates_with_gemini(
            question=question,
            scene=scene,
            candidates=ranked,
            query_understanding=query_understanding,
            k=k,
        )
    else:
        ranked = ranked[:k]

    # Attach query understanding for debugging in notebooks.
    for chunk in ranked:
        chunk["_query_understanding"] = query_understanding

    return ranked
