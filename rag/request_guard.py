"""Pre-RAG request guard.

Decides whether a user message should go through the printer RAG pipeline
or be handled as general chat / unrelated input.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional

from .gemini_client import generate_json_answer


GREETING_ONLY_PHRASES = {
    "hi",
    "hello",
    "hey",
    "yo",
    "sup",
    "good morning",
    "good afternoon",
    "good evening",
    "how are you",
    "how are you doing",
}

PRINTER_TERMS = {
    "printer",
    "3d printer",
    "cr-10",
    "cr10",
    "creality",
    "filament",
    "spool",
    "pla",
    "abs",
    "petg",
    "tpu",
    "nozzle",
    "hotend",
    "hot end",
    "extruder",
    "bed",
    "print bed",
    "build plate",
    "platform",
    "sd card",
    "tf card",
    "usb",
    "wifi",
    "wi-fi",
    "qr code",
    "gcode",
    "g-code",
    "slicer",
    "cura",
    "leveling",
    "bed leveling",
    "preheat",
    "preheating",
    "temperature",
    "axis",
    "z axis",
    "x axis",
    "y axis",
    "belt",
    "motor",
    "gantry",
    "pull rod",
    "support rod",
    "limit switch",
    "filament sensor",
    "material breakage",
    "print file",
    "printing",
    "print quality",
    "clog",
    "jam",
    "stringing",
    "retraction",
    "support",
    "overhang",
    "tool box", 
    "toolbox",
    "network interface",
    "wlan", 
    "lan port", 
    "ethernet",
    "material rack", 
    "rack", 
    "spool holder",
    "glass print surface", 
    "print surface",
    "guide rails", 
    "wheels", 
    "dust",
    "cotton gloves", 
    "gloves",
    "remove print", 
    "remove the print", 
    "print finishes",
    "spotlight",
    "x-axis limit switch",
    "photoelectric switch",
    "heated bed thermistor",
    "thermistor",
}

CARD_TERMS = {
    "storage card",
    "memory card",
    "sd card",
    "tf card",
    "card slot",
    "card port",
    "sd/tf",
    "insert card",
    "remove card",
    "gcode",
    "g-code",
    "print file",
}

MULTILINGUAL_PRINTER_TERMS = [
    "tiskárna", "tlačiareň", "nyomtató", "drucker", "filament", "vlákno", "szál",
    "tryska", "dýza", "fúvóka", "düse", "sd-kártya", "sd-karte", "karta sd",
    "vyrovnávání", "vyrovnávanie", "szintezés", "nivellierung", "předehřev", "predohrev", "előmelegítés", "vorheizen"
]


def _has_card_signal(q: str) -> bool:
    return _contains_any(q, CARD_TERMS)

OFF_TOPIC_TERMS = {
    "weather",
    "rain",
    "temperature outside",
    "forecast",
    "football",
    "basketball",
    "stock price",
    "bitcoin",
    "btc",
    "election",
    "president",
    "recipe",
    "movie",
    "music",
    "joke",
    "capital of",
    "translate",
    "history of",
}

DEICTIC_VISUAL_QUESTIONS = {
    "what is this",
    "what's this",
    "what is that",
    "what's that",
    "what part is this",
    "which part is this",
    "identify this",
    "is this correct",
    "is it correct",
    "is this right",
    "is it right",
}


def _normalize(text: str) -> str:
    text = (text or "").lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text.strip(" .?!")


def _contains_phrase(text: str, phrase: str) -> bool:
    phrase = phrase.lower().strip()
    if not phrase:
        return False

    if " " in phrase:
        return phrase in text

    return re.search(rf"(?<![a-zA-Z0-9]){re.escape(phrase)}(?![a-zA-Z0-9])", text) is not None


def _contains_any(text: str, phrases: set[str]) -> bool:
    return any(_contains_phrase(text, phrase) for phrase in phrases)


def _looks_like_greeting_only(q: str) -> bool:
    if q in GREETING_ONLY_PHRASES:
        return True

    words = q.split()
    if len(words) <= 4 and any(word in {"hi", "hello", "hey", "yo"} for word in words):
        return True

    return False


def _has_visual_context(scene: Optional[Dict[str, Any]]) -> bool:
    if not scene:
        return False
    return bool(scene.get("visible_objects") or scene.get("detections"))


def _is_deictic_visual_question(q: str) -> bool:
    return _contains_any(q, DEICTIC_VISUAL_QUESTIONS)


def _has_printer_text_signal(q: str) -> bool:
    return _contains_any(q, PRINTER_TERMS)


def classify_request_scope_heuristic(
    question: str,
    scene: Optional[Dict[str, Any]] = None,
    session_state: Optional[Dict[str, Any]] = None,
    turn_type: str = "task_question",
) -> Dict[str, Any]:
    q = _normalize(question)
    session_state = session_state or {}

    has_active_action = bool(session_state.get("active_action_id"))
    has_visual_context = _has_visual_context(scene)
    has_printer_text = _has_printer_text_signal(q)
    has_off_topic_text = _contains_any(q, OFF_TOPIC_TERMS)

    if not q:
        return {
            "scope": "ambiguous",
            "confidence": 0.95,
            "should_use_rag": False,
            "should_answer_general": False,
            "should_use_yolo_annotations": False,
            "reason": "Empty user message.",
        }

    if _looks_like_greeting_only(q):
        return {
            "scope": "general_chat",
            "confidence": 0.95,
            "should_use_rag": False,
            "should_answer_general": True,
            "should_use_yolo_annotations": False,
            "reason": "Greeting/small-talk message.",
        }

    if has_off_topic_text and not has_printer_text:
        return {
            "scope": "unrelated",
            "confidence": 0.90,
            "should_use_rag": False,
            "should_answer_general": True,
            "should_use_yolo_annotations": False,
            "reason": "Clearly unrelated to the printer/manual.",
        }

    if turn_type in {"continuation", "progress_update", "completion_confirmation"}:
        if has_active_action:
            return {
                "scope": "printer_rag",
                "confidence": 0.90,
                "should_use_rag": True,
                "should_answer_general": False,
                "should_use_yolo_annotations": True,
                "reason": "Continuation/progress message for an active printer task.",
            }

        return {
            "scope": "ambiguous",
            "confidence": 0.85,
            "should_use_rag": False,
            "should_answer_general": False,
            "should_use_yolo_annotations": False,
            "reason": "Continuation-like message but there is no active printer task.",
        }

    if has_printer_text:
        return {
            "scope": "printer_rag",
            "confidence": 0.88,
            "should_use_rag": True,
            "should_answer_general": False,
            "should_use_yolo_annotations": True,
            "reason": "User message contains printer/manual terms.",
        }

    if has_visual_context and _is_deictic_visual_question(q):
        return {
            "scope": "printer_rag",
            "confidence": 0.86,
            "should_use_rag": True,
            "should_answer_general": False,
            "should_use_yolo_annotations": True,
            "reason": "User asks about visible object in the provided printer image.",
        }

    if _has_card_signal(q):
        return {
            "scope": "printer_rag",
            "confidence": 0.92,
            "should_use_rag": True,
            "should_answer_general": False,
            "should_use_yolo_annotations": True,
            "reason": "User asks about printer storage/SD/TF card handling.",
        }

    if "?" in question or q.startswith(("what", "why", "how", "when", "where", "who", "can you", "tell me")):
        return {
            "scope": "general_chat",
            "confidence": 0.55,
            "should_use_rag": False,
            "should_answer_general": True,
            "should_use_yolo_annotations": False,
            "reason": "Question has no strong printer signal from heuristic.",
        }

    return {
        "scope": "ambiguous",
        "confidence": 0.65,
        "should_use_rag": False,
        "should_answer_general": False,
        "should_use_yolo_annotations": False,
        "reason": "No strong printer/manual signal found.",
    }


def classify_request_scope_with_gemini(
    question: str,
    scene: Optional[Dict[str, Any]] = None,
    session_state: Optional[Dict[str, Any]] = None,
    turn_type: str = "task_question",
) -> Dict[str, Any]:
    scene = scene or {}
    session_state = session_state or {}

    prompt = f"""
You are a request router for a CR-10 Smart 3D printer assistant.

Decide whether the user message should use the printer RAG/manual pipeline.

User message:
{question}

Detected visible objects:
{json.dumps(scene.get("visible_objects", []), indent=2)}

Turn type:
{turn_type}

Active action id:
{session_state.get("active_action_id")}

Return ONLY valid JSON:
{{
  "scope": "printer_rag | general_chat | unrelated | ambiguous",
  "confidence": 0.0,
  "should_use_rag": true,
  "should_answer_general": false,
  "should_use_yolo_annotations": true,
  "reason": "short reason"
}}

Definitions:
- printer_rag: questions about operating, identifying, setting up, troubleshooting, maintaining, or safely using the CR-10 Smart / 3D printer.
- printer_rag: also includes continuation/progress messages if there is an active printer task.
- general_chat: greetings, small talk, or general questions not about the printer.
- unrelated: clearly unrelated factual requests such as weather, sports, recipes, history, politics, etc.
- ambiguous: too vague to safely choose a manual task, especially if there is no active task.

Rules:
- Do not send greetings like "hello" to RAG.
- Do not send weather/general knowledge questions to RAG.
- Do not choose printer_rag only because an image exists.
- If the text asks "what is this?" and printer parts are visible, choose printer_rag.
- If there is no active task and the user only says "next", "done", or "ok now", choose ambiguous.
""".strip()

    try:
        parsed = generate_json_answer(prompt)
    except Exception as exc:
        fallback = classify_request_scope_heuristic(question, scene, session_state, turn_type)
        fallback["router_error"] = str(exc)
        return fallback

    scope = parsed.get("scope", "ambiguous")
    if scope not in {"printer_rag", "general_chat", "unrelated", "ambiguous"}:
        scope = "ambiguous"

    return {
        "scope": scope,
        "confidence": float(parsed.get("confidence", 0.0)),
        "should_use_rag": bool(parsed.get("should_use_rag", scope == "printer_rag")),
        "should_answer_general": bool(parsed.get("should_answer_general", scope in {"general_chat", "unrelated"})),
        "should_use_yolo_annotations": bool(parsed.get("should_use_yolo_annotations", scope == "printer_rag")),
        "reason": parsed.get("reason", ""),
        "source": "gemini",
    }


def classify_request_scope(
    question: str,
    scene: Optional[Dict[str, Any]] = None,
    session_state: Optional[Dict[str, Any]] = None,
    turn_type: str = "task_question",
    use_gemini: bool = False,
    min_confidence: float = 0.80,
) -> Dict[str, Any]:
    decision = classify_request_scope_heuristic(
        question=question,
        scene=scene,
        session_state=session_state,
        turn_type=turn_type,
    )
    decision["source"] = "heuristic"

    if use_gemini and decision.get("confidence", 0.0) < min_confidence:
        return classify_request_scope_with_gemini(
            question=question,
            scene=scene,
            session_state=session_state,
            turn_type=turn_type,
        )

    return decision


def build_general_answer_prompt(question: str, request_scope: Dict[str, Any]) -> str:
    return f"""
You are answering as the general Gemini fallback for a headset assistant.

The user message was NOT routed to the 3D printer RAG/manual pipeline.

User message:
{question}

Router decision:
{json.dumps(request_scope, indent=2)}

Answer naturally and briefly.

Rules:
- Do not pretend this came from the printer manual.
- Do not mention retrieved chunks or RAG.
- If the user asks for live/current information such as weather, prices, or news, say that live data is needed unless a live tool is connected.
- If it is just a greeting, greet them and ask how you can help.
""".strip()


def build_non_rag_fallback_answer(question: str, request_scope: Dict[str, Any]) -> str:
    scope = request_scope.get("scope")

    if scope == "ambiguous":
        return (
            "I’m not sure which printer task you want to continue. "
            "Please ask a specific printer question, or say what you are trying to do."
        )

    if scope in {"general_chat", "unrelated"}:
        return (
            "This does not look related to the printer manual, so I did not route it to RAG. "
            "Enable Gemini general fallback to answer this normally."
        )

    return "I could not safely route this message."