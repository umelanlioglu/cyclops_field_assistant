"""Retrieval uncertainty and fallback routing helpers."""

from __future__ import annotations

import json
from typing import Any, Dict, List

from .gemini_client import generate_json_answer


BROAD_OR_GENERIC_ACTIONS = {
    "tools_and_parts",
    "specifications",
    "assembly_setup",
    "start_printing_software",
    "circuit_wiring",
}


def action_id_of(chunk: Dict[str, Any]) -> str:
    return str(chunk.get("action_id", chunk.get("id", "")))


def retrieval_gap(retrieved: List[Dict[str, Any]]) -> float:
    if len(retrieved) < 2:
        return 1.0

    top_score = float(retrieved[0].get("score", 0.0))
    second_score = float(retrieved[1].get("score", 0.0))
    return top_score - second_score


def is_retrieval_uncertain(retrieved: List[Dict[str, Any]]) -> bool:
    if len(retrieved) < 2:
        return False

    top = retrieved[0]
    top_action = action_id_of(top)
    gap = retrieval_gap(retrieved)

    query_understanding = top.get("_query_understanding", {})
    question_type = query_understanding.get("question_type", "unknown")
    boosts = query_understanding.get("boost_action_ids", [])

    if question_type == "identify":
        return False

    if question_type == "unknown" and gap < 0.20:
        return True

    if not boosts and gap < 0.10:
        return True

    if top_action in BROAD_OR_GENERIC_ACTIONS and gap < 0.15:
        return True

    return False


def fallback_route_with_gemini(
    question: str,
    scene: Dict[str, Any],
    retrieved: List[Dict[str, Any]],
) -> Dict[str, Any]:
    candidates = []

    for chunk in retrieved[:5]:
        candidates.append({
            "action_id": action_id_of(chunk),
            "title": chunk.get("title"),
            "score": round(float(chunk.get("score", 0.0)), 4),
            "source_type": chunk.get("source_type", ""),
            "text_preview": str(chunk.get("text", ""))[:500],
        })

    prompt = f"""
You are a fallback router for the Creality CR-10 Smart 3D printer manual assistant.

The normal retrieval system was uncertain. Choose the best action_id from the candidate chunks.

User question:
{question}

Detected visible objects:
{json.dumps(scene.get("visible_objects", []), indent=2)}

Candidate chunks:
{json.dumps(candidates, indent=2)}

Return ONLY valid JSON:
{{
  "selected_action_id": "one action_id from candidates, or clarify",
  "confidence": 0.0,
  "reason": "one short reason"
}}

Rules:
- Only choose an action_id that appears in Candidate chunks.
- If the user describes a symptom/failure, prefer a troubleshooting chunk.
- If the user asks how to perform an operation, prefer a setup/operation chunk.
- Prefer specific chunks over broad/generic chunks like tools, specifications, or assembly.
- If none clearly fits, return "clarify".
""".strip()

    try:
        parsed = generate_json_answer(prompt)
    except Exception as exc:
        return {
            "used": False,
            "selected_action_id": None,
            "confidence": 0.0,
            "reason": f"Fallback router failed: {exc}",
        }

    allowed = {c["action_id"] for c in candidates}
    selected = parsed.get("selected_action_id")

    if selected not in allowed and selected != "clarify":
        selected = None

    return {
        "used": True,
        "selected_action_id": selected,
        "confidence": parsed.get("confidence", 0.0),
        "reason": parsed.get("reason", ""),
    }


def move_selected_chunk_to_front(
    retrieved: List[Dict[str, Any]],
    selected_action_id: str,
) -> List[Dict[str, Any]]:
    selected = None
    rest = []

    for chunk in retrieved:
        if action_id_of(chunk) == selected_action_id and selected is None:
            selected = chunk
        else:
            rest.append(chunk)

    if selected is None:
        return retrieved

    return [selected] + rest