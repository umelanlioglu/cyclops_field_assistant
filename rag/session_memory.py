"""Session state helpers for multi-turn printer guidance."""

from __future__ import annotations

from typing import Any, Dict, List, Optional


def summarize_text(text: str, max_len: int = 220) -> str:
    text = (text or "").replace("\n", " ").strip()
    if len(text) <= max_len:
        return text
    return text[:max_len].rstrip() + "..."


def make_session_state(session_state: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    base = {
        "active_action_id": None,
        "active_title": None,
        "turn_count": 0,
        "last_question": None,
        "last_answer_summary": None,
        "last_detected_objects": [],
        "last_missing_objects": [],
        "last_referenced_objects": [],
        "conversation_summary": "",
    }

    if session_state:
        base.update(session_state)

    return base


def find_chunk_by_action_id(
    chunks: List[Dict[str, Any]],
    action_id: Optional[str],
) -> Optional[Dict[str, Any]]:
    if not action_id:
        return None

    for chunk in chunks:
        if chunk.get("action_id", chunk.get("id")) == action_id:
            return chunk

    return None


def update_session_for_closing(
    session_state: Dict[str, Any],
    question: str,
    answer_text: str,
) -> Dict[str, Any]:
    updated = dict(session_state)
    updated["turn_count"] = int(updated.get("turn_count", 0)) + 1
    updated["last_question"] = question
    updated["last_user_turn_type"] = "thanks_or_closing"
    updated["last_answer_summary"] = summarize_text(answer_text)
    updated["conversation_summary"] = (
        updated.get("conversation_summary", "")
        + " User thanked or closed the current exchange; do not advance the guide automatically."
    ).strip()
    return updated


def update_session_state(
    session_state: Dict[str, Any],
    question: str,
    selected_chunk: Dict[str, Any],
    scene: Dict[str, Any],
    verification: Dict[str, Any],
    answer_text: str,
    referenced_objects: List[str],
    turn_type: str,
) -> Dict[str, Any]:
    action_id = selected_chunk.get("action_id", selected_chunk.get("id"))
    title = selected_chunk.get("title", action_id)

    updated = dict(session_state)
    updated["active_action_id"] = action_id
    updated["active_title"] = title
    updated["turn_count"] = int(updated.get("turn_count", 0)) + 1
    updated["last_question"] = question
    updated["last_answer_summary"] = summarize_text(answer_text)
    updated["last_detected_objects"] = scene.get("visible_objects", [])
    updated["last_missing_objects"] = verification.get("missing_required", [])
    updated["last_referenced_objects"] = referenced_objects
    updated["last_user_turn_type"] = turn_type

    if turn_type in {"progress_update", "completion_confirmation"}:
        updated["last_progress_update"] = question

    updated["conversation_summary"] = (
        f"The user is working on '{title}' ({action_id}). "
        f"Last question: {question}. "
        f"Missing/not visible parts: {', '.join(verification.get('missing_required', [])) or 'none'}."
    )

    return updated