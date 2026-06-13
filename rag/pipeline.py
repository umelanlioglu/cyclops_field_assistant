"""Main orchestration pipeline for the CR-10 Smart assistant."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from .expectations import verify_expectations
from .gemini_client import generate_json_answer, generate_text_answer
from .prompts import build_gemini_prompt, build_simple_answer
from .retrieval import retrieve_best_chunks
from .vision import build_scene_json
from .visual_guidance import build_visual_targets, draw_guided_annotations
from .references import build_references_from_chunks

from .turn_state import classify_user_turn
from .session_memory import (
    make_session_state,
    find_chunk_by_action_id,
    update_session_for_closing,
    update_session_state,
)
from .routing_fallback import (
    is_retrieval_uncertain,
    fallback_route_with_gemini,
    move_selected_chunk_to_front,
)
from .source_scope import (
    has_secondary_context,
    detect_official_manual_scope_gap,
    build_scope_gap_answer,
    is_vague_failure_question,
    build_vague_failure_answer,
)
from .request_guard import (
    classify_request_scope,
    build_general_answer_prompt,
    build_non_rag_fallback_answer,
)


def _action_id(chunk: Dict[str, Any]) -> str:
    return str(chunk.get("action_id", chunk.get("id", "")))


def _draw_empty_annotation(
    image_path: str | Path,
    detections: List[Dict[str, Any]],
    output_path: str | Path,
) -> str:
    annotated_path = draw_guided_annotations(
        image_path=image_path,
        detections=detections,
        visual_targets=[],
        output_path=output_path,
    )
    return str(annotated_path)


def _safe_visible_references(
    referenced_objects: List[str],
    scene: Dict[str, Any],
) -> List[str]:
    visible = set(scene.get("visible_objects", []))
    return [obj for obj in referenced_objects if obj in visible]


def _format_reference_for_ui(ref: Dict[str, Any]) -> str:
    """Build the source string the frontend can display.

    Example:
    Creality CR-10 Smart User Manual, pp. 5–6, Basic Parameters
    """
    pages = ref.get("pages")
    page = ref.get("page")

    if isinstance(pages, list) and pages:
        if len(pages) == 1:
            page_text = f"p. {pages[0]}"
        else:
            page_text = f"pp. {pages[0]}–{pages[-1]}"
    elif page is not None:
        page_text = f"p. {page}"
    else:
        page_text = "page unknown"

    source = ref.get("source_title") or "Unknown source"
    section = ref.get("section") or ref.get("chunk_title") or "Unknown section"

    return f"{source}, {page_text}, {section}"


def _build_used_reference_objects(
    references: List[Dict[str, Any]],
    used_reference_ids: List[str],
) -> List[Dict[str, Any]]:
    """Map Gemini's used_reference_ids to deterministic source objects.

    Gemini should only choose IDs. Page numbers, source titles, and display text
    are generated here from retrieved chunk metadata so the UI does not depend on
    the LLM inventing citations.
    """
    used_ids = {str(ref_id) for ref_id in (used_reference_ids or []) if ref_id}
    if not used_ids:
        return []

    used_refs: List[Dict[str, Any]] = []

    for ref in references or []:
        ref_id = ref.get("ref_id")
        if not ref_id or str(ref_id) not in used_ids:
            continue

        display_ref = dict(ref)
        display_ref["display_text"] = _format_reference_for_ui(display_ref)
        used_refs.append(display_ref)

    return used_refs


def _normalize_used_reference_ids(
    llm_json: Optional[Dict[str, Any]],
    references: List[Dict[str, Any]],
) -> List[str]:
    """Keep only valid reference IDs and fall back to the selected chunk ref."""
    valid_ids = [
        str(ref.get("ref_id"))
        for ref in (references or [])
        if ref.get("ref_id")
    ]
    valid_id_set = set(valid_ids)

    raw_used = []
    if isinstance(llm_json, dict):
        raw_used = llm_json.get("used_reference_ids") or []

    if isinstance(raw_used, str):
        raw_used = [raw_used]

    used_ids = [
        str(ref_id)
        for ref_id in raw_used
        if str(ref_id) in valid_id_set
    ]

    # Deduplicate while preserving order.
    used_ids = list(dict.fromkeys(used_ids))

    # If Gemini forgot references but this is a RAG answer, use the selected/top ref.
    if not used_ids and valid_ids:
        used_ids = [valid_ids[0]]

    return used_ids


def _select_top_chunk(
    question: str,
    turn_type: str,
    session_state: Dict[str, Any],
    manual_chunks: List[Dict[str, Any]],
    retrieved: List[Dict[str, Any]],
) -> Dict[str, Any]:
    should_continue_active_action = (
        turn_type in {"continuation", "progress_update", "completion_confirmation"}
        and session_state.get("active_action_id")
    )

    if not should_continue_active_action:
        return retrieved[0]

    previous_chunk = find_chunk_by_action_id(
        manual_chunks,
        session_state.get("active_action_id"),
    )

    if previous_chunk:
        return previous_chunk.copy()

    return retrieved[0]


def _apply_active_action_to_retrieved(
    selected_chunk: Dict[str, Any],
    retrieved: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    selected_action = _action_id(selected_chunk)

    return [selected_chunk] + [
        chunk for chunk in retrieved
        if _action_id(chunk) != selected_action
    ]


def _summarize_text(text: str, max_len: int = 220) -> str:
    text = (text or "").replace("\n", " ").strip()
    if len(text) <= max_len:
        return text
    return text[:max_len].rstrip() + "..."


def _update_session_for_non_rag(
    session_state: Dict[str, Any],
    question: str,
    answer_text: str,
    guard_type: str,
) -> Dict[str, Any]:
    updated = dict(session_state)
    updated["turn_count"] = int(updated.get("turn_count", 0)) + 1
    updated["last_question"] = question
    updated["last_answer_summary"] = _summarize_text(answer_text)
    updated["last_user_turn_type"] = guard_type
    updated["conversation_summary"] = (
        updated.get("conversation_summary", "")
        + f" User message was handled outside printer RAG by {guard_type}; active printer task was not advanced."
    ).strip()
    return updated


def _build_guarded_result(
    question: str,
    image_path: str | Path,
    detections: List[Dict[str, Any]],
    output_path: str | Path,
    scene: Dict[str, Any],
    selected_chunk: Dict[str, Any],
    retrieved: List[Dict[str, Any]],
    verification: Dict[str, Any],
    session_state: Dict[str, Any],
    turn_type: str,
    fallback_info: Optional[Dict[str, Any]],
    request_scope: Optional[Dict[str, Any]],
    answer_text: str,
    memory_note: str,
    guard_name: str,
    guard_payload: Dict[str, Any],
    references=None,
) -> Dict[str, Any]:
    
    referenced_objects: List[str] = []
    visual_targets: List[Dict[str, str]] = []

    answer_references = references or build_references_from_chunks(
        retrieved_chunks=retrieved,
        selected_action_id=_action_id(selected_chunk),
        max_refs=5,
    )

    llm_json = {
        "answer": answer_text,
        "needs_visual_annotation": False,
        "referenced_objects": [],
        "missing_or_uncertain_objects": [],
        "safety_warning": "",
        "memory_note": memory_note,
        "used_reference_ids": [ref["ref_id"] for ref in answer_references if ref.get("ref_id")],
    }

    used_references = _build_used_reference_objects(
        references=answer_references,
        used_reference_ids=llm_json.get("used_reference_ids", []),
    )

    annotated_path = draw_guided_annotations(
        image_path=image_path,
        detections=detections,
        visual_targets=visual_targets,
        output_path=output_path,
    )


    updated_session_state = update_session_state(
        session_state=session_state,
        question=question,
        selected_chunk=selected_chunk,
        scene=scene,
        verification=verification,
        answer_text=answer_text,
        referenced_objects=referenced_objects,
        turn_type=turn_type,
    )

    result = {
        "question": question,
        "scene": scene,
        "selected_action": _action_id(selected_chunk),
        "selected_chunk": selected_chunk,
        "retrieved_chunks": retrieved,
        "references": answer_references,
        "used_references": used_references,
        "verification": verification,
        "llm_json": llm_json,
        "prompt": None,
        "answer": answer_text,
        "referenced_objects": referenced_objects,
        "visual_targets": visual_targets,
        "annotated_image_path": str(annotated_path),
        "session_state": updated_session_state,
        "updated_session_state": updated_session_state,
        "turn_type": turn_type,
        "fallback_info": fallback_info,
        "request_scope_guard": request_scope,
    }

    result[guard_name] = guard_payload
    return result


def run_pipeline(
    question: str,
    image_path: str | Path,
    detections: List[Dict[str, Any]],
    manual_chunks: List[Dict[str, Any]],
    output_path: str | Path = "outputs/result.jpg",
    rag_index: Optional[Dict[str, Any]] = None,
    use_gemini: bool = False,
    use_gemini_request_guard: Optional[bool] = None,
    use_gemini_retrieval: bool = False,
    use_gemini_rerank: bool = False,
    use_gemini_answer: Optional[bool] = None,
    use_gemini_fallback_router: Optional[bool] = None,
    top_k: int = 5,
    session_state: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run the full CR-10 Smart assistant pipeline.

    Gemini is split into separate switches so evaluation can isolate routing,
    retrieval, reranking, and final answer generation.
    """
    use_gemini_request_guard = (
        use_gemini if use_gemini_request_guard is None else use_gemini_request_guard
    )
    use_gemini_answer = (
        use_gemini if use_gemini_answer is None else use_gemini_answer
    )
    use_gemini_fallback_router = (
        use_gemini if use_gemini_fallback_router is None else use_gemini_fallback_router
    )

    session_state = make_session_state(session_state)
    scene = build_scene_json(detections)
    turn_type = classify_user_turn(question, session_state)

    # 1. Closing / thanks turn
    if turn_type == "thanks_or_closing":
        answer_text = (
            "You're welcome! I’ll keep the current task in memory. "
            "When you want to continue, send another image or ask “what next?”"
        )

        annotated_path = _draw_empty_annotation(
            image_path=image_path,
            detections=detections,
            output_path=output_path,
        )

        updated_session_state = update_session_for_closing(
            session_state=session_state,
            question=question,
            answer_text=answer_text,
        )

        return {
            "question": question,
            "scene": scene,
            "selected_action": session_state.get("active_action_id"),
            "selected_chunk": None,
            "retrieved_chunks": [],
            "verification": {
                "action_id": session_state.get("active_action_id"),
                "found_required": [],
                "missing_required": session_state.get("last_missing_objects", []),
                "referenced_objects": [],
                "is_visually_ready": None,
                "cannot_verify_from_image": [],
                "warnings": [],
            },
            "llm_json": {
                "answer": answer_text,
                "turn_type": turn_type,
                "memory_note": "User thanked/closed; task was not advanced.",
                "used_reference_ids": [],
            },
            "prompt": None,
            "answer": answer_text,
            "referenced_objects": [],
            "references": [],
            "used_references": [],
            "visual_targets": [],
            "annotated_image_path": annotated_path,
            "session_state": updated_session_state,
            "updated_session_state": updated_session_state,
            "turn_type": turn_type,
            "fallback_info": None,
            "request_scope_guard": None,
        }

    # 2. Pre-RAG request/domain guard
    request_scope = classify_request_scope(
        question=question,
        scene=scene,
        session_state=session_state,
        turn_type=turn_type,
        use_gemini=use_gemini_request_guard,
    )

    if not request_scope.get("should_use_rag", True):
        if use_gemini_answer and request_scope.get("should_answer_general", False):
            answer_text = generate_text_answer(
                build_general_answer_prompt(question, request_scope)
            )
        else:
            answer_text = build_non_rag_fallback_answer(question, request_scope)

        annotated_path = _draw_empty_annotation(
            image_path=image_path,
            detections=detections,
            output_path=output_path,
        )

        updated_session_state = _update_session_for_non_rag(
            session_state=session_state,
            question=question,
            answer_text=answer_text,
            guard_type=f"request_scope_{request_scope.get('scope', 'unknown')}",
        )

        return {
            "question": question,
            "scene": scene,
            "selected_action": session_state.get("active_action_id"),
            "selected_chunk": None,
            "retrieved_chunks": [],
            "verification": {
                "action_id": session_state.get("active_action_id"),
                "found_required": [],
                "missing_required": [],
                "referenced_objects": [],
                "is_visually_ready": None,
                "cannot_verify_from_image": [],
                "warnings": [],
            },
            "llm_json": {
                "answer": answer_text,
                "turn_type": turn_type,
                "memory_note": "Message handled outside printer RAG; active task was not advanced.",
                "used_reference_ids": [],
            },
            "prompt": None,
            "answer": answer_text,
            "referenced_objects": [],
            "references": [],
            "used_references": [],
            "visual_targets": [],
            "annotated_image_path": annotated_path,
            "session_state": updated_session_state,
            "updated_session_state": updated_session_state,
            "turn_type": turn_type,
            "fallback_info": None,
            "request_scope_guard": request_scope,
        }

    # 3. Retrieval
    retrieved = retrieve_best_chunks(
        question=question,
        scene=scene,
        manual_chunks=manual_chunks,
        rag_index=rag_index,
        k=max(top_k, 5),
        use_llm_query_understanding=use_gemini_retrieval,
        use_llm_rerank=use_gemini_rerank,
    )

    if not retrieved:
        answer_text = (
            "I could not find a relevant CR-10 Smart manual chunk for this question. "
            "Please rephrase the question or add a relevant manual chunk."
        )

        annotated_path = _draw_empty_annotation(
            image_path=image_path,
            detections=detections,
            output_path=output_path,
        )

        updated_session_state = _update_session_for_non_rag(
            session_state=session_state,
            question=question,
            answer_text=answer_text,
            guard_type="retrieval_empty",
        )

        return {
            "question": question,
            "scene": scene,
            "selected_action": None,
            "selected_chunk": None,
            "retrieved_chunks": [],
            "verification": {
                "action_id": None,
                "found_required": [],
                "missing_required": [],
                "referenced_objects": [],
                "is_visually_ready": None,
                "cannot_verify_from_image": [],
                "warnings": [],
            },
            "llm_json": {
                "answer": answer_text,
                "turn_type": turn_type,
                "memory_note": "Retrieval returned no chunks.",
                "used_reference_ids": [],
            },
            "prompt": None,
            "answer": answer_text,
            "referenced_objects": [],
            "references": [],
            "used_references": [],
            "visual_targets": [],
            "annotated_image_path": annotated_path,
            "session_state": updated_session_state,
            "updated_session_state": updated_session_state,
            "turn_type": turn_type,
            "fallback_info": {"used": False, "reason": "retrieval_empty"},
            "request_scope_guard": request_scope,
        }

    fallback_info = None
    should_continue_active_action = (
        turn_type in {"continuation", "progress_update", "completion_confirmation"}
        and session_state.get("active_action_id")
    )

    # 4. Optional Gemini fallback router
    if (
        use_gemini_fallback_router
        and not should_continue_active_action
        and is_retrieval_uncertain(retrieved)
    ):
        fallback_info = fallback_route_with_gemini(
            question=question,
            scene=scene,
            retrieved=retrieved,
        )

        selected_action_id = fallback_info.get("selected_action_id")

        if selected_action_id and selected_action_id != "clarify":
            retrieved = move_selected_chunk_to_front(retrieved, selected_action_id)

    # 5. Select active/current chunk
    top_chunk = _select_top_chunk(
        question=question,
        turn_type=turn_type,
        session_state=session_state,
        manual_chunks=manual_chunks,
        retrieved=retrieved,
    )

    if should_continue_active_action:
        retrieved = _apply_active_action_to_retrieved(top_chunk, retrieved)

    retrieved = retrieved[:top_k]

    answer_references = build_references_from_chunks(
        retrieved_chunks=retrieved,
        selected_action_id=_action_id(top_chunk),
        max_refs=5,
    )

    # 6. Visual expectation checking
    verification = verify_expectations(top_chunk, scene)

    # 7. Manual-scope guard
    scope_gap_topic = detect_official_manual_scope_gap(question)

    if use_gemini_answer and scope_gap_topic and not has_secondary_context(retrieved):
        answer_text = build_scope_gap_answer(scope_gap_topic, retrieved)

        return _build_guarded_result(
            question=question,
            image_path=image_path,
            detections=detections,
            output_path=output_path,
            scene=scene,
            selected_chunk=top_chunk,
            retrieved=retrieved,
            references=answer_references,
            verification=verification,
            session_state=session_state,
            turn_type=turn_type,
            fallback_info=fallback_info,
            request_scope=request_scope,
            answer_text=answer_text,
            memory_note=f"Official manual scope gap: {scope_gap_topic}",
            guard_name="manual_scope_guard",
            guard_payload={
                "triggered": True,
                "topic": scope_gap_topic,
            },
        )

    # 8. Vague-failure guard
    if use_gemini_answer and is_vague_failure_question(question):
        answer_text = build_vague_failure_answer()

        return _build_guarded_result(
            question=question,
            image_path=image_path,
            detections=detections,
            output_path=output_path,
            scene=scene,
            selected_chunk=top_chunk,
            retrieved=retrieved,
            references=answer_references,
            verification=verification,
            session_state=session_state,
            turn_type=turn_type,
            fallback_info=fallback_info,
            request_scope=request_scope,
            answer_text=answer_text,
            memory_note="Asked clarification for vague failure report.",
            guard_name="vague_failure_guard",
            guard_payload={"triggered": True},
        )

    # 9. Final answer generation
    if use_gemini_answer:
        prompt = build_gemini_prompt(
            question=question,
            scene=scene,
            retrieved_chunks=retrieved,
            verification=verification,
            session_state=session_state,
            turn_type=turn_type,
        )
        llm_json = generate_json_answer(prompt)
        if not isinstance(llm_json.get("visual_captions"), dict):
            llm_json["visual_captions"] = {}
        answer_text = llm_json.get("answer", "")
        referenced_objects = llm_json.get("referenced_objects") or verification["referenced_objects"]

        used_reference_ids = _normalize_used_reference_ids(llm_json, answer_references)
        llm_json["used_reference_ids"] = used_reference_ids
    else:
        prompt = None
        answer_text = build_simple_answer(question, retrieved, scene, verification)
        referenced_objects = verification["referenced_objects"]

        used_reference_ids = _normalize_used_reference_ids(None, answer_references)
        llm_json = {
            "answer": answer_text,
            "needs_visual_annotation": bool(referenced_objects),
            "referenced_objects": referenced_objects,
            "visual_captions": {},
            "used_reference_ids": used_reference_ids,
            "missing_or_uncertain_objects": verification.get("missing_required", []),
            "safety_warning": "",
            "memory_note": "Deterministic simple answer.",
        }

    used_references = _build_used_reference_objects(
        references=answer_references,
        used_reference_ids=llm_json.get("used_reference_ids", []),
    )

    # 10. Safety: only reference actually visible objects
    referenced_objects = _safe_visible_references(referenced_objects, scene)

    # 11. Session update
    updated_session_state = update_session_state(
        session_state=session_state,
        question=question,
        selected_chunk=top_chunk,
        scene=scene,
        verification=verification,
        answer_text=answer_text,
        referenced_objects=referenced_objects,
        turn_type=turn_type,
    )

    # 12. Visual annotation
    visual_targets = build_visual_targets(
        llm_json=llm_json or {},
        verification=verification,
        selected_chunk=top_chunk,
        detections=detections,
        max_targets=4,
    )

    referenced_objects = [target["label"] for target in visual_targets]

    annotated_path = draw_guided_annotations(
        image_path=image_path,
        detections=detections,
        visual_targets=visual_targets,
        output_path=output_path,
    )

    # 13. Final result
    return {
        "question": question,
        "scene": scene,
        "selected_action": _action_id(top_chunk),
        "selected_chunk": top_chunk,
        "retrieved_chunks": retrieved,
        "references": answer_references,
        "used_references": used_references,
        "verification": verification,
        "llm_json": llm_json,
        "prompt": prompt,
        "answer": answer_text,
        "referenced_objects": referenced_objects,
        "visual_targets": visual_targets,
        "annotated_image_path": str(annotated_path),
        "session_state": updated_session_state,
        "updated_session_state": updated_session_state,
        "turn_type": turn_type,
        "fallback_info": fallback_info,
        "request_scope_guard": request_scope,
        "gemini_config": {
            "use_gemini": use_gemini,
            "use_gemini_request_guard": use_gemini_request_guard,
            "use_gemini_retrieval": use_gemini_retrieval,
            "use_gemini_rerank": use_gemini_rerank,
            "use_gemini_answer": use_gemini_answer,
            "use_gemini_fallback_router": use_gemini_fallback_router,
        },
    }
