"""Reference helpers for retrieved chunks and final answer JSON."""

from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Optional


def _stable_ref_id(chunk: Dict[str, Any], index: int = 0) -> str:
    action_id = str(chunk.get("action_id", chunk.get("id", "chunk")))
    title = str(chunk.get("title", ""))
    raw = f"{action_id}:{title}:{index}"
    digest = hashlib.md5(raw.encode("utf-8")).hexdigest()[:8]
    return f"{action_id}_{digest}"


def normalize_chunk_references(chunk: Dict[str, Any]) -> List[Dict[str, Any]]:
    refs = chunk.get("references") or []

    if isinstance(refs, dict):
        refs = [refs]

    normalized = []

    for i, ref in enumerate(refs):
        if not isinstance(ref, dict):
            continue

        source_type = ref.get("source_type") or chunk.get("source_type", "unknown")
        source_trust = ref.get("source_trust") or chunk.get("source_trust", "unknown")

        normalized.append({
            "ref_id": ref.get("ref_id") or _stable_ref_id(chunk, i),
            "source_id": ref.get("source_id", "unknown"),
            "source_title": ref.get("source_title", "Unknown source"),
            "source_type": source_type,
            "source_trust": source_trust,
            "section": ref.get("section") or chunk.get("title"),
            "page": ref.get("page"),
            "pages": ref.get("pages", ref.get("source_pages", chunk.get("source_pages"))),
            "figure": ref.get("figure"),
            "locator": ref.get("locator"),
            "url": ref.get("url"),
            "quote": ref.get("quote"),
            "chunk_id": chunk.get("id"),
            "action_id": chunk.get("action_id", chunk.get("id")),
            "chunk_title": chunk.get("title"),
        })

    # Fallback: every chunk should still have some reference object.
    if not normalized:
        normalized.append({
            "ref_id": _stable_ref_id(chunk),
            "source_id": chunk.get("source_id", "cr10smart_manual"),
            "source_title": chunk.get("source_title", "Creality CR-10 Smart User Manual"),
            "source_type": chunk.get("source_type", "official_manual"),
            "source_trust": chunk.get("source_trust", "high"),
            "section": chunk.get("title"),
            "page": chunk.get("page", chunk.get("source_page")),
            "pages": chunk.get("pages", chunk.get("source_pages")),
            "figure": chunk.get("figure"),
            "locator": chunk.get("locator") or f"Official manual / {chunk.get('title')}",
            "url": chunk.get("url"),
            "quote": None,
            "chunk_id": chunk.get("id"),
            "action_id": chunk.get("action_id", chunk.get("id")),
            "chunk_title": chunk.get("title"),
        })

    return normalized

def format_reference_for_ui(ref: dict) -> str:
    pages = ref.get("pages")
    page = ref.get("page")

    if pages:
        if len(pages) == 1:
            page_text = f"p. {pages[0]}"
        else:
            page_text = f"pp. {pages[0]}–{pages[-1]}"
    elif page:
        page_text = f"p. {page}"
    else:
        page_text = "page unknown"

    source = ref.get("source_title") or "Unknown source"
    section = ref.get("section") or "Unknown section"

    return f"{source}, {page_text}, {section}"


def build_used_reference_objects(references: list[dict], used_reference_ids: list[str]) -> list[dict]:
    used_ids = set(used_reference_ids or [])

    used_refs = [
        dict(ref)
        for ref in references
        if ref.get("ref_id") in used_ids
    ]

    for ref in used_refs:
        ref["display_text"] = format_reference_for_ui(ref)

    return used_refs


def build_references_from_chunks(
    retrieved_chunks: List[Dict[str, Any]],
    selected_action_id: Optional[str] = None,
    max_refs: int = 5,
) -> List[Dict[str, Any]]:
    refs = []
    seen = set()

    # Put selected chunk refs first.
    ordered_chunks = []

    if selected_action_id:
        ordered_chunks.extend([
            c for c in retrieved_chunks
            if str(c.get("action_id", c.get("id", ""))) == str(selected_action_id)
        ])

    ordered_chunks.extend([
        c for c in retrieved_chunks
        if str(c.get("action_id", c.get("id", ""))) != str(selected_action_id)
    ])

    for chunk in ordered_chunks:
        for ref in normalize_chunk_references(chunk):
            ref_id = ref["ref_id"]
            if ref_id in seen:
                continue

            refs.append(ref)
            seen.add(ref_id)

            if len(refs) >= max_refs:
                return refs

    return refs