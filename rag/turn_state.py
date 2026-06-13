"""Turn classification helpers for the printer assistant.

The classifier is intentionally simple and rule-based:
- explicit thanks/closing stays closing
- explicit continuation phrases stay continuation
- explicit task questions stay task_question
- short completion confirmations stay completion_confirmation
- first-person completed actions become progress_update

The important principle is to avoid over-triggering progress updates from broad words
like "already". A message such as "the print already started, can I remove the TF
card?" is a new task/safety question, not a progress update.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional


THANKS_PHRASES = [
    "thank you",
    "thanks",
    "appreciate it",
    "got it thanks",
    "ok thanks",
    "okay thanks",
]

# Exact / near-exact continuation requests.
CONTINUATION_PHRASES = {
    "okay now",
    "ok now",
    "what next",
    "next",
    "continue",
    "go on",
    "now what",
    "is this correct",
    "is it correct",
    "is it right",
    "is this right",
    "is this correct now",
    "is it correct now",
}

# Exact / near-exact completion confirmations.
COMPLETION_PHRASES = {
    "done",
    "finished",
    "completed",
    "i did it",
    "i have done it",
    "got it",
}

# Question cues that should usually make the turn a new task question.
# These prevent phrases like "already" from turning safety questions into progress updates.
TASK_QUESTION_CUES = [
    "can i",
    "should i",
    "do i",
    "does it",
    "does the",
    "is it okay",
    "is it safe",
    "is this safe",
    "how do i",
    "how can i",
    "what should",
    "what do i",
    "what is",
    "what's",
    "where do",
    "where should",
    "why",
    "when",
    "which",
]

# First-person progress patterns. Keep these concrete; avoid generic words like "already".
PROGRESS_PATTERNS = [
    r"\bi\s+(already\s+)?inserted\b",
    r"\bi\s+(already\s+)?put\b",
    r"\bi\s+(already\s+)?placed\b",
    r"\bi\s+(already\s+)?connected\b",
    r"\bi\s+(already\s+)?mounted\b",
    r"\bi\s+(already\s+)?loaded\b",
    r"\bi\s+(already\s+)?pressed\b",
    r"\bi\s+(already\s+)?renamed\b",
    r"\bi\s+(already\s+)?turned\b",
    r"\bi\s+(already\s+)?scanned\b",
    r"\bi\s+(already\s+)?tightened\b",
    r"\bi\s+(already\s+)?loosened\b",
    r"\bi\s+(already\s+)?adjusted\b",
    r"\bi\s+(already\s+)?selected\b",
    r"\bi\s+(already\s+)?clicked\b",
    r"\bi\s+(already\s+)?chose\b",
    r"\bi\s+(already\s+)?removed\b",
    r"\bi\s+(already\s+)?installed\b",
    r"\bi\s+(already\s+)?attached\b",
    r"\bi\s+(already\s+)?fed\b",
    r"\bi\s+(already\s+)?cut\b",
    r"\bi\s+(already\s+)?saved\b",
    r"\bi\s+(already\s+)?generated\b",
    r"\bi\s+(already\s+)?opened\b",
    r"\bit\s+is\s+heating\b",
    r"\bit's\s+heating\b",
    r"\bit\s+is\s+already\s+heating\b",
    r"\bit's\s+already\s+heating\b",
    r"\bit\s+is\s+in\b",
    r"\bit's\s+in\b",
    r"\bnow\s+it\s+is\b",
    r"\bnow\s+it's\b",
    r"\bturned\s+it\s+off\b",
    r"\bthe\s+card\s+is\s+inserted\b",
]


def _normalize(question: str) -> str:
    q = (question or "").lower().strip()
    q = re.sub(r"\s+", " ", q)
    return q.strip(" .?!")


def _has_task_question_intent(q: str, original_question: str) -> bool:
    if "?" in original_question:
        # Exact continuation checks are handled before this function.
        return True

    return any(cue in q for cue in TASK_QUESTION_CUES)


def _is_short_continuation(q: str) -> bool:
    if q in CONTINUATION_PHRASES:
        return True

    # Allows casual variants like "ok next" or "okay, now" after normalization.
    if len(q.split()) <= 3 and any(word in q.split() for word in ["ok", "okay", "next", "now"]):
        return True

    return False


def _is_completion_confirmation(q: str) -> bool:
    return q in COMPLETION_PHRASES


def _is_progress_update(q: str) -> bool:
    return any(re.search(pattern, q) for pattern in PROGRESS_PATTERNS)


def classify_user_turn(
    question: str,
    session_state: Optional[Dict[str, Any]] = None,
) -> str:
    q = _normalize(question)

    # 1. Pure thanks/closing should not advance the guide.
    if any(phrase in q for phrase in THANKS_PHRASES):
        if not _is_short_continuation(q):
            return "thanks_or_closing"

    # 2. Explicit continuation wins over question punctuation.
    if _is_short_continuation(q):
        return "continuation"

    # 3. Safety/procedure questions should not be classified as progress updates.
    if _has_task_question_intent(q, question):
        return "task_question"

    # 4. Short completion confirmations.
    if _is_completion_confirmation(q):
        return "completion_confirmation"

    # 5. Concrete first-person progress updates.
    if _is_progress_update(q):
        return "progress_update"

    return "task_question"
