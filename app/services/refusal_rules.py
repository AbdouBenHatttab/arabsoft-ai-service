"""
refusal_rules.py
----------------
Determines whether a user question must be refused.
Rules are intentionally simple and keyword-based (v1 / local mode).

Refused categories:
  - Automatic approve / reject / submit of any workflow item
  - Sending / forwarding requests on the user's behalf
  - Role changes or privilege escalation
  - Account deactivation
  - Bypassing any workflow step
  - Questions completely unrelated to the HR/ERP platform

Instructional questions are NOT refused:
  - "How do I submit a leave request?" — asking how, not commanding
  - "Where can I submit a leave request?" — navigation question
  These are distinguished by checking for instructional prefixes before
  applying action-directive checks.
"""

from typing import Optional

# ---------------------------------------------------------------------------
# Instructional prefixes — questions starting with these are NOT action
# directives; they ask HOW to do something, not to do it.
# ---------------------------------------------------------------------------

_INSTRUCTIONAL_PREFIXES: tuple[str, ...] = (
    "how ",
    "how do ",
    "how can ",
    "how should ",
    "where ",
    "where can ",
    "where do ",
    "where should ",
    "what ",
    "what is ",
    "what are ",
    "can i ",
    "could i ",
    "is it possible",
    "which ",
    "when ",
    "show me where",
    "tell me how",
)


# ---------------------------------------------------------------------------
# Phrase sets
# ---------------------------------------------------------------------------

_ADMIN_ACTION_PHRASES = [
    # Approval / rejection
    "approve",
    "reject",
    # Submission on behalf of the user — direct command form only.
    # "submit" alone is caught here; instructional forms ("how do I submit")
    # are excluded by the _is_instructional() guard applied before this check.
    "submit my",
    "submit this",
    "submit the",
    "submit it",
    "send my",
    "send this",
    "send the",
    "send it for me",
    "send this for me",
    "submit this for me",
    "submit it for me",
    "automatically submit",
    "auto submit",
    "auto-submit",
    # Role / privilege changes
    "change role",
    "change my role",
    "deactivate",
    "deactivate account",
    "bypass workflow",
    "bypass the workflow",
    "skip approval",
    "grant permission",
    "give me admin",
    "escalate privilege",
]

_UNRELATED_PHRASES = [
    "weather",
    "sports",
    "football",
    "movie",
    "recipe",
    "stock price",
    "stock market",
    "cryptocurrency",
    "bitcoin",
    "tell me a joke",
]


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def get_refusal(question: str) -> Optional[str]:
    """
    Return a refusal message string if the question should be refused,
    or None if it is allowed to proceed.

    Instructional questions ("How do I submit…", "Where can I submit…")
    are never refused — they are navigation/guidance questions answered by
    the platform help layer.
    """
    q = question.lower().strip()

    # Instructional questions are never action directives.
    if _is_instructional(q):
        # Still check unrelated phrases even for instructional questions.
        if _matches_any(q, _UNRELATED_PHRASES):
            return (
                "I am an HR platform assistant. "
                "I cannot answer questions unrelated to the ArabSoft platform or its HR processes."
            )
        return None

    if _matches_any(q, _ADMIN_ACTION_PHRASES):
        return (
            "I cannot perform or trigger administrative actions such as approving requests, "
            "rejecting requests, submitting requests on your behalf, or changing roles. "
            "These actions must be completed by an authorized user through the proper workflow."
        )

    if _matches_any(q, _UNRELATED_PHRASES):
        return (
            "I am an HR platform assistant. "
            "I cannot answer questions unrelated to the ArabSoft platform or its HR processes."
        )

    return None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _matches_any(text: str, phrases: list[str]) -> bool:
    return any(phrase in text for phrase in phrases)


def _is_instructional(text: str) -> bool:
    """
    Return True if the question is an instructional / navigation question
    (asking HOW or WHERE to do something) rather than a direct action command.
    """
    return text.startswith(_INSTRUCTIONAL_PREFIXES)
