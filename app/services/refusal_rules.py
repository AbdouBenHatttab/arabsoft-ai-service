"""
refusal_rules.py
----------------
Determines whether a user question must be refused.
Rules are intentionally simple and keyword-based (v1 / local mode).

Refused categories:
  - Automatic approve / reject of any workflow item
  - Role changes or privilege escalation
  - Account deactivation
  - Bypassing any workflow step
  - Questions completely unrelated to the HR/ERP platform
"""

from typing import Optional

# ---------------------------------------------------------------------------
# Phrase sets
# ---------------------------------------------------------------------------

_ADMIN_ACTION_PHRASES = [
    "approve",
    "reject",
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
    """
    q = question.lower()

    if _matches_any(q, _ADMIN_ACTION_PHRASES):
        return (
            "I cannot perform or trigger administrative actions such as approving requests, "
            "rejecting requests, changing roles, or deactivating accounts. "
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
