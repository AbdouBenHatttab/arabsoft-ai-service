"""
response_sanitizer.py
---------------------
Validates and cleans a ChatResponse before it is returned to the caller.

Responsibilities:
  1. Remove relatedPages whose route is not in TRUSTED_ROUTES.
  2. Remove relatedPages whose route is not allowed for the requesting role.
  3. Append a warning for every route that was removed.
  4. Scan the answer text for unsafe action-directive phrases and add a warning
     if any are detected (does not modify the answer text itself).
  5. Never invent replacement routes or modify answer/reasons/disclaimer.

Usage:
  from app.services.response_sanitizer import sanitize_response
  clean = sanitize_response(raw_response, role="EMPLOYEE")
"""

from app.schemas import ChatResponse
from app.data.trusted_routes import filter_related_pages_for_role


# ---------------------------------------------------------------------------
# Action-directive phrases that should not appear in AI-generated answers.
# These are advisory checks only — the answer is preserved, a warning is added.
# ---------------------------------------------------------------------------

_UNSAFE_DIRECTIVES: list[str] = [
    "approve the request",
    "approve this request",
    "reject the request",
    "reject this request",
    "click approve",
    "click reject",
    "bypass the workflow",
    "bypass workflow",
    "skip approval",
    "grant permission",
    "change the role",
    "deactivate the account",
]


def sanitize_response(response: ChatResponse, role: str) -> ChatResponse:
    """
    Return a sanitized copy of *response* with unsafe or wrong-role routes removed.
    The original object is not mutated.
    """
    warnings = list(response.warnings)

    # ------------------------------------------------------------------
    # 1 & 2. Strip routes that are invented or not allowed for this role
    # ------------------------------------------------------------------
    kept_pages, removed_routes = filter_related_pages_for_role(response.relatedPages, role)

    for route in removed_routes:
        warnings.append(
            f"Route '{route}' was removed because it is not a valid route for role '{role}'."
        )

    # ------------------------------------------------------------------
    # 3. Scan for unsafe action directives in the answer text
    # ------------------------------------------------------------------
    answer_lower = response.answer.lower()
    for phrase in _UNSAFE_DIRECTIVES:
        if phrase in answer_lower:
            warnings.append(
                f"The assistant response may contain an action directive ('{phrase}'). "
                "Final decisions must be made by authorized users only."
            )
            break  # one warning is enough; don't flood

    return ChatResponse(
        answer=response.answer,
        reasons=response.reasons,
        warnings=warnings,
        relatedPages=kept_pages,
        disclaimer=response.disclaimer,
        aiGenerated=response.aiGenerated,
        source=response.source,
    )
