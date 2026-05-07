"""
assistant_service.py
--------------------
Central router for the /assistant/chat endpoint.

Pipeline (v2 / Phase 3):
  1. Refusal check        -> refusal_rules.py          (source="refusal")
  2. Drafting intent      -> drafting_service.py        (source="external_ai" or "local_rules")
       Fires BEFORE local platform rules so that drafting questions containing
       platform keywords ("leave", "loan", "request" ...) are not swallowed by
       the platform handlers. get_draft_response() returns None immediately for
       non-drafting questions, so non-drafting traffic is unaffected.
  3. Local rule handler   -> platform_help_service.py  (source="local_rules")
       Only reached when no drafting intent was detected.
  4. Gemini Q&A           -> gemini_client.py           (source="external_ai")
       Only reached when:
         - no drafting intent detected, AND
         - local rules produced no answer, AND
         - GEMINI_ENABLED=true
       Response is always sanitized before returning.
  5. Generic external AI  -> external_agent_client.py  (source="external_ai")
       Only reached when Gemini is disabled or returns None.
  6. Fallback             -> generic guidance response  (source="fallback")

Invariants:
  - Refusals always fire before any AI or drafting call.
  - Drafting intent is checked before local platform rules to prevent platform
    keyword handlers from intercepting drafting-style questions.
  - Drafting responses carry a populated `draft` field and no relatedPages.
  - Drafting responses are NOT run through sanitize_response (no routes to strip).
  - All non-drafting external AI responses are sanitized (invented/wrong-role routes stripped).
  - No database access. No workflow mutations.
"""

from app.schemas import ChatRequest, ChatResponse
from app.services.refusal_rules import get_refusal
from app.services.decision_support_service import get_decision_support_response
from app.services.drafting_service import get_draft_response
from app.services.platform_help_service import get_platform_help
from app.services.response_sanitizer import sanitize_response
from app.clients.external_agent_client import call_external_agent
from app.clients.gemini_client import call_gemini


def process_chat(request: ChatRequest) -> ChatResponse:

    # --- Step 0: Team Leader selected-leave decision support --------------
    # Narrow exception before refusal: "Should I approve this leave?" is an
    # assessment request, not a command to perform approval. Direct action
    # commands such as "approve this leave" still fall through to refusal.
    decision_support_response = get_decision_support_response(request)
    if decision_support_response:
        return decision_support_response

    # --- Step 1: Refusal check ------------------------------------------
    # Must always run first. Prevents action directives from reaching any
    # downstream handler, including drafting and the external AI.
    refusal_message = get_refusal(request.question)
    if refusal_message:
        return ChatResponse(
            answer=refusal_message,
            warnings=["This type of action is not supported by the assistant."],
            source="refusal",
        )

    # --- Step 2: Drafting intent -----------------------------------------
    # Must run BEFORE local platform rules.
    #
    # Why: platform_help_service handlers match on keywords like "leave", "loan",
    # and "request" — exactly the same words that appear in drafting questions
    # such as "Help me draft a leave request reason" or "Write a loan justification".
    # If platform rules ran first, those handlers would return a navigation
    # response and drafting_service would never be reached.
    #
    # get_draft_response() calls detect_drafting_intent() internally and returns
    # None immediately for questions that are not drafting requests, so non-
    # drafting traffic passes through this step with zero cost.
    #
    # Drafting responses carry a `draft` field and are NOT sanitized
    # (they contain no relatedPages routes to strip).
    draft_response = get_draft_response(request)
    if draft_response:
        return draft_response

    # --- Step 3: Role-aware local platform guidance ---------------------
    # Only reached when no drafting intent was detected.
    # Deterministic, fast, free. Covers all currently known handler cases.
    platform_response = get_platform_help(request)
    if platform_response:
        if not platform_response.source:
            platform_response.source = "local_rules"
        return platform_response

    # --- Step 4: Gemini Q&A (only when enabled) --------------------------
    # Only reached when no drafting intent detected AND local rules produced
    # no answer. The Gemini response is always sanitized before being returned.
    gemini_response = call_gemini(request)
    if gemini_response:
        sanitized = sanitize_response(gemini_response, role=request.role)
        sanitized.source = "external_ai"
        return sanitized

    # --- Step 5: Generic external AI (fallback provider) -----------------
    # Only reached when Gemini is disabled or returned None.
    ai_response = call_external_agent(request)
    if ai_response:
        sanitized = sanitize_response(ai_response, role=request.role)
        sanitized.source = "external_ai"
        return sanitized

    # --- Step 6: Fallback -----------------------------------------------
    return ChatResponse(
        answer=(
            "I can help you with HR processes such as leave requests, loan applications, "
            "and platform navigation. Please describe what you need help with."
        ),
        source="fallback",
    )
