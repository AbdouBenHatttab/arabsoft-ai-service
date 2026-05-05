"""
assistant_service.py
--------------------
Central router for the /assistant/chat endpoint.

Pipeline (v2 / Phase 2):
  1. Refusal check        -> refusal_rules.py          (source="refusal")
  2. Local rule handler   -> platform_help_service.py  (source="local_rules")
  3. Gemini Q&A           -> gemini_client.py           (source="external_ai")
       Only reached when:
         - local rules produced no answer, AND
         - GEMINI_ENABLED=true
       Response is always sanitized before returning.
  4. Generic external AI  -> external_agent_client.py  (source="external_ai")
       Only reached when Gemini is disabled or returns None.
  5. Fallback             -> generic guidance response  (source="fallback")

Invariants:
  - Refusals always fire before any AI call.
  - Local deterministic handlers always fire before any AI call.
  - All external AI responses are sanitized (invented/wrong-role routes stripped).
  - No database access. No workflow mutations.
"""

from app.schemas import ChatRequest, ChatResponse
from app.services.refusal_rules import get_refusal
from app.services.platform_help_service import get_platform_help
from app.services.response_sanitizer import sanitize_response
from app.clients.external_agent_client import call_external_agent
from app.clients.gemini_client import call_gemini


def process_chat(request: ChatRequest) -> ChatResponse:

    # --- Step 1: Refusal check ------------------------------------------
    # Must always run first. Prevents action directives from reaching any
    # downstream handler, including the external AI.
    refusal_message = get_refusal(request.question)
    if refusal_message:
        return ChatResponse(
            answer=refusal_message,
            warnings=["This type of action is not supported by the assistant."],
            source="refusal",
        )

    # --- Step 2: Role-aware local platform guidance ---------------------
    # Deterministic, fast, free. Covers all currently known handler cases.
    platform_response = get_platform_help(request)
    if platform_response:
        if not platform_response.source:
            platform_response.source = "local_rules"
        return platform_response

    # --- Step 3: Gemini Q&A (only when enabled) --------------------------
    # Only reached when local rules produced no answer.
    # The Gemini response is always sanitized before being returned.
    gemini_response = call_gemini(request)
    if gemini_response:
        sanitized = sanitize_response(gemini_response, role=request.role)
        sanitized.source = "external_ai"
        return sanitized

    # --- Step 4: Generic external AI (fallback provider) -----------------
    # Only reached when Gemini is disabled or returned None.
    ai_response = call_external_agent(request)
    if ai_response:
        sanitized = sanitize_response(ai_response, role=request.role)
        sanitized.source = "external_ai"
        return sanitized

    # --- Step 5: Fallback -----------------------------------------------
    return ChatResponse(
        answer=(
            "I can help you with HR processes such as leave requests, loan applications, "
            "and platform navigation. Please describe what you need help with."
        ),
        source="fallback",
    )
