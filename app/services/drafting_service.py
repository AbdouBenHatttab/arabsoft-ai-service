"""
drafting_service.py
-------------------
V2 Phase 3: Drafting Assistant.

Responsibilities:
  1. Detect drafting intent from the user's question.
  2. When Gemini is enabled, call Gemini with a drafting-specific prompt and
     return the generated draft text in ChatResponse.draft.
  3. When Gemini is disabled or fails, return a safe local template draft so
     the caller never receives an error or an empty response.

Draft types handled:
  - leave_request     : leave request reason / justification
  - loan_justification: loan application justification
  - authorization     : authorization request explanation
  - document_request  : official document request letter
  - improve_text      : improve / make professional an existing text snippet

Invariants:
  - Never submits, approves, or performs any workflow action.
  - Never invents dates, amounts, salaries, balances, or medical details.
  - Placeholders ([date], [reason], [amount] …) are used when real data is absent.
  - Always includes a review disclaimer in the response.
  - source="external_ai"  when Gemini produces the draft.
  - source="local_rules"  when a local template is returned.
  - source="fallback"     when both Gemini and local template fail (should not happen).

The caller (assistant_service.py) must NOT run sanitize_response on drafting
responses — drafting responses carry no relatedPages and need no route-stripping.
"""

import json
import logging
from typing import Optional

import httpx

from app.config import settings
from app.schemas import ChatRequest, ChatResponse

logger = logging.getLogger(__name__)

_GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta/models"

# ---------------------------------------------------------------------------
# Drafting intent detection
# ---------------------------------------------------------------------------

# Trigger verbs that signal the user wants text to be composed / improved.
_DRAFT_VERBS: list[str] = [
    "draft",
    "write",
    "compose",
    "help me write",
    "help me draft",
    "help me compose",
    "improve",
    "make it more professional",
    "make this more professional",
    "make this message more professional",
    "make this professional",
    "rephrase",
    "rewrite",
    "polish",
    "refine",
    "suggest wording",
    "suggest text",
    "how should i phrase",
    "how do i phrase",
    "what should i write",
    "what should i say",
    "professional version",
    "professional wording",
    "formal version",
    "formal wording",
]

# Request-type nouns that anchor the drafting intent.
_DRAFT_SUBJECTS: list[str] = [
    "leave request",
    "leave reason",
    "leave justification",
    "loan request",
    "loan justification",
    "loan reason",
    "loan application",
    "authorization request",
    "authorization explanation",
    "authorization justification",
    "authorization reason",
    "document request",
    "document justification",
    "request text",
    "request reason",
    "request letter",
    "request explanation",
    "justification",
    "this request text",
    "this text",
    "this message",
    "my message",
    "my text",
    "my request",
]


def detect_drafting_intent(question: str) -> bool:
    """
    Return True if the question is a drafting request.

    A question is classified as a drafting request when it contains:
      - at least one drafting verb, AND
      - at least one drafting subject noun.

    This two-part check keeps precision high and avoids false positives
    on normal navigation or information questions.
    """
    q = question.lower()
    has_verb = any(verb in q for verb in _DRAFT_VERBS)
    has_subject = any(subj in q for subj in _DRAFT_SUBJECTS)
    return has_verb and has_subject


def _classify_draft_type(question: str) -> str:
    """Return the most specific draft type name for use in the Gemini prompt."""
    q = question.lower()
    if "loan" in q:
        return "loan_justification"
    if "authorization" in q:
        return "authorization"
    if "document" in q:
        return "document_request"
    if "improve" in q or "professional" in q or "rephrase" in q or "rewrite" in q or "polish" in q or "refine" in q:
        return "improve_text"
    return "leave_request"


# ---------------------------------------------------------------------------
# Local template drafts  (used when Gemini is disabled or fails)
# ---------------------------------------------------------------------------

_REVIEW_DISCLAIMER = (
    "\n\n⚠️ Please review and personalise this draft before submitting. "
    "The assistant cannot submit or send anything on your behalf."
)

_LOCAL_TEMPLATES: dict[str, str] = {
    "leave_request": (
        "Subject: Leave Request — [Your Name]\n\n"
        "Dear [Manager / HR Team],\n\n"
        "I am writing to formally request leave from [start date] to [end date] "
        "([number of days] working day(s)).\n\n"
        "Reason: [Briefly describe your reason, e.g., personal matter, family commitment, medical appointment].\n\n"
        "I will ensure all outstanding tasks are handed over before my absence and I am "
        "happy to discuss any arrangements needed to cover my responsibilities during this period.\n\n"
        "Thank you for considering my request.\n\n"
        "Regards,\n[Your Name]"
    ),
    "loan_justification": (
        "Subject: Loan Request Justification — [Your Name]\n\n"
        "Dear HR Team,\n\n"
        "I am respectfully submitting a loan request in the amount of [amount] "
        "to be repaid over [repayment period].\n\n"
        "Purpose: [Briefly describe the reason, e.g., medical expenses, home repairs, "
        "educational fees — do not include sensitive personal details unless required].\n\n"
        "I confirm that I am aware of the repayment terms and conditions and I agree to "
        "the applicable deduction schedule.\n\n"
        "Thank you for your consideration.\n\n"
        "Regards,\n[Your Name]"
    ),
    "authorization": (
        "Subject: Authorization Request — [Your Name]\n\n"
        "Dear [Manager / HR Team],\n\n"
        "I am writing to request authorization for [describe the action or access needed].\n\n"
        "Reason: [Explain briefly why the authorization is needed and how it relates to your role].\n\n"
        "I understand this request will be reviewed by the appropriate authority and I am "
        "available to provide any additional information required.\n\n"
        "Thank you.\n\n"
        "Regards,\n[Your Name]"
    ),
    "document_request": (
        "Subject: Document Request — [Document Name]\n\n"
        "Dear HR Team,\n\n"
        "I am writing to formally request [name of document, e.g., employment certificate, "
        "payslip, experience letter].\n\n"
        "Purpose: [State the reason, e.g., bank loan application, visa processing, personal records].\n\n"
        "Please let me know if any additional information is required to process this request.\n\n"
        "Thank you.\n\n"
        "Regards,\n[Your Name]"
    ),
    "improve_text": (
        "Below is a professionally rephrased version of your text. "
        "Replace the bracketed placeholders with your actual details before submitting.\n\n"
        "--- Suggested professional version ---\n"
        "I am writing to formally communicate [your main point or request]. "
        "[Briefly provide any relevant context or justification]. "
        "I would appreciate your consideration and am happy to provide further information if needed.\n\n"
        "Regards,\n[Your Name]\n"
        "--- End of suggestion ---"
    ),
}


def _local_draft(draft_type: str) -> ChatResponse:
    """Build a ChatResponse with a local template draft."""
    template = _LOCAL_TEMPLATES.get(draft_type, _LOCAL_TEMPLATES["leave_request"])
    draft_text = template + _REVIEW_DISCLAIMER
    return ChatResponse(
        answer=(
            "Here is a template draft you can personalise before submitting. "
            "Fill in the bracketed placeholders with your actual details."
        ),
        draft=draft_text,
        warnings=[
            "This is a locally generated template. Review all details before submitting."
        ],
        source="local_rules",
    )


# ---------------------------------------------------------------------------
# Gemini drafting prompt
# ---------------------------------------------------------------------------

_DRAFTING_SYSTEM_PROMPT = """\
You are a professional writing assistant for ArabSoft HR platform users.
Your ONLY task is to help users draft or improve HR-related request texts.

STRICT RULES — never break these:
1. Generate professional, concise draft text only.
2. Never submit, approve, reject, or perform any action.
3. Never invent specific dates, salary figures, leave balances, loan amounts, or medical details.
   Use placeholders in square brackets instead: [date], [amount], [reason], [your name], etc.
4. Never include fake company policy, policy numbers, or regulatory references.
5. Keep the draft under 250 words.
6. Always end your JSON response with a disclaimer reminding the user to review before submitting.
7. Respond ONLY with valid JSON — no Markdown fences, no extra keys.

JSON format (respond with exactly this structure):
{
  "answer": "<one sentence describing what the draft covers>",
  "draft": "<the full draft text with placeholders where real data is missing>",
  "disclaimer": "Please review and personalise this draft before submitting. The assistant cannot submit anything on your behalf."
}
"""


def _build_drafting_user_message(question: str, draft_type: str) -> str:
    return (
        f"Draft type requested: {draft_type}\n"
        f"User request: {question}\n\n"
        "Write a professional draft following the rules above. "
        "Use placeholders like [date], [reason], [amount] wherever specific details are missing."
    )


# ---------------------------------------------------------------------------
# Gemini call for drafting
# ---------------------------------------------------------------------------

def _call_gemini_for_draft(question: str, draft_type: str) -> Optional[ChatResponse]:
    """
    Call Gemini specifically for drafting. Returns a ChatResponse with draft
    populated, or None on any failure.
    """
    if not settings.gemini_enabled:
        logger.debug("Gemini disabled — skipping drafting call.")
        return None
    if not settings.gemini_api_key:
        logger.warning("GEMINI_ENABLED=true but GEMINI_API_KEY is empty — skipping drafting call.")
        return None

    url = f"{_GEMINI_BASE}/{settings.gemini_model}:generateContent"
    headers = {
        "x-goog-api-key": settings.gemini_api_key,
        "Content-Type": "application/json",
    }
    payload = {
        "system_instruction": {
            "parts": [{"text": _DRAFTING_SYSTEM_PROMPT}]
        },
        "contents": [
            {
                "role": "user",
                "parts": [{"text": _build_drafting_user_message(question, draft_type)}],
            }
        ],
        "generationConfig": {
            "temperature": 0.4,
            "maxOutputTokens": 600,
        },
    }

    try:
        with httpx.Client(timeout=settings.gemini_timeout_seconds) as client:
            resp = client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()

        raw_text = (
            data.get("candidates", [{}])[0]
            .get("content", {})
            .get("parts", [{}])[0]
            .get("text", "")
            .strip()
        )

        # Strip Markdown code fences if Gemini wrapped the JSON
        if raw_text.startswith("```"):
            raw_text = raw_text.split("```")[1]
            if raw_text.startswith("json"):
                raw_text = raw_text[4:]
            raw_text = raw_text.strip()

        parsed = json.loads(raw_text)

        draft_text = parsed.get("draft", "").strip()
        if not draft_text:
            logger.warning("Gemini drafting response contained no 'draft' field.")
            return None

        # Always append the review disclaimer so it cannot be omitted by the model
        if "review" not in draft_text.lower() and "disclaimer" not in draft_text.lower():
            draft_text += _REVIEW_DISCLAIMER

        return ChatResponse(
            answer=parsed.get(
                "answer",
                "Here is a professional draft. Please review before submitting.",
            ),
            draft=draft_text,
            warnings=[],
            relatedPages=[],
            aiGenerated=True,
            source="external_ai",
        )

    except httpx.TimeoutException:
        logger.warning("Gemini drafting call timed out.")
        return None
    except httpx.HTTPStatusError as exc:
        logger.warning("Gemini drafting returned HTTP %s.", exc.response.status_code)
        return None
    except (json.JSONDecodeError, KeyError, IndexError) as exc:
        logger.warning("Gemini drafting response could not be parsed: %s", exc)
        return None
    except Exception as exc:  # noqa: BLE001
        logger.warning("Gemini drafting call failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def get_draft_response(request: ChatRequest) -> Optional[ChatResponse]:
    """
    If the user's question is a drafting request, return a ChatResponse with
    the draft populated. Returns None if no drafting intent is detected.

    Pipeline:
      1. Detect intent  — return None immediately if not a drafting question.
      2. Try Gemini     — returns a response with source="external_ai" on success.
      3. Local template — always succeeds; source="local_rules".
    """
    if not detect_drafting_intent(request.question):
        return None

    draft_type = _classify_draft_type(request.question)
    logger.debug("Drafting intent detected. type=%s", draft_type)

    # Attempt AI-generated draft first
    gemini_result = _call_gemini_for_draft(request.question, draft_type)
    if gemini_result is not None:
        return gemini_result

    # Gemini disabled or failed — use local template (never crashes)
    return _local_draft(draft_type)
