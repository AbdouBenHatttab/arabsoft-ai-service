"""
drafting_service.py
-------------------
V2 Phase 3 / Phase 3.1: Drafting Assistant with Structured Field Extraction.

Responsibilities:
  1. Detect drafting intent from the user's question.
  2. Extract structured fields from the user's question (local regex, no network).
  3. When Gemini is enabled, call Gemini with a drafting-specific prompt that also
     requests structuredFields; merge Gemini's structured output with local extraction
     as a fallback for missing/invalid Gemini structure.
  4. When Gemini is disabled or fails, return a safe local template draft with
     locally extracted structured fields.

Draft types handled:
  - LEAVE_REQUEST         : leave request reason / justification
  - LOAN_REQUEST          : loan application justification
  - AUTHORIZATION_REQUEST : authorization request explanation
  - DOCUMENT_REQUEST      : official document request letter
  - IMPROVE_TEXT          : improve / make professional an existing text snippet

Supported leave types (ONLY these — never invent others):
  ANNUAL | SICK | UNPAID | MATERNITY | PATERNITY
  If the user says "emergency", "family emergency", etc., leaveType stays null
  and the detail is kept in reason. "leaveType" is added to missingFields.

Invariants:
  - Never submits, approves, or performs any workflow action.
  - Never invents dates, amounts, salaries, balances, or medical details.
  - Never calculates working days, leave balance, repayment months, eligibility,
    overlap, or approval rules.
  - Dates stay raw when ambiguous ("tomorrow", "May 12", "next Monday").
    Only normalize when input is unambiguously numeric (e.g. "2026-05-12").
  - draftFields uses a stable shape with null values for missing fields for all
    structured types. draftFields=null only for IMPROVE_TEXT.
  - missingFields lists every field that could not be extracted.
  - relatedPages is always [] for drafting responses.
  - Always includes a review disclaimer in the response.
  - source="external_ai"  when Gemini produces the draft text.
  - source="local_rules"  when a local template is used.

The caller (assistant_service.py) must NOT run sanitize_response on drafting
responses — drafting responses carry no relatedPages and need no route-stripping.
"""

import json
import logging
import re
from typing import Optional

import httpx

from app.config import settings
from app.schemas import ChatRequest, ChatResponse

logger = logging.getLogger(__name__)

_GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta/models"

# ---------------------------------------------------------------------------
# Drafting intent detection
# ---------------------------------------------------------------------------

# Trigger verbs/phrases that signal the user wants text to be composed,
# prepared, or improved — including request-preparation phrasing.
#
# Precision rule: these verbs must appear alongside a drafting subject
# (see _DRAFT_SUBJECTS) to avoid false positives on navigation questions.
# e.g. "How do I request a loan?" — no drafting verb match → not a draft.
#      "Help me request a loan for 2000 TND" — "help me request" matches → draft.
_DRAFT_VERBS: list[str] = [
    # Classic composition verbs
    "draft",
    "write",
    "compose",
    "help me write",
    "help me draft",
    "help me compose",
    # Request-preparation phrasing — user wants to build / prepare a request,
    # not just navigate to the request page.
    "help me request",
    "help me create",
    "help me prepare",
    "help me submit",
    "i want to request",
    "i need to request",
    "i would like to request",
    "prepare a",
    "prepare my",
    "create a",
    "create my",
    # Text-improvement verbs
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
    # Bare type nouns used alongside preparation verbs
    # e.g. "Help me request a loan …" / "I want to request annual leave …"
    "a loan",
    "annual leave",
    "sick leave",
    "unpaid leave",
    "maternity leave",
    "paternity leave",
    "a leave",
    "authorization",
    "a document",
    "a salary certificate",
    "an employment certificate",
    "a payslip",
    "an experience letter",
]


def detect_drafting_intent(question: str) -> bool:
    """
    Return True if the question is a drafting / request-preparation request.

    A question is classified as a drafting request when it contains:
      - at least one drafting verb, AND
      - at least one drafting subject noun.

    This two-part check keeps precision high:
      "How do I request a loan?"          → no verb match → False (navigation)
      "Help me request a loan for 2000"   → "help me request" + "a loan" → True
      "I want to request annual leave"    → "i want to request" + "annual leave" → True
    """
    q = question.lower()
    has_verb = any(verb in q for verb in _DRAFT_VERBS)
    has_subject = any(subj in q for subj in _DRAFT_SUBJECTS)
    return has_verb and has_subject


# Internal type constants — used throughout this module.
_TYPE_LEAVE = "LEAVE_REQUEST"
_TYPE_LOAN = "LOAN_REQUEST"
_TYPE_AUTH = "AUTHORIZATION_REQUEST"
_TYPE_DOC = "DOCUMENT_REQUEST"
_TYPE_IMPROVE = "IMPROVE_TEXT"

# Strong document-type signals — these phrases unambiguously identify a
# DOCUMENT_REQUEST even when "loan" appears as a purpose context.
# e.g. "document request letter for a salary certificate for a bank loan"
#   → "document request" and "salary certificate" are present → DOCUMENT_REQUEST
#   → "bank loan" is purpose, not request type.
_STRONG_DOCUMENT_SIGNALS: list[str] = [
    "document request",
    "salary certificate",
    "employment certificate",
    "work certificate",
    "payslip",
    "pay slip",
    "experience letter",
    "attestation",
    "work cert",
    "salary cert",
    "employment cert",
]


def _classify_draft_type(question: str) -> str:
    """
    Return the draftType constant for this question.

    Priority order (explicit > implicit):
      1. DOCUMENT_REQUEST — checked first when strong document signals are
         present, so that "document request for a salary cert for a bank loan"
         is not misclassified as LOAN_REQUEST.
      2. AUTHORIZATION_REQUEST
      3. LOAN_REQUEST — checked after document signals are cleared.
      4. IMPROVE_TEXT
      5. LEAVE_REQUEST (default)
    """
    q = question.lower()

    # 1. Document signals take precedence over loan keyword, because "loan"
    #    may appear as a PURPOSE inside a document request (e.g. "bank loan
    #    application" as the reason for needing a salary certificate).
    if any(sig in q for sig in _STRONG_DOCUMENT_SIGNALS):
        return _TYPE_DOC

    # 2. Authorization
    if "authorization" in q:
        return _TYPE_AUTH

    # 3. Loan — only after document signals have been ruled out
    if "loan" in q:
        return _TYPE_LOAN

    # 4. Text-improvement
    if (
        "improve" in q
        or "professional" in q
        or "rephrase" in q
        or "rewrite" in q
        or "polish" in q
        or "refine" in q
    ):
        return _TYPE_IMPROVE

    # 5. Default: leave request
    return _TYPE_LEAVE


# ---------------------------------------------------------------------------
# Structured field extraction  (local, no network, no DB, no date arithmetic)
# ---------------------------------------------------------------------------

# Supported leave types — ONLY these. Never add EMERGENCY or other unsupported values.
_SUPPORTED_LEAVE_TYPES: dict[str, list[str]] = {
    "ANNUAL": ["annual", "paid leave", "vacation", "holiday", "yearly leave"],
    "SICK": ["sick", "medical", "illness", "health", "unwell", "doctor"],
    "UNPAID": ["unpaid"],
    "MATERNITY": ["maternity"],
    "PATERNITY": ["paternity"],
}

# Document type keywords
_DOCUMENT_TYPES: list[tuple[str, list[str]]] = [
    ("salary certificate", ["salary certificate", "salary cert"]),
    ("employment certificate", ["employment certificate", "work certificate", "employment cert"]),
    ("payslip", ["payslip", "pay slip", "pay stub"]),
    ("experience letter", ["experience letter", "experience cert"]),
    ("work certificate", ["work cert"]),
    ("attestation", ["attestation"]),
]

# Authorization type keywords
_AUTH_TYPES: list[tuple[str, list[str]]] = [
    ("departure", ["early departure", "departure", "leave early", "leave the office"]),
    ("late arrival", ["late arrival", "arrive late", "coming in late", "late to work"]),
    ("external", ["external", "outside the office", "off-site"]),
    ("medical", ["medical", "doctor", "hospital", "clinic"]),
]

# Date patterns — extract raw strings; only ISO dates are normalized
_DATE_PATTERNS = [
    # ISO date: 2026-05-12 — safe to keep as-is
    re.compile(r"\b(\d{4}-\d{2}-\d{2})\b"),
    # DD/MM/YYYY or MM/DD/YYYY — keep as raw string, don't interpret
    re.compile(r"\b(\d{1,2}/\d{1,2}/\d{4})\b"),
    # "May 12", "12 May", "May 12th"
    re.compile(
        r"\b((?:january|february|march|april|may|june|july|august|september|october|november|december)"
        r"\s+\d{1,2}(?:st|nd|rd|th)?|\d{1,2}(?:st|nd|rd|th)?\s+"
        r"(?:january|february|march|april|may|june|july|august|september|october|november|december))\b",
        re.IGNORECASE,
    ),
    # Relative: tomorrow, next Monday, next week, etc.
    re.compile(
        r"\b(tomorrow|next\s+(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday|week|month))\b",
        re.IGNORECASE,
    ),
    # "the 15th", "the 3rd"
    re.compile(r"\bthe\s+(\d{1,2}(?:st|nd|rd|th))\b", re.IGNORECASE),
]

# Time patterns: "10h", "10:00", "10am", "10 AM", "10h30"
_TIME_PATTERN = re.compile(
    r"\b(\d{1,2})(?::(\d{2})|h(\d{2})?|)\s*([ap]m)?\b",
    re.IGNORECASE,
)

# Range patterns: "from 10 to 12", "10 to 12", "between 10 and 12"
_TIME_RANGE_PATTERN = re.compile(
    r"(?:from\s+)?(\d{1,2}(?::\d{2})?(?:h\d{0,2})?(?:\s*[ap]m)?)"
    r"\s+(?:to|until|till|-)\s+"
    r"(\d{1,2}(?::\d{2})?(?:h\d{0,2})?(?:\s*[ap]m)?)",
    re.IGNORECASE,
)

# Reason extraction triggers
_REASON_TRIGGERS = re.compile(
    r"(?:because|for|reason:|reason is|due to|since|as|to attend|to go to)\s+(.+?)(?:\.|$)",
    re.IGNORECASE,
)

# Amount pattern: digits optionally followed by currency
_AMOUNT_PATTERN = re.compile(
    r"\b(\d[\d\s,\.]*(?:\s*(?:TND|EUR|USD|DZD|MAD|€|\$|£))?)\b",
    re.IGNORECASE,
)


def _extract_dates(text: str) -> list[str]:
    """Extract all date-like strings from text, preserving raw form."""
    found: list[str] = []
    for pattern in _DATE_PATTERNS:
        for m in pattern.finditer(text):
            val = m.group(1) if m.lastindex and m.lastindex >= 1 else m.group(0)
            val = val.strip()
            if val and val not in found:
                found.append(val)
    return found


def _normalize_time(raw: str) -> str:
    """Convert a raw time string to HH:MM format when safely possible."""
    raw = raw.strip().lower()
    m = re.match(
        r"(\d{1,2})(?::(\d{2})|h(\d{2})?)?(?:\s*([ap]m))?$",
        raw,
        re.IGNORECASE,
    )
    if not m:
        return raw
    hour = int(m.group(1))
    minutes = int(m.group(2) or m.group(3) or 0)
    meridiem = (m.group(4) or "").lower()
    if meridiem == "pm" and hour < 12:
        hour += 12
    elif meridiem == "am" and hour == 12:
        hour = 0
    return f"{hour:02d}:{minutes:02d}"


def _extract_time_range(text: str) -> tuple[Optional[str], Optional[str]]:
    """Extract fromTime and toTime from a time range expression."""
    m = _TIME_RANGE_PATTERN.search(text)
    if m:
        return _normalize_time(m.group(1)), _normalize_time(m.group(2))
    return None, None


def _extract_reason(text: str) -> Optional[str]:
    """Extract the reason clause from text using common trigger words."""
    m = _REASON_TRIGGERS.search(text)
    if m:
        return m.group(1).strip().rstrip(".")
    return None


def _extract_amount(text: str) -> Optional[str]:
    """Extract the first numeric amount (with optional currency) from text."""
    m = _AMOUNT_PATTERN.search(text)
    if m:
        val = m.group(1).strip().rstrip(",.")
        # Must contain at least one digit — skip if just whitespace/symbols
        if re.search(r"\d", val):
            return val
    return None


def _extract_leave_type(text: str) -> Optional[str]:
    """
    Return a supported leave type keyword or None.
    NEVER returns unsupported types like EMERGENCY.
    Emergency/family-emergency language stays in reason; leaveType=None.
    """
    q = text.lower()
    for leave_type, keywords in _SUPPORTED_LEAVE_TYPES.items():
        if any(kw in q for kw in keywords):
            return leave_type
    return None


def _extract_document_type(text: str) -> Optional[str]:
    """Return the best matching document type string, or None."""
    q = text.lower()
    for doc_type, keywords in _DOCUMENT_TYPES:
        if any(kw in q for kw in keywords):
            return doc_type
    return None


def _extract_auth_type(text: str) -> Optional[str]:
    """Return the best matching authorization type string, or None."""
    q = text.lower()
    for auth_type, keywords in _AUTH_TYPES:
        if any(kw in q for kw in keywords):
            return auth_type
    return None


def _extract_purpose(text: str) -> Optional[str]:
    """Extract purpose from document request context."""
    m = re.search(
        r"(?:for|purpose[:\s]+|needed for|to use for|to be used for)\s+(.+?)(?:\.|$)",
        text,
        re.IGNORECASE,
    )
    if m:
        return m.group(1).strip().rstrip(".")
    return None


def extract_draft_fields(
    question: str, draft_type: str
) -> tuple[Optional[dict], list[str]]:
    """
    Extract structured fields from the user's question using local heuristics.

    Returns:
        (fields_dict, missing_fields_list)

    Rules:
    - For structured types (LEAVE_REQUEST, LOAN_REQUEST, AUTHORIZATION_REQUEST,
      DOCUMENT_REQUEST): always returns a dict with all expected keys; missing
      values are None, and their key names are added to missing_fields.
    - For IMPROVE_TEXT: returns (None, []) — no structured fields apply.
    - Never calculates working days, validates dates, checks leave balance,
      determines eligibility, or applies any business rule.
    - Dates are returned as raw strings when ambiguous.
    """
    if draft_type == _TYPE_IMPROVE:
        return None, []

    if draft_type == _TYPE_LEAVE:
        leave_type = _extract_leave_type(question)
        dates = _extract_dates(question)
        start_date = dates[0] if len(dates) > 0 else None
        end_date = dates[1] if len(dates) > 1 else None
        reason = _extract_reason(question)

        fields: dict = {
            "leaveType": leave_type,
            "startDate": start_date,
            "endDate": end_date,
            "reason": reason,
        }
        missing: list[str] = []
        if leave_type is None:
            missing.append("leaveType")
        if start_date is None:
            missing.append("startDate")
        if end_date is None:
            missing.append("endDate")
        if reason is None:
            missing.append("reason")
        return fields, missing

    if draft_type == _TYPE_LOAN:
        amount = _extract_amount(question)
        reason = _extract_reason(question)

        fields = {
            "amount": amount,
            "reason": reason,
        }
        missing = []
        if amount is None:
            missing.append("amount")
        if reason is None:
            missing.append("reason")
        return fields, missing

    if draft_type == _TYPE_AUTH:
        dates = _extract_dates(question)
        date_val = dates[0] if dates else None
        from_time, to_time = _extract_time_range(question)
        auth_type = _extract_auth_type(question)
        reason = _extract_reason(question)

        fields = {
            "authorizationType": auth_type,
            "date": date_val,
            "fromTime": from_time,
            "toTime": to_time,
            "reason": reason,
        }
        missing = []
        # authorizationType is optional metadata — not required, not added to missing
        if date_val is None:
            missing.append("date")
        if from_time is None:
            missing.append("fromTime")
        if to_time is None:
            missing.append("toTime")
        if reason is None:
            missing.append("reason")
        return fields, missing

    if draft_type == _TYPE_DOC:
        doc_type = _extract_document_type(question)
        purpose = _extract_purpose(question)

        fields = {
            "documentType": doc_type,
            "purpose": purpose,
            "extraDetails": None,
        }
        missing = []
        if doc_type is None:
            missing.append("documentType")
        if purpose is None:
            missing.append("purpose")
        return fields, missing

    # Fallback: unknown type — treat as IMPROVE_TEXT (no structured fields)
    return None, []


# ---------------------------------------------------------------------------
# Local template drafts  (used when Gemini is disabled or fails)
# ---------------------------------------------------------------------------

_REVIEW_DISCLAIMER = (
    "\n\n⚠️ Please review and personalise this draft before submitting. "
    "The assistant cannot submit or send anything on your behalf."
)

_LOCAL_TEMPLATES: dict[str, str] = {
    _TYPE_LEAVE: (
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
    _TYPE_LOAN: (
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
    _TYPE_AUTH: (
        "Subject: Authorization Request — [Your Name]\n\n"
        "Dear [Manager / HR Team],\n\n"
        "I am writing to request authorization for [describe the action or access needed].\n\n"
        "Reason: [Explain briefly why the authorization is needed and how it relates to your role].\n\n"
        "I understand this request will be reviewed by the appropriate authority and I am "
        "available to provide any additional information required.\n\n"
        "Thank you.\n\n"
        "Regards,\n[Your Name]"
    ),
    _TYPE_DOC: (
        "Subject: Document Request — [Document Name]\n\n"
        "Dear HR Team,\n\n"
        "I am writing to formally request [name of document, e.g., employment certificate, "
        "payslip, experience letter].\n\n"
        "Purpose: [State the reason, e.g., bank loan application, visa processing, personal records].\n\n"
        "Please let me know if any additional information is required to process this request.\n\n"
        "Thank you.\n\n"
        "Regards,\n[Your Name]"
    ),
    _TYPE_IMPROVE: (
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

# Keep legacy keys mapping (old snake_case internal names -> new constants)
# so that any residual callers using the old string keys still resolve correctly.
_LOCAL_TEMPLATES["leave_request"] = _LOCAL_TEMPLATES[_TYPE_LEAVE]
_LOCAL_TEMPLATES["loan_justification"] = _LOCAL_TEMPLATES[_TYPE_LOAN]
_LOCAL_TEMPLATES["authorization"] = _LOCAL_TEMPLATES[_TYPE_AUTH]
_LOCAL_TEMPLATES["document_request"] = _LOCAL_TEMPLATES[_TYPE_DOC]
_LOCAL_TEMPLATES["improve_text"] = _LOCAL_TEMPLATES[_TYPE_IMPROVE]


def _local_draft(question: str, draft_type: str) -> ChatResponse:
    """Build a ChatResponse with a local template draft and extracted structured fields."""
    template = _LOCAL_TEMPLATES.get(draft_type, _LOCAL_TEMPLATES[_TYPE_LEAVE])
    draft_text = template + _REVIEW_DISCLAIMER

    draft_fields, missing_fields = extract_draft_fields(question, draft_type)

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
        draftType=draft_type,
        draftFields=draft_fields,
        missingFields=missing_fields,
    )


# ---------------------------------------------------------------------------
# Gemini drafting prompt  (Phase 3.1: now requests structuredFields)
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
8. For structuredFields: extract ONLY values explicitly stated by the user.
   Use null (JSON null) for any field the user did not mention.
   Never guess, infer, or invent field values.
   Never calculate working days, leave balance, repayment months, salary, eligibility,
   overlap, or approval rules.

SUPPORTED LEAVE TYPES (only these — never use others):
  ANNUAL | SICK | UNPAID | MATERNITY | PATERNITY
  If the user says "emergency" or "family emergency", set leaveType to null.

structuredFields schema by draftType:
  LEAVE_REQUEST:
    { "leaveType": null, "startDate": null, "endDate": null, "reason": null }
  LOAN_REQUEST:
    { "amount": null, "reason": null }
  AUTHORIZATION_REQUEST:
    { "authorizationType": null, "date": null, "fromTime": null, "toTime": null, "reason": null }
  DOCUMENT_REQUEST:
    { "documentType": null, "purpose": null, "extraDetails": null }
  IMPROVE_TEXT:
    structuredFields must be null (no structured fields for text improvement).

Dates: return raw strings as stated by the user (e.g. "May 12", "tomorrow", "next Monday").
Times: normalize to HH:MM format when safely possible (e.g. "10h" -> "10:00").
Amounts: include the currency unit if stated (e.g. "2000 TND").

JSON format (respond with exactly this structure — no extra keys):
{
  "answer": "<one sentence describing what the draft covers>",
  "draft": "<the full draft text with placeholders where real data is missing>",
  "disclaimer": "Please review and personalise this draft before submitting. The assistant cannot submit anything on your behalf.",
  "structuredFields": { <fields per schema above, or null for IMPROVE_TEXT> }
}
"""


def _build_drafting_user_message(question: str, draft_type: str) -> str:
    return (
        f"Draft type: {draft_type}\n"
        f"User request: {question}\n\n"
        "Write a professional draft following the rules above. "
        "Use placeholders like [date], [reason], [amount] wherever specific details are missing. "
        "For structuredFields, extract only what the user explicitly stated; use null for the rest."
    )


# ---------------------------------------------------------------------------
# Gemini call for drafting
# ---------------------------------------------------------------------------

def _call_gemini_for_draft(
    question: str, draft_type: str
) -> Optional[ChatResponse]:
    """
    Call Gemini for drafting. Returns a ChatResponse with draft and structured
    fields populated, or None on any failure.

    If Gemini returns structuredFields, those are used directly.
    If Gemini omits structuredFields or returns invalid structured data,
    the local extractor is used as a fallback for the structured fields
    (but Gemini's draft text is still used).
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
            "maxOutputTokens": 800,
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

        # --- Structured fields from Gemini ---
        gemini_structured = parsed.get("structuredFields")

        draft_fields, missing_fields = _resolve_structured_fields(
            question=question,
            draft_type=draft_type,
            gemini_structured=gemini_structured,
        )

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
            draftType=draft_type,
            draftFields=draft_fields,
            missingFields=missing_fields,
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


def _resolve_structured_fields(
    question: str,
    draft_type: str,
    gemini_structured: Optional[dict],
) -> tuple[Optional[dict], list[str]]:
    """
    Determine the final draftFields and missingFields.

    Strategy:
    1. If draft_type is IMPROVE_TEXT: always (None, []).
    2. If Gemini provided a dict with at least one expected key: use Gemini's
       structured data. Compute missingFields from null/missing values.
    3. Otherwise: fall back to local extraction.

    Never invents values. Never applies business rules.
    """
    if draft_type == _TYPE_IMPROVE:
        return None, []

    expected_keys = _expected_keys_for_type(draft_type)

    if isinstance(gemini_structured, dict) and any(
        k in gemini_structured for k in expected_keys
    ):
        # Use Gemini's structured output. Fill in any missing expected keys with None.
        fields: dict = {}
        missing: list[str] = []
        for key in expected_keys:
            val = gemini_structured.get(key)  # None if absent
            fields[key] = val
            if val is None:
                missing.append(key)
        return fields, missing

    # Gemini omitted structuredFields or returned invalid data — local extractor
    logger.debug(
        "Gemini structuredFields absent or invalid for type=%s; using local extractor.",
        draft_type,
    )
    return extract_draft_fields(question, draft_type)


def _expected_keys_for_type(draft_type: str) -> list[str]:
    """Return the ordered list of expected field keys for a given draft type."""
    _EXPECTED: dict[str, list[str]] = {
        _TYPE_LEAVE: ["leaveType", "startDate", "endDate", "reason"],
        _TYPE_LOAN: ["amount", "reason"],
        _TYPE_AUTH: ["authorizationType", "date", "fromTime", "toTime", "reason"],
        _TYPE_DOC: ["documentType", "purpose", "extraDetails"],
    }
    return _EXPECTED.get(draft_type, [])


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def get_draft_response(request: ChatRequest) -> Optional[ChatResponse]:
    """
    If the user's question is a drafting request, return a ChatResponse with
    draft text and structured fields populated. Returns None if no drafting
    intent is detected.

    Pipeline:
      1. Detect intent   — return None immediately if not a drafting question.
      2. Classify type   — determine draftType.
      3. Try Gemini      — returns source="external_ai" on success, with structured
                           fields from Gemini (or local extractor fallback for fields).
      4. Local template  — always succeeds; source="local_rules" with local
                           extracted structured fields.
    """
    if not detect_drafting_intent(request.question):
        return None

    draft_type = _classify_draft_type(request.question)
    logger.debug("Drafting intent detected. draftType=%s", draft_type)

    # Attempt AI-generated draft first
    gemini_result = _call_gemini_for_draft(request.question, draft_type)
    if gemini_result is not None:
        return gemini_result

    # Gemini disabled or failed — use local template (never crashes)
    return _local_draft(request.question, draft_type)
