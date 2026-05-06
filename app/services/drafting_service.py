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
  - Always includes a review disclaimer in the response (except IMPROVE_TEXT).
  - source="external_ai"  when Gemini produces the draft text.
  - source="local_rules"  when a local template is used.

The caller (assistant_service.py) must NOT run sanitize_response on drafting
responses — drafting responses carry no relatedPages and need no route-stripping.
"""

import json
import logging
import re
from datetime import date, datetime, timedelta
from typing import Optional

import httpx

from app.config import settings
from app.schemas import ChatRequest, ChatResponse

logger = logging.getLogger(__name__)

_GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta/models"

# ---------------------------------------------------------------------------
# Drafting intent detection
# ---------------------------------------------------------------------------

_DRAFT_VERBS: list[str] = [
    "draft",
    "write",
    "compose",
    "help me write",
    "help me draft",
    "help me compose",
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
    "i want annual",
    "i want sick",
    "i want unpaid",
    "i want maternity",
    "i want paternity",
    "i want to take",
    "i need annual",
    "i need sick",
    "i need unpaid",
    "i need maternity",
    "i need paternity",
    "i need to take",
    "i would like annual",
    "i would like sick",
    "i would like unpaid",
    "i would like maternity",
    "i would like paternity",
    "i would like to take",
    "i need",
    "i want",
    "i will be",
    "i am taking",
    "i am going to take",
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
    "time off",
    "day off",
    "days off",
    "vacation",
    "away from work",
    "away from the office",
]


def detect_drafting_intent(question: str) -> bool:
    q = question.lower()
    has_verb = any(verb in q for verb in _DRAFT_VERBS)
    has_subject = any(subj in q for subj in _DRAFT_SUBJECTS)
    return has_verb and has_subject


# Internal type constants
_TYPE_LEAVE = "LEAVE_REQUEST"
_TYPE_LOAN = "LOAN_REQUEST"
_TYPE_AUTH = "AUTHORIZATION_REQUEST"
_TYPE_DOC = "DOCUMENT_REQUEST"
_TYPE_IMPROVE = "IMPROVE_TEXT"

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

# Improve-text leading verbs — checked FIRST in _classify_draft_type.
# When the message STARTS with one of these, it's always IMPROVE_TEXT,
# regardless of what words appear in the body being improved.
_IMPROVE_LEADING_VERBS: list[str] = [
    "rephrase",
    "rewrite",
    "improve",
    "polish",
    "refine",
    "correct this",
    "make this better",
    "make it better",
    "make this more professional",
    "make this message more professional",
    "make it more professional",
    "make this professional",
]


def _classify_draft_type(question: str) -> str:
    """
    Return the draftType constant for this question.

    Priority order:
      0. IMPROVE_TEXT (leading verb) — checked FIRST.
      1. DOCUMENT_REQUEST
      2. AUTHORIZATION_REQUEST
      3. LOAN_REQUEST
      4. IMPROVE_TEXT (anywhere)
      5. LEAVE_REQUEST (default)
    """
    q = question.lower().strip()

    # 0. Leading improvement verb wins over everything
    q_start = q[:40]
    if any(q_start.startswith(v) for v in _IMPROVE_LEADING_VERBS):
        return _TYPE_IMPROVE

    if any(sig in q for sig in _STRONG_DOCUMENT_SIGNALS):
        return _TYPE_DOC

    if "authorization" in q:
        return _TYPE_AUTH

    if "loan" in q:
        return _TYPE_LOAN

    if (
        "improve" in q
        or "professional" in q
        or "rephrase" in q
        or "rewrite" in q
        or "polish" in q
        or "refine" in q
    ):
        return _TYPE_IMPROVE

    return _TYPE_LEAVE


# ---------------------------------------------------------------------------
# Structured field extraction
# ---------------------------------------------------------------------------

_SUPPORTED_LEAVE_TYPES: dict[str, list[str]] = {
    "ANNUAL": ["annual", "paid leave", "vacation", "holiday", "yearly leave"],
    "SICK": ["sick", "medical", "illness", "health", "unwell", "doctor"],
    "UNPAID": ["unpaid"],
    "MATERNITY": ["maternity"],
    "PATERNITY": ["paternity"],
}

_DOCUMENT_TYPES: list[tuple[str, list[str]]] = [
    ("salary certificate", ["salary certificate", "salary cert"]),
    ("employment certificate", ["employment certificate", "work certificate", "employment cert"]),
    ("payslip", ["payslip", "pay slip", "pay stub"]),
    ("experience letter", ["experience letter", "experience cert"]),
    ("work certificate", ["work cert"]),
    ("attestation", ["attestation"]),
]

_AUTH_TYPES: list[tuple[str, list[str]]] = [
    ("departure", ["early departure", "departure", "leave early", "leave the office"]),
    ("late arrival", ["late arrival", "arrive late", "coming in late", "late to work"]),
    ("external", ["external", "outside the office", "off-site"]),
    ("medical", ["medical", "doctor", "hospital", "clinic"]),
]

# ---------------------------------------------------------------------------
# Date normalization
# ---------------------------------------------------------------------------

_MONTH_MAP: dict[str, int] = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
}

_ISO_RE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})$")

_MONTH_DAY_RE = re.compile(
    r"^(?:"
    r"(january|february|march|april|may|june|july|august|september|october|november|december)"
    r"\s+(\d{1,2})(?:st|nd|rd|th)?"
    r"|(\d{1,2})(?:st|nd|rd|th)?\s+"
    r"(january|february|march|april|may|june|july|august|september|october|november|december)"
    r")$",
    re.IGNORECASE,
)

_MONTH_DAY_YEAR_RE = re.compile(
    r"^(?:"
    r"(january|february|march|april|may|june|july|august|september|october|november|december)"
    r"\s+(\d{1,2})(?:st|nd|rd|th)?\s+(\d{4})"
    r"|(\d{1,2})(?:st|nd|rd|th)?\s+"
    r"(january|february|march|april|may|june|july|august|september|october|november|december)"
    r"\s+(\d{4})"
    r")$",
    re.IGNORECASE,
)

_NUMERIC_DMY_RE = re.compile(r"^(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{4})$")

_RELATIVE_TODAY_RE = re.compile(r"^today$", re.IGNORECASE)
_RELATIVE_TOMORROW_RE = re.compile(r"^tomorrow$", re.IGNORECASE)
_RELATIVE_AFTER_TOMORROW_RE = re.compile(
    r"^(?:day\s+after\s+tomorrow|after\s+tomorrow)$", re.IGNORECASE
)
_RELATIVE_IN_N_DAYS_RE = re.compile(r"^in\s+(\d+)\s+days?$", re.IGNORECASE)

_VAGUE_RELATIVE_RE = re.compile(
    r"^(?:next\s+week|next\s+month|soon|later|sometime|sometime\s+next|in\s+a\s+few|in\s+a\s+week)",
    re.IGNORECASE,
)


def normalize_date_to_iso(raw: Optional[str]) -> Optional[str]:
    if not raw:
        return None
    raw = raw.strip()

    m = _ISO_RE.match(raw)
    if m:
        try:
            date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
            return raw
        except ValueError:
            return None

    if _VAGUE_RELATIVE_RE.match(raw):
        return None

    today_date = date.today()
    if _RELATIVE_TODAY_RE.match(raw):
        return today_date.strftime("%Y-%m-%d")
    if _RELATIVE_TOMORROW_RE.match(raw):
        return (today_date + timedelta(days=1)).strftime("%Y-%m-%d")
    if _RELATIVE_AFTER_TOMORROW_RE.match(raw):
        return (today_date + timedelta(days=2)).strftime("%Y-%m-%d")
    m = _RELATIVE_IN_N_DAYS_RE.match(raw)
    if m:
        n = int(m.group(1))
        if 1 <= n <= 365:
            return (today_date + timedelta(days=n)).strftime("%Y-%m-%d")
        return None

    m = _NUMERIC_DMY_RE.match(raw)
    if m:
        day, month, year = int(m.group(1)), int(m.group(2)), int(m.group(3))
        try:
            return date(year, month, day).strftime("%Y-%m-%d")
        except ValueError:
            return None

    m = _MONTH_DAY_YEAR_RE.match(raw)
    if m:
        if m.group(1):
            month_name, day, year = m.group(1).lower(), int(m.group(2)), int(m.group(3))
        else:
            month_name, day, year = m.group(5).lower(), int(m.group(4)), int(m.group(6))
        month_num = _MONTH_MAP.get(month_name)
        if month_num is None:
            return None
        try:
            return date(year, month_num, day).strftime("%Y-%m-%d")
        except ValueError:
            return None

    m = _MONTH_DAY_RE.match(raw)
    if m:
        if m.group(1):
            month_name = m.group(1).lower()
            day = int(m.group(2))
        else:
            month_name = m.group(4).lower()
            day = int(m.group(3))
        month_num = _MONTH_MAP.get(month_name)
        if month_num is None:
            return None
        year = today_date.year
        try:
            candidate = date(year, month_num, day)
        except ValueError:
            return None
        if candidate < today_date:
            try:
                candidate = date(year + 1, month_num, day)
            except ValueError:
                return None
        return candidate.strftime("%Y-%m-%d")

    return None


_DATE_PATTERNS = [
    re.compile(r"\b(\d{4}-\d{2}-\d{2})\b"),
    re.compile(r"\b(\d{1,2}[/\-.]\d{1,2}[/\-.]\d{4})\b"),
    re.compile(
        r"\b((?:january|february|march|april|may|june|july|august|september|october|november|december)"
        r"\s+\d{1,2}(?:st|nd|rd|th)?\s+\d{4}"
        r"|\d{1,2}(?:st|nd|rd|th)?\s+"
        r"(?:january|february|march|april|may|june|july|august|september|october|november|december)"
        r"\s+\d{4})\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b((?:january|february|march|april|may|june|july|august|september|october|november|december)"
        r"\s+\d{1,2}(?:st|nd|rd|th)?"
        r"|\d{1,2}(?:st|nd|rd|th)?\s+"
        r"(?:january|february|march|april|may|june|july|august|september|october|november|december))\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(day\s+after\s+tomorrow|after\s+tomorrow|tomorrow|today|in\s+\d+\s+days?)\b",
        re.IGNORECASE,
    ),
]

_TIME_RANGE_PATTERN = re.compile(
    r"(?:from\s+)?(\d{1,2}(?::\d{2})?(?:h\d{0,2})?(?:\s*[ap]m)?)"
    r"\s+(?:to|until|till|-)\s+"
    r"(\d{1,2}(?::\d{2})?(?:h\d{0,2})?(?:\s*[ap]m)?)",
    re.IGNORECASE,
)

_REASON_TRIGGERS = re.compile(
    r"(?:because|for|reason:|reason is|due to|since|as|to attend|to go to)\s+(.+?)(?:\.|$)",
    re.IGNORECASE,
)

_AMOUNT_PATTERN = re.compile(
    r"\b(\d[\d\s,\.]*(?:\s*(?:TND|EUR|USD|DZD|MAD|€|\$|£))?)\b",
    re.IGNORECASE,
)


def _extract_dates(text: str) -> list[str]:
    found: list[str] = []
    for pattern in _DATE_PATTERNS:
        for m in pattern.finditer(text):
            val = m.group(1) if m.lastindex and m.lastindex >= 1 else m.group(0)
            val = val.strip()
            if val and val not in found:
                found.append(val)
    return found


def _normalize_time(raw: str) -> str:
    raw = raw.strip().lower()
    m = re.match(r"(\d{1,2})(?::(\d{2})|h(\d{2})?)?(?:\s*([ap]m))?$", raw, re.IGNORECASE)
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
    m = _TIME_RANGE_PATTERN.search(text)
    if m:
        return _normalize_time(m.group(1)), _normalize_time(m.group(2))
    return None, None


def _extract_reason(text: str) -> Optional[str]:
    m = _REASON_TRIGGERS.search(text)
    if m:
        return m.group(1).strip().rstrip(".")
    return None


def _extract_amount(text: str) -> Optional[str]:
    m = _AMOUNT_PATTERN.search(text)
    if m:
        val = m.group(1).strip().rstrip(",.")
        if re.search(r"\d", val):
            return val
    return None


def _extract_leave_type(text: str) -> Optional[str]:
    q = text.lower()
    for leave_type, keywords in _SUPPORTED_LEAVE_TYPES.items():
        if any(kw in q for kw in keywords):
            return leave_type
    return None


def _extract_document_type(text: str) -> Optional[str]:
    q = text.lower()
    for doc_type, keywords in _DOCUMENT_TYPES:
        if any(kw in q for kw in keywords):
            return doc_type
    return None


def _extract_auth_type(text: str) -> Optional[str]:
    q = text.lower()
    for auth_type, keywords in _AUTH_TYPES:
        if any(kw in q for kw in keywords):
            return auth_type
    return None


def _extract_purpose(text: str) -> Optional[str]:
    m = re.search(
        r"(?:for|purpose[:\s]+|needed for|to use for|to be used for)\s+(.+?)(?:\.|$)",
        text, re.IGNORECASE,
    )
    if m:
        return m.group(1).strip().rstrip(".")
    return None


# ---------------------------------------------------------------------------
# Leave date range extraction
# ---------------------------------------------------------------------------

_RANGE_FROM_TO_RE = re.compile(
    r"(?:from\s+)?(.+?)\s+(?:to|until|till|--)\s+(.+?)(?:\s+(?:for|because|reason|to\s+attend)|$)",
    re.IGNORECASE,
)

_START_PLUS_DURATION_RE = re.compile(
    r"(?:starting|from)\s+(.+?)\s+for\s+(\d+)\s+days?", re.IGNORECASE,
)

_DURATION_THEN_START_RE = re.compile(
    r"for\s+(\d+)\s+days?\s+starting\s+(.+?)(?:\s+(?:because|reason|to\s+attend)|$)",
    re.IGNORECASE,
)

_FROM_X_TO_N_DAYS_LATER_RE = re.compile(
    r"from\s+(.+?)\s+(?:to|for)\s+(\d+)\s+days?\s+later", re.IGNORECASE,
)

_UNTIL_ONLY_RE = re.compile(
    r"(?:until|till)\s+(.+?)(?:\s+(?:for|because|reason)|$)", re.IGNORECASE,
)

_SINGLE_LEAVE_DAY_RE = re.compile(
    r"(?:leave|absence)\s+(today|tomorrow|day\s+after\s+tomorrow|after\s+tomorrow|in\s+\d+\s+days?)",
    re.IGNORECASE,
)


def _raw_date_token(text: str) -> Optional[str]:
    text = text.strip()
    for pattern in _DATE_PATTERNS:
        m = pattern.search(text)
        if m:
            val = m.group(1) if m.lastindex and m.lastindex >= 1 else m.group(0)
            return val.strip()
    return None


def _compute_end_from_start_and_duration(start_iso: str, n_days: int) -> Optional[str]:
    try:
        start = date.fromisoformat(start_iso)
        end = start + timedelta(days=n_days - 1)
        return end.strftime("%Y-%m-%d")
    except (ValueError, OverflowError):
        return None


def _compute_end_from_start_plus_n(start_iso: str, n_days: int) -> Optional[str]:
    try:
        start = date.fromisoformat(start_iso)
        end = start + timedelta(days=n_days)
        return end.strftime("%Y-%m-%d")
    except (ValueError, OverflowError):
        return None


def _extract_leave_date_range(question: str) -> tuple[Optional[str], Optional[str]]:
    q = question

    m = _FROM_X_TO_N_DAYS_LATER_RE.search(q)
    if m:
        raw_start = _raw_date_token(m.group(1))
        n = int(m.group(2))
        start_iso = normalize_date_to_iso(raw_start)
        if start_iso:
            return start_iso, _compute_end_from_start_plus_n(start_iso, n)

    m = _START_PLUS_DURATION_RE.search(q)
    if m:
        raw_start = _raw_date_token(m.group(1))
        n = int(m.group(2))
        start_iso = normalize_date_to_iso(raw_start)
        if start_iso:
            return start_iso, _compute_end_from_start_and_duration(start_iso, n)

    m = _DURATION_THEN_START_RE.search(q)
    if m:
        n = int(m.group(1))
        raw_start = _raw_date_token(m.group(2))
        start_iso = normalize_date_to_iso(raw_start)
        if start_iso:
            return start_iso, _compute_end_from_start_and_duration(start_iso, n)

    m = _RANGE_FROM_TO_RE.search(q)
    if m:
        raw_start = _raw_date_token(m.group(1))
        raw_end = _raw_date_token(m.group(2))
        if raw_start and raw_end:
            start_iso = normalize_date_to_iso(raw_start)
            end_iso = normalize_date_to_iso(raw_end)
            if start_iso or end_iso:
                return start_iso, end_iso

    m = _UNTIL_ONLY_RE.search(q)
    if m:
        raw_end = _raw_date_token(m.group(1))
        end_iso = normalize_date_to_iso(raw_end)
        if end_iso:
            return None, end_iso

    m = _SINGLE_LEAVE_DAY_RE.search(q)
    if m:
        iso = normalize_date_to_iso(m.group(1))
        if iso:
            return iso, iso

    raw_dates = _extract_dates(q)
    if len(raw_dates) >= 2:
        return normalize_date_to_iso(raw_dates[0]), normalize_date_to_iso(raw_dates[1])
    if len(raw_dates) == 1:
        return normalize_date_to_iso(raw_dates[0]), None

    return None, None


def extract_draft_fields(
    question: str, draft_type: str
) -> tuple[Optional[dict], list[str]]:
    if draft_type == _TYPE_IMPROVE:
        return None, []

    if draft_type == _TYPE_LEAVE:
        leave_type = _extract_leave_type(question)
        reason = _extract_reason(question)
        start_date, end_date = _extract_leave_date_range(question)
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
        fields = {"amount": amount, "reason": reason}
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
        fields = {"documentType": doc_type, "purpose": purpose, "extraDetails": None}
        missing = []
        if doc_type is None:
            missing.append("documentType")
        if purpose is None:
            missing.append("purpose")
        return fields, missing

    return None, []


# ---------------------------------------------------------------------------
# Local template drafts
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
        # Intentionally empty — IMPROVE_TEXT drafts are built dynamically
        # in _local_draft by extracting the user's actual text and polishing it.
        # This key is kept only for legacy-alias compatibility below.
        ""
    ),
}

_LOCAL_TEMPLATES["leave_request"] = _LOCAL_TEMPLATES[_TYPE_LEAVE]
_LOCAL_TEMPLATES["loan_justification"] = _LOCAL_TEMPLATES[_TYPE_LOAN]
_LOCAL_TEMPLATES["authorization"] = _LOCAL_TEMPLATES[_TYPE_AUTH]
_LOCAL_TEMPLATES["document_request"] = _LOCAL_TEMPLATES[_TYPE_DOC]
_LOCAL_TEMPLATES["improve_text"] = _LOCAL_TEMPLATES[_TYPE_IMPROVE]


# ---------------------------------------------------------------------------
# IMPROVE_TEXT local rewrite helpers
# ---------------------------------------------------------------------------

# Strips the improvement instruction prefix from the user's message, leaving
# only the text they actually want polished.
# e.g. "rephrase this message: I need a day off" -> "I need a day off"
_IMPROVE_PREFIX_RE = re.compile(
    r'^(rephrase|rewrite|improve|fix|polish|refine|correct)'
    r'(?:\s+(?:this|the|my))?'
    r'(?:\s+(?:message|sentence|text|email|paragraph|writing))?'
    r'[\s:]*',
    re.IGNORECASE,
)


def _polish_text(raw: str) -> str:
    """
    Produce a polished professional rewrite of the user's raw text.
    No placeholders. No template scaffolding. No workflow language.
    """
    if not raw:
        return raw

    text = raw.strip()

    # "I need a day off tomorrow" -> "I would like to request a day off tomorrow."
    m = re.match(r'^i need (.+)$', text, re.IGNORECASE)
    if m:
        rest = m.group(1).rstrip('.')
        result = f"I would like to request {rest}."
        return result[0].upper() + result[1:]

    # "I want ..." -> "I would like to ..."
    m = re.match(r'^i want (.+)$', text, re.IGNORECASE)
    if m:
        rest = m.group(1).rstrip('.')
        result = f"I would like to {rest}."
        return result[0].upper() + result[1:]

    # Fallback: capitalise and punctuate
    result = text
    if result:
        result = result[0].upper() + result[1:]
    if result and not result.endswith(('.', '?', '!')):
        result += '.'
    return result


def _local_draft(question: str, draft_type: str) -> ChatResponse:
    """Build a ChatResponse — polished rewrite for IMPROVE_TEXT, template for all others."""
    if draft_type == _TYPE_IMPROVE:
        # Strip the instruction prefix to isolate the text to be polished,
        # then rewrite it naturally. No placeholders. No workflow language.
        body = _IMPROVE_PREFIX_RE.sub('', question.strip()).strip()
        polished = _polish_text(body) if body else question.strip()
        return ChatResponse(
            answer="Here is a polished rewrite of your text.",
            draft=polished,
            warnings=[],
            source="local_rules",
            draftType=_TYPE_IMPROVE,
            draftFields=None,
            missingFields=[],
        )

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
6. For non-IMPROVE_TEXT drafts: end your JSON response with a disclaimer reminding the user to review before submitting.
7. Respond ONLY with valid JSON — no Markdown fences, no extra keys.
8. For structuredFields: extract ONLY values explicitly stated by the user.
   Use null (JSON null) for any field the user did not mention.
   Never guess, infer, or invent field values.

SUPPORTED LEAVE TYPES (only these — never use others):
  ANNUAL | SICK | UNPAID | MATERNITY | PATERNITY
  If the user says "emergency" or "family emergency", set leaveType to null.
  CRITICAL: "time off", "day off", "days off", "away from work", "away from the office"
  are GENERIC absence phrases — they do NOT identify a leave type. Set leaveType to null
  whenever the user uses these phrases without also stating an explicit type keyword.
  Only map to ANNUAL when the user explicitly says "annual leave" or "vacation".
  Only map to SICK when the user explicitly says "sick leave", "sick", "medical", "illness".
  Never infer leave type from generic absence language.

For IMPROVE_TEXT: produce ONLY a polished rewrite of the text the user provided.
  Do NOT produce a leave request template, a loan template, or any HR form.
  Do NOT add placeholders, disclaimers, or "please submit" language.
  structuredFields must be null.

structuredFields schema by draftType:
  LEAVE_REQUEST:    { "leaveType": null, "startDate": null, "endDate": null, "reason": null }
  LOAN_REQUEST:     { "amount": null, "reason": null }
  AUTHORIZATION_REQUEST: { "authorizationType": null, "date": null, "fromTime": null, "toTime": null, "reason": null }
  DOCUMENT_REQUEST: { "documentType": null, "purpose": null, "extraDetails": null }
  IMPROVE_TEXT:     structuredFields must be null.

Dates for LEAVE_REQUEST: normalize to ISO yyyy-MM-dd.
Times: normalize to HH:MM when safely possible.
Amounts: include currency if stated.

JSON format:
{
  "answer": "<one sentence>",
  "draft": "<draft text or polished rewrite>",
  "disclaimer": "Please review before submitting.",
  "structuredFields": { ... or null }
}
"""


def _build_drafting_user_message(question: str, draft_type: str) -> str:
    if draft_type == _TYPE_IMPROVE:
        return (
            f"Draft type: {draft_type}\n"
            f"User request: {question}\n\n"
            "IMPORTANT: The user wants you to rephrase or improve the text they provided. "
            "Do NOT produce a leave request template, a loan template, or any HR request form. "
            "Do NOT add placeholders like [Your Name] or workflow language like 'please submit'. "
            "Produce only a polished rewrite of what the user wrote. "
            "structuredFields must be null."
        )
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
        "system_instruction": {"parts": [{"text": _DRAFTING_SYSTEM_PROMPT}]},
        "contents": [
            {
                "role": "user",
                "parts": [{"text": _build_drafting_user_message(question, draft_type)}],
            }
        ],
        "generationConfig": {"temperature": 0.4, "maxOutputTokens": 800},
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

        # Append the review disclaimer for non-IMPROVE_TEXT drafts only.
        # IMPROVE_TEXT is a polished rewrite — no workflow language.
        if draft_type != _TYPE_IMPROVE:
            if "review" not in draft_text.lower() and "disclaimer" not in draft_text.lower():
                draft_text += _REVIEW_DISCLAIMER

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

    Safety guard for LEAVE_REQUEST: leaveType returned by Gemini is accepted ONLY
    when the question contains an explicit keyword for that type.
    """
    if draft_type == _TYPE_IMPROVE:
        return None, []

    expected_keys = _expected_keys_for_type(draft_type)

    if isinstance(gemini_structured, dict) and any(
        k in gemini_structured for k in expected_keys
    ):
        fields: dict = {}
        missing: list[str] = []
        for key in expected_keys:
            val = gemini_structured.get(key)
            if draft_type == _TYPE_LEAVE and key in ("startDate", "endDate") and val is not None:
                val = normalize_date_to_iso(val)
            fields[key] = val
            if val is None:
                missing.append(key)

        # Safety guard: reject Gemini-inferred leaveType for generic absence language
        if draft_type == _TYPE_LEAVE and fields.get("leaveType") is not None:
            if not _is_explicit_leave_type(question, fields["leaveType"]):
                fields["leaveType"] = None
                if "leaveType" not in missing:
                    missing.append("leaveType")

        return fields, missing

    logger.debug(
        "Gemini structuredFields absent or invalid for type=%s; using local extractor.",
        draft_type,
    )
    return extract_draft_fields(question, draft_type)


# ---------------------------------------------------------------------------
# Explicit leave-type keyword check
# ---------------------------------------------------------------------------

_EXPLICIT_LEAVE_TYPE_KEYWORDS: dict[str, list[str]] = {
    "ANNUAL":    ["annual leave", "annual", "vacation", "paid leave", "holiday", "yearly leave"],
    "SICK":      ["sick leave", "sick", "medical", "illness", "health", "unwell", "doctor"],
    "UNPAID":    ["unpaid leave", "unpaid"],
    "MATERNITY": ["maternity leave", "maternity"],
    "PATERNITY": ["paternity leave", "paternity"],
}


def _is_explicit_leave_type(question: str, leave_type: str) -> bool:
    q = question.lower()
    keywords = _EXPLICIT_LEAVE_TYPE_KEYWORDS.get(leave_type, [])
    return any(kw in q for kw in keywords)


def _expected_keys_for_type(draft_type: str) -> list[str]:
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
    if not detect_drafting_intent(request.question):
        return None

    draft_type = _classify_draft_type(request.question)
    logger.debug("Drafting intent detected. draftType=%s", draft_type)

    gemini_result = _call_gemini_for_draft(request.question, draft_type)
    if gemini_result is not None:
        return gemini_result

    return _local_draft(request.question, draft_type)
