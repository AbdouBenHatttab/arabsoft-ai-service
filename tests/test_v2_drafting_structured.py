"""
tests/test_v2_drafting_structured.py
--------------------------------------
V2 Phase 3.1: Structured Draft Extraction tests.

All external HTTP calls are mocked — no real API key or network required.
The conftest autouse fixture disables Gemini for every test. Tests that
need Gemini enabled patch it themselves inside their own `with patch(...)` block.

Coverage:
  1.  Non-drafting response has draftType=None, draftFields=None, missingFields=[].
  2.  Leave with annual keyword, startDate, endDate, reason all extracted.
  3.  Emergency/family-emergency does NOT produce unsupported EMERGENCY leave type.
  4.  Sick leave with one date has endDate=null and "endDate" in missingFields.
  5.  Leave request with no leave type: leaveType=null, "leaveType" in missingFields.
  6.  Loan request extracts amount and reason.
  7.  Loan request without amount: "amount" in missingFields.
  8.  Authorization request extracts absenceDate, fromTime, toTime, reason (V3.2: TIME_PERMISSION sub-type).
  9.  Document request extracts documentType and purpose.
  10. improve_text gives draftType=IMPROVE_TEXT, draftFields=None, missingFields=[].
  11. Gemini enabled with structuredFields: draftFields populated from Gemini.
  12. Gemini enabled with null structuredFields values: missingFields computed correctly.
  13. Gemini omitted structuredFields: local extractor used as fallback.
  14. Gemini disabled: local extractor used.
  15. relatedPages remains [] for all drafting responses.
  16. Refusal wins over drafting for submit/approve/change workflow requests.
  17. draftType is set for all structured drafting responses.
  18. draftFields has stable shape (all expected keys present) for structured types.
  19. missingFields is always a list, never null.
  20. LOAN_REQUEST draftFields always has "amount" and "reason" keys.
  21. AUTHORIZATION_REQUEST draftFields always has all five expected keys.
  22. DOCUMENT_REQUEST draftFields always has "documentType", "purpose", "extraDetails".
  23. Local extractor unit — extract_draft_fields LEAVE_REQUEST with full details.
  24. Local extractor unit — extract_draft_fields LOAN_REQUEST with amount and reason.
  25. Local extractor unit — extract_draft_fields AUTHORIZATION_REQUEST time range.
  26. Local extractor unit — extract_draft_fields IMPROVE_TEXT returns (None, []).
  27. Gemini response missing structuredFields key entirely uses local fallback.
  28. Non-drafting route responses still have draftType=None.
"""

import json
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

import httpx

from app.main import app
from app.schemas import ChatRequest, ContextInfo
from app.services.drafting_service import (
    detect_drafting_intent,
    extract_draft_fields,
    get_draft_response,
    normalize_date_to_iso,
)

client = TestClient(app)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def post_chat(role: str, question: str) -> dict:
    return client.post(
        "/assistant/chat",
        json={"role": role, "question": question, "context": {}},
    ).json()


def _make_request(question: str, role: str = "EMPLOYEE") -> ChatRequest:
    return ChatRequest(role=role, question=question, context=ContextInfo())


def _mock_gemini_settings(mock_settings, *, enabled: bool = True, api_key: str = "test-key"):
    mock_settings.gemini_enabled = enabled
    mock_settings.gemini_api_key = api_key
    mock_settings.gemini_model = "gemini-2.5-flash"
    mock_settings.gemini_timeout_seconds = 10


def _mock_gemini_http_drafting(mock_http, *, draft_text: str, answer: str, structured_fields):
    """
    Configure mock_http to return a valid Gemini drafting JSON response
    that includes the structuredFields key.
    structured_fields may be a dict or None.
    """
    payload = json.dumps({
        "answer": answer,
        "draft": draft_text,
        "disclaimer": "Please review before submitting.",
        "structuredFields": structured_fields,
    })
    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "candidates": [{"content": {"parts": [{"text": payload}]}}]
    }
    mock_resp.raise_for_status = MagicMock()
    mock_http.return_value.__enter__.return_value.post.return_value = mock_resp


def _mock_gemini_http_no_structured(mock_http, *, draft_text: str, answer: str):
    """Configure mock_http to return a Gemini response WITHOUT structuredFields key."""
    payload = json.dumps({
        "answer": answer,
        "draft": draft_text,
        "disclaimer": "Please review before submitting.",
        # structuredFields key intentionally omitted
    })
    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "candidates": [{"content": {"parts": [{"text": payload}]}}]
    }
    mock_resp.raise_for_status = MagicMock()
    mock_http.return_value.__enter__.return_value.post.return_value = mock_resp


# ===========================================================================
# 1. Non-drafting responses have draftType=None, draftFields=None, missingFields=[]
# ===========================================================================

def test_non_drafting_response_has_no_structured_fields():
    """A plain navigation question must have draftType=None and empty structured fields."""
    data = post_chat("EMPLOYEE", "How do I request a loan?")
    assert data["draftType"] is None
    assert data["draftFields"] is None
    assert data["missingFields"] == []


def test_non_drafting_leave_balance_has_no_structured_fields():
    data = post_chat("EMPLOYEE", "What is my leave balance?")
    assert data["draftType"] is None
    assert data["draftFields"] is None
    assert data["missingFields"] == []


def test_refusal_response_has_no_structured_fields():
    data = post_chat("EMPLOYEE", "approve my leave automatically")
    assert data["source"] == "refusal"
    assert data["draftType"] is None
    assert data["draftFields"] is None
    assert data["missingFields"] == []


# ===========================================================================
# 2. Leave extraction: annual leave, both dates, reason
# ===========================================================================

def test_leave_annual_full_details_extracts_correctly():
    """Annual leave with both dates and a reason — all fields populated, missingFields=[]."""
    data = post_chat(
        "EMPLOYEE",
        "Help me draft a leave request for annual leave from May 12 to May 14 "
        "because I have a family appointment.",
    )
    assert data["draftType"] == "LEAVE_REQUEST"
    fields = data["draftFields"]
    assert fields is not None
    assert fields["leaveType"] == "ANNUAL"
    assert fields["startDate"] is not None
    assert fields["endDate"] is not None
    assert fields["reason"] is not None
    assert "appointment" in fields["reason"].lower() or "family" in fields["reason"].lower()
    assert data["missingFields"] == []


def test_leave_annual_keyword_sets_annual_type():
    """The word 'annual' in the question must produce leaveType=ANNUAL."""
    fields, _ = extract_draft_fields(
        "Help me draft a leave request for annual leave from June 1 to June 3 "
        "because of a personal matter.",
        "LEAVE_REQUEST",
    )
    assert fields["leaveType"] == "ANNUAL"


def test_leave_sick_keyword_sets_sick_type():
    """The word 'sick' must produce leaveType=SICK."""
    fields, _ = extract_draft_fields(
        "Write a leave request reason for sick leave starting tomorrow because I am unwell.",
        "LEAVE_REQUEST",
    )
    assert fields["leaveType"] == "SICK"


def test_leave_reason_extracted_after_because():
    """Reason clause after 'because' is extracted."""
    fields, _ = extract_draft_fields(
        "Draft a leave request for annual leave from May 5 to May 7 because of a dental appointment.",
        "LEAVE_REQUEST",
    )
    assert fields["reason"] is not None
    assert "dental" in fields["reason"].lower() or "appointment" in fields["reason"].lower()


# ===========================================================================
# 3. Emergency / family emergency does NOT produce unsupported leave type
# ===========================================================================

def test_emergency_does_not_produce_emergency_leave_type():
    """
    'emergency' must NOT map to leaveType=EMERGENCY (unsupported).
    leaveType must be None and "leaveType" must appear in missingFields.
    """
    fields, missing = extract_draft_fields(
        "Help me draft a leave request for a family emergency from May 10 to May 11.",
        "LEAVE_REQUEST",
    )
    assert fields["leaveType"] is None
    assert "leaveType" in missing


def test_emergency_via_api_leave_type_is_null():
    data = post_chat(
        "EMPLOYEE",
        "Help me draft a leave request for a family emergency starting tomorrow.",
    )
    assert data["draftType"] == "LEAVE_REQUEST"
    assert data["draftFields"]["leaveType"] is None
    assert "leaveType" in data["missingFields"]


def test_family_emergency_leave_type_null_and_missing():
    """Family emergency must keep leaveType=null and add leaveType to missingFields."""
    fields, missing = extract_draft_fields(
        "Write a leave request for a family emergency from June 2 to June 3 "
        "because of an urgent family situation.",
        "LEAVE_REQUEST",
    )
    assert fields["leaveType"] is None
    assert "leaveType" in missing


# ===========================================================================
# 4. Sick leave with one date: endDate missing
# ===========================================================================

def test_sick_leave_one_date_enddate_in_missing():
    """Only one date provided: startDate extracted, endDate=None, 'endDate' in missingFields."""
    fields, missing = extract_draft_fields(
        "Write a leave request for sick leave starting tomorrow because I have a medical appointment.",
        "LEAVE_REQUEST",
    )
    assert fields["leaveType"] == "SICK"
    assert fields["startDate"] is not None
    assert fields["endDate"] is None
    assert "endDate" in missing


def test_sick_leave_no_date_both_dates_missing():
    """No dates at all: both startDate and endDate in missingFields."""
    fields, missing = extract_draft_fields(
        "Help me draft a leave request for sick leave because I have a doctor's appointment.",
        "LEAVE_REQUEST",
    )
    assert "startDate" in missing
    assert "endDate" in missing


# ===========================================================================
# 5. Leave request with missing leave type
# ===========================================================================

def test_leave_request_missing_type_leaveType_null():
    """No recognizable leave type keyword: leaveType=None, 'leaveType' in missingFields."""
    fields, missing = extract_draft_fields(
        "Help me draft a leave request from May 20 to May 22 because of a personal matter.",
        "LEAVE_REQUEST",
    )
    assert fields["leaveType"] is None
    assert "leaveType" in missing


# ===========================================================================
# 6. Loan request: amount and reason extracted
# ===========================================================================

def test_loan_request_extracts_amount_and_reason():
    """Amount and reason must both be extracted."""
    fields, missing = extract_draft_fields(
        "Help me request a loan for 2000 TND because of family expenses.",
        "LOAN_REQUEST",
    )
    assert fields["amount"] is not None
    assert "2000" in fields["amount"]
    assert fields["reason"] is not None
    assert "family" in fields["reason"].lower() or "expenses" in fields["reason"].lower()
    assert missing == []


def test_loan_request_via_api_extracts_amount():
    data = post_chat(
        "EMPLOYEE",
        "Help me request a loan for 2000 TND because of family expenses.",
    )
    assert data["draftType"] == "LOAN_REQUEST"
    assert data["draftFields"]["amount"] is not None
    assert "2000" in data["draftFields"]["amount"]
    assert data["draftFields"]["reason"] is not None
    assert data["missingFields"] == []


def test_loan_with_currency_extracted():
    fields, missing = extract_draft_fields(
        "Write a loan justification for 500 EUR because of home repairs.",
        "LOAN_REQUEST",
    )
    assert fields["amount"] is not None
    assert "500" in fields["amount"]


# ===========================================================================
# 7. Loan request without amount
# ===========================================================================

def test_loan_request_without_amount_adds_to_missing():
    """No amount mentioned: 'amount' must be in missingFields."""
    fields, missing = extract_draft_fields(
        "Write a professional loan justification because I need it for medical treatment.",
        "LOAN_REQUEST",
    )
    assert fields["amount"] is None
    assert "amount" in missing


# ===========================================================================
# 8. Authorization request: date, fromTime, toTime, reason
# ===========================================================================

def test_authorization_extracts_time_range():
    """'from 10 to 12' must produce fromTime='10:00' and toTime='12:00'."""
    fields, _ = extract_draft_fields(
        "Help me request authorization tomorrow from 10 to 12 for an appointment.",
        "AUTHORIZATION_REQUEST",
    )
    assert fields["fromTime"] == "10:00"
    assert fields["toTime"] == "12:00"


def test_authorization_extracts_date_and_reason():
    # V3.2: date field renamed to absenceDate for TIME_PERMISSION sub-type.
    # absenceDate is now normalized to ISO yyyy-MM-dd when the date is unambiguous
    # ("tomorrow" normalizes to today+1). Check it's a non-None string.
    import re as _re
    fields, missing = extract_draft_fields(
        "Draft an authorization request tomorrow from 10 to 12 for a doctor appointment.",
        "AUTHORIZATION_REQUEST",
    )
    assert fields["absenceDate"] is not None
    # absenceDate is either an ISO date (tomorrow normalized) or the raw token "tomorrow"
    iso_re = _re.compile(r"^\d{4}-\d{2}-\d{2}$")
    absence = fields["absenceDate"]
    assert iso_re.match(absence) or "tomorrow" in absence.lower(), (
        f"absenceDate should be ISO date or 'tomorrow', got: {absence!r}"
    )
    assert fields["reason"] is not None
    assert "doctor" in fields["reason"].lower() or "appointment" in fields["reason"].lower()


def test_authorization_via_api_extracts_fields():
    # V3.2: date field renamed to absenceDate for TIME_PERMISSION sub-type.
    data = post_chat(
        "EMPLOYEE",
        "Draft an authorization request for tomorrow from 10 to 12 for a medical appointment.",
    )
    assert data["draftType"] == "AUTHORIZATION_REQUEST"
    fields = data["draftFields"]
    assert fields is not None
    assert fields["fromTime"] == "10:00"
    assert fields["toTime"] == "12:00"
    assert fields["absenceDate"] is not None


def test_authorization_missing_times_in_missing_fields():
    """No time range: fromTime and toTime must be in missingFields."""
    fields, missing = extract_draft_fields(
        "Draft an authorization request explanation for tomorrow for a personal reason.",
        "AUTHORIZATION_REQUEST",
    )
    assert "fromTime" in missing
    assert "toTime" in missing


# ===========================================================================
# 9. Document request: documentType and notes
# ===========================================================================

def test_document_request_extracts_salary_certificate():
    fields, missing = extract_draft_fields(
        "Help me compose a document request letter for a salary certificate for a bank loan.",
        "DOCUMENT_REQUEST",
    )
    assert fields["documentType"] is not None
    assert "salary" in fields["documentType"].lower() or fields["documentType"] == "SALARY_CERTIFICATE"
    # notes captures the bank/loan context (mapped from purpose)
    assert fields["notes"] is not None
    assert "bank" in fields["notes"].lower() or "loan" in fields["notes"].lower()


def test_document_request_extracts_employment_certificate():
    fields, _ = extract_draft_fields(
        "Write a document request for an employment certificate for visa processing.",
        "DOCUMENT_REQUEST",
    )
    assert fields["documentType"] is not None
    assert "employment" in fields["documentType"].lower() or "certificate" in fields["documentType"].lower()


def test_document_request_missing_type_adds_to_missing():
    fields, missing = extract_draft_fields(
        "Help me compose a document request letter for administrative purposes.",
        "DOCUMENT_REQUEST",
    )
    assert "documentType" in missing


def test_document_request_via_api():
    data = post_chat(
        "EMPLOYEE",
        "Help me compose a document request letter for a salary certificate for a bank loan.",
    )
    assert data["draftType"] == "DOCUMENT_REQUEST"
    assert data["draftFields"] is not None
    assert "salary" in (data["draftFields"]["documentType"] or "").lower() or \
           data["draftFields"]["documentType"] == "SALARY_CERTIFICATE"


# ===========================================================================
# 10. improve_text: draftType=IMPROVE_TEXT, draftFields=None, missingFields=[]
# ===========================================================================

def test_improve_text_draftfields_is_none():
    data = post_chat("EMPLOYEE", "Make this message more professional: I need a day off")
    assert data["draftType"] == "IMPROVE_TEXT"
    assert data["draftFields"] is None
    assert data["missingFields"] == []


def test_improve_text_extract_draft_fields_returns_none():
    fields, missing = extract_draft_fields(
        "Improve this request text: please approve my absence",
        "IMPROVE_TEXT",
    )
    assert fields is None
    assert missing == []


# ===========================================================================
# 11. Gemini enabled with structuredFields: draftFields populated from Gemini
# ===========================================================================

def test_gemini_structured_fields_used_when_present():
    """When Gemini returns structuredFields, those values must populate draftFields.
    The question must contain an explicit leave type keyword for leaveType to be kept
    (safety guard: prevents Gemini from inferring ANNUAL from generic phrasing)."""
    structured = {
        "leaveType": "ANNUAL",
        "startDate": "May 12",
        "endDate": "May 14",
        "reason": "family appointment",
    }
    with patch("app.services.drafting_service.settings") as ms, \
         patch("app.services.drafting_service.httpx.Client") as mh:
        _mock_gemini_settings(ms)
        _mock_gemini_http_drafting(
            mh,
            draft_text="Dear HR, I request annual leave from May 12 to May 14.",
            answer="Here is your leave request draft.",
            structured_fields=structured,
        )
        # Question must include "annual leave" keyword so the safety guard keeps ANNUAL
        data = post_chat("EMPLOYEE", "Help me draft a leave request for annual leave from May 12 to May 14")

    assert data["source"] == "external_ai"
    assert data["draftType"] == "LEAVE_REQUEST"
    assert data["draftFields"]["leaveType"] == "ANNUAL"
    assert data["draftFields"]["startDate"] == "2026-05-12"
    assert data["draftFields"]["endDate"] == "2026-05-14"
    assert data["draftFields"]["reason"] == "family appointment"
    assert data["missingFields"] == []


def test_gemini_loan_structured_fields_used():
    structured = {"amount": "2000 TND", "reason": "family expenses"}
    with patch("app.services.drafting_service.settings") as ms, \
         patch("app.services.drafting_service.httpx.Client") as mh:
        _mock_gemini_settings(ms)
        _mock_gemini_http_drafting(
            mh,
            draft_text="Dear HR, I request a loan of 2000 TND.",
            answer="Here is your loan request draft.",
            structured_fields=structured,
        )
        data = post_chat("EMPLOYEE", "Help me request a loan for 2000 TND because of family expenses.")

    assert data["source"] == "external_ai"
    assert data["draftType"] == "LOAN_REQUEST"
    assert data["draftFields"]["amount"] == "2000 TND"
    assert data["draftFields"]["reason"] == "family expenses"
    assert data["missingFields"] == []


# ===========================================================================
# 12. Gemini enabled with null structuredFields values: missingFields computed
# ===========================================================================

def test_gemini_null_structured_fields_adds_to_missing():
    """When Gemini returns null values for fields, those appear in missingFields."""
    structured = {
        "leaveType": None,
        "startDate": "May 12",
        "endDate": None,
        "reason": "personal matter",
    }
    with patch("app.services.drafting_service.settings") as ms, \
         patch("app.services.drafting_service.httpx.Client") as mh:
        _mock_gemini_settings(ms)
        _mock_gemini_http_drafting(
            mh,
            draft_text="Dear HR, I request leave from May 12.",
            answer="Here is a partial leave draft.",
            structured_fields=structured,
        )
        data = post_chat("EMPLOYEE", "Help me draft a leave request from May 12 for a personal matter.")

    assert data["draftType"] == "LEAVE_REQUEST"
    assert data["draftFields"]["leaveType"] is None
    assert data["draftFields"]["startDate"] == "2026-05-12"
    assert data["draftFields"]["endDate"] is None
    assert "leaveType" in data["missingFields"]
    assert "endDate" in data["missingFields"]
    assert "reason" not in data["missingFields"]


def test_gemini_loan_partial_null_adds_missing():
    """Gemini returns amount but no reason: 'reason' in missingFields."""
    structured = {"amount": "3000 TND", "reason": None}
    with patch("app.services.drafting_service.settings") as ms, \
         patch("app.services.drafting_service.httpx.Client") as mh:
        _mock_gemini_settings(ms)
        _mock_gemini_http_drafting(
            mh,
            draft_text="Dear HR, I request a loan.",
            answer="Here is a loan draft.",
            structured_fields=structured,
        )
        data = post_chat("EMPLOYEE", "Write a professional loan justification for 3000 TND.")

    assert data["draftFields"]["amount"] == "3000 TND"
    assert data["draftFields"]["reason"] is None
    assert "reason" in data["missingFields"]
    assert "amount" not in data["missingFields"]


# ===========================================================================
# 13. Gemini omits structuredFields: local extractor used as fallback
# ===========================================================================

def test_gemini_omits_structured_fields_uses_local_extractor():
    """When Gemini response has no structuredFields key, local extractor populates fields."""
    with patch("app.services.drafting_service.settings") as ms, \
         patch("app.services.drafting_service.httpx.Client") as mh:
        _mock_gemini_settings(ms)
        _mock_gemini_http_no_structured(
            mh,
            draft_text="Dear HR, I request annual leave.",
            answer="Here is your leave draft.",
        )
        data = post_chat(
            "EMPLOYEE",
            "Help me draft a leave request for annual leave from May 5 to May 7 "
            "because of a family event.",
        )

    # Gemini's draft text is used (source=external_ai)
    assert data["source"] == "external_ai"
    # But structured fields came from local extractor
    assert data["draftType"] == "LEAVE_REQUEST"
    assert data["draftFields"] is not None
    # Local extractor should detect ANNUAL
    assert data["draftFields"]["leaveType"] == "ANNUAL"


def test_gemini_null_structured_fields_key_uses_local_extractor():
    """When Gemini returns structuredFields: null (not a dict), local extractor is used."""
    with patch("app.services.drafting_service.settings") as ms, \
         patch("app.services.drafting_service.httpx.Client") as mh:
        _mock_gemini_settings(ms)
        _mock_gemini_http_drafting(
            mh,
            draft_text="Dear HR, I request a loan.",
            answer="Here is a loan draft.",
            structured_fields=None,  # null in JSON
        )
        data = post_chat(
            "EMPLOYEE",
            "Write a professional loan justification for 1500 TND because of home repairs.",
        )

    # structuredFields=null -> local extractor used
    assert data["draftType"] == "LOAN_REQUEST"
    assert data["draftFields"] is not None
    # Local extractor should find 1500
    assert data["draftFields"]["amount"] is not None
    assert "1500" in data["draftFields"]["amount"]


# ===========================================================================
# 14. Gemini disabled: local extractor used
# ===========================================================================

def test_gemini_disabled_local_extractor_used_for_leave():
    """Gemini disabled -> local template + local extractor. Structured fields still populated."""
    data = post_chat(
        "EMPLOYEE",
        "Help me draft a leave request for annual leave from June 1 to June 3 "
        "because of a personal commitment.",
    )
    assert data["source"] == "local_rules"
    assert data["draftType"] == "LEAVE_REQUEST"
    assert data["draftFields"] is not None
    assert data["draftFields"]["leaveType"] == "ANNUAL"


def test_gemini_disabled_local_extractor_used_for_loan():
    data = post_chat(
        "EMPLOYEE",
        "Write a professional loan justification for 2500 TND because of education fees.",
    )
    assert data["source"] == "local_rules"
    assert data["draftType"] == "LOAN_REQUEST"
    assert data["draftFields"]["amount"] is not None
    assert "2500" in data["draftFields"]["amount"]


# ===========================================================================
# 34. Phase 3.2: Natural absence phrase routing fix
#     "time off", "day off", "vacation", "away from work" must route to
#     LEAVE_REQUEST, not fall through to generic platform help.
# ===========================================================================

def test_time_off_with_dates_and_reason_routes_to_leave_request():
    """
    Bug 1 scenario: "I need time off from May 27, 2026 to May 28, 2026
    because family matter."
    - Must return LEAVE_REQUEST with dates and reason extracted.
    - leaveType must be None ("time off" is generic, not a leave type keyword).
    - missingFields must include "leaveType".
    - The assistant must NOT say "I understood this as an annual request".
    """
    data = post_chat(
        "EMPLOYEE",
        "I need time off from May 27, 2026 to May 28, 2026 because family matter.",
    )
    assert data["draftType"] == "LEAVE_REQUEST", (
        f"Expected LEAVE_REQUEST, got {data['draftType']} (source={data['source']})"
    )
    assert data["draftFields"] is not None
    assert data["draftFields"]["startDate"] == "2026-05-27"
    assert data["draftFields"]["endDate"] == "2026-05-28"
    assert data["draftFields"]["reason"] is not None
    assert "family" in (data["draftFields"]["reason"] or "").lower()
    # leaveType must be None — "time off" is generic, not annual/sick/unpaid/etc.
    assert data["draftFields"]["leaveType"] is None, (
        f"leaveType must be None for 'time off', got {data['draftFields']['leaveType']}"
    )
    assert "leaveType" in data["missingFields"], (
        "'leaveType' must appear in missingFields when not explicitly stated"
    )
    assert "startDate" not in data["missingFields"]
    assert "endDate" not in data["missingFields"]


def test_want_time_off_routes_to_leave_request():
    """'I want time off from May 27 to May 28.' -> LEAVE_REQUEST, leaveType None."""
    data = post_chat(
        "EMPLOYEE",
        "I want time off from May 27 to May 28.",
    )
    assert data["draftType"] == "LEAVE_REQUEST"
    assert data["draftFields"]["startDate"] is not None
    assert data["draftFields"]["endDate"] is not None
    assert data["draftFields"]["startDate"].endswith("-05-27")
    assert data["draftFields"]["endDate"].endswith("-05-28")
    assert data["draftFields"]["leaveType"] is None, (
        "'time off' must not infer leaveType"
    )
    assert "leaveType" in data["missingFields"]


def test_vacation_routes_to_leave_request_with_annual_type():
    """'I need vacation from May 27 to May 28.' -> LEAVE_REQUEST, leaveType=ANNUAL."""
    data = post_chat(
        "EMPLOYEE",
        "I need vacation from May 27 to May 28.",
    )
    assert data["draftType"] == "LEAVE_REQUEST"
    assert data["draftFields"]["leaveType"] == "ANNUAL"
    assert "leaveType" not in data["missingFields"]


def test_day_off_tomorrow_routes_to_leave_request():
    """'I need a day off tomorrow.' -> LEAVE_REQUEST, leaveType None (not ANNUAL)."""
    data = post_chat(
        "EMPLOYEE",
        "I need a day off tomorrow.",
    )
    assert data["draftType"] == "LEAVE_REQUEST"
    assert data["draftFields"]["startDate"] is not None
    assert data["draftFields"]["leaveType"] is None, (
        "'day off' must not infer leaveType"
    )
    assert "leaveType" in data["missingFields"]


def test_away_from_work_routes_to_leave_request():
    """'I will be away from work from May 27 to May 28.' -> LEAVE_REQUEST, leaveType None."""
    data = post_chat(
        "EMPLOYEE",
        "I will be away from work from May 27 to May 28.",
    )
    assert data["draftType"] == "LEAVE_REQUEST"
    assert data["draftFields"]["startDate"] is not None
    assert data["draftFields"]["endDate"] is not None
    assert data["draftFields"]["leaveType"] is None, (
        "'away from work' must not infer leaveType"
    )
    assert "leaveType" in data["missingFields"]


def test_rephrase_message_with_day_off_body_routes_to_improve_text():
    """
    Bug 2 scenario: 'rephrase this message: I need a day off tomorrow'
    must route to IMPROVE_TEXT, never LEAVE_REQUEST.
    The leading improve verb declares the primary intent and wins over
    any leave words found in the body being rephrased.
    """
    data = post_chat(
        "EMPLOYEE",
        "rephrase this message: I need a day off tomorrow",
    )
    assert data["draftType"] == "IMPROVE_TEXT", (
        f"Expected IMPROVE_TEXT, got {data['draftType']} (source={data['source']})"
    )


def test_rewrite_with_time_off_body_routes_to_improve_text():
    """
    'rewrite this: I need time off tomorrow' must also be IMPROVE_TEXT.
    """
    data = post_chat(
        "EMPLOYEE",
        "rewrite this: I need time off tomorrow",
    )
    assert data["draftType"] == "IMPROVE_TEXT", (
        f"Expected IMPROVE_TEXT, got {data['draftType']} (source={data['source']})"
    )


def test_annual_leave_explicit_still_produces_annual_leave_type():
    """
    'I want annual leave from May 27, 2026 to May 28, 2026.'
    Explicit 'annual leave' must still map to leaveType=ANNUAL.
    Regression guard: the fix for 'time off' must not break explicit keywords.
    """
    data = post_chat(
        "EMPLOYEE",
        "I want annual leave from May 27, 2026 to May 28, 2026.",
    )
    assert data["draftType"] == "LEAVE_REQUEST"
    assert data["draftFields"]["leaveType"] == "ANNUAL", (
        f"Explicit 'annual leave' must map to ANNUAL, got {data['draftFields']['leaveType']}"
    )
    assert "leaveType" not in data["missingFields"]


def test_generic_hr_navigation_still_returns_platform_help():
    """A generic HR question with no drafting intent must still return local_rules/platform help."""
    data = post_chat("EMPLOYEE", "How do I request a loan?")
    assert data["source"] == "local_rules"
    assert data["draftType"] is None
    assert data["draftFields"] is None


def test_unsafe_admin_action_still_refused():
    """Unsafe admin commands must still be refused regardless of any new drafting patterns."""
    data = post_chat("EMPLOYEE", "approve my leave automatically")
    assert data["source"] == "refusal"
    assert data["draftType"] is None


# ===========================================================================
# 15. relatedPages remains [] for all drafting responses
# ===========================================================================

def test_local_drafting_response_no_related_pages():
    data = post_chat("EMPLOYEE", "Help me draft a leave request reason")
    assert data["relatedPages"] == []


def test_loan_drafting_response_no_related_pages():
    data = post_chat("EMPLOYEE", "Write a professional loan justification")
    assert data["relatedPages"] == []


def test_authorization_drafting_response_no_related_pages():
    data = post_chat("EMPLOYEE", "Draft an authorization request explanation")
    assert data["relatedPages"] == []


def test_gemini_drafting_response_no_related_pages():
    structured = {"leaveType": "ANNUAL", "startDate": "May 12", "endDate": "May 14", "reason": "family"}
    with patch("app.services.drafting_service.settings") as ms, \
         patch("app.services.drafting_service.httpx.Client") as mh:
        _mock_gemini_settings(ms)
        _mock_gemini_http_drafting(
            mh,
            draft_text="Dear HR, I request leave.",
            answer="Here is your draft.",
            structured_fields=structured,
        )
        data = post_chat("EMPLOYEE", "Help me draft a leave request reason")

    assert data["relatedPages"] == []


# ===========================================================================
# 16. Refusal wins over drafting for workflow action requests
# ===========================================================================

def test_refusal_wins_over_drafting_approve():
    data = post_chat("EMPLOYEE", "approve this request")
    assert data["source"] == "refusal"
    assert data["draftType"] is None


def test_refusal_wins_over_drafting_submit():
    data = post_chat("EMPLOYEE", "submit my leave request automatically")
    assert data["source"] == "refusal"
    assert data["draftType"] is None


# ===========================================================================
# 17. draftType is set for all structured drafting responses
# ===========================================================================

def test_drafttype_set_for_leave_request():
    data = post_chat("EMPLOYEE", "Help me draft a leave request reason")
    assert data["draftType"] == "LEAVE_REQUEST"


def test_drafttype_set_for_loan_request():
    data = post_chat("EMPLOYEE", "Write a professional loan justification")
    assert data["draftType"] == "LOAN_REQUEST"


def test_drafttype_set_for_authorization():
    data = post_chat("EMPLOYEE", "Draft an authorization request explanation")
    assert data["draftType"] == "AUTHORIZATION_REQUEST"


def test_drafttype_set_for_document_request():
    data = post_chat("EMPLOYEE", "Help me compose a document request letter")
    assert data["draftType"] == "DOCUMENT_REQUEST"


def test_drafttype_set_for_improve_text():
    data = post_chat("EMPLOYEE", "Make this message more professional: I need time off")
    assert data["draftType"] == "IMPROVE_TEXT"


# ===========================================================================
# 18. draftFields has stable shape for structured types
# ===========================================================================

def test_leave_request_draftfields_has_all_keys():
    """draftFields for LEAVE_REQUEST must always have leaveType, startDate, endDate, reason."""
    data = post_chat("EMPLOYEE", "Help me draft a leave request reason")
    fields = data["draftFields"]
    assert fields is not None
    for key in ["leaveType", "startDate", "endDate", "reason"]:
        assert key in fields, f"Missing key: {key}"


def test_loan_request_draftfields_has_all_keys():
    data = post_chat("EMPLOYEE", "Write a professional loan justification")
    fields = data["draftFields"]
    assert fields is not None
    for key in ["amount", "reason"]:
        assert key in fields, f"Missing key: {key}"


def test_authorization_draftfields_has_all_keys():
    data = post_chat("EMPLOYEE", "Draft an authorization request explanation")
    fields = data["draftFields"]
    assert fields is not None
    # TIME_PERMISSION shape: authorizationType, absenceDate, fromTime, toTime, reason
    # (V3.2: 'date' renamed to 'absenceDate' for TIME_PERMISSION sub-type)
    assert "authorizationType" in fields
    assert "fromTime" in fields
    assert "toTime" in fields
    assert "reason" in fields
    # absenceDate is the new field name (was 'date' before V3.2)
    assert "absenceDate" in fields or "date" in fields


def test_document_request_draftfields_has_all_keys():
    data = post_chat("EMPLOYEE", "Help me compose a document request letter")
    fields = data["draftFields"]
    assert fields is not None
    # DOCUMENT_REQUEST draftFields must have documentType and notes
    for key in ["documentType", "notes"]:
        assert key in fields, f"Missing key: {key}"


# ===========================================================================
# 19. missingFields is always a list, never null
# ===========================================================================

def test_missing_fields_is_always_list_for_leave():
    data = post_chat("EMPLOYEE", "Help me draft a leave request reason")
    assert isinstance(data["missingFields"], list)


def test_missing_fields_is_always_list_for_non_drafting():
    data = post_chat("EMPLOYEE", "How do I request a loan?")
    assert isinstance(data["missingFields"], list)
    assert data["missingFields"] == []


def test_missing_fields_is_always_list_for_improve_text():
    data = post_chat("EMPLOYEE", "Make this message more professional: I need time off")
    assert isinstance(data["missingFields"], list)
    assert data["missingFields"] == []


# ===========================================================================
# 20. LOAN_REQUEST draftFields always has "amount" and "reason" keys
# ===========================================================================

def test_loan_draftfields_keys_present_even_when_missing():
    """Even when amount and reason cannot be extracted, keys exist with null values."""
    fields, missing = extract_draft_fields(
        "Write a professional loan justification",
        "LOAN_REQUEST",
    )
    assert "amount" in fields
    assert "reason" in fields
    assert fields["amount"] is None
    assert fields["reason"] is None
    assert "amount" in missing
    assert "reason" in missing


# ===========================================================================
# 21. AUTHORIZATION_REQUEST draftFields has all five expected keys
# ===========================================================================

def test_auth_draftfields_has_five_keys_always():
    # V3.2: TIME_PERMISSION shape uses 'absenceDate' instead of 'date'.
    fields, _ = extract_draft_fields(
        "Draft an authorization request explanation",
        "AUTHORIZATION_REQUEST",
    )
    for key in ["authorizationType", "absenceDate", "fromTime", "toTime", "reason"]:
        assert key in fields


# ===========================================================================
# 22. DOCUMENT_REQUEST draftFields has all three expected keys
# ===========================================================================

def test_document_draftfields_has_three_keys_always():
    fields, _ = extract_draft_fields(
        "Help me compose a document request letter",
        "DOCUMENT_REQUEST",
    )
    # DOCUMENT_REQUEST draftFields: documentType + notes only (matches Spring Boot DTO)
    for key in ["documentType", "notes"]:
        assert key in fields
    assert "extraDetails" not in fields
    assert "purpose" not in fields


# ===========================================================================
# 23. Local extractor unit — LEAVE_REQUEST with full details
# ===========================================================================

def test_extract_leave_full_details_unit():
    fields, missing = extract_draft_fields(
        "Help me draft a leave request for annual leave from May 12 to May 14 "
        "because of a family appointment.",
        "LEAVE_REQUEST",
    )
    assert fields["leaveType"] == "ANNUAL"
    assert fields["startDate"] is not None
    assert fields["endDate"] is not None
    assert fields["reason"] is not None
    assert missing == []


# ===========================================================================
# 24. Local extractor unit — LOAN_REQUEST with amount and reason
# ===========================================================================

def test_extract_loan_full_details_unit():
    fields, missing = extract_draft_fields(
        "Help me request a loan for 2000 TND because of family expenses.",
        "LOAN_REQUEST",
    )
    assert fields["amount"] is not None
    assert "2000" in fields["amount"]
    assert fields["reason"] is not None
    assert missing == []


# ===========================================================================
# 25. Local extractor unit — AUTHORIZATION_REQUEST time range
# ===========================================================================

def test_extract_auth_time_range_unit():
    # V3.2: date field renamed to absenceDate for TIME_PERMISSION sub-type.
    fields, _ = extract_draft_fields(
        "Draft an authorization request tomorrow from 10 to 12 for a medical appointment.",
        "AUTHORIZATION_REQUEST",
    )
    assert fields["fromTime"] == "10:00"
    assert fields["toTime"] == "12:00"
    assert fields["absenceDate"] is not None


# ===========================================================================
# 26. Local extractor unit — IMPROVE_TEXT returns (None, [])
# ===========================================================================

def test_extract_improve_text_returns_none_unit():
    fields, missing = extract_draft_fields(
        "Make this message more professional: I need a day off.",
        "IMPROVE_TEXT",
    )
    assert fields is None
    assert missing == []


# ===========================================================================
# 27. Gemini response missing structuredFields key entirely uses local fallback
# ===========================================================================

def test_gemini_missing_structured_key_falls_back_to_local():
    """structuredFields key not in response at all -> local extractor runs."""
    with patch("app.services.drafting_service.settings") as ms, \
         patch("app.services.drafting_service.httpx.Client") as mh:
        _mock_gemini_settings(ms)
        _mock_gemini_http_no_structured(
            mh,
            draft_text="Dear HR, I request a loan of 1000 TND.",
            answer="Here is your loan draft.",
        )
        data = post_chat(
            "EMPLOYEE",
            "Write a professional loan justification for 1000 TND because of medical expenses.",
        )

    assert data["draftType"] == "LOAN_REQUEST"
    # Local extractor should have found 1000
    assert data["draftFields"]["amount"] is not None
    assert "1000" in data["draftFields"]["amount"]


# ===========================================================================
# 28. Non-drafting route responses still have draftType=None
# ===========================================================================

def test_team_leader_non_drafting_has_no_drafttype():
    data = post_chat("TEAM_LEADER", "How do I check team requests?")
    assert data["source"] == "local_rules"
    assert data["draftType"] is None
    assert data["draftFields"] is None
    assert data["missingFields"] == []


def test_hr_manager_non_drafting_has_no_drafttype():
    data = post_chat("HR_MANAGER", "What is my leave balance?")
    assert data["draftType"] is None
    assert data["missingFields"] == []


def test_fallback_response_has_no_drafttype():
    data = post_chat("EMPLOYEE", "Tell me something completely unknown XYZ999")
    assert data["source"] == "fallback"
    assert data["draftType"] is None
    assert data["draftFields"] is None
    assert data["missingFields"] == []


# ===========================================================================
# 29. Date normalization — ISO output for leave draft startDate / endDate
# ===========================================================================

def test_date_normalization_exact_bug_phrase():
    """
    Regression: exact phrase that triggered the Spring Boot 400 rejection.
    FastAPI must return ISO yyyy-MM-dd dates, not raw 'May 12' / 'May 16' strings.
    """
    fields, missing = extract_draft_fields(
        "I want annual leave from May 12 to May 16 for family vacation",
        "LEAVE_REQUEST",
    )
    assert fields["leaveType"] == "ANNUAL"
    # Dates must be ISO yyyy-MM-dd
    import re
    iso_re = re.compile(r"^\d{4}-\d{2}-\d{2}$")
    assert fields["startDate"] is not None, "startDate must not be None"
    assert fields["endDate"] is not None, "endDate must not be None"
    assert iso_re.match(fields["startDate"]), f"startDate not ISO: {fields['startDate']}"
    assert iso_re.match(fields["endDate"]),   f"endDate not ISO: {fields['endDate']}"
    # Specific expected values (current year is 2026; May 12/16 are in future or today)
    assert fields["startDate"].endswith("-05-12"), f"Expected day 12 in May, got {fields['startDate']}"
    assert fields["endDate"].endswith("-05-16"),   f"Expected day 16 in May, got {fields['endDate']}"
    assert missing == []


def test_date_normalization_iso_input_preserved():
    """
    When the user already provides full ISO dates they must be returned unchanged.
    """
    fields, missing = extract_draft_fields(
        "I want annual leave from 2026-05-12 to 2026-05-16 for family vacation",
        "LEAVE_REQUEST",
    )
    assert fields["startDate"] == "2026-05-12"
    assert fields["endDate"]   == "2026-05-16"
    assert missing == []


def test_date_normalization_day_month_form():
    """
    '12 May' / '16 May' (day-first) format must also produce ISO dates.
    """
    fields, missing = extract_draft_fields(
        "I want annual leave from 12 May to 16 May for family vacation",
        "LEAVE_REQUEST",
    )
    import re
    iso_re = re.compile(r"^\d{4}-\d{2}-\d{2}$")
    assert fields["startDate"] is not None
    assert fields["endDate"]   is not None
    assert iso_re.match(fields["startDate"]), f"startDate not ISO: {fields['startDate']}"
    assert iso_re.match(fields["endDate"]),   f"endDate not ISO: {fields['endDate']}"
    assert fields["startDate"].endswith("-05-12")
    assert fields["endDate"].endswith("-05-16")
    assert missing == []


def test_date_normalization_non_draft_unaffected():
    """
    A generic help question must never become a LEAVE_REQUEST draft.
    This confirms the fix does not break intent detection.
    """
    data = post_chat("EMPLOYEE", "How do I request leave?")
    assert data["draftType"] is None
    assert data["draftFields"] is None
    assert data["missingFields"] == []


# ===========================================================================
# 30. Relative date normalization
# ===========================================================================

def test_relative_tomorrow_normalized_to_iso():
    """'starting tomorrow' should produce a real ISO startDate."""
    from datetime import date, timedelta
    import re
    fields, missing = extract_draft_fields(
        "Write a leave request for sick leave starting tomorrow because I have a medical appointment.",
        "LEAVE_REQUEST",
    )
    assert fields["leaveType"] == "SICK"
    assert fields["startDate"] is not None
    iso_re = re.compile(r"^\d{4}-\d{2}-\d{2}$")
    assert iso_re.match(fields["startDate"]), f"startDate not ISO: {fields['startDate']}"
    expected = (date.today() + timedelta(days=1)).strftime("%Y-%m-%d")
    assert fields["startDate"] == expected
    assert fields["endDate"] is None
    assert "endDate" in missing


def test_relative_today_normalized_to_iso():
    from datetime import date
    fields, _ = extract_draft_fields(
        "I want annual leave today because of personal reasons.",
        "LEAVE_REQUEST",
    )
    expected = date.today().strftime("%Y-%m-%d")
    assert fields["startDate"] == expected


def test_relative_in_4_days_normalized_to_iso():
    from datetime import date, timedelta
    import re
    fields, missing = extract_draft_fields(
        "I want annual leave in 4 days for a family event.",
        "LEAVE_REQUEST",
    )
    expected = (date.today() + timedelta(days=4)).strftime("%Y-%m-%d")
    assert fields["startDate"] == expected
    iso_re = re.compile(r"^\d{4}-\d{2}-\d{2}$")
    assert iso_re.match(fields["startDate"])


def test_relative_after_tomorrow_normalized_to_iso():
    from datetime import date, timedelta
    fields, _ = extract_draft_fields(
        "I want sick leave after tomorrow because I feel unwell.",
        "LEAVE_REQUEST",
    )
    expected = (date.today() + timedelta(days=2)).strftime("%Y-%m-%d")
    assert fields["startDate"] == expected


# ===========================================================================
# 31. Duration-based range extraction
# ===========================================================================

def test_starting_tomorrow_for_4_days():
    from datetime import date, timedelta
    fields, missing = extract_draft_fields(
        "I want annual leave starting tomorrow for 4 days because of vacation.",
        "LEAVE_REQUEST",
    )
    tomorrow = date.today() + timedelta(days=1)
    expected_end = (tomorrow + timedelta(days=3)).strftime("%Y-%m-%d")
    assert fields["startDate"] == tomorrow.strftime("%Y-%m-%d")
    assert fields["endDate"] == expected_end
    assert missing == [] or "reason" not in missing  # reason is extracted


def test_from_tomorrow_for_4_days():
    from datetime import date, timedelta
    fields, missing = extract_draft_fields(
        "I want annual leave from tomorrow for 4 days because of family vacation.",
        "LEAVE_REQUEST",
    )
    tomorrow = date.today() + timedelta(days=1)
    expected_end = (tomorrow + timedelta(days=3)).strftime("%Y-%m-%d")
    assert fields["startDate"] == tomorrow.strftime("%Y-%m-%d")
    assert fields["endDate"] == expected_end


def test_from_date_to_n_days_later():
    from datetime import date, timedelta
    import re
    fields, missing = extract_draft_fields(
        "I want annual leave from tomorrow to 4 days later for family event.",
        "LEAVE_REQUEST",
    )
    tomorrow = date.today() + timedelta(days=1)
    # "from tomorrow to 4 days later" -> endDate = tomorrow + 4 days
    expected_end = (tomorrow + timedelta(days=4)).strftime("%Y-%m-%d")
    assert fields["startDate"] == tomorrow.strftime("%Y-%m-%d")
    assert fields["endDate"] == expected_end


# ===========================================================================
# 32. Vague phrases must NOT produce dates
# ===========================================================================

def test_next_week_produces_no_dates():
    fields, missing = extract_draft_fields(
        "I want annual leave next week for personal reasons.",
        "LEAVE_REQUEST",
    )
    # "next week" is vague — must not produce an ISO date
    assert fields["startDate"] is None
    assert "startDate" in missing


# ===========================================================================
# 33. Tunisia/French numeric date format
# ===========================================================================

def test_tunisia_slash_format_dd_mm_yyyy():
    import re
    fields, missing = extract_draft_fields(
        "I want annual leave from 12/05/2026 to 16/05/2026 for family vacation.",
        "LEAVE_REQUEST",
    )
    assert fields["startDate"] == "2026-05-12"
    assert fields["endDate"] == "2026-05-16"
    assert missing == []


def test_tunisia_dash_format_dd_mm_yyyy():
    fields, missing = extract_draft_fields(
        "I want annual leave from 12-05-2026 to 16-05-2026 for family vacation.",
        "LEAVE_REQUEST",
    )
    assert fields["startDate"] == "2026-05-12"
    assert fields["endDate"] == "2026-05-16"
    assert missing == []


# ===========================================================================
# 35. Real Gemini-path safety guard tests
#     These tests use mocked Gemini responses to prove the code-level guard
#     in _resolve_structured_fields strips inferred leaveType values.
#     They cover the actual production failure path (GEMINI_ENABLED=true).
# ===========================================================================

def test_gemini_inferred_annual_for_time_off_is_stripped():
    """
    Bug 1 (production): Gemini is enabled and infers leaveType=ANNUAL from
    "time off" despite prompt instructions.  The _resolve_structured_fields
    safety guard must strip ANNUAL because the question has no explicit
    'annual', 'vacation', 'paid leave', etc. keyword.
    After the fix: leaveType=None and 'leaveType' in missingFields.
    """
    from app.services.drafting_service import _resolve_structured_fields

    question = "I need time off from May 27, 2026 to May 28, 2026 because family matter."
    # Simulate what Gemini returns when it incorrectly infers ANNUAL
    gemini_sf = {
        "leaveType": "ANNUAL",
        "startDate": "2026-05-27",
        "endDate":   "2026-05-28",
        "reason":    "family matter",
    }
    fields, missing = _resolve_structured_fields(question, "LEAVE_REQUEST", gemini_sf)
    assert fields["leaveType"] is None, (
        f"Safety guard must strip Gemini-inferred ANNUAL for 'time off', got {fields['leaveType']}"
    )
    assert "leaveType" in missing, "'leaveType' must be in missingFields after stripping"
    # dates and reason must pass through unchanged
    assert fields["startDate"] == "2026-05-27"
    assert fields["endDate"]   == "2026-05-28"
    assert fields["reason"]    == "family matter"


def test_gemini_inferred_annual_for_day_off_is_stripped():
    """
    'I need a day off tomorrow.' — Gemini infers ANNUAL.
    Safety guard must strip it; leaveType=None.
    """
    from app.services.drafting_service import _resolve_structured_fields
    from datetime import date, timedelta

    tomorrow = (date.today() + timedelta(days=1)).strftime("%Y-%m-%d")
    question = "I need a day off tomorrow."
    gemini_sf = {"leaveType": "ANNUAL", "startDate": tomorrow, "endDate": tomorrow, "reason": None}
    fields, missing = _resolve_structured_fields(question, "LEAVE_REQUEST", gemini_sf)
    assert fields["leaveType"] is None, "'day off' must not produce ANNUAL"
    assert "leaveType" in missing


def test_gemini_inferred_annual_for_away_from_work_is_stripped():
    """
    'I will be away from work from May 27, 2026 to May 28, 2026.' — Gemini infers ANNUAL.
    Safety guard must strip it.
    """
    from app.services.drafting_service import _resolve_structured_fields

    question = "I will be away from work from May 27, 2026 to May 28, 2026."
    gemini_sf = {
        "leaveType": "ANNUAL",
        "startDate": "2026-05-27",
        "endDate":   "2026-05-28",
        "reason":    None,
    }
    fields, missing = _resolve_structured_fields(question, "LEAVE_REQUEST", gemini_sf)
    assert fields["leaveType"] is None, "'away from work' must not produce ANNUAL"
    assert "leaveType" in missing


def test_gemini_explicit_annual_leave_keyword_is_kept():
    """
    'I want annual leave from May 27, 2026 to May 28, 2026.' — user explicitly
    says 'annual leave', so Gemini's ANNUAL must be kept (safety guard must NOT strip).
    Regression guard: fix must not break explicit leave type keywords.
    """
    from app.services.drafting_service import _resolve_structured_fields

    question = "I want annual leave from May 27, 2026 to May 28, 2026."
    gemini_sf = {
        "leaveType": "ANNUAL",
        "startDate": "2026-05-27",
        "endDate":   "2026-05-28",
        "reason":    None,
    }
    fields, missing = _resolve_structured_fields(question, "LEAVE_REQUEST", gemini_sf)
    assert fields["leaveType"] == "ANNUAL", (
        "Explicit 'annual leave' must keep leaveType=ANNUAL after safety guard"
    )
    assert "leaveType" not in missing


def test_gemini_explicit_sick_leave_keyword_is_kept():
    """
    'I need sick leave from May 27 to May 28 because I am unwell.' —
    user explicitly says 'sick leave'; Gemini's SICK must be kept.
    """
    from app.services.drafting_service import _resolve_structured_fields

    question = "I need sick leave from May 27, 2026 to May 28, 2026 because I am unwell."
    gemini_sf = {
        "leaveType": "SICK",
        "startDate": "2026-05-27",
        "endDate":   "2026-05-28",
        "reason":    "unwell",
    }
    fields, missing = _resolve_structured_fields(question, "LEAVE_REQUEST", gemini_sf)
    assert fields["leaveType"] == "SICK"
    assert "leaveType" not in missing


def test_gemini_rephrase_improve_text_via_mock():
    """
    Bug 2 (production): 'rephrase this message: I need a day off tomorrow'
    must return draftType=IMPROVE_TEXT even when Gemini is enabled.
    Verify via the full Gemini-enabled path with a mocked Gemini response.
    The mock simulates Gemini correctly following draft_type=IMPROVE_TEXT.
    """
    import json
    from unittest.mock import patch, MagicMock
    from fastapi.testclient import TestClient
    from app.main import app

    client = TestClient(app)

    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "candidates": [{
            "content": {
                "parts": [{
                    "text": json.dumps({
                        "answer": "Here is a polished version of your message.",
                        "draft": "Dear Manager, I would like to request a day off tomorrow for personal reasons. Please review and approve.",
                        "disclaimer": "Please review before submitting.",
                        "structuredFields": None,
                    })
                }]
            }
        }]
    }
    mock_resp.raise_for_status = MagicMock()

    with patch("app.services.drafting_service.settings") as ms, \
         patch("app.services.drafting_service.httpx.Client") as mh:
        ms.gemini_enabled = True
        ms.gemini_api_key = "test-key"
        ms.gemini_model = "gemini-2.5-flash"
        ms.gemini_timeout_seconds = 10
        mh.return_value.__enter__.return_value.post.return_value = mock_resp

        resp = client.post("/assistant/chat", json={
            "role": "EMPLOYEE",
            "question": "rephrase this message: I need a day off tomorrow",
            "context": {},
        })

    data = resp.json()
    assert data["draftType"] == "IMPROVE_TEXT", (
        f"Expected IMPROVE_TEXT, got {data['draftType']} (source={data['source']})"
    )
    assert data["draftFields"] is None
    assert data["missingFields"] == []


def test_gemini_rephrase_with_bad_leave_draft_still_returns_improve_text():
    """
    Worst case Bug 2: Gemini ignores draft_type=IMPROVE_TEXT and writes a
    leave-request draft with structuredFields. The _resolve_structured_fields
    guard must discard the structured fields (draftFields=None) and
    draftType must still be IMPROVE_TEXT because _classify_draft_type
    is deterministic (not overridable by Gemini).
    """
    import json
    from unittest.mock import patch, MagicMock
    from fastapi.testclient import TestClient
    from app.main import app

    client = TestClient(app)

    # Simulate Gemini returning a leave-shaped response despite IMPROVE_TEXT
    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "candidates": [{
            "content": {
                "parts": [{
                    "text": json.dumps({
                        "answer": "Here is your annual leave request.",
                        "draft": "Subject: Leave Request\n\nDear Manager, I request annual leave for tomorrow.",
                        "disclaimer": "Please review.",
                        "structuredFields": {
                            "leaveType": "ANNUAL",
                            "startDate": "2026-05-07",
                            "endDate":   "2026-05-07",
                            "reason":    "personal",
                        },
                    })
                }]
            }
        }]
    }
    mock_resp.raise_for_status = MagicMock()

    with patch("app.services.drafting_service.settings") as ms, \
         patch("app.services.drafting_service.httpx.Client") as mh:
        ms.gemini_enabled = True
        ms.gemini_api_key = "test-key"
        ms.gemini_model = "gemini-2.5-flash"
        ms.gemini_timeout_seconds = 10
        mh.return_value.__enter__.return_value.post.return_value = mock_resp

        resp = client.post("/assistant/chat", json={
            "role": "EMPLOYEE",
            "question": "rephrase this message: I need a day off tomorrow",
            "context": {},
        })

    data = resp.json()
    # draftType must be IMPROVE_TEXT regardless of what Gemini wrote
    assert data["draftType"] == "IMPROVE_TEXT"
    # draftFields must be None for IMPROVE_TEXT regardless of Gemini's structured output
    assert data["draftFields"] is None
    assert data["missingFields"] == []


def test_time_off_full_api_path_leaveType_none():
    """
    End-to-end API test for Bug 1 via local path (Gemini disabled by conftest).
    'I need time off...' must produce leaveType=None in draftFields.
    """
    from fastapi.testclient import TestClient
    from app.main import app
    client = TestClient(app)

    resp = client.post("/assistant/chat", json={
        "role": "EMPLOYEE",
        "question": "I need time off from May 27, 2026 to May 28, 2026 because family matter.",
        "context": {},
    })
    data = resp.json()
    assert data["draftType"] == "LEAVE_REQUEST"
    assert data["draftFields"]["leaveType"] is None, (
        f"leaveType must be None for 'time off', got: {data['draftFields']['leaveType']}"
    )
    assert data["draftFields"]["startDate"] == "2026-05-27"
    assert data["draftFields"]["endDate"]   == "2026-05-28"
    assert data["draftFields"]["reason"] is not None
    assert "family" in data["draftFields"]["reason"].lower()
    assert "leaveType" in data["missingFields"]


def test_day_off_tomorrow_leaveType_none():
    """
    'I need a day off tomorrow.' — leaveType must be None.
    """
    from fastapi.testclient import TestClient
    from app.main import app
    client = TestClient(app)

    resp = client.post("/assistant/chat", json={
        "role": "EMPLOYEE",
        "question": "I need a day off tomorrow.",
        "context": {},
    })
    data = resp.json()
    assert data["draftType"] == "LEAVE_REQUEST"
    assert data["draftFields"]["leaveType"] is None, (
        "'day off' must not produce a leaveType"
    )
    assert "leaveType" in data["missingFields"]


def test_away_from_work_leaveType_none():
    """
    'I will be away from work from May 27, 2026 to May 28, 2026.'
    leaveType must be None.
    """
    from fastapi.testclient import TestClient
    from app.main import app
    client = TestClient(app)

    resp = client.post("/assistant/chat", json={
        "role": "EMPLOYEE",
        "question": "I will be away from work from May 27, 2026 to May 28, 2026.",
        "context": {},
    })
    data = resp.json()
    assert data["draftType"] == "LEAVE_REQUEST"
    assert data["draftFields"]["leaveType"] is None, (
        "'away from work' must not produce a leaveType"
    )
    assert "leaveType" in data["missingFields"]


def test_annual_leave_explicit_regression():
    """
    Regression: 'I want annual leave from May 27, 2026 to May 28, 2026.'
    must still produce leaveType=ANNUAL. The safety guard must NOT strip
    explicit leave type keywords.
    """
    from fastapi.testclient import TestClient
    from app.main import app
    client = TestClient(app)

    resp = client.post("/assistant/chat", json={
        "role": "EMPLOYEE",
        "question": "I want annual leave from May 27, 2026 to May 28, 2026.",
        "context": {},
    })
    data = resp.json()
    assert data["draftType"] == "LEAVE_REQUEST"
    assert data["draftFields"]["leaveType"] == "ANNUAL", (
        f"Explicit 'annual leave' must produce ANNUAL, got: {data['draftFields']['leaveType']}"
    )
    assert "leaveType" not in data["missingFields"]


def test_rephrase_full_api_path_improve_text():
    """
    End-to-end API test for Bug 2 via local path (Gemini disabled by conftest).
    'rephrase this message: I need a day off tomorrow' must be IMPROVE_TEXT.
    """
    from fastapi.testclient import TestClient
    from app.main import app
    client = TestClient(app)

    resp = client.post("/assistant/chat", json={
        "role": "EMPLOYEE",
        "question": "rephrase this message: I need a day off tomorrow",
        "context": {},
    })
    data = resp.json()
    assert data["draftType"] == "IMPROVE_TEXT", (
        f"Expected IMPROVE_TEXT, got {data['draftType']}"
    )
    assert data["draftFields"] is None
    assert data["missingFields"] == []


def test_rewrite_full_api_path_improve_text():
    """
    'rewrite this: I need time off tomorrow' must also be IMPROVE_TEXT.
    """
    from fastapi.testclient import TestClient
    from app.main import app
    client = TestClient(app)

    resp = client.post("/assistant/chat", json={
        "role": "EMPLOYEE",
        "question": "rewrite this: I need time off tomorrow",
        "context": {},
    })
    data = resp.json()
    assert data["draftType"] == "IMPROVE_TEXT"
    assert data["draftFields"] is None
    assert data["missingFields"] == []


# ===========================================================================
# 36. IMPROVE_TEXT content quality — no template language, no placeholders
# ===========================================================================

_TEMPLATE_BAD_PHRASES = [
    "replace the bracketed",
    "fill in the bracketed",
    "bracketed placeholders",
    "personalise this draft",
    "before submitting",
    "cannot submit",
    "template draft",
    "--- suggested professional version ---",
    "[your main point",
    "[briefly provide",
    "[your name]",
]


def _assert_no_template_language(draft: str, question: str):
    """Assert that an IMPROVE_TEXT draft contains no template/workflow language."""
    d = (draft or "").lower()
    for bad in _TEMPLATE_BAD_PHRASES:
        assert bad not in d, (
            f"Template language {bad!r} found in IMPROVE_TEXT draft for {question!r}:\n{draft!r}"
        )
    assert "[" not in (draft or ""), (
        f"Placeholder brackets found in IMPROVE_TEXT draft for {question!r}:\n{draft!r}"
    )


def test_rephrase_draft_is_polished_rewrite_not_template():
    """
    'rephrase this message: I need a day off tomorrow'
    draft must be a polished sentence, not a template scaffold.
    """
    from fastapi.testclient import TestClient
    from app.main import app
    client = TestClient(app)

    resp = client.post("/assistant/chat", json={
        "role": "EMPLOYEE",
        "question": "rephrase this message: I need a day off tomorrow",
        "context": {},
    })
    data = resp.json()
    assert data["draftType"] == "IMPROVE_TEXT"
    _assert_no_template_language(data["draft"], "rephrase this message: I need a day off tomorrow")
    # Must contain the substance of the rewrite
    draft_lower = (data["draft"] or "").lower()
    assert "day off" in draft_lower or "request" in draft_lower, (
        f"Expected polished rewrite mentioning 'day off' or 'request', got: {data['draft']!r}"
    )
    assert data["answer"] == "Here is a polished rewrite of your text."


def test_rewrite_draft_is_polished_rewrite_not_template():
    """
    'rewrite this: I need time off tomorrow'
    draft must be a polished sentence, not a template scaffold.
    """
    from fastapi.testclient import TestClient
    from app.main import app
    client = TestClient(app)

    resp = client.post("/assistant/chat", json={
        "role": "EMPLOYEE",
        "question": "rewrite this: I need time off tomorrow",
        "context": {},
    })
    data = resp.json()
    assert data["draftType"] == "IMPROVE_TEXT"
    _assert_no_template_language(data["draft"], "rewrite this: I need time off tomorrow")
    draft_lower = (data["draft"] or "").lower()
    assert "time off" in draft_lower or "request" in draft_lower, (
        f"Expected polished rewrite mentioning 'time off' or 'request', got: {data['draft']!r}"
    )


def test_improve_sentence_draft_is_polished_rewrite_not_template():
    """
    'improve this sentence: I need a day off tomorrow'
    draft must be a polished sentence, not a template scaffold.
    """
    from fastapi.testclient import TestClient
    from app.main import app
    client = TestClient(app)

    resp = client.post("/assistant/chat", json={
        "role": "EMPLOYEE",
        "question": "improve this sentence: I need a day off tomorrow",
        "context": {},
    })
    data = resp.json()
    assert data["draftType"] == "IMPROVE_TEXT"
    _assert_no_template_language(data["draft"], "improve this sentence: I need a day off tomorrow")
    draft_lower = (data["draft"] or "").lower()
    assert "day off" in draft_lower or "request" in draft_lower


def test_local_polish_text_function_directly():
    """Unit test for _polish_text — core rewrite logic."""
    from app.services.drafting_service import _polish_text

    cases = [
        ("I need a day off tomorrow",   "I would like to request a day off tomorrow."),
        ("I need time off tomorrow",    "I would like to request time off tomorrow."),
        ("I want a day off next week",  "I would like to a day off next week."),
    ]
    for raw, expected in cases:
        result = _polish_text(raw)
        assert result == expected, f"_polish_text({raw!r}) = {result!r}, expected {expected!r}"


def test_local_improve_prefix_re_strips_correctly():
    """Unit test: _IMPROVE_PREFIX_RE strips instruction prefix leaving the body."""
    from app.services.drafting_service import _IMPROVE_PREFIX_RE

    cases = [
        ("rephrase this message: I need a day off tomorrow", "I need a day off tomorrow"),
        ("rewrite this: I need time off tomorrow",           "I need time off tomorrow"),
        ("improve this sentence: I need a day off tomorrow", "I need a day off tomorrow"),
        ("rephrase: I need a day off",                       "I need a day off"),
        ("polish this text: hello world",                   "hello world"),
    ]
    for q, expected_body in cases:
        body = _IMPROVE_PREFIX_RE.sub('', q.strip()).strip()
        assert body == expected_body, (
            f"Prefix strip of {q!r} = {body!r}, expected {expected_body!r}"
        )


def test_gemini_improve_text_bad_response_falls_back_to_local_polish():
    """
    When Gemini returns a leave-template draft for an IMPROVE_TEXT request,
    the fallback _local_draft must produce a clean polished rewrite.
    This test simulates Gemini failing (timeout) so _local_draft is used.
    """
    import httpx
    from unittest.mock import patch, MagicMock
    from fastapi.testclient import TestClient
    from app.main import app
    client = TestClient(app)

    with patch("app.services.drafting_service.settings") as ms, \
         patch("app.services.drafting_service.httpx.Client") as mh:
        ms.gemini_enabled = True
        ms.gemini_api_key = "test-key"
        ms.gemini_model = "gemini-2.5-flash"
        ms.gemini_timeout_seconds = 10
        mh.return_value.__enter__.return_value.post.side_effect = httpx.TimeoutException("timeout")

        resp = client.post("/assistant/chat", json={
            "role": "EMPLOYEE",
            "question": "rephrase this message: I need a day off tomorrow",
            "context": {},
        })

    data = resp.json()
    assert data["draftType"] == "IMPROVE_TEXT"
    _assert_no_template_language(data["draft"], "rephrase this message: I need a day off tomorrow")
    assert data["source"] == "local_rules"


def test_leave_request_draft_still_has_template_language():
    """
    Regression: LEAVE_REQUEST drafts must still include template placeholders
    and the review disclaimer. The fix must not remove them from non-IMPROVE_TEXT types.
    """
    from fastapi.testclient import TestClient
    from app.main import app
    client = TestClient(app)

    resp = client.post("/assistant/chat", json={
        "role": "EMPLOYEE",
        "question": "draft a leave request for tomorrow",
        "context": {},
    })
    data = resp.json()
    assert data["draftType"] == "LEAVE_REQUEST"
    draft_lower = (data["draft"] or "").lower()
    # Leave template must still contain its structural elements
    assert "[" in (data["draft"] or ""), "Leave template must still contain placeholders"
    assert "review" in draft_lower or "personalise" in draft_lower, (
        "Leave template must still contain review disclaimer"
    )

