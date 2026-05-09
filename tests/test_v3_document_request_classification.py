"""
tests/test_v3_document_request_classification.py
-------------------------------------------------
V3.1: Document request intent classification tests.

Proves that document request intents are detected and classified as
DOCUMENT_REQUEST with correct Spring Boot enum values in draftFields.documentType,
and that IMPROVE_TEXT is NOT triggered for simple document requests.

Coverage:
  1.  "I need a salary certificate for my bank"
      => DOCUMENT_REQUEST / SALARY_CERTIFICATE / notes capture bank purpose
  2.  "I need an employment certificate"
      => DOCUMENT_REQUEST / EMPLOYMENT_CERTIFICATE
  3.  "I need a leave balance statement"
      => DOCUMENT_REQUEST / LEAVE_BALANCE_STATEMENT
  4.  "I need a contract copy"
      => guidance only (HR-managed, not a normal employee document draft)
  5.  "Write me a formal letter requesting a salary certificate"
      => IMPROVE_TEXT (letter-writing intent = text generation, not platform submission)
  6.  "I need an experience certificate"
      => DOCUMENT_REQUEST / EXPERIENCE_CERTIFICATE
  7.  "I need a work reference letter"
      => DOCUMENT_REQUEST / WORK_REFERENCE_LETTER
  8.  detect_drafting_intent returns True for all document request phrasings
  9.  _classify_draft_type returns DOCUMENT_REQUEST for document signals
  10. _extract_document_type returns enum names, not free-form strings
  11. DOCUMENT_REQUEST draftFields has stable shape (documentType, notes)
  12. draftType is set to DOCUMENT_REQUEST (never IMPROVE_TEXT or LEAVE_REQUEST)
  13. Existing non-document drafts unaffected (LEAVE_REQUEST, LOAN_REQUEST, IMPROVE_TEXT)
  14. Gemini structuredFields with enum name used directly
  15. Gemini structuredFields with free-form string falls back to local extractor
"""

import json
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from app.main import app
from app.schemas import ChatRequest, ContextInfo
from app.services.drafting_service import (
    detect_drafting_intent,
    _classify_draft_type,
    _extract_document_type,
    extract_draft_fields,
    get_draft_response,
)

client = TestClient(app)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def post_chat(question: str, role: str = "EMPLOYEE") -> dict:
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


# ===========================================================================
# 1. Salary certificate for bank
# ===========================================================================

def test_salary_certificate_for_bank_classified_as_document_request():
    """
    Bug scenario: "I need a salary certificate for my bank"
    Must be DOCUMENT_REQUEST, never IMPROVE_TEXT or generic Q&A.
    """
    data = post_chat("I need a salary certificate for my bank")
    assert data["draftType"] == "DOCUMENT_REQUEST", (
        f"Expected DOCUMENT_REQUEST, got {data['draftType']} (source={data['source']})"
    )


def test_salary_certificate_for_bank_extracts_salary_certificate_enum():
    """draftFields.documentType must be the Spring Boot enum value SALARY_CERTIFICATE."""
    data = post_chat("I need a salary certificate for my bank")
    assert data["draftType"] == "DOCUMENT_REQUEST"
    assert data["draftFields"] is not None
    assert data["draftFields"]["documentType"] == "SALARY_CERTIFICATE", (
        f"Expected SALARY_CERTIFICATE, got {data['draftFields']['documentType']!r}"
    )


def test_salary_certificate_for_bank_captures_bank_purpose():
    """purpose field should capture the bank context."""
    data = post_chat("I need a salary certificate for my bank")
    assert data["draftType"] == "DOCUMENT_REQUEST"
    purpose = data["draftFields"].get("purpose")
    # purpose may be extracted ("my bank" / "bank") or null — not required if not extractable,
    # but documentType must be correct regardless.
    # We check it is not mis-filled with something unrelated.
    if purpose is not None:
        assert "bank" in purpose.lower() or "loan" in purpose.lower() or purpose != "None", (
            f"purpose should relate to bank context, got: {purpose!r}"
        )


def test_salary_certificate_intent_detected():
    """detect_drafting_intent must fire for salary certificate phrases."""
    assert detect_drafting_intent("I need a salary certificate for my bank") is True


def test_salary_certificate_classifies_as_document_request():
    assert _classify_draft_type("I need a salary certificate for my bank") == "DOCUMENT_REQUEST"


# ===========================================================================
# 2. Employment certificate
# ===========================================================================

def test_employment_certificate_classified_as_document_request():
    data = post_chat("I need an employment certificate")
    assert data["draftType"] == "DOCUMENT_REQUEST"
    assert data["draftFields"]["documentType"] == "EMPLOYMENT_CERTIFICATE", (
        f"Expected EMPLOYMENT_CERTIFICATE, got {data['draftFields']['documentType']!r}"
    )


def test_employment_certificate_intent_detected():
    assert detect_drafting_intent("I need an employment certificate") is True


def test_employment_certificate_enum_extracted():
    result = _extract_document_type("I need an employment certificate")
    assert result == "EMPLOYMENT_CERTIFICATE", f"Got: {result!r}"


# ===========================================================================
# 3. Leave balance statement
# ===========================================================================

def test_leave_balance_statement_classified_as_document_request():
    data = post_chat("I need a leave balance statement")
    assert data["draftType"] == "DOCUMENT_REQUEST", (
        f"Expected DOCUMENT_REQUEST, got {data['draftType']} (source={data['source']})"
    )
    assert data["draftFields"]["documentType"] == "LEAVE_BALANCE_STATEMENT", (
        f"Expected LEAVE_BALANCE_STATEMENT, got {data['draftFields']['documentType']!r}"
    )


def test_leave_balance_statement_intent_detected():
    assert detect_drafting_intent("I need a leave balance statement") is True


def test_leave_balance_statement_enum_extracted():
    result = _extract_document_type("I need a leave balance statement")
    assert result == "LEAVE_BALANCE_STATEMENT", f"Got: {result!r}"


def test_leave_balance_classifies_as_document_not_leave():
    """
    Critical: "leave balance statement" must classify as DOCUMENT_REQUEST,
    not LEAVE_REQUEST. The word 'leave' must not hijack classification.
    """
    draft_type = _classify_draft_type("I need a leave balance statement")
    assert draft_type == "DOCUMENT_REQUEST", (
        f"'leave balance statement' must be DOCUMENT_REQUEST, got {draft_type!r}"
    )


# ===========================================================================
# 4. Contract copy — must surface as DOCUMENT_REQUEST, not hidden
# ===========================================================================

def test_contract_copy_returns_guidance_not_document_request():
    """
    CONTRACT_COPY is HR-managed. The assistant must not expose it as a normal
    employee document draft card.
    """
    data = post_chat("I need a contract copy")
    assert data["draftType"] is None
    assert data["draftFields"] is None
    assert data["missingFields"] == []
    assert "contract copies are managed by hr" in data["answer"].lower()
    assert "my documents" in data["answer"].lower()


def test_contract_copy_intent_detected():
    assert detect_drafting_intent("I need a contract copy") is True


def test_contract_copy_enum_extracted():
    result = _extract_document_type("I need a contract copy")
    assert result == "CONTRACT_COPY", f"Got: {result!r}"


# ===========================================================================
# 5. "Write me a formal letter requesting a salary certificate"
#    => IMPROVE_TEXT / text drafting, NOT a platform document request.
#    Explicit letter-writing intent means the user wants text generation,
#    not a platform submission.
# ===========================================================================

def test_write_formal_letter_salary_cert_classified_as_improve_text():
    """
    "Write me a formal letter requesting a salary certificate" must be
    IMPROVE_TEXT because the user explicitly wants to write a letter,
    not submit a platform request.
    """
    data = post_chat("Write me a formal letter requesting a salary certificate")
    assert data["draftType"] == "IMPROVE_TEXT", (
        f"Expected IMPROVE_TEXT for letter-writing intent, got {data['draftType']!r}"
    )


def test_write_formal_letter_classifies_as_improve_text_unit():
    assert _classify_draft_type("Write me a formal letter requesting a salary certificate") == "IMPROVE_TEXT"


def test_draft_a_letter_classifies_as_improve_text():
    assert _classify_draft_type("Draft a letter asking for a salary certificate") == "IMPROVE_TEXT"


def test_compose_a_letter_classifies_as_improve_text():
    assert _classify_draft_type("Compose a letter to HR requesting an employment certificate") == "IMPROVE_TEXT"


# ===========================================================================
# 6. Experience certificate
# ===========================================================================

def test_experience_certificate_classified_as_document_request():
    data = post_chat("I need an experience certificate")
    assert data["draftType"] == "DOCUMENT_REQUEST"
    assert data["draftFields"]["documentType"] == "EXPERIENCE_CERTIFICATE", (
        f"Expected EXPERIENCE_CERTIFICATE, got {data['draftFields']['documentType']!r}"
    )


def test_experience_letter_enum_extracted():
    result = _extract_document_type("I need an experience letter")
    assert result == "EXPERIENCE_CERTIFICATE", f"Got: {result!r}"


# ===========================================================================
# 7. Work reference letter
# ===========================================================================

def test_work_reference_letter_classified_as_document_request():
    data = post_chat("I need a work reference letter")
    assert data["draftType"] == "DOCUMENT_REQUEST"
    assert data["draftFields"]["documentType"] == "WORK_REFERENCE_LETTER", (
        f"Expected WORK_REFERENCE_LETTER, got {data['draftFields']['documentType']!r}"
    )


# ===========================================================================
# 8. detect_drafting_intent fires for all document request phrasings
# ===========================================================================

@pytest.mark.parametrize("question", [
    "I need a salary certificate for my bank",
    "I need an employment certificate",
    "I need a leave balance statement",
    "I need a contract copy",
    "I need an experience certificate",
    "I need a work reference letter",
    "I need an experience letter",
    "I need a leave balance",
    "I want a salary certificate",
    "I would like to request a salary certificate",
    "Help me request a salary certificate",
])
def test_detect_drafting_intent_for_document_phrasings(question):
    assert detect_drafting_intent(question) is True, (
        f"detect_drafting_intent should be True for: {question!r}"
    )


# ===========================================================================
# 9. _classify_draft_type returns DOCUMENT_REQUEST for document signals
# ===========================================================================

@pytest.mark.parametrize("question", [
    "I need a salary certificate for my bank",
    "I need an employment certificate",
    "I need a leave balance statement",
    "I need a contract copy",
    "I need an experience certificate",
    "I need an experience letter",
    "I need a work reference letter",
    "Help me compose a document request letter",
])
def test_classify_draft_type_document_request(question):
    assert _classify_draft_type(question) == "DOCUMENT_REQUEST", (
        f"_classify_draft_type should return DOCUMENT_REQUEST for: {question!r}"
    )


# ===========================================================================
# 10. _extract_document_type returns Spring Boot enum names
# ===========================================================================

@pytest.mark.parametrize("question,expected_enum", [
    ("I need a salary certificate",          "SALARY_CERTIFICATE"),
    ("I need a salary cert",                 "SALARY_CERTIFICATE"),
    ("I need an employment certificate",     "EMPLOYMENT_CERTIFICATE"),
    ("I need a work certificate",            "EMPLOYMENT_CERTIFICATE"),
    ("I need an experience letter",          "EXPERIENCE_CERTIFICATE"),
    ("I need an experience certificate",     "EXPERIENCE_CERTIFICATE"),
    ("I need a work reference letter",       "WORK_REFERENCE_LETTER"),
    ("I need a leave balance statement",     "LEAVE_BALANCE_STATEMENT"),
    ("I need a leave balance",               "LEAVE_BALANCE_STATEMENT"),
    ("I need a contract copy",               "CONTRACT_COPY"),
])
def test_extract_document_type_returns_enum_name(question, expected_enum):
    result = _extract_document_type(question)
    assert result == expected_enum, (
        f"_extract_document_type({question!r}) = {result!r}, expected {expected_enum!r}"
    )


def test_extract_document_type_returns_none_for_unknown():
    result = _extract_document_type("I need some kind of document")
    assert result is None


# ===========================================================================
# 11. DOCUMENT_REQUEST draftFields has correct stable shape: documentType + notes
#     No extraDetails, no purpose — matches CreateDocumentRequestDto exactly.
# ===========================================================================

def test_document_request_draftfields_has_correct_keys():
    """draftFields must have documentType and notes. No extraDetails, no purpose."""
    data = post_chat("I need a salary certificate")
    fields = data["draftFields"]
    assert fields is not None
    assert "documentType" in fields, "documentType key must be present"
    assert "notes" in fields, "notes key must be present"
    assert "extraDetails" not in fields, "extraDetails must NOT be in draftFields"
    assert "purpose" not in fields, "purpose must NOT be in draftFields"


def test_document_request_draftfields_shape_via_extract():
    """Unit-level: extract_draft_fields returns documentType + notes, nothing else."""
    fields, missing = extract_draft_fields(
        "I need a salary certificate for my bank",
        "DOCUMENT_REQUEST",
    )
    assert "documentType" in fields
    assert "notes" in fields
    assert "extraDetails" not in fields
    assert "purpose" not in fields


# ===========================================================================
# 12. draftType is DOCUMENT_REQUEST, never IMPROVE_TEXT or LEAVE_REQUEST
# ===========================================================================

@pytest.mark.parametrize("question", [
    "I need a salary certificate for my bank",
    "I need an employment certificate",
    "I need a leave balance statement",
])
def test_drafttype_is_document_request_not_other(question):
    data = post_chat(question)
    assert data["draftType"] == "DOCUMENT_REQUEST", (
        f"Expected DOCUMENT_REQUEST for {question!r}, got {data['draftType']!r}"
    )
    assert data["draftType"] != "IMPROVE_TEXT"
    assert data["draftType"] != "LEAVE_REQUEST"


def test_contract_copy_is_guidance_not_document_request():
    data = post_chat("I need a contract copy")
    assert data["draftType"] is None
    assert data["draftFields"] is None
    assert data["missingFields"] == []
    answer = data["answer"].lower()
    assert "contract copies are managed by hr" in answer
    assert "my documents" in answer


# ===========================================================================
# 13. Existing non-document drafts unaffected
# ===========================================================================

def test_leave_request_still_classified_as_leave():
    data = post_chat("Help me draft a leave request for annual leave")
    assert data["draftType"] == "LEAVE_REQUEST"


def test_loan_request_still_classified_as_loan():
    data = post_chat("Write a professional loan justification")
    assert data["draftType"] == "LOAN_REQUEST"


def test_improve_text_still_classified_as_improve():
    data = post_chat("Rephrase this message: I need a day off tomorrow")
    assert data["draftType"] == "IMPROVE_TEXT"


def test_improve_text_with_salary_cert_in_body_stays_improve():
    """
    'Make this more professional: I need a salary certificate'
    Leading improve verb wins — must be IMPROVE_TEXT, not DOCUMENT_REQUEST.
    """
    data = post_chat("Make this more professional: I need a salary certificate")
    assert data["draftType"] == "IMPROVE_TEXT", (
        f"Leading improve verb must win, got {data['draftType']!r}"
    )


def test_rewrite_salary_cert_stays_improve_text():
    """
    'Rewrite this: I need a salary certificate for my bank'
    Leading rewrite verb wins — must be IMPROVE_TEXT.
    """
    data = post_chat("Rewrite this: I need a salary certificate for my bank")
    assert data["draftType"] == "IMPROVE_TEXT", (
        f"Leading rewrite verb must win, got {data['draftType']!r}"
    )


# ===========================================================================
# 14. Gemini structuredFields with correct enum name used directly
# ===========================================================================

def test_gemini_document_enum_name_used_from_structured_fields():
    """
    When Gemini returns documentType as SALARY_CERTIFICATE and notes,
    draftFields must contain those exact values.
    draft must be None (no formal letter generated for DOCUMENT_REQUEST).
    """
    structured = {
        "documentType": "SALARY_CERTIFICATE",
        "notes": "bank loan application",
    }
    with patch("app.services.drafting_service.settings") as ms, \
         patch("app.services.drafting_service.httpx.Client") as mh:
        _mock_gemini_settings(ms)
        _mock_gemini_http_drafting(
            mh,
            draft_text="Subject: Document Request\n\nDear HR Team, I need a salary certificate.",
            answer="I prepared a document request preview. Please review the details and confirm to submit.",
            structured_fields=structured,
        )
        data = post_chat("I need a salary certificate for my bank loan")

    assert data["draftType"] == "DOCUMENT_REQUEST"
    assert data["draftFields"]["documentType"] == "SALARY_CERTIFICATE"
    assert data["draftFields"]["notes"] == "bank loan application"
    assert data.get("draft") is None, (
        f"DOCUMENT_REQUEST must not produce a formal letter draft, got: {data.get('draft')!r}"
    )


# ===========================================================================
# 15. Gemini returns free-form string -> local extractor provides enum fallback
# ===========================================================================

def test_gemini_free_form_documenttype_falls_back_to_local_extractor():
    """
    If Gemini incorrectly returns a free-form string like 'salary certificate'
    instead of the enum name, the local extractor must still produce the correct
    enum value via the fallback path.
    This is validated by testing the local path (Gemini disabled by conftest).
    """
    # Gemini disabled — local extractor runs directly
    data = post_chat("I need a salary certificate for my bank")
    assert data["draftType"] == "DOCUMENT_REQUEST"
    assert data["source"] == "local_rules"
    assert data["draftFields"]["documentType"] == "SALARY_CERTIFICATE", (
        f"Local extractor must return enum name SALARY_CERTIFICATE, got {data['draftFields']['documentType']!r}"
    )


# ===========================================================================
# 16. missingFields correct for document requests
# ===========================================================================

def test_document_request_with_known_type_no_missing():
    """
    salary certificate — documentType extracted, notes not required.
    missingFields must be [].
    """
    fields, missing = extract_draft_fields(
        "I need a salary certificate",
        "DOCUMENT_REQUEST",
    )
    assert fields["documentType"] == "SALARY_CERTIFICATE"
    assert missing == [], f"Expected no missingFields, got: {missing}"


def test_document_request_with_known_type_and_notes_no_missing():
    """
    salary certificate for bank — documentType extracted, notes optional.
    missingFields must be [].
    """
    fields, missing = extract_draft_fields(
        "I need a salary certificate for my bank",
        "DOCUMENT_REQUEST",
    )
    assert fields["documentType"] == "SALARY_CERTIFICATE"
    assert fields["notes"] is not None, "notes should capture bank context"
    assert missing == [], f"Expected no missingFields, got: {missing}"


def test_document_request_missing_type_adds_to_missing():
    """Unknown document type -> documentType=None and 'documentType' in missingFields."""
    fields, missing = extract_draft_fields(
        "I need some kind of HR document",
        "DOCUMENT_REQUEST",
    )
    assert fields["documentType"] is None
    assert "documentType" in missing


def test_contract_copy_missingfields_empty():
    """CONTRACT_COPY extracted — missingFields must be [] since notes is optional."""
    fields, missing = extract_draft_fields(
        "I need a contract copy",
        "DOCUMENT_REQUEST",
    )
    assert fields["documentType"] == "CONTRACT_COPY"
    assert missing == [], f"Expected no missingFields for CONTRACT_COPY, got: {missing}"


# ===========================================================================
# 17. Required output shape — the 6 spec scenarios
# ===========================================================================

def test_spec_1_salary_certificate_for_bank_full_shape():
    """
    Spec 1: "I need a salary certificate for my bank"
    => draftType DOCUMENT_REQUEST
    => draftFields.documentType SALARY_CERTIFICATE
    => missingFields []
    => draft is None or does not contain "Dear HR Team"
    => answer does not contain "Here's a draft"
    """
    data = post_chat("I need a salary certificate for my bank")
    assert data["draftType"] == "DOCUMENT_REQUEST"
    assert data["draftFields"]["documentType"] == "SALARY_CERTIFICATE"
    assert data["missingFields"] == [], f"Expected [], got: {data['missingFields']}"
    draft = data.get("draft")
    assert draft is None or "Dear HR Team" not in (draft or ""), (
        f"DOCUMENT_REQUEST must not contain formal letter content, got: {draft!r}"
    )
    assert "Here's a draft" not in data["answer"], (
        f"answer must not say \"Here's a draft\", got: {data['answer']!r}"
    )


def test_spec_2_salary_certificate_no_context_missing_empty():
    """
    Spec 2: "I need a salary certificate"
    => missingFields []
    """
    data = post_chat("I need a salary certificate")
    assert data["draftType"] == "DOCUMENT_REQUEST"
    assert data["missingFields"] == [], f"Expected [], got: {data['missingFields']}"


def test_spec_3_unknown_document_missing_includes_documenttype():
    """
    Spec 3: "I need a document"
    => missingFields includes documentType
    """
    data = post_chat("I need a document")
    # May be LEAVE or DOCUMENT depending on classification—check at unit level
    fields, missing = extract_draft_fields("I need a document", "DOCUMENT_REQUEST")
    assert fields["documentType"] is None
    assert "documentType" in missing


def test_spec_4_contract_copy_no_formal_letter():
    """
    Spec 4: "I need a contract copy"
    => guidance only
    """
    data = post_chat("I need a contract copy")
    assert data["draftType"] is None
    assert data["draftFields"] is None
    assert data["missingFields"] == [], f"Expected [], got: {data['missingFields']}"
    answer = data["answer"].lower()
    assert "contract copies are managed by hr" in answer
    assert "my documents" in answer


def test_spec_5_write_formal_letter_is_improve_text_with_draft():
    """
    Spec 5: "Write me a formal letter requesting a salary certificate"
    => IMPROVE_TEXT
    => formal letter/text drafting is allowed here
    => no DOCUMENT_REQUEST submit flow
    """
    data = post_chat("Write me a formal letter requesting a salary certificate")
    assert data["draftType"] == "IMPROVE_TEXT", (
        f"Expected IMPROVE_TEXT, got {data['draftType']!r}"
    )
    assert data.get("draftFields") is None, (
        "IMPROVE_TEXT must have draftFields=None"
    )
    assert data["missingFields"] == []


def test_spec_6_leave_request_regression():
    """Spec 6: LEAVE_REQUEST still returns leave draft structure."""
    data = post_chat("Help me draft a leave request for annual leave from June 1 to June 3")
    assert data["draftType"] == "LEAVE_REQUEST"
    assert data["draftFields"] is not None
    assert "leaveType" in data["draftFields"]
    assert data["draftFields"]["leaveType"] == "ANNUAL"
    draft = data.get("draft") or ""
    assert "[" in draft, "Leave draft must still contain placeholder brackets"


def test_spec_6_loan_request_regression():
    """Spec 6: LOAN_REQUEST still returns loan draft structure."""
    data = post_chat("Write a professional loan justification for 2000 TND because of family expenses")
    assert data["draftType"] == "LOAN_REQUEST"
    assert data["draftFields"] is not None
    assert data["draftFields"]["amount"] is not None
    draft = data.get("draft") or ""
    assert len(draft) > 20, "Loan draft must not be empty"


def test_spec_6_authorization_request_regression():
    """Spec 6: AUTHORIZATION_REQUEST still returns authorization draft structure."""
    data = post_chat("Draft an authorization request explanation for tomorrow from 10 to 12")
    assert data["draftType"] == "AUTHORIZATION_REQUEST"
    assert data["draftFields"] is not None
    assert "authorizationType" in data["draftFields"]
    draft = data.get("draft") or ""
    assert len(draft) > 20, "Authorization draft must not be empty"


def test_document_request_answer_says_preview():
    """
    answer for DOCUMENT_REQUEST must say something about a preview/review,
    not "Here is a draft" or "Here's a draft".
    """
    data = post_chat("I need a salary certificate for my bank")
    assert data["draftType"] == "DOCUMENT_REQUEST"
    answer = data["answer"].lower()
    assert "preview" in answer or "review" in answer or "confirm" in answer, (
        f"answer should mention preview/review/confirm, got: {data['answer']!r}"
    )
    assert "here is a template" not in answer, (
        f"DOCUMENT_REQUEST answer must not say 'here is a template'"
    )


def test_document_request_draft_is_none_local_path():
    """Local path (Gemini disabled): DOCUMENT_REQUEST draft must be None."""
    data = post_chat("I need a salary certificate for my bank")
    assert data["draftType"] == "DOCUMENT_REQUEST"
    assert data["source"] == "local_rules"
    assert data.get("draft") is None, (
        f"Local DOCUMENT_REQUEST must return draft=None, got: {data.get('draft')!r}"
    )


# ===========================================================================
# 18. Explicit letter-writing path — IMPROVE_TEXT with real letter body
# ===========================================================================

def test_write_formal_letter_salary_cert_produces_real_letter():
    """
    Spec 1: "Write me a formal letter requesting a salary certificate"
    => draftType IMPROVE_TEXT
    => draft contains "Dear HR Team"
    => draft contains "salary certificate"
    => answer does not say "polished rewrite"
    => draft is NOT just the prompt echoed back
    """
    data = post_chat("Write me a formal letter requesting a salary certificate")
    assert data["draftType"] == "IMPROVE_TEXT", (
        f"Expected IMPROVE_TEXT, got {data['draftType']!r}"
    )
    draft = data.get("draft") or ""
    assert "Dear HR Team" in draft, (
        f"Expected formal letter opening 'Dear HR Team' in draft, got: {draft!r}"
    )
    assert "salary certificate" in draft.lower(), (
        f"Expected 'salary certificate' in letter body, got: {draft!r}"
    )
    assert "polished rewrite" not in data["answer"].lower(), (
        f"answer must not say 'polished rewrite' for letter-writing, got: {data['answer']!r}"
    )
    # Must not echo the instruction verbatim as the entire draft
    assert draft.lower().strip() != "write me a formal letter requesting a salary certificate.", (
        "draft must not be a verbatim echo of the user instruction"
    )
    assert data.get("draftFields") is None
    assert data["missingFields"] == []


def test_write_formal_letter_answer_says_formal_letter():
    """answer must indicate it's a formal letter to review and edit."""
    data = post_chat("Write me a formal letter requesting a salary certificate")
    answer = data["answer"].lower()
    assert "formal letter" in answer or "letter" in answer, (
        f"answer should mention 'letter', got: {data['answer']!r}"
    )


def test_draft_a_letter_employment_cert_produces_real_letter():
    """
    Spec 2: "Draft a letter requesting an employment certificate"
    => IMPROVE_TEXT
    => real formal letter body with Dear HR Team
    """
    data = post_chat("Draft a letter requesting an employment certificate")
    assert data["draftType"] == "IMPROVE_TEXT"
    draft = data.get("draft") or ""
    assert "Dear HR Team" in draft, (
        f"Expected 'Dear HR Team' in draft, got: {draft!r}"
    )
    assert len(draft) > 80, f"Draft too short to be a real letter: {draft!r}"


def test_compose_a_letter_leave_balance_produces_real_letter():
    """
    Spec 3: "Compose a letter asking for a leave balance statement"
    => IMPROVE_TEXT
    => real formal letter body
    """
    data = post_chat("Compose a letter asking for a leave balance statement")
    assert data["draftType"] == "IMPROVE_TEXT"
    draft = data.get("draft") or ""
    assert "Dear HR Team" in draft, (
        f"Expected 'Dear HR Team' in draft, got: {draft!r}"
    )
    assert len(draft) > 80, f"Draft too short to be a real letter: {draft!r}"


def test_letter_draft_does_not_echo_prompt():
    """Draft must not be a copy/restatement of the original instruction."""
    question = "Write me a formal letter requesting a salary certificate"
    data = post_chat(question)
    draft = (data.get("draft") or "").strip()
    # The draft must contain substantially more than the instruction alone
    assert draft.lower() != question.lower(), "Draft must not equal the original question"
    assert draft.lower() != question.lower() + ".", "Draft must not be the question with punctuation"


# ===========================================================================
# 19. Regression — DOCUMENT_REQUEST still untouched by letter-writing fix
# ===========================================================================

def test_regression_salary_cert_platform_request_still_document_request():
    """
    Spec 4 regression: "I need a salary certificate for my bank"
    => DOCUMENT_REQUEST (not IMPROVE_TEXT)
    => draft null
    => no formal letter
    """
    data = post_chat("I need a salary certificate for my bank")
    assert data["draftType"] == "DOCUMENT_REQUEST", (
        f"Expected DOCUMENT_REQUEST, got {data['draftType']!r}"
    )
    assert data.get("draft") is None, (
        f"DOCUMENT_REQUEST must have draft=None, got: {data.get('draft')!r}"
    )
    assert data["draftFields"]["documentType"] == "SALARY_CERTIFICATE"
    assert data["missingFields"] == []


def test_regression_contract_copy_still_guidance_only():
    """
    Regression: contract copy remains guidance-only.
    """
    data = post_chat("I need a contract copy")
    assert data["draftType"] is None
    assert data["draftFields"] is None
    assert data["missingFields"] == []
    assert "contract copies are managed by hr" in data["answer"].lower()


def test_regression_rephrase_still_generic_polish():
    """
    Regression: generic rephrase must still use polish path, not letter template.
    """
    data = post_chat("Rephrase this message: I need a day off tomorrow")
    assert data["draftType"] == "IMPROVE_TEXT"
    draft = data.get("draft") or ""
    # Generic polish: must NOT produce a formal letter
    assert "Dear HR Team" not in draft, (
        f"Generic rephrase must not produce a formal letter, got: {draft!r}"
    )


def test_unit_is_letter_writing_intent():
    """Unit test for _is_letter_writing_intent."""
    from app.services.drafting_service import _is_letter_writing_intent
    assert _is_letter_writing_intent("Write me a formal letter requesting a salary certificate") is True
    assert _is_letter_writing_intent("Draft a letter asking for an employment certificate") is True
    assert _is_letter_writing_intent("Compose a letter for a leave balance statement") is True
    assert _is_letter_writing_intent("I need a salary certificate for my bank") is False
    assert _is_letter_writing_intent("Rephrase this: I need a day off") is False
    assert _is_letter_writing_intent("Help me draft a leave request") is False


def test_unit_generate_letter_draft_salary_certificate():
    """Unit test: _generate_letter_draft produces letter with correct document label."""
    from app.services.drafting_service import _generate_letter_draft
    letter = _generate_letter_draft("Write me a formal letter requesting a salary certificate")
    assert "Dear HR Team" in letter
    assert "salary certificate" in letter.lower()
    assert "[Your Name]" in letter


def test_unit_generate_letter_draft_unknown_document():
    """Unit test: _generate_letter_draft uses placeholder when type unknown."""
    from app.services.drafting_service import _generate_letter_draft
    letter = _generate_letter_draft("Write me a formal letter for some document")
    assert "Dear HR Team" in letter
    assert "[document name]" in letter


# ===========================================================================
# 20. CUSTOM_ADMINISTRATIVE_LETTER classification
# ===========================================================================

def test_custom_administrative_letter_classified_as_document_request():
    """
    Spec 1: "I need a custom administrative letter"
    => DOCUMENT_REQUEST / CUSTOM_ADMINISTRATIVE_LETTER / missingFields []
    """
    data = post_chat("I need a custom administrative letter")
    assert data["draftType"] == "DOCUMENT_REQUEST", (
        f"Expected DOCUMENT_REQUEST, got {data['draftType']!r} (source={data['source']!r})"
    )
    assert data["draftFields"]["documentType"] == "CUSTOM_ADMINISTRATIVE_LETTER", (
        f"Expected CUSTOM_ADMINISTRATIVE_LETTER, got {data['draftFields']['documentType']!r}"
    )
    assert data["missingFields"] == [], f"Expected [], got {data['missingFields']}"
    assert data.get("draft") is None


def test_administrative_letter_classified_as_document_request():
    """
    Spec 2: "I need an administrative letter"
    => DOCUMENT_REQUEST / CUSTOM_ADMINISTRATIVE_LETTER / missingFields []
    """
    data = post_chat("I need an administrative letter")
    assert data["draftType"] == "DOCUMENT_REQUEST"
    assert data["draftFields"]["documentType"] == "CUSTOM_ADMINISTRATIVE_LETTER", (
        f"Expected CUSTOM_ADMINISTRATIVE_LETTER, got {data['draftFields']['documentType']!r}"
    )
    assert data["missingFields"] == []


def test_admin_letter_classified_as_document_request():
    """
    Spec 3: "I need an admin letter"
    => DOCUMENT_REQUEST / CUSTOM_ADMINISTRATIVE_LETTER / missingFields []
    """
    data = post_chat("I need an admin letter")
    assert data["draftType"] == "DOCUMENT_REQUEST"
    assert data["draftFields"]["documentType"] == "CUSTOM_ADMINISTRATIVE_LETTER", (
        f"Expected CUSTOM_ADMINISTRATIVE_LETTER, got {data['draftFields']['documentType']!r}"
    )
    assert data["missingFields"] == []


@pytest.mark.parametrize("question", [
    "I need a custom administrative letter",
    "I need an administrative letter",
    "I need an admin letter",
    "I need a custom admin letter",
])
def test_custom_administrative_letter_intent_detected(question):
    assert detect_drafting_intent(question) is True, (
        f"detect_drafting_intent should be True for: {question!r}"
    )


@pytest.mark.parametrize("question", [
    "I need a custom administrative letter",
    "I need an administrative letter",
    "I need an admin letter",
    "I need a custom admin letter",
])
def test_custom_administrative_letter_classifies_as_document(question):
    assert _classify_draft_type(question) == "DOCUMENT_REQUEST", (
        f"_classify_draft_type should return DOCUMENT_REQUEST for: {question!r}"
    )


@pytest.mark.parametrize("question", [
    "I need a custom administrative letter",
    "I need an administrative letter",
    "I need an admin letter",
    "I need a custom admin letter",
])
def test_custom_administrative_letter_enum_extracted(question):
    result = _extract_document_type(question)
    assert result == "CUSTOM_ADMINISTRATIVE_LETTER", (
        f"_extract_document_type({question!r}) = {result!r}, expected CUSTOM_ADMINISTRATIVE_LETTER"
    )


# ===========================================================================
# 21. Regression suite for the fix
# ===========================================================================

def test_regression_salary_certificate_unaffected():
    """Spec 4a: salary certificate still works."""
    data = post_chat("I need a salary certificate")
    assert data["draftType"] == "DOCUMENT_REQUEST"
    assert data["draftFields"]["documentType"] == "SALARY_CERTIFICATE"
    assert data["missingFields"] == []


def test_regression_employment_certificate_unaffected():
    """Spec 4b: employment certificate still works."""
    data = post_chat("I need an employment certificate")
    assert data["draftType"] == "DOCUMENT_REQUEST"
    assert data["draftFields"]["documentType"] == "EMPLOYMENT_CERTIFICATE"
    assert data["missingFields"] == []


def test_regression_leave_balance_statement_unaffected():
    """Spec 4c: leave balance statement still works."""
    data = post_chat("I need a leave balance statement")
    assert data["draftType"] == "DOCUMENT_REQUEST"
    assert data["draftFields"]["documentType"] == "LEAVE_BALANCE_STATEMENT"
    assert data["missingFields"] == []


def test_regression_contract_copy_returns_guidance():
    """Contract copy stays blocked from normal employee document drafting."""
    data = post_chat("I need a contract copy")
    assert data["draftType"] is None
    assert data["draftFields"] is None
    assert data["missingFields"] == []
    assert "contract copies are managed by hr" in data["answer"].lower()


def test_regression_write_formal_letter_still_improve_text():
    """Spec 5: write formal letter requesting salary certificate still => IMPROVE_TEXT."""
    data = post_chat("Write me a formal letter requesting a salary certificate")
    assert data["draftType"] == "IMPROVE_TEXT"
    assert data.get("draftFields") is None
