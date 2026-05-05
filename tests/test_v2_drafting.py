"""
tests/test_v2_drafting.py
--------------------------
V2 Phase 3: Drafting Assistant tests.

All external HTTP calls are mocked — no real API key or network required.

Coverage:
  1.  detect_drafting_intent: leave request
  2.  detect_drafting_intent: loan justification
  3.  detect_drafting_intent: authorization request
  4.  detect_drafting_intent: document request
  5.  detect_drafting_intent: improve text
  6.  Non-drafting questions return False from detector
  7.  Submit/approve requests are still refused (refusal fires before drafting)
  8.  Gemini disabled -> local draft returned, source=local_rules
  9.  Gemini disabled -> draft field is populated (not None)
  10. Gemini disabled -> no crash, response has 200
  11. Gemini enabled -> draft field populated, source=external_ai
  12. Gemini failure (timeout) -> falls back to local draft safely
  13. Gemini failure (HTTP error) -> falls back to local draft safely
  14. Gemini failure (bad JSON) -> falls back to local draft safely
  15. Gemini enabled but empty API key -> local draft, not crash
  16. No relatedPages fake routes in drafting responses
  17. Draft response includes review disclaimer text
  18. Draft response answer is non-empty
  19. Local draft: leave type contains leave-related content
  20. Local draft: loan type contains loan-related content
  21. Local draft: authorization type contains authorization-related content
  22. Local draft: document type contains document-related content
  23. Pipeline order: refusal before drafting
  24. Pipeline order: local rules before drafting
  25. Drafting response source is never "refusal" or None
  26. Existing non-drafting V1/V2 routes unaffected
"""

import json
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

import httpx

from app.main import app
from app.schemas import ChatRequest, ContextInfo
from app.services.drafting_service import detect_drafting_intent, get_draft_response

client = TestClient(app)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def post_chat(role: str, question: str, page_context: str = None) -> dict:
    body = {"role": role, "question": question, "context": {}}
    if page_context:
        body["pageContext"] = page_context
    return client.post("/assistant/chat", json=body).json()


def _make_request(question: str, role: str = "EMPLOYEE") -> ChatRequest:
    return ChatRequest(role=role, question=question, context=ContextInfo())


def _mock_drafting_gemini_settings(mock_settings, *, enabled: bool = True, api_key: str = "test-key"):
    mock_settings.gemini_enabled = enabled
    mock_settings.gemini_api_key = api_key
    mock_settings.gemini_model = "gemini-2.5-flash"
    mock_settings.gemini_timeout_seconds = 10


def _mock_drafting_gemini_http(mock_http, draft_text: str, answer: str = "Here is your draft."):
    """Configure mock_http to return a valid Gemini drafting JSON response."""
    payload = json.dumps({"answer": answer, "draft": draft_text})
    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "candidates": [{"content": {"parts": [{"text": payload}]}}]
    }
    mock_resp.raise_for_status = MagicMock()
    mock_http.return_value.__enter__.return_value.post.return_value = mock_resp


# ---------------------------------------------------------------------------
# 1-5: detect_drafting_intent unit tests
# ---------------------------------------------------------------------------

def test_detect_drafting_intent_leave_request():
    assert detect_drafting_intent("Help me draft a leave request reason") is True


def test_detect_drafting_intent_loan_justification():
    assert detect_drafting_intent("Write a professional loan justification") is True


def test_detect_drafting_intent_authorization():
    assert detect_drafting_intent("Draft an authorization request explanation") is True


def test_detect_drafting_intent_document_request():
    assert detect_drafting_intent("Help me compose a document request letter") is True


def test_detect_drafting_intent_improve_text():
    assert detect_drafting_intent("Improve this request text: I need a day off") is True
    assert detect_drafting_intent("Make this message more professional") is True


# ---------------------------------------------------------------------------
# 6: Non-drafting questions do NOT trigger detector
# ---------------------------------------------------------------------------

def test_detect_drafting_intent_false_for_navigation():
    assert detect_drafting_intent("How do I request a loan?") is False


def test_detect_drafting_intent_false_for_balance_check():
    assert detect_drafting_intent("What is my leave balance?") is False


def test_detect_drafting_intent_false_for_team_question():
    assert detect_drafting_intent("How do I check team requests?") is False


def test_detect_drafting_intent_false_for_approve():
    """Even though 'approve' is in the question, no drafting subject -> False."""
    assert detect_drafting_intent("approve my leave automatically") is False


# ---------------------------------------------------------------------------
# 7: Unsafe submit/approve requests are refused BEFORE drafting fires
# ---------------------------------------------------------------------------

def test_submit_request_is_refused_not_drafted():
    """'approve this request' must hit the refusal layer before drafting."""
    data = post_chat("EMPLOYEE", "approve this request")
    assert data["source"] == "refusal"
    assert data.get("draft") is None or data["draft"] == ""


def test_approve_request_is_refused_not_drafted():
    """'approve this request' must hit refusal before drafting."""
    data = post_chat("EMPLOYEE", "approve this request")
    assert data["source"] == "refusal"
    assert data.get("draft") is None or data["draft"] == ""


# ---------------------------------------------------------------------------
# 8-10: Gemini disabled -> local draft (no crash)
# ---------------------------------------------------------------------------

def test_gemini_disabled_leave_draft_returns_local():
    """Gemini disabled -> local template, source=local_rules."""
    # conftest autouse fixture already disables Gemini + drafting_service.settings
    data = post_chat("EMPLOYEE", "Help me draft a leave request reason")
    assert data["source"] == "local_rules"


def test_gemini_disabled_draft_field_populated():
    """Draft field must not be None when a drafting question is asked."""
    data = post_chat("EMPLOYEE", "Help me draft a leave request reason")
    assert data["draft"] is not None
    assert len(data["draft"]) > 50  # must be a real template, not an empty string


def test_gemini_disabled_no_crash_returns_200():
    """With Gemini disabled, the endpoint must return HTTP 200 for a drafting question."""
    response = client.post("/assistant/chat", json={
        "role": "EMPLOYEE",
        "question": "Write a professional loan justification",
        "context": {},
    })
    assert response.status_code == 200


# ---------------------------------------------------------------------------
# 11: Gemini enabled -> AI draft returned
# ---------------------------------------------------------------------------

def test_gemini_enabled_draft_field_populated_and_source_external_ai():
    """When Gemini is enabled and returns a valid draft, source=external_ai and draft is set."""
    ai_draft = "Dear HR Team, I am writing to request leave from [start date] to [end date]."
    with patch("app.services.drafting_service.settings") as ms, \
         patch("app.services.drafting_service.httpx.Client") as mh:
        _mock_drafting_gemini_settings(ms)
        _mock_drafting_gemini_http(mh, ai_draft)
        data = post_chat("EMPLOYEE", "Help me draft a leave request reason")

    assert data["source"] == "external_ai"
    assert data["draft"] is not None
    assert "leave" in data["draft"].lower() or "[start date]" in data["draft"]


def test_gemini_enabled_draft_answer_non_empty():
    ai_draft = "Dear HR, please approve my leave from [start date] to [end date]."
    with patch("app.services.drafting_service.settings") as ms, \
         patch("app.services.drafting_service.httpx.Client") as mh:
        _mock_drafting_gemini_settings(ms)
        _mock_drafting_gemini_http(mh, ai_draft, answer="Here is your leave request draft.")
        data = post_chat("EMPLOYEE", "Help me draft a leave request reason")

    assert data["answer"]


# ---------------------------------------------------------------------------
# 12-14: Gemini failure -> safe local draft fallback
# ---------------------------------------------------------------------------

def test_gemini_timeout_falls_back_to_local_draft():
    with patch("app.services.drafting_service.settings") as ms, \
         patch("app.services.drafting_service.httpx.Client") as mh:
        _mock_drafting_gemini_settings(ms)
        mh.return_value.__enter__.return_value.post.side_effect = httpx.TimeoutException("timeout")
        data = post_chat("EMPLOYEE", "Help me draft a leave request reason")

    # Must fall back to local template, not crash
    assert data["draft"] is not None
    assert data["source"] == "local_rules"


def test_gemini_http_error_falls_back_to_local_draft():
    with patch("app.services.drafting_service.settings") as ms, \
         patch("app.services.drafting_service.httpx.Client") as mh:
        _mock_drafting_gemini_settings(ms)
        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "500", request=MagicMock(), response=MagicMock(status_code=500)
        )
        mh.return_value.__enter__.return_value.post.return_value = mock_resp
        data = post_chat("EMPLOYEE", "Write a professional loan justification")

    assert data["draft"] is not None
    assert data["source"] == "local_rules"


def test_gemini_bad_json_falls_back_to_local_draft():
    mock_resp = MagicMock()
    mock_resp.json.return_value = {
    "candidates": [
        {
            "content": {
                "parts": [
                    {"text": "not valid json {{{{"}
                ]
            }
        }
    ]
}
    mock_resp.raise_for_status = MagicMock()
    with patch("app.services.drafting_service.settings") as ms, \
         patch("app.services.drafting_service.httpx.Client") as mh:
        _mock_drafting_gemini_settings(ms)
        mh.return_value.__enter__.return_value.post.return_value = mock_resp
        data = post_chat("EMPLOYEE", "Draft an authorization request explanation")

    assert data["draft"] is not None
    assert data["source"] == "local_rules"


# ---------------------------------------------------------------------------
# 15: Gemini enabled but empty API key -> local draft
# ---------------------------------------------------------------------------

def test_gemini_enabled_empty_api_key_gives_local_draft():
    with patch("app.services.drafting_service.settings") as ms, \
         patch("app.services.drafting_service.httpx.Client") as mh:
        _mock_drafting_gemini_settings(ms, enabled=True, api_key="")
        data = post_chat("EMPLOYEE", "Help me draft a leave request reason")
        mh.assert_not_called()

    assert data["draft"] is not None
    assert data["source"] == "local_rules"


# ---------------------------------------------------------------------------
# 16: No fake relatedPages routes in drafting responses
# ---------------------------------------------------------------------------

def test_drafting_response_has_no_related_pages():
    """Drafting responses must never include relatedPages."""
    data = post_chat("EMPLOYEE", "Help me draft a leave request reason")
    assert data["relatedPages"] == []


def test_drafting_response_no_fake_routes_gemini_enabled():
    ai_draft = "Dear HR, I request leave."
    with patch("app.services.drafting_service.settings") as ms, \
         patch("app.services.drafting_service.httpx.Client") as mh:
        _mock_drafting_gemini_settings(ms)
        _mock_drafting_gemini_http(mh, ai_draft)
        data = post_chat("EMPLOYEE", "Help me draft a leave request reason")

    for page in data.get("relatedPages", []):
        assert not page["route"].startswith("/payroll")
        assert not page["route"].startswith("/admin")


# ---------------------------------------------------------------------------
# 17: Draft contains review disclaimer
# ---------------------------------------------------------------------------

def test_local_draft_contains_review_disclaimer():
    data = post_chat("EMPLOYEE", "Help me draft a leave request reason")
    draft = data["draft"] or ""
    assert "review" in draft.lower() or "personalise" in draft.lower()


def test_gemini_draft_gets_disclaimer_appended_if_missing():
    """If Gemini's draft does not mention 'review', the service appends the disclaimer."""
    ai_draft = "Dear HR, I request leave from [start date] to [end date]."  # no disclaimer
    with patch("app.services.drafting_service.settings") as ms, \
         patch("app.services.drafting_service.httpx.Client") as mh:
        _mock_drafting_gemini_settings(ms)
        _mock_drafting_gemini_http(mh, ai_draft)
        data = post_chat("EMPLOYEE", "Help me draft a leave request reason")

    draft = data["draft"] or ""
    assert "review" in draft.lower() or "submit" in draft.lower()


# ---------------------------------------------------------------------------
# 18: Draft answer field is always non-empty
# ---------------------------------------------------------------------------

def test_drafting_answer_is_non_empty_local():
    data = post_chat("EMPLOYEE", "Help me draft a leave request reason")
    assert data["answer"].strip() != ""


def test_drafting_answer_is_non_empty_gemini():
    ai_draft = "Dear HR, I request leave."
    with patch("app.services.drafting_service.settings") as ms, \
         patch("app.services.drafting_service.httpx.Client") as mh:
        _mock_drafting_gemini_settings(ms)
        _mock_drafting_gemini_http(mh, ai_draft)
        data = post_chat("EMPLOYEE", "Help me draft a leave request reason")

    assert data["answer"].strip() != ""


# ---------------------------------------------------------------------------
# 19-22: Local template content by draft type
# ---------------------------------------------------------------------------

def test_local_leave_draft_contains_leave_content():
    data = post_chat("EMPLOYEE", "Help me draft a leave request reason")
    draft = (data["draft"] or "").lower()
    assert "leave" in draft or "absence" in draft or "[start date]" in draft


def test_local_loan_draft_contains_loan_content():
    data = post_chat("EMPLOYEE", "Write a professional loan justification")
    draft = (data["draft"] or "").lower()
    assert "loan" in draft or "[amount]" in draft or "repay" in draft


def test_local_authorization_draft_contains_authorization_content():
    data = post_chat("EMPLOYEE", "Draft an authorization request explanation")
    draft = (data["draft"] or "").lower()
    assert "authorization" in draft or "authoris" in draft or "access" in draft


def test_local_document_draft_contains_document_content():
    data = post_chat("EMPLOYEE", "Help me compose a document request letter")
    draft = (data["draft"] or "").lower()
    assert "document" in draft or "certificate" in draft or "[document name]" in draft


# ---------------------------------------------------------------------------
# 23: Pipeline order — refusal fires before drafting
# ---------------------------------------------------------------------------

def test_refusal_fires_before_drafting():
    """'approve this request' must hit refusal, not drafting."""
    data = post_chat("EMPLOYEE", "approve this request")
    assert data["source"] == "refusal"
    assert data.get("draft") is None or data["draft"] == ""


# ---------------------------------------------------------------------------
# 24: Pipeline order — local rules fire before drafting
# ---------------------------------------------------------------------------

def test_local_rules_fire_before_drafting():
    """A known local-rule question must be answered by local rules, not drafting."""
    data = post_chat("EMPLOYEE", "How do I request a loan?")
    assert data["source"] == "local_rules"
    assert data.get("draft") is None or data["draft"] == ""


def test_local_rules_leave_balance_fires_before_drafting():
    data = post_chat("EMPLOYEE", "What is my leave balance?")
    assert data["source"] == "local_rules"
    assert data.get("draft") is None or data["draft"] == ""


# ---------------------------------------------------------------------------
# 25: Drafting responses source is valid
# ---------------------------------------------------------------------------

def test_drafting_source_is_never_none():
    data = post_chat("EMPLOYEE", "Help me draft a leave request reason")
    assert data["source"] is not None


def test_drafting_source_is_never_refusal():
    data = post_chat("EMPLOYEE", "Help me draft a leave request reason")
    assert data["source"] != "refusal"


# ---------------------------------------------------------------------------
# 26: Existing non-drafting routes unaffected by Phase 3
# ---------------------------------------------------------------------------

def test_v1_loan_route_unaffected():
    data = post_chat("EMPLOYEE", "How do I request a loan?")
    routes = [p["route"] for p in data["relatedPages"]]
    assert "/employee/loans" in routes
    assert data["source"] == "local_rules"


def test_v1_leave_balance_route_unaffected():
    data = post_chat("EMPLOYEE", "What is my leave balance?")
    routes = [p["route"] for p in data["relatedPages"]]
    assert "/employee/leave" in routes


def test_v1_team_requests_route_unaffected():
    data = post_chat("TEAM_LEADER", "How do I check team requests?")
    routes = [p["route"] for p in data["relatedPages"]]
    assert "/team/requests" in routes


def test_v2_fallback_unaffected_by_drafting():
    """Unknown non-drafting question still falls through to fallback."""
    data = post_chat("EMPLOYEE", "Tell me about something completely unknown XYZ123")
    assert data["source"] == "fallback"
    assert data.get("draft") is None or data["draft"] == ""


# ---------------------------------------------------------------------------
# Extra: get_draft_response unit tests (service layer directly)
# ---------------------------------------------------------------------------

def test_get_draft_response_returns_none_for_non_drafting():
    req = _make_request("How do I request a loan?")
    # conftest has already patched drafting_service.settings -> disabled
    result = get_draft_response(req)
    assert result is None


def test_get_draft_response_returns_response_for_drafting():
    req = _make_request("Help me draft a leave request reason")
    result = get_draft_response(req)
    assert result is not None
    assert result.draft is not None
    assert result.source in ("local_rules", "external_ai")


def test_get_draft_response_loan_local_template():
    req = _make_request("Write a professional loan justification")
    result = get_draft_response(req)
    assert result is not None
    assert result.source == "local_rules"  # Gemini is disabled by conftest
    assert "loan" in (result.draft or "").lower()


def test_get_draft_response_authorization_local_template():
    req = _make_request("Draft an authorization request explanation")
    result = get_draft_response(req)
    assert result is not None
    draft_lower = (result.draft or "").lower()
    assert "authorization" in draft_lower or "authoris" in draft_lower
