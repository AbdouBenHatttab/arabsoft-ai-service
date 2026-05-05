"""
tests/test_v2_gemini.py
-----------------------
Phase 2 tests: Gemini platform Q&A.

All HTTP calls are mocked — no real API key or network required.

Coverage:
  1. Gemini disabled by default -> no HTTP call made, falls through to fallback.
  2. Gemini enabled + valid JSON response -> source=external_ai, answer returned.
  3. Gemini enabled + invented route -> route stripped by sanitizer, warning added.
  4. Gemini enabled + wrong-role route -> route stripped by sanitizer.
  5. Gemini enabled + Markdown-fenced JSON -> parsed correctly.
  6. Gemini timeout -> falls through to fallback.
  7. Gemini HTTP 429 -> falls through to fallback.
  8. Gemini bad JSON -> falls through to fallback.
  9. Gemini empty relatedPages -> answer still returned.
 10. Gemini disabled + external_agent disabled -> source=fallback.
 11. Local rules still fire before Gemini.
 12. Refusal still fires before Gemini.
 13. Prompt builder: system prompt contains role-appropriate routes.
 14. Prompt builder: system prompt does NOT contain routes from other roles.
 15. Gemini enabled but empty API key -> no HTTP call, falls through to fallback.
"""

import json
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

import httpx

from app.main import app
from app.services.gemini_prompt_builder import build_system_prompt, build_user_message
from app.schemas import ChatRequest, ContextInfo

client = TestClient(app)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def post_chat(role: str, question: str, page_context: str = None) -> dict:
    body = {"role": role, "question": question, "context": {}}
    if page_context:
        body["pageContext"] = page_context
    return client.post("/assistant/chat", json=body).json()


def _mock_gemini_settings(mock_settings, *, enabled: bool = True, api_key: str = "test-key"):
    mock_settings.gemini_enabled = enabled
    mock_settings.gemini_api_key = api_key
    mock_settings.gemini_model = "gemini-2.5-flash"
    mock_settings.gemini_timeout_seconds = 10


def _mock_gemini_http(mock_http, answer_json: dict):
    """Configure mock_http to return answer_json as Gemini REST response."""
    raw_text = json.dumps(answer_json)
    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "candidates": [
            {"content": {"parts": [{"text": raw_text}]}}
        ]
    }
    mock_resp.raise_for_status = MagicMock()
    mock_http.return_value.__enter__.return_value.post.return_value = mock_resp
    return mock_http


# ===========================================================================
# 1. Gemini disabled by default — no HTTP call
# ===========================================================================

def test_gemini_disabled_by_default_no_http_call():
    """
    When Gemini is disabled, httpx.Client must never be instantiated.
    The conftest autouse fixture already disables Gemini; this test only
    spies on the HTTP layer to confirm no network call escapes.
    """
    # conftest disables both providers; just assert the HTTP layer is untouched.
    with patch("app.clients.gemini_client.httpx.Client") as mock_http:
        data = post_chat("EMPLOYEE", "Completely unknown question XYZ987")
        mock_http.assert_not_called()
    assert data["source"] == "fallback"


# ===========================================================================
# 2. Gemini enabled + valid response
# ===========================================================================

def test_gemini_enabled_returns_external_ai_source():
    answer_payload = {
        "answer": "You can submit a leave request from the Leave section.",
        "reasons": [],
        "relatedPages": [{"label": "My Leave", "route": "/employee/leave"}],
    }
    with patch("app.clients.gemini_client.settings") as ms, \
         patch("app.clients.gemini_client.httpx.Client") as mh:
        _mock_gemini_settings(ms)
        _mock_gemini_http(mh, answer_payload)
        data = post_chat("EMPLOYEE", "Completely unknown question XYZ987")

    assert data["source"] == "external_ai"
    assert "leave" in data["answer"].lower()
    routes = [p["route"] for p in data["relatedPages"]]
    assert "/employee/leave" in routes


def test_gemini_enabled_answer_text_preserved():
    answer_payload = {
        "answer": "Navigate to the Loans section under your personal menu.",
        "reasons": [],
        "relatedPages": [],
    }
    with patch("app.clients.gemini_client.settings") as ms, \
         patch("app.clients.gemini_client.httpx.Client") as mh:
        _mock_gemini_settings(ms)
        _mock_gemini_http(mh, answer_payload)
        data = post_chat("EMPLOYEE", "Completely unknown question XYZ987")

    assert data["answer"] == "Navigate to the Loans section under your personal menu."


# ===========================================================================
# 2b. Gemini call uses correct URL shape and x-goog-api-key header
# ===========================================================================

def test_gemini_uses_correct_url_and_header():
    """URL must be .../v1beta/models/gemini-2.5-flash:generateContent
    with no ?key= param. API key must be in x-goog-api-key header."""
    answer_payload = {
        "answer": "Here is the answer.",
        "reasons": [],
        "relatedPages": [],
    }
    with patch("app.clients.gemini_client.settings") as ms, \
         patch("app.clients.gemini_client.httpx.Client") as mh:
        _mock_gemini_settings(ms)
        _mock_gemini_http(mh, answer_payload)
        post_chat("EMPLOYEE", "Completely unknown question XYZ987")

        call_args = mh.return_value.__enter__.return_value.post.call_args
        called_url: str = call_args[0][0] if call_args[0] else call_args[1]["url"]
        called_headers: dict = call_args[1].get("headers", {})

    # URL must contain the model path and action — no ?key= query param
    assert "generativelanguage.googleapis.com" in called_url
    assert "v1beta/models/gemini-2.5-flash:generateContent" in called_url
    assert "?key=" not in called_url
    assert "key=" not in called_url

    # API key must travel in the header, not the URL
    assert called_headers.get("x-goog-api-key") == "test-key"
    assert "Content-Type" in called_headers


# ===========================================================================
# 3. Gemini returns an invented route -> sanitizer strips it
# ===========================================================================

def test_gemini_invented_route_stripped():
    answer_payload = {
        "answer": "Check the payroll section.",
        "reasons": [],
        "relatedPages": [{"label": "Payroll", "route": "/payroll"}],
    }
    with patch("app.clients.gemini_client.settings") as ms, \
         patch("app.clients.gemini_client.httpx.Client") as mh:
        _mock_gemini_settings(ms)
        _mock_gemini_http(mh, answer_payload)
        data = post_chat("EMPLOYEE", "Completely unknown question XYZ987")

    routes = [p["route"] for p in data["relatedPages"]]
    assert "/payroll" not in routes
    assert any("/payroll" in w for w in data["warnings"])


# ===========================================================================
# 4. Gemini returns a wrong-role route -> sanitizer strips it
# ===========================================================================

def test_gemini_wrong_role_route_stripped():
    answer_payload = {
        "answer": "Go to HR users.",
        "reasons": [],
        "relatedPages": [{"label": "Users", "route": "/hr/users"}],
    }
    with patch("app.clients.gemini_client.settings") as ms, \
         patch("app.clients.gemini_client.httpx.Client") as mh:
        _mock_gemini_settings(ms)
        _mock_gemini_http(mh, answer_payload)
        data = post_chat("EMPLOYEE", "Completely unknown question XYZ987")

    routes = [p["route"] for p in data["relatedPages"]]
    assert "/hr/users" not in routes
    assert any("/hr/users" in w for w in data["warnings"])


# ===========================================================================
# 5. Gemini returns Markdown-fenced JSON -> parsed correctly
# ===========================================================================

def test_gemini_markdown_fenced_json_parsed():
    inner = json.dumps({
        "answer": "Use the calendar to track leave.",
        "reasons": [],
        "relatedPages": [{"label": "Calendar", "route": "/employee/calendar"}],
    })
    fenced = f"```json\n{inner}\n```"

    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "candidates": [{"content": {"parts": [{"text": fenced}]}}]
    }
    mock_resp.raise_for_status = MagicMock()

    with patch("app.clients.gemini_client.settings") as ms, \
         patch("app.clients.gemini_client.httpx.Client") as mh:
        _mock_gemini_settings(ms)
        mh.return_value.__enter__.return_value.post.return_value = mock_resp
        data = post_chat("EMPLOYEE", "Completely unknown question XYZ987")

    assert data["answer"] == "Use the calendar to track leave."
    routes = [p["route"] for p in data["relatedPages"]]
    assert "/employee/calendar" in routes


# ===========================================================================
# 6. Gemini timeout -> falls through to fallback
# ===========================================================================

def test_gemini_timeout_falls_back():
    with patch("app.clients.gemini_client.settings") as ms, \
         patch("app.clients.gemini_client.httpx.Client") as mh:
        _mock_gemini_settings(ms)
        mh.return_value.__enter__.return_value.post.side_effect = httpx.TimeoutException("timeout")
        data = post_chat("EMPLOYEE", "Completely unknown question XYZ987")

    assert data["source"] == "fallback"


# ===========================================================================
# 7. Gemini HTTP error -> falls through to fallback
# ===========================================================================

def test_gemini_http_error_falls_back():
    with patch("app.clients.gemini_client.settings") as ms, \
         patch("app.clients.gemini_client.httpx.Client") as mh:
        _mock_gemini_settings(ms)
        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "429", request=MagicMock(), response=MagicMock(status_code=429)
        )
        mh.return_value.__enter__.return_value.post.return_value = mock_resp
        data = post_chat("EMPLOYEE", "Completely unknown question XYZ987")

    assert data["source"] == "fallback"


# ===========================================================================
# 8. Gemini bad JSON -> falls through to fallback
# ===========================================================================

def test_gemini_bad_json_falls_back():
    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "candidates": [{"content": {"parts": [{"text": "not valid json {{{{"}]}}]
    }
    mock_resp.raise_for_status = MagicMock()

    with patch("app.clients.gemini_client.settings") as ms, \
         patch("app.clients.gemini_client.httpx.Client") as mh:
        _mock_gemini_settings(ms)
        mh.return_value.__enter__.return_value.post.return_value = mock_resp
        data = post_chat("EMPLOYEE", "Completely unknown question XYZ987")

    assert data["source"] == "fallback"


# ===========================================================================
# 9. Gemini returns empty relatedPages -> answer still returned
# ===========================================================================

def test_gemini_empty_related_pages_still_returns_answer():
    answer_payload = {
        "answer": "Please contact your HR administrator for assistance.",
        "reasons": [],
        "relatedPages": [],
    }
    with patch("app.clients.gemini_client.settings") as ms, \
         patch("app.clients.gemini_client.httpx.Client") as mh:
        _mock_gemini_settings(ms)
        _mock_gemini_http(mh, answer_payload)
        data = post_chat("EMPLOYEE", "Completely unknown question XYZ987")

    assert data["source"] == "external_ai"
    assert data["answer"]
    assert data["relatedPages"] == []


# ===========================================================================
# 10. Both Gemini and external_agent disabled -> fallback
# ===========================================================================

def test_both_providers_disabled_gives_fallback():
    """
    With both providers off (conftest default), unknown question -> fallback.
    No patching needed here; this test documents the baseline pipeline state.
    """
    data = post_chat("EMPLOYEE", "Completely unknown question XYZ987")
    assert data["source"] == "fallback"


# ===========================================================================
# 11. Local rules still fire before Gemini
# ===========================================================================

def test_local_rules_fire_before_gemini():
    with patch("app.clients.gemini_client.settings") as ms, \
         patch("app.clients.gemini_client.httpx.Client") as mh:
        _mock_gemini_settings(ms)
        # Gemini would return something, but local rule should answer first
        data = post_chat("EMPLOYEE", "How do I request a loan?")
        # Gemini HTTP must NOT have been called
        mh.assert_not_called()

    assert data["source"] == "local_rules"
    routes = [p["route"] for p in data["relatedPages"]]
    assert "/employee/loans" in routes


# ===========================================================================
# 12. Refusal still fires before Gemini
# ===========================================================================

def test_refusal_fires_before_gemini():
    """
    Refusal layer must fire before Gemini is ever consulted.
    We assert Gemini's HTTP client is never instantiated even if Gemini
    were hypothetically enabled (settings are already off via conftest).
    """
    with patch("app.clients.gemini_client.httpx.Client") as mh:
        data = post_chat("EMPLOYEE", "approve my leave automatically")
        mh.assert_not_called()

    assert data["source"] == "refusal"


# ===========================================================================
# 13. Prompt builder: system prompt contains role routes
# ===========================================================================

def test_prompt_builder_employee_routes_in_system_prompt():
    prompt = build_system_prompt("EMPLOYEE")
    assert "/employee/leave" in prompt
    assert "/employee/loans" in prompt
    assert "/employee/profile" in prompt


def test_prompt_builder_hr_manager_routes_in_system_prompt():
    prompt = build_system_prompt("HR_MANAGER")
    assert "/hr/dashboard" in prompt
    assert "/hr/users" in prompt
    assert "/hr/approvals" in prompt


def test_prompt_builder_team_leader_routes_in_system_prompt():
    prompt = build_system_prompt("TEAM_LEADER")
    assert "/team/dashboard" in prompt
    assert "/team/requests" in prompt


# ===========================================================================
# 14. Prompt builder: system prompt does NOT include wrong-role routes
# ===========================================================================

def test_prompt_builder_employee_prompt_excludes_hr_routes():
    prompt = build_system_prompt("EMPLOYEE")
    assert "/hr/dashboard" not in prompt
    assert "/hr/users" not in prompt


def test_prompt_builder_hr_manager_prompt_excludes_employee_routes():
    prompt = build_system_prompt("HR_MANAGER")
    assert "/employee/leave" not in prompt
    assert "/employee/loans" not in prompt


def test_prompt_builder_role_prefix_normalised():
    prompt_with_prefix = build_system_prompt("ROLE_EMPLOYEE")
    prompt_without = build_system_prompt("EMPLOYEE")
    assert prompt_with_prefix == prompt_without


# ===========================================================================
# 15. Gemini enabled but empty API key -> no HTTP call
# ===========================================================================

def test_gemini_enabled_but_empty_api_key_no_http_call():
    with patch("app.clients.gemini_client.settings") as ms, \
         patch("app.clients.gemini_client.httpx.Client") as mh:
        _mock_gemini_settings(ms, enabled=True, api_key="")
        data = post_chat("EMPLOYEE", "Completely unknown question XYZ987")
        mh.assert_not_called()

    assert data["source"] == "fallback"
