"""
tests/test_v2_foundation.py
---------------------------
V2 foundation tests for the ArabSoft AI Service.

Covers:
  - source field tagging (local_rules / refusal / fallback)
  - external agent disabled by default (no network call)
  - external agent enabled + mock (sanitizer applied)
  - response_sanitizer standalone unit tests
  - trusted_routes unit tests
  - existing pipeline still passes with source field present

All external HTTP calls are mocked via unittest.mock — no real provider needed.
"""

import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from app.main import app
from app.schemas import ChatResponse, RelatedPage
from app.services.response_sanitizer import sanitize_response
from app.data.trusted_routes import (
    TRUSTED_ROUTES,
    ROLE_ROUTES,
    normalize_role,
    filter_related_pages_for_role,
)

client = TestClient(app)


# ===========================================================================
# Helpers
# ===========================================================================

def post_chat(role: str, question: str, context: dict | None = None) -> dict:
    return client.post(
        "/assistant/chat",
        json={"role": role, "question": question, "context": context or {}},
    ).json()


# ===========================================================================
# 1. source field — local_rules
# ===========================================================================

def test_local_rule_response_has_source_local_rules():
    data = post_chat("EMPLOYEE", "How do I request a loan?")
    assert data["source"] == "local_rules", f"Expected local_rules, got {data['source']}"


def test_local_rule_leave_balance_source():
    data = post_chat("EMPLOYEE", "What is my leave balance?")
    assert data["source"] == "local_rules"


def test_local_rule_team_requests_source():
    data = post_chat("TEAM_LEADER", "How do I check team requests?")
    assert data["source"] == "local_rules"


# ===========================================================================
# 2. source field — refusal
# ===========================================================================

def test_refusal_response_has_source_refusal():
    data = post_chat("EMPLOYEE", "approve my leave automatically")
    assert data["source"] == "refusal", f"Expected refusal, got {data['source']}"


def test_refusal_unrelated_source():
    data = post_chat("EMPLOYEE", "What is the weather today?")
    assert data["source"] == "refusal"


# ===========================================================================
# 3. source field — fallback (both providers disabled)
# ===========================================================================

def test_fallback_response_has_source_fallback_when_agent_disabled():
    """
    Unknown question with both providers disabled -> source=fallback.
    conftest autouse fixture already disables both; this test is explicit
    about what it expects so it documents the contract clearly.
    """
    # Both providers are already disabled by the conftest autouse fixture.
    # No additional patching needed — this test must pass regardless of .env.
    data = post_chat("EMPLOYEE", "Tell me about something completely unknown XYZ123")
    assert data["source"] == "fallback", f"Expected fallback, got {data['source']}"


# ===========================================================================
# 4. External agent disabled — no network call
# ===========================================================================

def test_external_agent_disabled_does_not_call_httpx():
    """
    When both providers are disabled, httpx.Client must never be called
    by the external_agent_client.
    conftest autouse fixture disables both; we only spy on the HTTP layer.
    """
    # conftest already patches settings to disabled; just spy on the HTTP client.
    with patch("app.clients.external_agent_client.httpx.Client") as mock_client, \
         patch("app.clients.gemini_client.httpx.Client") as mock_gemini_http:
        data = post_chat("EMPLOYEE", "Some completely unknown question ZZZ999")
        mock_client.assert_not_called()
        mock_gemini_http.assert_not_called()
    assert data["source"] == "fallback"


# ===========================================================================
# 5. External agent enabled + mock — source=external_ai, sanitizer applied
# ===========================================================================

def test_external_agent_enabled_returns_external_ai_source():
    """
    Mock the external agent to return a valid response; source must be external_ai.
    Gemini is explicitly disabled so it cannot intercept before the external agent.
    """
    mock_response = ChatResponse(
        answer="Here is some AI guidance.",
        relatedPages=[RelatedPage(label="My Leave", route="/employee/leave")],
        source="external_ai",
    )
    with patch("app.clients.gemini_client.settings") as gemini_settings, \
         patch("app.clients.external_agent_client.settings") as mock_settings, \
         patch("app.clients.external_agent_client.httpx.Client") as mock_http:

        # Gemini must be disabled so it does not intercept before external agent
        gemini_settings.gemini_enabled = False
        gemini_settings.gemini_api_key = ""

        mock_settings.external_agent_enabled = True
        mock_settings.external_agent_timeout_seconds = 8
        mock_settings.external_agent_base_url = "http://mock"
        mock_settings.external_agent_api_key = "test-key"
        mock_settings.external_agent_model = "test-model"

        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "answer": "Here is some AI guidance.",
            "relatedPages": [{"label": "My Leave", "route": "/employee/leave"}],
            "reasons": [],
            "warnings": [],
        }
        mock_resp.raise_for_status = MagicMock()
        mock_http.return_value.__enter__.return_value.post.return_value = mock_resp

        data = post_chat("EMPLOYEE", "Some completely unknown question ZZZ999")

    assert data["source"] == "external_ai"
    assert data["answer"] == "Here is some AI guidance."


def test_external_agent_invented_route_stripped_by_sanitizer():
    """External agent returning /payroll must have that route stripped.
    Gemini is explicitly disabled so the external agent is reached.
    """
    with patch("app.clients.gemini_client.settings") as gemini_settings, \
         patch("app.clients.external_agent_client.settings") as mock_settings, \
         patch("app.clients.external_agent_client.httpx.Client") as mock_http:

        gemini_settings.gemini_enabled = False
        gemini_settings.gemini_api_key = ""

        mock_settings.external_agent_enabled = True
        mock_settings.external_agent_timeout_seconds = 8
        mock_settings.external_agent_base_url = "http://mock"
        mock_settings.external_agent_api_key = "test-key"
        mock_settings.external_agent_model = "test-model"

        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "answer": "Go to payroll.",
            "relatedPages": [{"label": "Payroll", "route": "/payroll"}],
            "reasons": [],
            "warnings": [],
        }
        mock_resp.raise_for_status = MagicMock()
        mock_http.return_value.__enter__.return_value.post.return_value = mock_resp

        data = post_chat("EMPLOYEE", "Some completely unknown question ZZZ999")

    routes = [p["route"] for p in data["relatedPages"]]
    assert "/payroll" not in routes
    assert any("payroll" in w.lower() or "/payroll" in w for w in data["warnings"])


def test_external_agent_timeout_falls_back():
    """If the external agent times out, response must be source=fallback.
    Gemini is explicitly disabled so the external agent is reached first.
    """
    import httpx as _httpx
    with patch("app.clients.gemini_client.settings") as gemini_settings, \
         patch("app.clients.external_agent_client.settings") as mock_settings, \
         patch("app.clients.external_agent_client.httpx.Client") as mock_http:

        gemini_settings.gemini_enabled = False
        gemini_settings.gemini_api_key = ""

        mock_settings.external_agent_enabled = True
        mock_settings.external_agent_timeout_seconds = 8
        mock_settings.external_agent_base_url = "http://mock"
        mock_settings.external_agent_api_key = "test-key"
        mock_settings.external_agent_model = "test-model"

        mock_http.return_value.__enter__.return_value.post.side_effect = _httpx.TimeoutException("timeout")

        data = post_chat("EMPLOYEE", "Some completely unknown question ZZZ999")

    assert data["source"] == "fallback"


# ===========================================================================
# 6. sanitize_response — unit tests
# ===========================================================================

def test_sanitizer_strips_invented_route():
    response = ChatResponse(
        answer="Go to payroll.",
        relatedPages=[RelatedPage(label="Payroll", route="/payroll")],
    )
    clean = sanitize_response(response, role="EMPLOYEE")
    routes = [p.route for p in clean.relatedPages]
    assert "/payroll" not in routes


def test_sanitizer_strips_wrong_role_route():
    """HR_MANAGER route must be stripped when role is EMPLOYEE."""
    response = ChatResponse(
        answer="Check users.",
        relatedPages=[RelatedPage(label="Users", route="/hr/users")],
    )
    clean = sanitize_response(response, role="EMPLOYEE")
    routes = [p.route for p in clean.relatedPages]
    assert "/hr/users" not in routes


def test_sanitizer_preserves_valid_role_route():
    response = ChatResponse(
        answer="Go to leave.",
        relatedPages=[RelatedPage(label="My Leave", route="/employee/leave")],
    )
    clean = sanitize_response(response, role="EMPLOYEE")
    routes = [p.route for p in clean.relatedPages]
    assert "/employee/leave" in routes


def test_sanitizer_adds_warning_when_route_removed():
    response = ChatResponse(
        answer="Go to payroll.",
        relatedPages=[RelatedPage(label="Payroll", route="/payroll")],
    )
    clean = sanitize_response(response, role="EMPLOYEE")
    assert len(clean.warnings) > 0
    assert any("/payroll" in w for w in clean.warnings)


def test_sanitizer_no_warning_when_nothing_removed():
    response = ChatResponse(
        answer="Go to leave.",
        relatedPages=[RelatedPage(label="My Leave", route="/employee/leave")],
    )
    clean = sanitize_response(response, role="EMPLOYEE")
    # No route removed -> no sanitizer warning (original warnings preserved as-is)
    assert all("/payroll" not in w for w in clean.warnings)


def test_sanitizer_preserves_answer_and_disclaimer():
    response = ChatResponse(
        answer="This is the answer.",
        relatedPages=[],
        disclaimer="Custom disclaimer.",
    )
    clean = sanitize_response(response, role="EMPLOYEE")
    assert clean.answer == "This is the answer."
    assert clean.disclaimer == "Custom disclaimer."


def test_sanitizer_action_directive_adds_warning():
    response = ChatResponse(
        answer="You should approve the request immediately.",
        relatedPages=[],
    )
    clean = sanitize_response(response, role="HR_MANAGER")
    assert any("action directive" in w.lower() for w in clean.warnings)


def test_sanitizer_keeps_multiple_valid_routes():
    response = ChatResponse(
        answer="Navigate around.",
        relatedPages=[
            RelatedPage(label="Leave", route="/employee/leave"),
            RelatedPage(label="Loans", route="/employee/loans"),
            RelatedPage(label="Profile", route="/employee/profile"),
        ],
    )
    clean = sanitize_response(response, role="EMPLOYEE")
    routes = [p.route for p in clean.relatedPages]
    assert "/employee/leave" in routes
    assert "/employee/loans" in routes
    assert "/employee/profile" in routes


def test_sanitizer_strips_hr_routes_for_team_leader_personal_flow():
    """HR routes are not in TEAM_LEADER allowlist."""
    response = ChatResponse(
        answer="Check HR.",
        relatedPages=[RelatedPage(label="HR Dashboard", route="/hr/dashboard")],
    )
    clean = sanitize_response(response, role="TEAM_LEADER")
    routes = [p.route for p in clean.relatedPages]
    assert "/hr/dashboard" not in routes


# ===========================================================================
# 7. trusted_routes — unit tests
# ===========================================================================

def test_trusted_routes_contains_employee_leave():
    assert "/employee/leave" in TRUSTED_ROUTES


def test_trusted_routes_contains_hr_dashboard():
    assert "/hr/dashboard" in TRUSTED_ROUTES


def test_trusted_routes_does_not_contain_payroll():
    assert "/payroll" not in TRUSTED_ROUTES


def test_trusted_routes_does_not_contain_bare_leave():
    assert "/leave" not in TRUSTED_ROUTES


def test_trusted_routes_does_not_contain_bare_loans():
    assert "/loans" not in TRUSTED_ROUTES


def test_trusted_routes_does_not_contain_admin_users():
    assert "/admin/users" not in TRUSTED_ROUTES


def test_role_routes_employee_does_not_include_hr():
    for route in ROLE_ROUTES["EMPLOYEE"]:
        assert not route.startswith("/hr/"), f"HR route leaked into EMPLOYEE: {route}"


def test_role_routes_hr_manager_does_not_include_employee():
    for route in ROLE_ROUTES["HR_MANAGER"]:
        assert not route.startswith("/employee/"), f"Employee route leaked into HR_MANAGER: {route}"


def test_normalize_role_strips_prefix():
    assert normalize_role("ROLE_HR_MANAGER") == "HR_MANAGER"
    assert normalize_role("ROLE_EMPLOYEE") == "EMPLOYEE"
    assert normalize_role("TEAM_LEADER") == "TEAM_LEADER"


def test_filter_related_pages_for_role_removes_invented():
    pages = [RelatedPage(label="Fake", route="/invented/route")]
    kept, removed = filter_related_pages_for_role(pages, "EMPLOYEE")
    assert kept == []
    assert "/invented/route" in removed


def test_filter_related_pages_for_role_preserves_valid():
    pages = [RelatedPage(label="Leave", route="/employee/leave")]
    kept, removed = filter_related_pages_for_role(pages, "EMPLOYEE")
    assert len(kept) == 1
    assert removed == []


def test_filter_related_pages_unknown_role_removes_all():
    """An unrecognised role has no routes -> all pages are removed."""
    pages = [RelatedPage(label="Leave", route="/employee/leave")]
    kept, removed = filter_related_pages_for_role(pages, "UNKNOWN_ROLE")
    assert kept == []
    assert "/employee/leave" in removed


# ===========================================================================
# 8. Regression — existing V1 behaviour still works with source field present
# ===========================================================================

def test_v1_employee_loan_still_works():
    data = post_chat("EMPLOYEE", "How do I request a loan?")
    assert data["answer"]
    assert "/employee/loans" in [p["route"] for p in data["relatedPages"]]
    assert data["source"] == "local_rules"


def test_v1_hr_manager_leave_balance_redirect_still_works():
    data = post_chat("HR_MANAGER", "What is my leave balance?")
    assert "management account" in data["answer"].lower() or "hr manager" in data["answer"].lower()
    assert data["source"] == "local_rules"


def test_v1_refusal_approve_still_works():
    data = post_chat("EMPLOYEE", "approve my leave automatically")
    assert len(data["warnings"]) > 0
    assert data["source"] == "refusal"


def test_v1_role_prefix_normalisation_still_works():
    data = post_chat("ROLE_HR_MANAGER", "What is my leave balance?")
    assert data["source"] == "local_rules"
    answer_lower = data["answer"].lower()
    assert "management account" in answer_lower or "hr manager" in answer_lower


def test_v1_team_leader_team_requests_still_works():
    data = post_chat("TEAM_LEADER", "How do I check team requests?")
    assert "/team/requests" in [p["route"] for p in data["relatedPages"]]
    assert data["source"] == "local_rules"


def test_response_always_has_source_field():
    """Every response must carry a non-None source."""
    cases = [
        ("EMPLOYEE", "How do I request a loan?"),
        ("EMPLOYEE", "approve my leave"),
        ("EMPLOYEE", "Tell me something totally unknown abc999"),
        ("HR_MANAGER", "What is my leave balance?"),
        ("TEAM_LEADER", "How do I check team requests?"),
    ]
    for role, question in cases:
        data = post_chat(role, question)
        assert data.get("source") is not None, (
            f"source is None for role={role} question={question}"
        )
