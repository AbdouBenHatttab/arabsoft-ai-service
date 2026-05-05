"""
tests/test_assistant.py
-----------------------
v1.2 test suite for the ArabSoft AI Service.

All relatedPages routes verified against the real React Router table:
  EMPLOYEE    -> /employee/*
  TEAM_LEADER -> /team/* (management) + /employee/* (personal)
  HR_MANAGER  -> /hr/*

Fake routes that must NEVER appear in any response:
  /payroll, /admin/users, /loans, /leave, /manager, /profile (standalone)
"""

import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


# ---------------------------------------------------------------------------
# Fake-route guard helpers
# ---------------------------------------------------------------------------

# Routes that must never appear in any assistant relatedPages.
# /leave and /loans without the /employee prefix are fake.
# /manager/* does not exist (role does not exist).
# /payroll does not exist.
# /admin/users does not exist.
FAKE_ROUTE_PREFIXES = [
    "/payroll",
    "/admin/users",
    "/manager",
]


def _routes_from(response) -> list:
    return [p["route"] for p in response.json().get("relatedPages", [])]


def _assert_no_fake_routes(response):
    """Assert no relatedPages route is a fake/dead path."""
    routes = _routes_from(response)
    for route in routes:
        for fake in FAKE_ROUTE_PREFIXES:
            assert not route.startswith(fake), (
                f"Fake route '{route}' found (matches fake prefix '{fake}')"
            )
        # bare /leave/* (not /employee/leave) is fake
        if route.startswith("/leave"):
            raise AssertionError(f"Bare /leave* route found: '{route}'")
        # bare /loans/* (not /employee/loans) is fake
        if route.startswith("/loans"):
            raise AssertionError(f"Bare /loans* route found: '{route}'")
        # standalone /profile (not /employee/profile) is fake
        if route == "/profile":
            raise AssertionError("Standalone /profile route found in relatedPages")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

MANAGEMENT_MARKERS = [
    "management account",
    "hr manager",
    "leave requests",        # redirected to all-employee view
    "loan applications",
    "hr management",
    "administer",
    "manage employee",
    "hr reports",
    "management sections",
]


def _is_management_redirect(answer: str) -> bool:
    """Return True if the answer contains any management-redirect marker."""
    lower = answer.lower()
    return any(m in lower for m in MANAGEMENT_MARKERS)


# ---------------------------------------------------------------------------
# Infrastructure endpoints
# ---------------------------------------------------------------------------

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "docs" in data


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "UP", "service": "arabsoft-ai-service"}


# ---------------------------------------------------------------------------
# Employee: loan help — route assertions
# ---------------------------------------------------------------------------

def test_employee_loan_help():
    response = client.post("/assistant/chat", json={
        "role": "EMPLOYEE",
        "question": "How do I request a loan?",
        "pageContext": "loans",
        "context": {},
    })
    assert response.status_code == 200
    data = response.json()
    assert "loan" in data["answer"].lower()
    assert data["aiGenerated"] is True


def test_employee_loan_route_is_employee_loans():
    """EMPLOYEE loan relatedPages must point to /employee/loans."""
    response = client.post("/assistant/chat", json={
        "role": "EMPLOYEE",
        "question": "How do I request a loan?",
        "context": {},
    })
    assert response.status_code == 200
    routes = _routes_from(response)
    assert "/employee/loans" in routes, f"Expected /employee/loans, got {routes}"
    _assert_no_fake_routes(response)


def test_employee_loan_related_pages_shape():
    response = client.post("/assistant/chat", json={
        "role": "EMPLOYEE",
        "question": "How do I request a loan?",
        "context": {},
    })
    assert response.status_code == 200
    pages = response.json()["relatedPages"]
    assert isinstance(pages, list)
    assert len(pages) > 0
    for page in pages:
        assert "label" in page, f"Missing 'label' key in relatedPage: {page}"
        assert "route" in page, f"Missing 'route' key in relatedPage: {page}"
        assert isinstance(page["label"], str)
        assert isinstance(page["route"], str)


# ---------------------------------------------------------------------------
# Employee: leave balance — route assertions
# ---------------------------------------------------------------------------

def test_employee_leave_balance_with_context():
    response = client.post("/assistant/chat", json={
        "role": "EMPLOYEE",
        "question": "What is my leave balance?",
        "context": {"leave": {"balance": 15}},
    })
    assert response.status_code == 200
    data = response.json()
    assert "15" in data["answer"]


def test_employee_leave_balance_without_context():
    response = client.post("/assistant/chat", json={
        "role": "EMPLOYEE",
        "question": "What is my leave balance?",
        "context": {},
    })
    assert response.status_code == 200
    data = response.json()
    assert "leave" in data["answer"].lower()


def test_employee_leave_balance_route_is_employee_leave():
    """EMPLOYEE leave balance must link to /employee/leave, not bare /leave/*."""
    response = client.post("/assistant/chat", json={
        "role": "EMPLOYEE",
        "question": "What is my leave balance?",
        "context": {},
    })
    assert response.status_code == 200
    routes = _routes_from(response)
    assert "/employee/leave" in routes, f"Expected /employee/leave, got {routes}"
    _assert_no_fake_routes(response)


def test_leave_balance_related_pages_shape():
    response = client.post("/assistant/chat", json={
        "role": "EMPLOYEE",
        "question": "What is my leave balance?",
        "context": {},
    })
    pages = response.json()["relatedPages"]
    assert isinstance(pages, list)
    for page in pages:
        assert "label" in page
        assert "route" in page


# ---------------------------------------------------------------------------
# Employee: leave request — route assertions
# ---------------------------------------------------------------------------

def test_employee_leave_request_route_is_employee_leave():
    """EMPLOYEE leave submission must link to /employee/leave."""
    response = client.post("/assistant/chat", json={
        "role": "EMPLOYEE",
        "question": "How do I submit a leave request?",
        "context": {},
    })
    assert response.status_code == 200
    routes = _routes_from(response)
    assert "/employee/leave" in routes, f"Expected /employee/leave, got {routes}"
    _assert_no_fake_routes(response)


# ---------------------------------------------------------------------------
# Employee: profile — route assertions
# ---------------------------------------------------------------------------

def test_employee_profile_route_is_employee_profile():
    """EMPLOYEE profile must link to /employee/profile, not bare /profile."""
    response = client.post("/assistant/chat", json={
        "role": "EMPLOYEE",
        "question": "How do I view my profile?",
        "context": {},
    })
    assert response.status_code == 200
    routes = _routes_from(response)
    assert "/employee/profile" in routes, f"Expected /employee/profile, got {routes}"
    assert "/profile" not in routes, "Standalone /profile must not appear"
    _assert_no_fake_routes(response)


# ---------------------------------------------------------------------------
# HR: user setup help  (kept for fallback — HR role not valid in prod)
# ---------------------------------------------------------------------------

def test_hr_user_setup_help():
    """Legacy test using role HR — falls through to fallback, still returns 200."""
    response = client.post("/assistant/chat", json={
        "role": "HR",
        "question": "How do I setup a new user?",
        "context": {},
    })
    assert response.status_code == 200


# ---------------------------------------------------------------------------
# TEAM_LEADER: personal flows use /employee/*
# ---------------------------------------------------------------------------

def test_team_leader_loan_route_is_employee_loans():
    """TEAM_LEADER personal loan must route to /employee/loans."""
    response = client.post("/assistant/chat", json={
        "role": "TEAM_LEADER",
        "question": "How do I request a loan?",
        "context": {},
    })
    assert response.status_code == 200
    routes = _routes_from(response)
    assert "/employee/loans" in routes, f"Expected /employee/loans, got {routes}"
    _assert_no_fake_routes(response)


def test_team_leader_leave_balance_route_is_employee_leave():
    """TEAM_LEADER personal leave balance must route to /employee/leave."""
    response = client.post("/assistant/chat", json={
        "role": "TEAM_LEADER",
        "question": "What is my leave balance?",
        "context": {},
    })
    assert response.status_code == 200
    routes = _routes_from(response)
    assert "/employee/leave" in routes, f"Expected /employee/leave, got {routes}"
    _assert_no_fake_routes(response)


def test_team_leader_leave_request_route_is_employee_leave():
    """TEAM_LEADER personal leave submission must route to /employee/leave."""
    response = client.post("/assistant/chat", json={
        "role": "TEAM_LEADER",
        "question": "How do I submit my own leave?",
        "context": {},
    })
    assert response.status_code == 200
    routes = _routes_from(response)
    assert "/employee/leave" in routes, f"Expected /employee/leave, got {routes}"
    _assert_no_fake_routes(response)


def test_team_leader_profile_route_is_employee_profile():
    """TEAM_LEADER profile must route to /employee/profile."""
    response = client.post("/assistant/chat", json={
        "role": "TEAM_LEADER",
        "question": "How do I view my profile?",
        "context": {},
    })
    assert response.status_code == 200
    routes = _routes_from(response)
    assert "/employee/profile" in routes, f"Expected /employee/profile, got {routes}"
    assert "/profile" not in routes, "Standalone /profile must not appear"
    _assert_no_fake_routes(response)


# ---------------------------------------------------------------------------
# TEAM_LEADER: team management uses /team/*
# ---------------------------------------------------------------------------

def test_team_leader_team_requests_route():
    """TEAM_LEADER team-request question returns /team/requests."""
    response = client.post("/assistant/chat", json={
        "role": "TEAM_LEADER",
        "question": "How do I check team requests?",
        "context": {},
    })
    assert response.status_code == 200
    routes = _routes_from(response)
    assert "/team/requests" in routes, f"Expected /team/requests, got {routes}"
    _assert_no_fake_routes(response)


def test_team_leader_team_calendar_route():
    """TEAM_LEADER team calendar question returns /team/calendar."""
    response = client.post("/assistant/chat", json={
        "role": "TEAM_LEADER",
        "question": "Where can I see the team leave calendar?",
        "context": {},
    })
    assert response.status_code == 200
    routes = _routes_from(response)
    assert "/team/calendar" in routes, f"Expected /team/calendar, got {routes}"
    _assert_no_fake_routes(response)


def test_team_leader_team_members_route():
    """TEAM_LEADER team-members question returns /team/members."""
    response = client.post("/assistant/chat", json={
        "role": "TEAM_LEADER",
        "question": "How do I manage my team members?",
        "context": {},
    })
    assert response.status_code == 200
    routes = _routes_from(response)
    assert "/team/members" in routes, f"Expected /team/members, got {routes}"
    _assert_no_fake_routes(response)


def test_team_leader_team_tasks_route():
    """TEAM_LEADER team-tasks question returns /team/tasks."""
    response = client.post("/assistant/chat", json={
        "role": "TEAM_LEADER",
        "question": "Where can I view team tasks?",
        "context": {},
    })
    assert response.status_code == 200
    routes = _routes_from(response)
    assert "/team/tasks" in routes, f"Expected /team/tasks, got {routes}"
    _assert_no_fake_routes(response)


def test_team_leader_team_handler_does_not_fire_for_employee():
    """EMPLOYEE asking about team requests must NOT get team-leader /team/* routes."""
    response = client.post("/assistant/chat", json={
        "role": "EMPLOYEE",
        "question": "How do I check team requests?",
        "context": {},
    })
    routes = _routes_from(response)
    assert "/team/requests" not in routes, "Team route leaked to EMPLOYEE role"


# ---------------------------------------------------------------------------
# HR_MANAGER: personal-flow redirects  (requirement tests 1–5, 10, 11)
# ---------------------------------------------------------------------------

def test_hr_manager_leave_balance_redirect():
    """HR_MANAGER asking about personal leave balance must get a management redirect."""
    response = client.post("/assistant/chat", json={
        "role": "HR_MANAGER",
        "question": "How much leave balance do I have?",
        "context": {"leave": {"balance": 10}},  # balance in context must be ignored
    })
    assert response.status_code == 200
    data = response.json()
    assert _is_management_redirect(data["answer"]), (
        f"Expected management redirect, got: {data['answer']}"
    )
    # Must NOT expose the personal balance from context
    assert "10" not in data["answer"], "Personal leave balance leaked into HR_MANAGER response"
    assert "your current leave balance is" not in data["answer"].lower()


def test_hr_manager_loan_request_redirect():
    """HR_MANAGER asking to request a personal loan must get a management redirect."""
    response = client.post("/assistant/chat", json={
        "role": "HR_MANAGER",
        "question": "Can I request a loan?",
        "context": {},
    })
    assert response.status_code == 200
    data = response.json()
    assert _is_management_redirect(data["answer"]), (
        f"Expected management redirect, got: {data['answer']}"
    )
    # Must NOT give employee loan-submission instructions
    assert "complete the loan request form" not in data["answer"].lower()


def test_hr_manager_submit_own_leave_redirect():
    """HR_MANAGER asking to submit their own leave must get a management redirect."""
    response = client.post("/assistant/chat", json={
        "role": "HR_MANAGER",
        "question": "How do I submit my own leave?",
        "context": {},
    })
    assert response.status_code == 200
    assert _is_management_redirect(response.json()["answer"])


def test_hr_manager_submit_leave_request_redirect():
    """HR_MANAGER asking to submit a leave request (no 'own') must also be redirected."""
    response = client.post("/assistant/chat", json={
        "role": "HR_MANAGER",
        "question": "How do I submit a leave request?",
        "context": {},
    })
    assert response.status_code == 200
    assert _is_management_redirect(response.json()["answer"])


def test_hr_manager_personal_requests_redirect():
    """HR_MANAGER asking to see personal requests must get a management redirect."""
    response = client.post("/assistant/chat", json={
        "role": "HR_MANAGER",
        "question": "Show my personal requests",
        "context": {},
    })
    assert response.status_code == 200
    assert _is_management_redirect(response.json()["answer"])


def test_hr_manager_redirect_related_pages_shape():
    response = client.post("/assistant/chat", json={
        "role": "HR_MANAGER",
        "question": "What is my leave balance?",
        "context": {},
    })
    assert response.status_code == 200
    pages = response.json()["relatedPages"]
    assert isinstance(pages, list)
    assert len(pages) > 0, "Redirect must return at least one relatedPage"
    for page in pages:
        assert "label" in page, f"Missing 'label' in relatedPage: {page}"
        assert "route" in page, f"Missing 'route' in relatedPage: {page}"
        assert isinstance(page["label"], str)
        assert isinstance(page["route"], str)
    _assert_no_fake_routes(response)


def test_hr_manager_post_returns_200():
    """POST /assistant/chat with role HR_MANAGER must return HTTP 200."""
    response = client.post("/assistant/chat", json={
        "role": "HR_MANAGER",
        "question": "What can I do here?",
        "context": {},
    })
    assert response.status_code == 200


# ---------------------------------------------------------------------------
# HR_MANAGER: legitimate management questions + route assertions
# ---------------------------------------------------------------------------

def test_hr_manager_review_pending_leave_not_redirected():
    response = client.post("/assistant/chat", json={
        "role": "HR_MANAGER",
        "question": "How do I review pending leave requests?",
        "context": {},
    })
    assert response.status_code == 200
    assert "not a personal employee account" not in response.json()["answer"].lower()


def test_hr_manager_user_setup_returns_hr_users():
    """HR_MANAGER user-setup must link to /hr/users, not /admin/users."""
    response = client.post("/assistant/chat", json={
        "role": "HR_MANAGER",
        "question": "How do I create a new user?",
        "context": {},
    })
    assert response.status_code == 200
    data = response.json()
    assert "not a personal employee account" not in data["answer"].lower()
    assert "user" in data["answer"].lower() or "admin" in data["answer"].lower()
    routes = _routes_from(response)
    assert "/hr/users" in routes, f"Expected /hr/users, got {routes}"
    _assert_no_fake_routes(response)


def test_hr_manager_loan_management_returns_hr_requests():
    """HR_MANAGER asking to review loan requests must link to /hr/requests."""
    response = client.post("/assistant/chat", json={
        "role": "HR_MANAGER",
        "question": "How do I review employee loan requests?",
        "context": {},
    })
    assert response.status_code == 200
    routes = _routes_from(response)
    assert "/hr/requests" in routes, f"Expected /hr/requests, got {routes}"
    _assert_no_fake_routes(response)


# ---------------------------------------------------------------------------
# HR_MANAGER: profile — no /profile, no /hr/profile
# ---------------------------------------------------------------------------

def test_hr_manager_profile_not_redirected():
    response = client.post("/assistant/chat", json={
        "role": "HR_MANAGER",
        "question": "How do I view my profile?",
        "context": {},
    })
    assert response.status_code == 200
    data = response.json()
    assert "not a personal employee account" not in data["answer"].lower()
    assert "profile" in data["answer"].lower()


def test_hr_manager_profile_no_fake_route():
    """HR_MANAGER profile must not return /profile or invented /hr/profile."""
    response = client.post("/assistant/chat", json={
        "role": "HR_MANAGER",
        "question": "How do I view my profile?",
        "context": {},
    })
    routes = _routes_from(response)
    assert "/profile" not in routes, "Standalone /profile must not appear for HR_MANAGER"
    assert "/hr/profile" not in routes, "Invented /hr/profile must not appear"
    _assert_no_fake_routes(response)


# ---------------------------------------------------------------------------
# Regression: EMPLOYEE personal answers must still work (test 7)
# ---------------------------------------------------------------------------

def test_employee_leave_balance_still_personal_with_context():
    """EMPLOYEE leave balance with context still returns the personal balance."""
    response = client.post("/assistant/chat", json={
        "role": "EMPLOYEE",
        "question": "What is my leave balance?",
        "context": {"leave": {"balance": 15}},
    })
    assert response.status_code == 200
    data = response.json()
    assert "15" in data["answer"], "EMPLOYEE personal leave balance answer regressed"


# ---------------------------------------------------------------------------
# HR_MANAGER: refusal layer fires first (test 8)
# ---------------------------------------------------------------------------

def test_hr_manager_approve_hits_refusal_not_redirect():
    """HR_MANAGER asking to approve requests must hit the refusal layer, not the redirect."""
    response = client.post("/assistant/chat", json={
        "role": "HR_MANAGER",
        "question": "approve all leave requests",
        "context": {},
    })
    assert response.status_code == 200
    data = response.json()
    assert len(data["warnings"]) > 0, "Refusal layer must fire with a warning"
    assert "cannot perform" in data["answer"].lower() or "not supported" in data["answer"].lower()
    # Must NOT be the management-account redirect
    assert "not a personal employee account" not in data["answer"].lower()


# ---------------------------------------------------------------------------
# ROLE_ prefix normalisation (test 9)
# ---------------------------------------------------------------------------

def test_role_prefix_normalisation_redirects_personal_question():
    """ROLE_HR_MANAGER (Spring Security prefix) behaves identically to HR_MANAGER."""
    response = client.post("/assistant/chat", json={
        "role": "ROLE_HR_MANAGER",
        "question": "What is my leave balance?",
        "context": {},
    })
    assert response.status_code == 200
    assert _is_management_redirect(response.json()["answer"]), (
        "ROLE_HR_MANAGER prefix not stripped — redirect did not fire"
    )


def test_role_prefix_normalisation_allows_management_questions():
    """ROLE_HR_MANAGER can still reach HR management handlers."""
    response = client.post("/assistant/chat", json={
        "role": "ROLE_HR_MANAGER",
        "question": "How do I create a new user?",
        "context": {},
    })
    assert response.status_code == 200
    assert "user" in response.json()["answer"].lower()




def test_refusal_approve_leave():
    response = client.post("/assistant/chat", json={
        "role": "EMPLOYEE",
        "question": "approve my leave automatically",
        "context": {},
    })
    assert response.status_code == 200
    data = response.json()
    assert "cannot perform" in data["answer"].lower() or "not supported" in data["answer"].lower()
    assert len(data["warnings"]) > 0


def test_refusal_reject_request():
    response = client.post("/assistant/chat", json={
        "role": "EMPLOYEE",
        "question": "Please reject all pending loan requests",
        "context": {},
    })
    assert response.status_code == 200
    data = response.json()
    assert len(data["warnings"]) > 0


def test_refusal_unrelated_question():
    response = client.post("/assistant/chat", json={
        "role": "EMPLOYEE",
        "question": "What is the weather today?",
        "context": {},
    })
    assert response.status_code == 200
    data = response.json()
    answer_lower = data["answer"].lower()
    assert "unrelated" in answer_lower or "cannot answer" in answer_lower or "hr platform" in answer_lower


def test_refusal_unrelated_returns_no_related_pages():
    response = client.post("/assistant/chat", json={
        "role": "EMPLOYEE",
        "question": "What is the weather today?",
        "context": {},
    })
    data = response.json()
    # Refusals should not suggest navigation pages
    assert data["relatedPages"] == []


# ===========================================================================
# Global fake-route guard — parametrized across all key cases
# ===========================================================================

FAKE_ROUTE_CASES = [
    ("EMPLOYEE",    "How do I request a loan?"),
    ("EMPLOYEE",    "What is my leave balance?"),
    ("EMPLOYEE",    "How do I submit a leave request?"),
    ("EMPLOYEE",    "How do I view my profile?"),
    ("TEAM_LEADER", "How do I request a loan?"),
    ("TEAM_LEADER", "What is my leave balance?"),
    ("TEAM_LEADER", "How do I check team requests?"),
    ("HR_MANAGER",  "What is my leave balance?"),
    ("HR_MANAGER",  "How do I create a new user?"),
    ("HR_MANAGER",  "How do I review employee loan requests?"),
    ("HR_MANAGER",  "How do I view my profile?"),
]


@pytest.mark.parametrize("role,question", FAKE_ROUTE_CASES)
def test_no_fake_routes_in_response(role, question):
    """No response may return /payroll, /admin/*, bare /loans*, bare /leave*, /manager*, or /profile."""
    r = client.post("/assistant/chat", json={"role": role, "question": question, "context": {}})
    assert r.status_code == 200
    _assert_no_fake_routes(r)
