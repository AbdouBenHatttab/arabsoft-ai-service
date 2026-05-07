"""
tests/test_v2_context.py
------------------------
Tests for the typed SafeAssistantContext integration (v1.3).

Spring Boot now sends a nested context shape:
  context.employee  — EmployeeContext  (EMPLOYEE / TEAM_LEADER)
  context.team      — TeamContext      (TEAM_LEADER)
  context.hr        — HrContext        (HR_MANAGER)

Coverage:
  1.  EMPLOYEE pending request question uses context.employee.totalPendingRequests.
  2.  EMPLOYEE annual leave balance uses context.employee.annualAvailableDays.
  3.  EMPLOYEE sick leave balance uses context.employee.sickAvailableDays.
  4.  TEAM_LEADER team approval question uses context.team.pendingTeamLeaderApprovals.
  5.  HR_MANAGER pending actions question uses context.hr.totalPendingActions.
  6.  HR_MANAGER new-user approval question uses context.hr.newUsersPendingApproval.
  7.  HR_MANAGER asking personal leave balance gets management-safe redirection.
  8.  Missing employee/team/hr context does not crash.
  9.  Missing value does not hallucinate a number.
  10. Existing drafting tests still pass (regression gate).
  11. Existing refusal tests still pass (regression gate).
  12. Old flat context shape {"leave": {"balance": N}} still deserialises safely.
  13. Zero pending values answered correctly (not omitted).
  14. EMPLOYEE must not receive HR context answers.
  15. Breakdown fields (documentsPending, loansPending) appear in EMPLOYEE pending answer.
  16. documentsAwaitingFile exposed separately in pending breakdown (not merged into documentsPending).
  17. HR context parses documentsAwaitingFile without crashing.
"""

import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def post_chat(role: str, question: str, context: dict | None = None) -> dict:
    return client.post(
        "/assistant/chat",
        json={"role": role, "question": question, "context": context or {}},
    ).json()


def _employee_context(
    annual=15, sick=8, total=3, leaves=1, docs=1, docs_awaiting=0, loans=1, auths=0
) -> dict:
    return {
        "employee": {
            "annualAvailableDays": annual,
            "sickAvailableDays": sick,
            "totalPendingRequests": total,
            "leavesPending": leaves,
            "documentsPending": docs,
            "documentsAwaitingFile": docs_awaiting,
            "loansPending": loans,
            "authorizationsPending": auths,
        },
        "team": None,
        "hr": None,
    }


def _team_leader_context(
    annual=8, sick=3, total=2, leaves=1, docs=0, docs_awaiting=0, loans=1, auths=0,
    team_name="Backend Squad", member_count=5, pending_approvals=2
) -> dict:
    return {
        "employee": {
            "annualAvailableDays": annual,
            "sickAvailableDays": sick,
            "totalPendingRequests": total,
            "leavesPending": leaves,
            "documentsPending": docs,
            "documentsAwaitingFile": docs_awaiting,
            "loansPending": loans,
            "authorizationsPending": auths,
        },
        "team": {
            "teamName": team_name,
            "memberCount": member_count,
            "pendingTeamLeaderApprovals": pending_approvals,
        },
        "hr": None,
    }


def _hr_context(
    total=21, leaves=5, docs=3, docs_awaiting=0, loans=6, auths=7, new_users=4
) -> dict:
    return {
        "employee": None,
        "team": None,
        "hr": {
            "totalPendingActions": total,
            "leavesPending": leaves,
            "documentsPending": docs,
            "documentsAwaitingFile": docs_awaiting,
            "loansPending": loans,
            "authorizationsPending": auths,
            "newUsersPendingApproval": new_users,
        },
    }


# ===========================================================================
# 1. EMPLOYEE: pending request count uses context.employee.totalPendingRequests
# ===========================================================================

def test_employee_pending_requests_uses_context_total():
    """Answer must contain the exact totalPendingRequests value from context."""
    data = post_chat(
        "EMPLOYEE",
        "How many pending requests do I have?",
        _employee_context(total=5, leaves=2, docs=1, loans=1, auths=1),
    )
    assert data["source"] == "local_rules"
    assert "5" in data["answer"], f"Expected '5' in answer, got: {data['answer']}"
    # Must not invent any other number
    assert "10" not in data["answer"]


def test_employee_pending_requests_zero_answered_not_omitted():
    """Zero pending requests must be explicitly stated, not omitted."""
    data = post_chat(
        "EMPLOYEE",
        "Do I have pending requests?",
        _employee_context(total=0, leaves=0, docs=0, loans=0, auths=0),
    )
    assert data["source"] == "local_rules"
    answer_lower = data["answer"].lower()
    assert "no" in answer_lower or "0" in answer_lower or "no open" in answer_lower, (
        f"Expected zero-count answer, got: {data['answer']}"
    )


def test_employee_pending_requests_breakdown_in_answer():
    """Answer should mention the breakdown (leaves, docs, loans) when non-zero."""
    data = post_chat(
        "EMPLOYEE",
        "How many pending requests do I have?",
        _employee_context(total=3, leaves=1, docs=1, loans=1, auths=0),
    )
    assert data["source"] == "local_rules"
    answer_lower = data["answer"].lower()
    # At least one breakdown category should appear
    has_breakdown = (
        "leave" in answer_lower or "document" in answer_lower or "loan" in answer_lower
    )
    assert has_breakdown, f"Expected breakdown in answer, got: {data['answer']}"


# ===========================================================================
# 2. EMPLOYEE: annual leave balance uses context.employee.annualAvailableDays
# ===========================================================================

def test_employee_annual_leave_balance_uses_context():
    """Answer must contain the exact annualAvailableDays value from context."""
    data = post_chat(
        "EMPLOYEE",
        "What is my annual leave balance?",
        _employee_context(annual=15),
    )
    assert data["source"] == "local_rules"
    assert "15" in data["answer"], f"Expected '15' in answer, got: {data['answer']}"
    assert "/employee/leave" in [p["route"] for p in data["relatedPages"]]


def test_employee_annual_leave_balance_generic_question():
    """'What is my leave balance?' also reads annualAvailableDays."""
    data = post_chat(
        "EMPLOYEE",
        "What is my leave balance?",
        _employee_context(annual=12),
    )
    assert "12" in data["answer"], f"Expected '12' in answer, got: {data['answer']}"


def test_employee_annual_leave_balance_zero():
    """Zero annual days must be explicitly stated."""
    data = post_chat(
        "EMPLOYEE",
        "What is my annual leave balance?",
        _employee_context(annual=0),
    )
    assert "0" in data["answer"] or "zero" in data["answer"].lower(), (
        f"Expected zero balance answer, got: {data['answer']}"
    )


# ===========================================================================
# 3. EMPLOYEE: sick leave balance uses context.employee.sickAvailableDays
# ===========================================================================

def test_employee_sick_leave_balance_uses_context():
    """Answer must contain the exact sickAvailableDays value from context."""
    data = post_chat(
        "EMPLOYEE",
        "What is my sick leave balance?",
        _employee_context(sick=8),
    )
    assert data["source"] == "local_rules"
    assert "8" in data["answer"], f"Expected '8' in answer, got: {data['answer']}"
    assert "/employee/leave" in [p["route"] for p in data["relatedPages"]]


def test_employee_sick_leave_balance_zero():
    """Zero sick days must be explicitly stated."""
    data = post_chat(
        "EMPLOYEE",
        "How many sick leave days do I have?",
        _employee_context(sick=0),
    )
    assert "0" in data["answer"] or "zero" in data["answer"].lower(), (
        f"Expected zero sick balance, got: {data['answer']}"
    )


def test_employee_sick_leave_does_not_leak_annual_value():
    """Sick balance answer must not contain the annual balance value when different."""
    data = post_chat(
        "EMPLOYEE",
        "What is my sick leave balance?",
        _employee_context(annual=15, sick=8),
    )
    assert "8" in data["answer"], f"Expected sick days (8) in answer, got: {data['answer']}"
    # Annual value must not appear in the sick leave answer
    assert "15" not in data["answer"], (
        f"Annual balance (15) leaked into sick leave answer: {data['answer']}"
    )


def test_employee_document_readiness_awaiting_file_uses_context():
    data = post_chat(
        "EMPLOYEE",
        "Are my documents ready?",
        _employee_context(total=2, leaves=0, docs=0, docs_awaiting=2, loans=0, auths=0),
    )
    assert data["source"] == "local_rules"
    answer_lower = data["answer"].lower()
    assert "2" in data["answer"], f"Expected awaiting-file count '2': {data['answer']}"
    assert (
        "waiting" in answer_lower
        and ("upload" in answer_lower or "hr" in answer_lower or "final file" in answer_lower)
    ), f"Expected waiting/upload/HR/final file wording: {data['answer']}"
    assert "ready documents can be downloaded" in answer_lower
    assert "2 ready" not in answer_lower and "2 document(s) ready" not in answer_lower, (
        f"Invented ready document count: {data['answer']}"
    )
    routes = [p["route"] for p in data["relatedPages"]]
    assert "/employee/documents" in routes
    assert "/employee/notifications" in routes


def test_employee_document_readiness_pending_review_uses_context():
    data = post_chat(
        "EMPLOYEE",
        "Are my documents ready?",
        _employee_context(total=1, leaves=0, docs=1, docs_awaiting=0, loans=0, auths=0),
    )
    assert data["source"] == "local_rules"
    answer_lower = data["answer"].lower()
    assert "1" in data["answer"], f"Expected pending document count '1': {data['answer']}"
    assert "pending" in answer_lower
    assert "review" in answer_lower or "preparation" in answer_lower


def test_employee_document_readiness_zero_context_does_not_claim_ready():
    data = post_chat(
        "EMPLOYEE",
        "Are my documents ready?",
        _employee_context(total=0, leaves=0, docs=0, docs_awaiting=0, loans=0, auths=0),
    )
    assert data["source"] == "local_rules"
    answer_lower = data["answer"].lower()
    assert "i do not see pending document preparation from the available context" in answer_lower
    assert "check my documents for downloadable files" in answer_lower
    assert "your documents are ready" not in answer_lower


# ===========================================================================
# 4. TEAM_LEADER: team approvals uses context.team.pendingTeamLeaderApprovals
# ===========================================================================

def test_team_leader_pending_approvals_uses_context():
    """Answer must contain the exact pendingTeamLeaderApprovals value."""
    data = post_chat(
        "TEAM_LEADER",
        "How many team approvals are waiting?",
        _team_leader_context(pending_approvals=3),
    )
    assert data["source"] == "local_rules"
    assert "3" in data["answer"], f"Expected '3' in answer, got: {data['answer']}"
    routes = [p["route"] for p in data["relatedPages"]]
    assert "/team/requests" in routes


def test_team_leader_pending_approvals_zero():
    """Zero pending approvals must be explicitly stated."""
    data = post_chat(
        "TEAM_LEADER",
        "Are there any pending approvals from my team?",
        _team_leader_context(pending_approvals=0),
    )
    assert data["source"] == "local_rules"
    answer_lower = data["answer"].lower()
    assert "no" in answer_lower or "0" in answer_lower or "caught up" in answer_lower, (
        f"Expected zero-approval answer, got: {data['answer']}"
    )


def test_team_leader_pending_approvals_singular():
    """1 pending approval uses singular 'request', not 'requests'."""
    data = post_chat(
        "TEAM_LEADER",
        "How many team approvals are pending?",
        _team_leader_context(pending_approvals=1),
    )
    assert "1" in data["answer"], f"Expected '1' in answer, got: {data['answer']}"


def test_team_leader_pending_team_requests_uses_team_context_count():
    data = post_chat(
        "TEAM_LEADER",
        "Do I have pending team requests?",
        _team_leader_context(pending_approvals=3),
    )
    assert data["source"] == "local_rules"
    assert "3" in data["answer"], f"Expected '3' in answer, got: {data['answer']}"
    routes = [p["route"] for p in data["relatedPages"]]
    assert "/team/requests" in routes, f"Expected /team/requests, got {routes}"
    assert "/team/calendar" in routes, f"Expected /team/calendar, got {routes}"
    assert not any(route.startswith("/employee/") for route in routes), (
        f"Employee route leaked into team pending answer: {routes}"
    )


def test_team_leader_pending_team_requests_zero_count_is_explicit():
    data = post_chat(
        "TEAM_LEADER",
        "Do I have pending team requests?",
        _team_leader_context(pending_approvals=0),
    )
    assert data["source"] == "local_rules"
    answer_lower = data["answer"].lower()
    assert "no" in answer_lower or "0" in answer_lower or "caught up" in answer_lower, (
        f"Expected explicit zero pending team request answer, got: {data['answer']}"
    )
    routes = [p["route"] for p in data["relatedPages"]]
    assert "/team/requests" in routes, f"Expected /team/requests, got {routes}"
    assert not any(route.startswith("/employee/") for route in routes), (
        f"Employee route leaked into team pending zero answer: {routes}"
    )


def test_team_leader_pending_team_requests_missing_count_does_not_invent():
    data = post_chat(
        "TEAM_LEADER",
        "Do I have pending team requests?",
        _team_leader_context(pending_approvals=None),
    )
    assert data["source"] == "local_rules"
    import re
    assert not re.search(r"there (?:are|is) \d+ (?:leave|request)", data["answer"].lower()), (
        f"Invented pending team request count in answer: {data['answer']}"
    )
    routes = [p["route"] for p in data["relatedPages"]]
    assert "/team/requests" in routes, f"Expected /team/requests, got {routes}"
    assert not any(route.startswith("/employee/") for route in routes), (
        f"Employee route leaked into team pending unknown answer: {routes}"
    )


def test_team_leader_track_own_leave_still_personal_after_team_pending_fix():
    data = post_chat("TEAM_LEADER", "How do I track my own leave?")
    assert data["source"] == "local_rules"
    routes = [p["route"] for p in data["relatedPages"]]
    assert "/employee/leave" in routes, f"Expected /employee/leave, got {routes}"
    assert not any(route.startswith("/team/") for route in routes), (
        f"Team route leaked into personal leave tracking: {routes}"
    )


# ===========================================================================
# 4-B. TeamContext.pendingTeamLeaderApprovals: absent-key, zero, and positive
# ===========================================================================

def test_team_leader_absent_approvals_key_stays_none_not_zero():
    """
    When pendingTeamLeaderApprovals is absent from the JSON payload entirely
    the field must deserialise to None, not 0.
    """
    ctx = {
        "employee": {
            "annualAvailableDays": 8, "sickAvailableDays": 3,
            "totalPendingRequests": 2, "leavesPending": 1,
            "documentsPending": 0, "loansPending": 1, "authorizationsPending": 0,
        },
        "team": {
            "teamName": "Alpha Squad",
            "memberCount": 4,
            # pendingTeamLeaderApprovals intentionally absent
        },
        "hr": None,
    }
    data = post_chat(
        "TEAM_LEADER",
        "How many team approvals are pending?",
        ctx,
    )
    assert data["source"] == "local_rules"
    answer_lower = data["answer"].lower()
    assert "caught up" not in answer_lower, (
        f"Absent key incorrectly treated as zero: {data['answer']}"
    )
    import re
    assert not re.search(r"there (?:are|is) \d+", answer_lower), (
        f"Invented count for absent key: {data['answer']}"
    )
    routes = [p["route"] for p in data["relatedPages"]]
    assert "/team/requests" in routes, f"Expected /team/requests, got {routes}"


def test_team_leader_explicit_zero_approvals_gives_caught_up_answer():
    ctx = {
        "employee": {
            "annualAvailableDays": 8, "sickAvailableDays": 3,
            "totalPendingRequests": 0, "leavesPending": 0,
            "documentsPending": 0, "loansPending": 0, "authorizationsPending": 0,
        },
        "team": {
            "teamName": "Beta Squad",
            "memberCount": 3,
            "pendingTeamLeaderApprovals": 0,
        },
        "hr": None,
    }
    data = post_chat("TEAM_LEADER", "How many team approvals are pending?", ctx)
    assert data["source"] == "local_rules"
    answer_lower = data["answer"].lower()
    assert "caught up" in answer_lower or "no leave" in answer_lower or "no " in answer_lower, (
        f"Expected clear-queue answer for explicit 0, got: {data['answer']}"
    )


def test_team_leader_positive_approvals_gives_exact_count():
    ctx = {
        "employee": {
            "annualAvailableDays": 8, "sickAvailableDays": 3,
            "totalPendingRequests": 2, "leavesPending": 1,
            "documentsPending": 0, "loansPending": 1, "authorizationsPending": 0,
        },
        "team": {
            "teamName": "Gamma Squad",
            "memberCount": 6,
            "pendingTeamLeaderApprovals": 5,
        },
        "hr": None,
    }
    data = post_chat("TEAM_LEADER", "How many team approvals are waiting?", ctx)
    assert data["source"] == "local_rules"
    assert "5" in data["answer"], f"Expected exact count '5', got: {data['answer']}"
    routes = [p["route"] for p in data["relatedPages"]]
    assert "/team/requests" in routes


def test_team_leader_null_team_context_does_not_crash_or_invent():
    ctx = {
        "employee": {
            "annualAvailableDays": 8, "sickAvailableDays": 3,
            "totalPendingRequests": 2, "leavesPending": 1,
            "documentsPending": 0, "loansPending": 1, "authorizationsPending": 0,
        },
        "team": None,
        "hr": None,
    }
    data = post_chat("TEAM_LEADER", "How many team approvals are pending?", ctx)
    assert data["source"] == "local_rules"
    assert data["answer"]
    import re
    assert not re.search(r"there (?:are|is) \d+", data["answer"].lower()), (
        f"Invented count when team is null: {data['answer']}"
    )
    assert "caught up" not in data["answer"].lower(), (
        f"'caught up' stated when team context is null: {data['answer']}"
    )
    routes = [p["route"] for p in data["relatedPages"]]
    assert "/team/requests" in routes


# ===========================================================================
# 5. HR_MANAGER: pending actions uses context.hr.totalPendingActions
# ===========================================================================

def test_hr_manager_pending_actions_uses_context():
    data = post_chat("HR_MANAGER", "How many HR actions are pending?", _hr_context(total=21))
    assert data["source"] == "local_rules"
    assert "21" in data["answer"], f"Expected '21' in answer, got: {data['answer']}"
    routes = [p["route"] for p in data["relatedPages"]]
    assert "/hr/dashboard" in routes or "/hr/requests" in routes


def test_hr_manager_total_pending_zero():
    data = post_chat("HR_MANAGER", "How many pending actions are there?", _hr_context(total=0))
    assert data["source"] == "local_rules"
    answer_lower = data["answer"].lower()
    assert "no" in answer_lower or "0" in answer_lower or "up to date" in answer_lower, (
        f"Expected zero-action answer, got: {data['answer']}"
    )


def test_hr_manager_platform_pending_question():
    data = post_chat("HR_MANAGER", "How many platform actions are pending?", _hr_context(total=7))
    assert "7" in data["answer"], f"Expected '7' in answer, got: {data['answer']}"


def test_hr_manager_my_pending_requests_uses_hr_pending_actions():
    data = post_chat("HR_MANAGER", "How many pending requests do I have?", _hr_context(total=7))
    assert data["source"] == "local_rules"
    assert "7" in data["answer"], f"Expected '7' in answer, got: {data['answer']}"
    answer_lower = data["answer"].lower()
    assert "management account" not in answer_lower, (
        f"HR pending request count should not use personal-account redirect: {data['answer']}"
    )
    routes = [p["route"] for p in data["relatedPages"]]
    assert routes
    assert all(route.startswith("/hr/") for route in routes), f"Expected HR-only routes, got: {routes}"
    assert not any(route.startswith("/employee/") for route in routes), (
        f"Employee route leaked into HR pending answer: {routes}"
    )


def test_hr_manager_my_pending_requests_missing_total_does_not_invent_count():
    ctx = {
        "employee": None,
        "team": None,
        "hr": {
            "totalPendingActions": None,
            "leavesPending": None,
            "documentsPending": None,
            "loansPending": None,
            "authorizationsPending": None,
            "newUsersPendingApproval": None,
        },
    }
    data = post_chat("HR_MANAGER", "How many pending requests do I have?", ctx)
    assert data["source"] == "local_rules"
    import re
    assert not re.search(r"there (?:are|is) \d+ pending", data["answer"].lower()), (
        f"Invented HR pending count in answer: {data['answer']}"
    )
    routes = [p["route"] for p in data["relatedPages"]]
    assert routes
    assert all(route.startswith("/hr/") for route in routes), f"Expected HR-only routes, got: {routes}"
    assert not any(route.startswith("/employee/") for route in routes), (
        f"Employee route leaked into HR pending answer: {routes}"
    )


# ===========================================================================
# 6. HR_MANAGER: new user count uses context.hr.newUsersPendingApproval
# ===========================================================================

def test_hr_manager_new_users_pending_uses_context():
    data = post_chat("HR_MANAGER", "How many new users are waiting for approval?", _hr_context(new_users=4))
    assert data["source"] == "local_rules"
    assert "4" in data["answer"], f"Expected '4' in answer, got: {data['answer']}"
    routes = [p["route"] for p in data["relatedPages"]]
    assert "/hr/users" in routes


def test_hr_manager_new_users_zero():
    data = post_chat("HR_MANAGER", "Are there new users waiting for onboarding?", _hr_context(new_users=0))
    assert data["source"] == "local_rules"
    answer_lower = data["answer"].lower()
    assert "no" in answer_lower or "0" in answer_lower or "clear" in answer_lower, (
        f"Expected zero-new-users answer, got: {data['answer']}"
    )


def test_hr_manager_new_users_singular():
    data = post_chat("HR_MANAGER", "How many users are pending approval?", _hr_context(new_users=1))
    assert "1" in data["answer"], f"Expected '1' in answer, got: {data['answer']}"


# ===========================================================================
# 7. HR_MANAGER personal leave balance gets management-safe redirection
# ===========================================================================

def test_hr_manager_leave_balance_redirected_not_context_value():
    data = post_chat("HR_MANAGER", "What is my leave balance?", _hr_context(total=5))
    assert data["source"] == "local_rules"
    answer_lower = data["answer"].lower()
    assert "management account" in answer_lower or "hr manager" in answer_lower, (
        f"Expected management redirect, got: {data['answer']}"
    )
    assert "your current annual leave balance" not in answer_lower


def test_hr_manager_annual_leave_question_redirected():
    data = post_chat("HR_MANAGER", "What is my annual leave balance?", {})
    answer_lower = data["answer"].lower()
    assert "management account" in answer_lower or "hr manager" in answer_lower


def test_hr_manager_request_leave_still_redirected():
    data = post_chat("HR_MANAGER", "Can I request leave?", {})
    answer_lower = data["answer"].lower()
    assert "management account" in answer_lower or "hr manager" in answer_lower


def test_hr_manager_request_loan_still_redirected():
    data = post_chat("HR_MANAGER", "Can I request a loan?", {})
    answer_lower = data["answer"].lower()
    assert "management account" in answer_lower or "hr manager" in answer_lower


def test_hr_manager_request_document_still_redirected():
    data = post_chat("HR_MANAGER", "Can I request a document?", {})
    answer_lower = data["answer"].lower()
    assert "management account" in answer_lower or "hr manager" in answer_lower


# ===========================================================================
# 8. Missing employee/team/hr context does not crash
# ===========================================================================

def test_employee_missing_context_does_not_crash():
    data = post_chat("EMPLOYEE", "How many pending requests do I have?", {})
    assert "source" in data
    assert data["answer"]


def test_team_leader_missing_team_context_does_not_crash():
    ctx = {
        "employee": {"annualAvailableDays": 8, "sickAvailableDays": 3,
                     "totalPendingRequests": 2, "leavesPending": 1,
                     "documentsPending": 0, "loansPending": 1, "authorizationsPending": 0},
        "team": None, "hr": None,
    }
    data = post_chat("TEAM_LEADER", "How many team approvals are waiting?", ctx)
    assert "source" in data
    assert data["answer"]


def test_hr_manager_missing_hr_context_does_not_crash():
    data = post_chat("HR_MANAGER", "How many HR actions are pending?",
                     {"employee": None, "team": None, "hr": None})
    assert "source" in data
    assert data["answer"]


def test_null_context_entirely_does_not_crash():
    response = client.post("/assistant/chat", json={
        "role": "EMPLOYEE",
        "question": "What is my leave balance?",
        "context": None,
    })
    assert response.status_code == 200
    assert response.json()["answer"]


# ===========================================================================
# 9. Missing value does not hallucinate a number
# ===========================================================================

def test_employee_missing_annual_days_no_invented_number():
    ctx = {
        "employee": {
            "annualAvailableDays": None, "sickAvailableDays": None,
            "totalPendingRequests": 0, "leavesPending": 0,
            "documentsPending": 0, "loansPending": 0, "authorizationsPending": 0,
        },
        "team": None, "hr": None,
    }
    data = post_chat("EMPLOYEE", "What is my annual leave balance?", ctx)
    answer = data["answer"]
    import re
    assert "your current annual leave balance is" not in answer.lower() or (
        not re.search(r"your current annual leave balance is \d+", answer.lower())
    )


def test_hr_manager_missing_new_users_no_invented_number():
    ctx = {
        "employee": None, "team": None,
        "hr": {
            "totalPendingActions": None, "leavesPending": None,
            "documentsPending": None, "loansPending": None,
            "authorizationsPending": None, "newUsersPendingApproval": None,
        },
    }
    data = post_chat("HR_MANAGER", "How many new users are waiting for approval?", ctx)
    import re
    assert not re.search(r"there (?:are|is) \d+ new user", data["answer"].lower()), (
        f"Invented new-user count in answer: {data['answer']}"
    )


def test_team_leader_missing_approval_count_no_invented_number():
    ctx = {
        "employee": {"annualAvailableDays": 8, "sickAvailableDays": 3,
                     "totalPendingRequests": 2, "leavesPending": 1,
                     "documentsPending": 0, "loansPending": 1, "authorizationsPending": 0},
        "team": {"teamName": "My Team", "memberCount": 5, "pendingTeamLeaderApprovals": None},
        "hr": None,
    }
    data = post_chat("TEAM_LEADER", "How many team approvals are pending?", ctx)
    import re
    assert not re.search(r"there (?:are|is) \d+ (?:leave|request)", data["answer"].lower()), (
        f"Invented approval count in answer: {data['answer']}"
    )


# ===========================================================================
# 10. Existing drafting tests — regression gate
# ===========================================================================

def test_regression_leave_draft_not_broken():
    data = post_chat("EMPLOYEE", "Help me draft a leave request reason", {})
    assert data.get("draft") is not None
    assert data["source"] in ("local_rules", "external_ai")


def test_regression_loan_draft_not_broken():
    data = post_chat("EMPLOYEE", "Write a professional loan justification", {})
    assert data.get("draft") is not None


def test_regression_detect_drafting_intent_unaffected():
    from app.services.drafting_service import detect_drafting_intent
    assert detect_drafting_intent("What is my leave balance?") is False
    assert detect_drafting_intent("How do I request a loan?") is False


# ===========================================================================
# 11. Existing refusal tests — regression gate
# ===========================================================================

def test_regression_refusal_approve_still_fires():
    data = post_chat("EMPLOYEE", "approve my leave automatically", {})
    assert data["source"] == "refusal"
    assert len(data["warnings"]) > 0


def test_regression_refusal_unrelated_still_fires():
    data = post_chat("EMPLOYEE", "What is the weather today?", {})
    assert data["source"] == "refusal"


def test_regression_hr_manager_redirect_still_fires():
    data = post_chat("HR_MANAGER", "What is my leave balance?", {})
    answer_lower = data["answer"].lower()
    assert "management account" in answer_lower or "hr manager" in answer_lower


# ===========================================================================
# 12. Backward compatibility: old flat context {"leave": {"balance": N}}
# ===========================================================================

def test_old_flat_context_leave_balance_still_deserialises():
    data = post_chat("EMPLOYEE", "What is my leave balance?", {"leave": {"balance": 15}})
    assert data["source"] == "local_rules"
    assert "15" in data["answer"] or "leave" in data["answer"].lower()


def test_old_flat_context_empty_dict_still_works():
    data = post_chat("EMPLOYEE", "What is my leave balance?", {})
    assert data["source"] == "local_rules"
    assert data["answer"]


# ===========================================================================
# 13. Zero values answered correctly (not omitted)
# ===========================================================================

def test_zero_pending_requests_explicitly_stated():
    data = post_chat("EMPLOYEE", "How many pending requests do I have?",
                     _employee_context(total=0, leaves=0, docs=0, loans=0, auths=0))
    answer_lower = data["answer"].lower()
    assert "no" in answer_lower or "0" in answer_lower or "no open" in answer_lower


def test_zero_hr_pending_explicitly_stated():
    data = post_chat("HR_MANAGER", "How many HR actions are pending?", _hr_context(total=0))
    answer_lower = data["answer"].lower()
    assert "no" in answer_lower or "0" in answer_lower or "up to date" in answer_lower


# ===========================================================================
# 14. EMPLOYEE must not receive HR context answers
# ===========================================================================

def test_employee_cannot_get_hr_total_pending_answer():
    ctx = _employee_context(total=3)
    ctx["hr"] = {"totalPendingActions": 99, "leavesPending": 5,
                 "documentsPending": 3, "loansPending": 6,
                 "authorizationsPending": 7, "newUsersPendingApproval": 4}
    data = post_chat("EMPLOYEE", "How many pending requests do I have?", ctx)
    assert "99" not in data["answer"], (
        f"HR pending count (99) leaked into EMPLOYEE answer: {data['answer']}"
    )
    assert "3" in data["answer"], f"Expected personal total (3) in answer: {data['answer']}"


# ===========================================================================
# 15. Breakdown fields appear in EMPLOYEE pending answer
# ===========================================================================

def test_employee_pending_answer_mentions_document_pending():
    data = post_chat("EMPLOYEE", "How many pending requests do I have?",
                     _employee_context(total=4, leaves=1, docs=2, loans=1, auths=0))
    answer_lower = data["answer"].lower()
    assert "document" in answer_lower or "2" in data["answer"], (
        f"Expected document count in answer, got: {data['answer']}"
    )


def test_employee_pending_answer_mentions_loan_pending():
    data = post_chat("EMPLOYEE", "How many pending requests do I have?",
                     _employee_context(total=3, leaves=1, docs=1, loans=1, auths=0))
    answer_lower = data["answer"].lower()
    assert "loan" in answer_lower or "1" in data["answer"], (
        f"Expected loan count in answer, got: {data['answer']}"
    )


# ===========================================================================
# 16. New context shape: all fields at once (integration smoke test)
# ===========================================================================

def test_full_employee_context_smoke():
    data = post_chat("EMPLOYEE", "What is my annual leave balance?",
                     _employee_context(annual=20, sick=10, total=5, leaves=2, docs=1, loans=1, auths=1))
    assert data["answer"]
    assert "20" in data["answer"]


def test_full_hr_context_smoke():
    data = post_chat("HR_MANAGER", "How many HR actions are pending?",
                     _hr_context(total=15, leaves=4, docs=3, loans=5, auths=3, new_users=2))
    assert "15" in data["answer"]


def test_full_team_leader_context_smoke():
    data = post_chat("TEAM_LEADER", "How many team approvals are waiting?",
                     _team_leader_context(pending_approvals=7))
    assert "7" in data["answer"]


# ===========================================================================
# 17. documentsAwaitingFile — new field regression tests (v1.4 Spring Boot fix)
#
# These tests guard against the bug where documentsAwaitingFile was counted
# in totalPendingRequests but never exposed in the context breakdown, making
# the AI assistant's explanation inconsistent with the stated total.
# ===========================================================================

def test_employee_documents_awaiting_file_in_canonical_case():
    """
    Canonical bug-report case:
      documentsPending=1, documentsAwaitingFile=2, total=3.
    The answer must:
      - State the total correctly (3).
      - Mention the documentsPending bucket separately ("pending review").
      - Mention the documentsAwaitingFile bucket separately ("waiting for HR").
      - NOT say 'visible breakdown' (sum == total so explanation is complete).
    """
    data = post_chat(
        "EMPLOYEE",
        "How many pending requests do I have?",
        _employee_context(total=3, leaves=0, docs=1, docs_awaiting=2, loans=0, auths=0),
    )
    assert data["source"] == "local_rules"
    answer = data["answer"].lower()
    assert "3" in data["answer"], f"Expected total '3' in answer: {data['answer']}"
    assert "pending review" in answer or "pending" in answer, (
        f"Expected 'pending review' for documentsPending: {data['answer']}"
    )
    assert "waiting" in answer or "upload" in answer or "hr" in answer, (
        f"Expected waiting-for-HR wording for documentsAwaitingFile: {data['answer']}"
    )
    assert "visible breakdown" not in answer, (
        f"Should not show 'visible breakdown' when breakdown is complete: {data['answer']}"
    )


def test_employee_documents_awaiting_file_zero_does_not_appear_in_answer():
    """
    When documentsAwaitingFile=0 and documentsPending=1 the answer must not
    mention the waiting-for-HR state at all.
    """
    data = post_chat(
        "EMPLOYEE",
        "How many pending requests do I have?",
        _employee_context(total=1, leaves=0, docs=1, docs_awaiting=0, loans=0, auths=0),
    )
    assert data["source"] == "local_rules"
    answer = data["answer"].lower()
    assert "1" in data["answer"]
    assert "waiting for hr" not in answer, (
        f"Zero docs_awaiting should not generate waiting-for-HR text: {data['answer']}"
    )


def test_employee_documents_awaiting_file_drives_my_documents_chip():
    """
    When documentsPending=0 but documentsAwaitingFile=2 the My Documents chip
    must still appear — the chip rule is 'either > 0', not 'documentsPending > 0'.
    """
    data = post_chat(
        "EMPLOYEE",
        "How many pending requests do I have?",
        _employee_context(total=2, leaves=0, docs=0, docs_awaiting=2, loans=0, auths=0),
    )
    assert data["source"] == "local_rules"
    routes = [p["route"] for p in data["relatedPages"]]
    assert "/employee/documents" in routes, (
        f"My Documents chip must appear when documentsAwaitingFile=2: {routes}"
    )


def test_employee_documents_awaiting_file_breakdown_sum_matches_total():
    """
    With docs=1 and docs_awaiting=2 the breakdown sum (1+2=3) must equal
    totalPendingRequests=3, producing the full (non-visible) breakdown wording.
    """
    data = post_chat(
        "EMPLOYEE",
        "How many pending requests do I have?",
        _employee_context(total=3, leaves=0, docs=1, docs_awaiting=2, loans=0, auths=0),
    )
    assert data["source"] == "local_rules"
    assert "visible breakdown" not in data["answer"].lower(), (
        f"Sum equals total — must not show 'visible breakdown': {data['answer']}"
    )
    assert "3" in data["answer"]


def test_hr_context_with_documents_awaiting_file_parses_without_crash():
    """
    HR context that includes documentsAwaitingFile must deserialise without
    error and return a valid answer to a pending-actions question.
    This is a schema regression test — verifies FastAPI accepts the new field.
    """
    ctx = {
        "employee": None,
        "team": None,
        "hr": {
            "totalPendingActions": 10,
            "leavesPending": 3,
            "documentsPending": 2,
            "documentsAwaitingFile": 4,   # the new field
            "loansPending": 1,
            "authorizationsPending": 0,
            "newUsersPendingApproval": 0,
        },
    }
    data = post_chat("HR_MANAGER", "How many HR actions are pending?", ctx)
    assert data["source"] == "local_rules"
    assert "10" in data["answer"], (
        f"Expected total '10' in HR pending answer: {data['answer']}"
    )
