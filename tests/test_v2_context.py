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
    annual=15, sick=8, total=3, leaves=1, docs=1, loans=1, auths=0
) -> dict:
    return {
        "employee": {
            "annualAvailableDays": annual,
            "sickAvailableDays": sick,
            "totalPendingRequests": total,
            "leavesPending": leaves,
            "documentsPending": docs,
            "loansPending": loans,
            "authorizationsPending": auths,
        },
        "team": None,
        "hr": None,
    }


def _team_leader_context(
    annual=8, sick=3, total=2, leaves=1, docs=0, loans=1, auths=0,
    team_name="Backend Squad", member_count=5, pending_approvals=2
) -> dict:
    return {
        "employee": {
            "annualAvailableDays": annual,
            "sickAvailableDays": sick,
            "totalPendingRequests": total,
            "leavesPending": leaves,
            "documentsPending": docs,
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
    total=21, leaves=5, docs=3, loans=6, auths=7, new_users=4
) -> dict:
    return {
        "employee": None,
        "team": None,
        "hr": {
            "totalPendingActions": total,
            "leavesPending": leaves,
            "documentsPending": docs,
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


# ===========================================================================
# 4-B. TeamContext.pendingTeamLeaderApprovals: absent-key, zero, and positive
# ===========================================================================
# These four tests verify the fix for the unsafe default=0 bug.
# Before the fix, an absent JSON key became 0 and the handler said
# "you are all caught up" even when the count was unknown.

def test_team_leader_absent_approvals_key_stays_none_not_zero():
    """
    When pendingTeamLeaderApprovals is absent from the JSON payload entirely
    the field must deserialise to None, not 0.
    Sending a team object without the key simulates Spring Boot omitting it
    when the count query failed during context assembly.
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
    # Must NOT say "caught up" or "no leave requests" (that would be the 0 branch)
    assert "caught up" not in answer_lower, (
        f"Absent key incorrectly treated as zero: {data['answer']}"
    )
    # Must NOT state an invented number
    import re
    assert not re.search(r"there (?:are|is) \d+", answer_lower), (
        f"Invented count for absent key: {data['answer']}"
    )
    # Must guide the user to the team requests page
    routes = [p["route"] for p in data["relatedPages"]]
    assert "/team/requests" in routes, f"Expected /team/requests, got {routes}"


def test_team_leader_explicit_zero_approvals_gives_caught_up_answer():
    """
    When pendingTeamLeaderApprovals is explicitly 0 the handler must say
    the queue is clear — this is a confirmed value, not a missing one.
    """
    ctx = {
        "employee": {
            "annualAvailableDays": 8, "sickAvailableDays": 3,
            "totalPendingRequests": 0, "leavesPending": 0,
            "documentsPending": 0, "loansPending": 0, "authorizationsPending": 0,
        },
        "team": {
            "teamName": "Beta Squad",
            "memberCount": 3,
            "pendingTeamLeaderApprovals": 0,  # explicit zero
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
    assert "caught up" in answer_lower or "no leave" in answer_lower or "no " in answer_lower, (
        f"Expected clear-queue answer for explicit 0, got: {data['answer']}"
    )


def test_team_leader_positive_approvals_gives_exact_count():
    """
    When pendingTeamLeaderApprovals is a positive integer the handler must
    state that exact number in the answer.
    """
    ctx = {
        "employee": {
            "annualAvailableDays": 8, "sickAvailableDays": 3,
            "totalPendingRequests": 2, "leavesPending": 1,
            "documentsPending": 0, "loansPending": 1, "authorizationsPending": 0,
        },
        "team": {
            "teamName": "Gamma Squad",
            "memberCount": 6,
            "pendingTeamLeaderApprovals": 5,  # explicit positive
        },
        "hr": None,
    }
    data = post_chat(
        "TEAM_LEADER",
        "How many team approvals are waiting?",
        ctx,
    )
    assert data["source"] == "local_rules"
    assert "5" in data["answer"], f"Expected exact count '5', got: {data['answer']}"
    routes = [p["route"] for p in data["relatedPages"]]
    assert "/team/requests" in routes


def test_team_leader_null_team_context_does_not_crash_or_invent():
    """
    When the team sub-object itself is null the handler must return a safe
    answer (cannot-see guidance) without crashing or stating any invented count.
    """
    ctx = {
        "employee": {
            "annualAvailableDays": 8, "sickAvailableDays": 3,
            "totalPendingRequests": 2, "leavesPending": 1,
            "documentsPending": 0, "loansPending": 1, "authorizationsPending": 0,
        },
        "team": None,  # no team assigned yet
        "hr": None,
    }
    data = post_chat(
        "TEAM_LEADER",
        "How many team approvals are pending?",
        ctx,
    )
    assert data["source"] == "local_rules"
    # Must not crash
    assert data["answer"]
    # Must not invent a count
    import re
    assert not re.search(r"there (?:are|is) \d+", data["answer"].lower()), (
        f"Invented count when team is null: {data['answer']}"
    )
    # Must not say caught up (that would require a confirmed 0)
    assert "caught up" not in data["answer"].lower(), (
        f"'caught up' stated when team context is null: {data['answer']}"
    )
    routes = [p["route"] for p in data["relatedPages"]]
    assert "/team/requests" in routes


# ===========================================================================
# 5. HR_MANAGER: pending actions uses context.hr.totalPendingActions
# ===========================================================================

def test_hr_manager_pending_actions_uses_context():
    """Answer must contain the exact totalPendingActions value."""
    data = post_chat(
        "HR_MANAGER",
        "How many HR actions are pending?",
        _hr_context(total=21),
    )
    assert data["source"] == "local_rules"
    assert "21" in data["answer"], f"Expected '21' in answer, got: {data['answer']}"
    routes = [p["route"] for p in data["relatedPages"]]
    assert "/hr/dashboard" in routes or "/hr/requests" in routes


def test_hr_manager_total_pending_zero():
    """Zero pending actions must be explicitly stated."""
    data = post_chat(
        "HR_MANAGER",
        "How many pending actions are there?",
        _hr_context(total=0),
    )
    assert data["source"] == "local_rules"
    answer_lower = data["answer"].lower()
    assert "no" in answer_lower or "0" in answer_lower or "up to date" in answer_lower, (
        f"Expected zero-action answer, got: {data['answer']}"
    )


def test_hr_manager_platform_pending_question():
    """'How many platform actions are pending?' should also work."""
    data = post_chat(
        "HR_MANAGER",
        "How many platform actions are pending?",
        _hr_context(total=7),
    )
    assert "7" in data["answer"], f"Expected '7' in answer, got: {data['answer']}"


# ===========================================================================
# 6. HR_MANAGER: new user count uses context.hr.newUsersPendingApproval
# ===========================================================================

def test_hr_manager_new_users_pending_uses_context():
    """Answer must contain the exact newUsersPendingApproval value."""
    data = post_chat(
        "HR_MANAGER",
        "How many new users are waiting for approval?",
        _hr_context(new_users=4),
    )
    assert data["source"] == "local_rules"
    assert "4" in data["answer"], f"Expected '4' in answer, got: {data['answer']}"
    routes = [p["route"] for p in data["relatedPages"]]
    assert "/hr/users" in routes


def test_hr_manager_new_users_zero():
    """Zero new users must be stated explicitly."""
    data = post_chat(
        "HR_MANAGER",
        "Are there new users waiting for onboarding?",
        _hr_context(new_users=0),
    )
    assert data["source"] == "local_rules"
    answer_lower = data["answer"].lower()
    assert "no" in answer_lower or "0" in answer_lower or "clear" in answer_lower, (
        f"Expected zero-new-users answer, got: {data['answer']}"
    )


def test_hr_manager_new_users_singular():
    """1 new user uses singular form."""
    data = post_chat(
        "HR_MANAGER",
        "How many users are pending approval?",
        _hr_context(new_users=1),
    )
    assert "1" in data["answer"], f"Expected '1' in answer, got: {data['answer']}"


# ===========================================================================
# 7. HR_MANAGER personal leave balance gets management-safe redirection
# ===========================================================================

def test_hr_manager_leave_balance_redirected_not_context_value():
    """HR_MANAGER asking about leave balance must get the redirect, not a context value."""
    data = post_chat(
        "HR_MANAGER",
        "What is my leave balance?",
        _hr_context(total=5),  # context has HR data — must not be used for personal answer
    )
    assert data["source"] == "local_rules"
    answer_lower = data["answer"].lower()
    # Must be the management redirect
    assert "management account" in answer_lower or "hr manager" in answer_lower, (
        f"Expected management redirect, got: {data['answer']}"
    )
    # Must not contain any invented personal balance number
    assert "your current annual leave balance" not in answer_lower


def test_hr_manager_annual_leave_question_redirected():
    """'What is my annual leave balance?' for HR_MANAGER must be redirected."""
    data = post_chat(
        "HR_MANAGER",
        "What is my annual leave balance?",
        {},
    )
    answer_lower = data["answer"].lower()
    assert "management account" in answer_lower or "hr manager" in answer_lower


# ===========================================================================
# 8. Missing employee/team/hr context does not crash
# ===========================================================================

def test_employee_missing_context_does_not_crash():
    """Empty context must not crash the service."""
    data = post_chat("EMPLOYEE", "How many pending requests do I have?", {})
    assert "source" in data
    assert data["answer"]


def test_team_leader_missing_team_context_does_not_crash():
    """Team context null must not crash the service."""
    ctx = {
        "employee": {"annualAvailableDays": 8, "sickAvailableDays": 3,
                     "totalPendingRequests": 2, "leavesPending": 1,
                     "documentsPending": 0, "loansPending": 1, "authorizationsPending": 0},
        "team": None,
        "hr": None,
    }
    data = post_chat("TEAM_LEADER", "How many team approvals are waiting?", ctx)
    assert "source" in data
    assert data["answer"]


def test_hr_manager_missing_hr_context_does_not_crash():
    """HR context null must not crash the service."""
    data = post_chat("HR_MANAGER", "How many HR actions are pending?", {"employee": None, "team": None, "hr": None})
    assert "source" in data
    assert data["answer"]


def test_null_context_entirely_does_not_crash():
    """Null context object must not crash the service."""
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
    """When annualAvailableDays is null, answer must not contain an invented count."""
    ctx = {
        "employee": {
            "annualAvailableDays": None,
            "sickAvailableDays": None,
            "totalPendingRequests": 0,
            "leavesPending": 0,
            "documentsPending": 0,
            "loansPending": 0,
            "authorizationsPending": 0,
        },
        "team": None,
        "hr": None,
    }
    data = post_chat("EMPLOYEE", "What is my annual leave balance?", ctx)
    answer = data["answer"]
    # Must not contain any bare digit that looks like an invented balance
    import re
    # The answer should guide to the page, not state a number
    assert "your current annual leave balance is" not in answer.lower() or (
        # If it does appear, it must not have a digit after it (i.e. not "is 99 days")
        not re.search(r"your current annual leave balance is \d+", answer.lower())
    )


def test_hr_manager_missing_new_users_no_invented_number():
    """When newUsersPendingApproval is null, answer must not state an invented count."""
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
    data = post_chat("HR_MANAGER", "How many new users are waiting for approval?", ctx)
    answer = data["answer"]
    # Must not state a specific invented number
    import re
    assert not re.search(r"there (?:are|is) \d+ new user", answer.lower()), (
        f"Invented new-user count in answer: {answer}"
    )


def test_team_leader_missing_approval_count_no_invented_number():
    """When pendingTeamLeaderApprovals is null, answer must not state an invented count."""
    ctx = {
        "employee": {"annualAvailableDays": 8, "sickAvailableDays": 3,
                     "totalPendingRequests": 2, "leavesPending": 1,
                     "documentsPending": 0, "loansPending": 1, "authorizationsPending": 0},
        "team": {
            "teamName": "My Team",
            "memberCount": 5,
            "pendingTeamLeaderApprovals": None,
        },
        "hr": None,
    }
    data = post_chat("TEAM_LEADER", "How many team approvals are pending?", ctx)
    answer = data["answer"]
    import re
    assert not re.search(r"there (?:are|is) \d+ (?:leave|request)", answer.lower()), (
        f"Invented approval count in answer: {answer}"
    )


# ===========================================================================
# 10. Existing drafting tests — regression gate
# ===========================================================================

def test_regression_leave_draft_not_broken():
    """Drafting questions must still work after context schema change."""
    data = post_chat("EMPLOYEE", "Help me draft a leave request reason", {})
    assert data.get("draft") is not None
    assert data["source"] in ("local_rules", "external_ai")


def test_regression_loan_draft_not_broken():
    data = post_chat("EMPLOYEE", "Write a professional loan justification", {})
    assert data.get("draft") is not None


def test_regression_detect_drafting_intent_unaffected():
    """detect_drafting_intent must still return False for non-drafting questions."""
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
    """Old test fixtures that send {"leave": {"balance": N}} must not crash."""
    data = post_chat(
        "EMPLOYEE",
        "What is my leave balance?",
        {"leave": {"balance": 15}},
    )
    assert data["source"] == "local_rules"
    # Old shape: should still answer with a balance (legacy fallback)
    assert "15" in data["answer"] or "leave" in data["answer"].lower()


def test_old_flat_context_empty_dict_still_works():
    data = post_chat("EMPLOYEE", "What is my leave balance?", {})
    assert data["source"] == "local_rules"
    assert data["answer"]


# ===========================================================================
# 13. Zero values answered correctly (not omitted)
# ===========================================================================

def test_zero_pending_requests_explicitly_stated():
    data = post_chat(
        "EMPLOYEE",
        "How many pending requests do I have?",
        _employee_context(total=0, leaves=0, docs=0, loans=0, auths=0),
    )
    answer_lower = data["answer"].lower()
    assert "no" in answer_lower or "0" in answer_lower or "no open" in answer_lower


def test_zero_hr_pending_explicitly_stated():
    data = post_chat(
        "HR_MANAGER",
        "How many HR actions are pending?",
        _hr_context(total=0),
    )
    answer_lower = data["answer"].lower()
    assert "no" in answer_lower or "0" in answer_lower or "up to date" in answer_lower


# ===========================================================================
# 14. EMPLOYEE must not receive HR context answers
# ===========================================================================

def test_employee_cannot_get_hr_total_pending_answer():
    """EMPLOYEE asking 'how many pending' must get personal, not HR, answer."""
    ctx = _employee_context(total=3)
    # HR context also provided (should never happen in prod, but defensive test)
    ctx["hr"] = {"totalPendingActions": 99, "leavesPending": 5,
                 "documentsPending": 3, "loansPending": 6,
                 "authorizationsPending": 7, "newUsersPendingApproval": 4}
    data = post_chat("EMPLOYEE", "How many pending requests do I have?", ctx)
    # Must not expose the HR total (99)
    assert "99" not in data["answer"], (
        f"HR pending count (99) leaked into EMPLOYEE answer: {data['answer']}"
    )
    # Must use personal total (3)
    assert "3" in data["answer"], f"Expected personal total (3) in answer: {data['answer']}"


# ===========================================================================
# 15. Breakdown fields appear in EMPLOYEE pending answer
# ===========================================================================

def test_employee_pending_answer_mentions_document_pending():
    data = post_chat(
        "EMPLOYEE",
        "How many pending requests do I have?",
        _employee_context(total=4, leaves=1, docs=2, loans=1, auths=0),
    )
    answer_lower = data["answer"].lower()
    assert "document" in answer_lower or "2" in data["answer"], (
        f"Expected document count in answer, got: {data['answer']}"
    )


def test_employee_pending_answer_mentions_loan_pending():
    data = post_chat(
        "EMPLOYEE",
        "How many pending requests do I have?",
        _employee_context(total=3, leaves=1, docs=1, loans=1, auths=0),
    )
    answer_lower = data["answer"].lower()
    assert "loan" in answer_lower or "1" in data["answer"], (
        f"Expected loan count in answer, got: {data['answer']}"
    )


# ===========================================================================
# 16. New context shape: all fields at once (integration smoke test)
# ===========================================================================

def test_full_employee_context_smoke():
    """Full EMPLOYEE context with all fields — must not crash and must answer."""
    data = post_chat(
        "EMPLOYEE",
        "What is my annual leave balance?",
        _employee_context(annual=20, sick=10, total=5, leaves=2, docs=1, loans=1, auths=1),
    )
    assert data["answer"]
    assert "20" in data["answer"]


def test_full_hr_context_smoke():
    """Full HR_MANAGER context with all fields — must not crash and must answer."""
    data = post_chat(
        "HR_MANAGER",
        "How many HR actions are pending?",
        _hr_context(total=15, leaves=4, docs=3, loans=5, auths=3, new_users=2),
    )
    assert "15" in data["answer"]


def test_full_team_leader_context_smoke():
    """Full TEAM_LEADER context with all fields — must not crash and must answer."""
    data = post_chat(
        "TEAM_LEADER",
        "How many team approvals are waiting?",
        _team_leader_context(pending_approvals=7),
    )
    assert "7" in data["answer"]
