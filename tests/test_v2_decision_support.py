"""
tests/test_v2_decision_support.py
---------------------------------
FastAPI-only tests for Team Leader selected leave decision support.
"""

from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def post_chat(role: str, question: str, context: dict | None = None) -> dict:
    return client.post(
        "/assistant/chat",
        json={"role": role, "question": question, "context": context or {}},
    ).json()


def _decision_context(
    *,
    available=True,
    unavailable_reason=None,
    overlapping_approved=0,
    overlapping_pending=0,
    overlap_available=True,
    workload_available=True,
    active=0,
    due_soon=0,
    overdue=0,
    high_priority=0,
) -> dict:
    return {
        "employee": None,
        "team": {"teamName": "Backend Squad", "memberCount": 5, "pendingTeamLeaderApprovals": 2},
        "hr": None,
        "teamLeaveDecision": {
            "available": available,
            "unavailableReason": unavailable_reason,
            "leaveRequestId": 99,
            "employeeDisplayName": "Ahmed Ben Ali",
            "leaveType": "ANNUAL",
            "startDate": "2026-06-10",
            "endDate": "2026-06-13",
            "deductedWorkingDays": 4,
            "status": "PENDING",
            "approvalStage": "PENDING_TL",
            "reason": "Family event",
            "overlappingApprovedLeaves": overlapping_approved,
            "overlappingPendingLeaves": overlapping_pending,
            "teamMemberCount": 5,
            "activeTaskCount": active,
            "dueSoonTaskCount": due_soon,
            "overdueTaskCount": overdue,
            "highPriorityTaskCount": high_priority,
            "workloadContextAvailable": workload_available,
            "overlapContextAvailable": overlap_available,
        },
    }


def _routes(data: dict) -> list[str]:
    return [p["route"] for p in data.get("relatedPages", [])]


def _assert_team_only_pages(data: dict) -> None:
    routes = _routes(data)
    assert "/team/requests" in routes
    assert "/team/calendar" in routes
    assert all(route.startswith("/team/") for route in routes), routes


def _assert_no_automatic_decision_wording(data: dict) -> None:
    answer = data["answer"].lower()
    forbidden = ("i approve", "i reject", "approved by ai", "rejected by ai")
    for phrase in forbidden:
        assert phrase not in answer, f"Automatic decision wording leaked: {data['answer']}"


def test_team_leader_complete_context_high_attention_and_reasons():
    data = post_chat(
        "TEAM_LEADER",
        "Should I approve this leave?",
        _decision_context(overlapping_approved=2, high_priority=1),
    )

    assert data["source"] == "local_rules"
    answer = data["answer"].lower()
    assert "ahmed ben ali" in answer
    assert "annual" in answer
    assert "2026-06-10" in answer and "2026-06-13" in answer
    assert "4 working" in answer
    assert "2 approved overlapping" in answer
    assert "1 high priority" in answer
    assert "attention level: high" in answer
    assert "team leader keeps the final decision" in answer
    assert any("high" in reason.lower() for reason in data["reasons"])
    _assert_team_only_pages(data)
    _assert_no_automatic_decision_wording(data)


def test_team_leader_medium_attention_for_single_overlap():
    data = post_chat(
        "TEAM_LEADER",
        "Can you help me decide on this leave?",
        _decision_context(overlapping_approved=1),
    )

    assert "attention level: medium" in data["answer"].lower()
    _assert_team_only_pages(data)
    _assert_no_automatic_decision_wording(data)


def test_team_leader_low_attention_when_all_confirmed_counts_are_zero():
    data = post_chat(
        "TEAM_LEADER",
        "What is the risk of this leave request?",
        _decision_context(),
    )

    answer = data["answer"].lower()
    assert "attention level: low" in answer
    assert "0 approved overlapping" in answer
    assert "0 pending overlapping" in answer
    assert "0 overdue" in answer
    _assert_team_only_pages(data)
    _assert_no_automatic_decision_wording(data)


def test_team_leader_missing_decision_context_asks_to_select_request_first():
    data = post_chat(
        "TEAM_LEADER",
        "Analyze this leave request",
        {"employee": None, "team": {"teamName": "Backend Squad"}, "hr": None},
    )

    answer = data["answer"].lower()
    assert "select" in answer or "open" in answer
    assert "leave request first" in answer
    assert "cannot decide for you" in answer
    _assert_team_only_pages(data)


def test_team_leader_unavailable_context_uses_reason_and_does_not_invent():
    data = post_chat(
        "TEAM_LEADER",
        "Give me decision support for this leave",
        _decision_context(available=False, unavailable_reason="LEAVE_REQUEST_NOT_VISIBLE"),
    )

    answer = data["answer"].lower()
    assert "unavailable" in answer
    assert "leave_request_not_visible" in answer
    assert "cannot invent" in answer
    assert "ahmed ben ali" not in answer
    assert "4 working" not in answer
    _assert_team_only_pages(data)
    _assert_no_automatic_decision_wording(data)


def test_team_leader_missing_overlap_and_workload_context_is_unknown_and_caveated():
    data = post_chat(
        "TEAM_LEADER",
        "Analyze this leave request",
        _decision_context(overlap_available=False, workload_available=False),
    )

    answer = data["answer"].lower()
    assert "attention level: unknown" in answer
    assert "overlap context is not available" in answer
    assert "workload context is not available" in answer
    assert "DECISION_CONTEXT_INCOMPLETE" in data["warnings"]
    _assert_team_only_pages(data)
    _assert_no_automatic_decision_wording(data)


def test_employee_and_hr_manager_cannot_use_team_leader_decision_support():
    for role in ("EMPLOYEE", "HR_MANAGER"):
        data = post_chat(
            role,
            "Analyze this leave request",
            _decision_context(overlapping_approved=2),
        )
        answer = data["answer"].lower()
        assert "only available for team leader" in answer
        assert "/team/requests" not in _routes(data)
        assert "/employee" not in " ".join(_routes(data))
        _assert_no_automatic_decision_wording(data)


def test_direct_action_command_still_refused():
    data = post_chat(
        "TEAM_LEADER",
        "Approve this leave",
        _decision_context(),
    )

    assert data["source"] == "refusal"
    assert "cannot perform or trigger administrative actions" in data["answer"].lower()
