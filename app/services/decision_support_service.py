"""
decision_support_service.py
---------------------------
Deterministic Team Leader leave decision-support responses.

FastAPI does not fetch data or make workflow decisions. It only explains the
safe teamLeaveDecision context that Spring Boot already authorized and sent.
"""

from typing import Optional

from app.schemas import ChatRequest, ChatResponse, RelatedPage, TeamLeaveDecisionContext


_DECISION_SUPPORT_PHRASES: tuple[str, ...] = (
    "should i approve this leave",
    "can you help me decide on this leave",
    "what is the risk of this leave request",
    "analyze this leave request",
    "give me decision support for this leave",
)


def is_decision_support_question(question: str) -> bool:
    """Return True for narrow Team Leader selected-leave assessment questions."""
    q = question.lower().strip()
    return any(phrase in q for phrase in _DECISION_SUPPORT_PHRASES)


def get_decision_support_response(request: ChatRequest) -> Optional[ChatResponse]:
    """
    Return a local deterministic decision-support response when the question
    matches this feature, otherwise None.
    """
    if not is_decision_support_question(request.question):
        return None

    role = _normalize_role(request.role)
    if role != "TEAM_LEADER":
        return ChatResponse(
            answer=(
                "Team Leader leave decision support is only available for Team Leader "
                "accounts reviewing their own team requests. I cannot expose that context "
                "for this role."
            ),
            warnings=["TEAM_LEADER_DECISION_CONTEXT_NOT_AVAILABLE"],
            relatedPages=[],
            source="local_rules",
        )

    context = request.context
    decision = context.teamLeaveDecision if context else None
    if decision is None:
        return ChatResponse(
            answer=(
                "Please select or open a team leave request first so I can explain the "
                "decision context from the safe information provided by the platform. "
                "I cannot decide for you. The Team Leader keeps the final decision."
            ),
            relatedPages=_team_pages(),
            source="local_rules",
        )

    if decision.available is False:
        reason = decision.unavailableReason
        reason_text = f" Reason: {reason}." if reason else ""
        return ChatResponse(
            answer=(
                "Decision context for the selected leave request is unavailable right now."
                f"{reason_text} I cannot invent request details or counts. "
                "I cannot decide for you. The Team Leader keeps the final decision."
            ),
            warnings=["TEAM_LEADER_DECISION_CONTEXT_UNAVAILABLE"],
            relatedPages=_team_pages(),
            source="local_rules",
        )

    return _available_decision_response(decision)


def _available_decision_response(decision: TeamLeaveDecisionContext) -> ChatResponse:
    attention = _attention_level(decision)

    employee = decision.employeeDisplayName or "the selected employee"
    leave_type = _pretty_value(decision.leaveType)
    start = decision.startDate or "the selected start date"
    end = decision.endDate or "the selected end date"
    days = decision.deductedWorkingDays
    day_text = f"{days} working day(s)" if days is not None else "the provided working-day count"

    lines = [
        (
            f"Decision context for {employee}: {leave_type} leave from {start} to {end}, "
            f"deducting {day_text}."
        )
    ]

    if decision.status:
        lines.append(f"Current status: {decision.status}.")
    if decision.approvalStage:
        lines.append(f"Current approval stage: {decision.approvalStage}.")
    if decision.reason:
        lines.append(f"Employee reason: {decision.reason}")

    if decision.overlapContextAvailable is True:
        approved = _count(decision.overlappingApprovedLeaves)
        pending = _count(decision.overlappingPendingLeaves)
        lines.append(
            f"Team overlap context: {approved} approved overlapping leave(s) and "
            f"{pending} pending overlapping leave request(s)."
        )
    else:
        lines.append("Team overlap context is not available, so I cannot confirm coverage impact.")

    if decision.workloadContextAvailable is True:
        active = _count(decision.activeTaskCount)
        due_soon = _count(decision.dueSoonTaskCount)
        overdue = _count(decision.overdueTaskCount)
        high_priority = _count(decision.highPriorityTaskCount)
        lines.append(
            f"Workload context: {active} active task(s), {due_soon} due soon, "
            f"{overdue} overdue, and {high_priority} high priority."
        )
    else:
        lines.append("Workload context is not available, so I cannot confirm task pressure.")

    lines.append(f"Attention level: {attention}.")
    lines.append(_risk_reason(decision, attention))
    lines.append("I cannot decide for you. The Team Leader keeps the final decision.")

    return ChatResponse(
        answer="\n".join(lines),
        reasons=_reasons(decision, attention),
        warnings=[] if attention != "unknown" else ["DECISION_CONTEXT_INCOMPLETE"],
        relatedPages=_team_pages(),
        source="local_rules",
    )


def _attention_level(decision: TeamLeaveDecisionContext) -> str:
    approved = _count(decision.overlappingApprovedLeaves)
    pending = _count(decision.overlappingPendingLeaves)
    due_soon = _count(decision.dueSoonTaskCount)
    overdue = _count(decision.overdueTaskCount)
    high_priority = _count(decision.highPriorityTaskCount)

    if approved >= 2 or pending >= 3 or overdue > 0 or high_priority > 0:
        return "high"
    if approved == 1 or pending >= 1 or due_soon > 0:
        return "medium"
    if (
        decision.overlapContextAvailable is True
        and decision.workloadContextAvailable is True
        and approved == 0
        and pending == 0
        and due_soon == 0
        and overdue == 0
        and high_priority == 0
    ):
        return "low"
    return "unknown"


def _risk_reason(decision: TeamLeaveDecisionContext, attention: str) -> str:
    if attention == "high":
        return "Risk factors: one or more strong coverage or workload signals need close review."
    if attention == "medium":
        return "Risk factors: there is some overlap or near-term workload pressure to review."
    if attention == "low":
        return "Risk factors: the supplied overlap and workload counts do not show pressure."
    return "Risk factors: important context is missing, so the attention level is caveated."


def _reasons(decision: TeamLeaveDecisionContext, attention: str) -> list[str]:
    reasons = [f"Attention level is {attention} based only on safe context from Spring Boot."]
    if decision.overlapContextAvailable is not True:
        reasons.append("Overlap context is not available.")
    if decision.workloadContextAvailable is not True:
        reasons.append("Workload context is not available.")
    if decision.overlapContextAvailable is True:
        reasons.append(
            f"Overlap counts: {_count(decision.overlappingApprovedLeaves)} approved, "
            f"{_count(decision.overlappingPendingLeaves)} pending."
        )
    if decision.workloadContextAvailable is True:
        reasons.append(
            f"Task counts: {_count(decision.activeTaskCount)} active, "
            f"{_count(decision.dueSoonTaskCount)} due soon, "
            f"{_count(decision.overdueTaskCount)} overdue, "
            f"{_count(decision.highPriorityTaskCount)} high priority."
        )
    return reasons


def _team_pages() -> list[RelatedPage]:
    return [
        RelatedPage(label="Team Requests", route="/team/requests"),
        RelatedPage(label="Team Leave Calendar", route="/team/calendar"),
    ]


def _normalize_role(role: str) -> str:
    normalized = role.upper()
    if normalized.startswith("ROLE_"):
        normalized = normalized[len("ROLE_"):]
    return normalized


def _count(value: Optional[int]) -> int:
    return value if value is not None else 0


def _pretty_value(value: Optional[str]) -> str:
    if not value:
        return "selected"
    return value.replace("_", " ").lower()
