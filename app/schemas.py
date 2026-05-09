from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional


class RelatedPage(BaseModel):
    """A navigable page suggestion returned with assistant responses."""
    label: str
    route: str


# ---------------------------------------------------------------------------
# Context sub-models matching the SafeAssistantContext Spring Boot DTO
# ---------------------------------------------------------------------------

class EmployeeContext(BaseModel):
    """
    Personal employee context — present for EMPLOYEE and TEAM_LEADER roles.
    All fields are optional because Spring Boot may omit them when a service
    fails during context assembly.  Defaults to None / 0 rather than failing.
    """
    annualAvailableDays: Optional[int] = None
    sickAvailableDays: Optional[int] = None
    totalPendingRequests: Optional[int] = 0
    leavesPending: Optional[int] = 0
    documentsPending: Optional[int] = 0
    # Approved document requests for which HR has not yet uploaded the file.
    # Exposed separately from documentsPending so the assistant can explain each
    # state precisely.  Counted in totalPendingRequests.
    documentsAwaitingFile: Optional[int] = 0
    loansPending: Optional[int] = 0
    authorizationsPending: Optional[int] = 0


class TeamContext(BaseModel):
    """
    Safe team summary — present for TEAM_LEADER role only.
    Sub-fields may be None when the Team Leader has no team assigned yet.

    pendingTeamLeaderApprovals defaults to None (not 0) so that an absent key
    is distinguishable from an explicit zero.  The Q&A handler treats:
      - None  -> value unknown; guide user to the page, never invent a count.
      - 0     -> confirmed zero; say the queue is clear.
      - N > 0 -> confirmed count; state the exact number.
    """
    teamName: Optional[str] = None
    memberCount: Optional[int] = None
    pendingTeamLeaderApprovals: Optional[int] = None


class TeamLeaveDecisionContext(BaseModel):
    """
    Safe selected leave request context for Team Leader decision support.
    Spring Boot performs authorization and data access; FastAPI only explains
    these already-safe fields and never triggers workflow actions.
    """
    available: Optional[bool] = None
    unavailableReason: Optional[str] = None
    leaveRequestId: Optional[int] = None
    employeeDisplayName: Optional[str] = None
    leaveType: Optional[str] = None
    startDate: Optional[str] = None
    endDate: Optional[str] = None
    deductedWorkingDays: Optional[int] = None
    status: Optional[str] = None
    approvalStage: Optional[str] = None
    reason: Optional[str] = None
    overlappingApprovedLeaves: Optional[int] = None
    overlappingPendingLeaves: Optional[int] = None
    teamMemberCount: Optional[int] = None
    activeTaskCount: Optional[int] = None
    dueSoonTaskCount: Optional[int] = None
    overdueTaskCount: Optional[int] = None
    highPriorityTaskCount: Optional[int] = None
    workloadContextAvailable: Optional[bool] = None
    overlapContextAvailable: Optional[bool] = None


class HrContext(BaseModel):
    """
    HR management context — present for HR_MANAGER role only.
    All fields are platform-wide aggregate counts — never individual employee data.
    """
    totalPendingActions: Optional[int] = 0
    leavesPending: Optional[int] = 0
    documentsPending: Optional[int] = 0
    # Approved document requests for which HR has not yet uploaded the file.
    # Exposed separately from documentsPending so the assistant can explain each
    # state precisely.  Counted in totalPendingActions.
    documentsAwaitingFile: Optional[int] = 0
    loansPending: Optional[int] = 0
    authorizationsPending: Optional[int] = 0
    newUsersPendingApproval: Optional[int] = 0


class ContextInfo(BaseModel):
    """
    Typed safe context forwarded from Spring Boot.

    Spring Boot now sends a nested SafeAssistantContext shape with three
    role-specific sub-objects.  The old flat shape (leave / requests / tasks / routes)
    is preserved as optional fields with empty defaults so that any existing test
    fixtures that pass the old shape continue to deserialise without error.

    Nullability contract (mirrors SafeAssistantContext.java):
      - employee : non-null for EMPLOYEE and TEAM_LEADER; null for HR_MANAGER / NEW_USER.
      - team     : non-null for TEAM_LEADER; null for all other roles.
      - hr       : non-null for HR_MANAGER; null for all other roles.
      - displayName: first + last name only — never used in answers but may be logged.

    Legacy flat fields (kept for backward-compatible test fixtures):
      - leave    : old {"balance": N} shape — still parsed but no longer used
                   by new handlers (they read employee.annualAvailableDays instead).
      - requests : old flat requests map — unused by new handlers.
      - tasks    : unused.
      - routes   : unused.
    """
    # New typed sub-objects
    displayName: Optional[str] = None
    employee: Optional[EmployeeContext] = None
    team: Optional[TeamContext] = None
    hr: Optional[HrContext] = None
    teamLeaveDecision: Optional[TeamLeaveDecisionContext] = None

    # Legacy flat fields — kept for backward compatibility with old test fixtures
    leave: Optional[Dict[str, Any]] = Field(default_factory=dict)
    requests: Optional[Dict[str, Any]] = Field(default_factory=dict)
    tasks: Optional[Dict[str, Any]] = Field(default_factory=dict)
    routes: Optional[Dict[str, Any]] = Field(default_factory=dict)


class ChatRequest(BaseModel):
    role: str
    question: str
    pageContext: Optional[str] = None
    context: Optional[ContextInfo] = Field(default_factory=ContextInfo)


class ChatResponse(BaseModel):
    answer: str
    reasons: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    relatedPages: List[RelatedPage] = Field(default_factory=list)
    disclaimer: str = "The assistant provides guidance only. Final decisions remain with authorized users."
    aiGenerated: bool = True
    # V2 traceability: identifies which pipeline step produced this response.
    # Values: "local_rules" | "external_ai" | "fallback" | "refusal"
    source: Optional[str] = None
    # V2 Phase 3: drafting assistant — populated only when a draft was generated.
    # None for all non-drafting responses (backward-compatible).
    draft: Optional[str] = None
    # V2 Phase 3.1: structured draft extraction.
    # draftType  — identifies the request type being drafted.
    #              Values: "LEAVE_REQUEST" | "LOAN_REQUEST" | "AUTHORIZATION_REQUEST"
    #                      | "DOCUMENT_REQUEST" | "IMPROVE_TEXT"
    #              None for all non-drafting responses (backward-compatible).
    # draftFields — extracted field values keyed by field name.
    #              Stable shape per draftType; null values for unextracted fields.
    #              For LOAN_REQUEST, repaymentMonths is optional and omitted from
    #              missingFields when the user does not specify a repayment period.
    #              None only for IMPROVE_TEXT (no structured fields apply).
    # missingFields — list of field names that could not be extracted from the
    #              user's input. Empty list for non-drafting and IMPROVE_TEXT.
    #              Spring Boot is responsible for final validation.
    draftType: Optional[str] = None
    draftFields: Optional[dict] = None
    missingFields: List[str] = Field(default_factory=list)
