"""
platform_help_service.py
------------------------
Role-aware, rule-based platform guidance for ArabSoft users (v1.4 / local mode).

Each handler receives the lowercased question and the original ChatRequest,
and returns a ChatResponse when it matches, or None to fall through.

Valid roles (Spring Boot TypeRole enum):
  EMPLOYEE    — personal employee self-service  -> /employee/*
  TEAM_LEADER — team management + personal flow -> /team/* and /employee/*
  HR_MANAGER  — HR administration only          -> /hr/*

Role normalization
------------------
Spring Boot sends the role as the enum name (e.g. "HR_MANAGER").
As a defensive measure the ROLE_ Spring Security prefix is also stripped,
so "ROLE_HR_MANAGER" is treated identically to "HR_MANAGER".

Context reading rules
---------------------
Spring Boot now sends a typed SafeAssistantContext shape (v1.3):
  context.employee  -- EmployeeContext  (EMPLOYEE / TEAM_LEADER)
  context.team      -- TeamContext      (TEAM_LEADER)
  context.hr        -- HrContext        (HR_MANAGER)

Handlers read ONLY the field they need via the safe helper functions below.
None/missing context is handled gracefully: the assistant tells the user it
cannot see that specific value and directs them to the correct platform page.
Numbers are NEVER invented.

All relatedPages routes are verified against the actual React Router
route table in App.jsx.  No invented or placeholder routes.
"""

from typing import Optional
from app.schemas import ChatRequest, ChatResponse, RelatedPage


# ---------------------------------------------------------------------------
# Personal employee-flow phrases that HR_MANAGER should NOT be given
# ---------------------------------------------------------------------------

_PERSONAL_EMPLOYEE_PHRASES: frozenset[str] = frozenset([
    # leave balance
    "my leave balance",
    "leave balance",
    "how much leave",
    "annual leave",
    "sick leave",
    "my sick",
    # loan
    "can i request a loan",
    "request a loan",
    "my loan request",
    "my loan",
    # leave submission
    "submit my own leave",
    "submit my leave",
    "submit a leave request",
    "how do i submit",
    "my leave request",
    # personal requests (MY requests, not general)
    "my personal requests",
    "show my requests",
    "my requests",
    "how many pending requests do i",
    "do i have pending",
    "my pending requests",
    "my open requests",
])


def _is_personal_employee_question(q: str) -> bool:
    """Return True if the lowercased question matches any personal employee-flow phrase."""
    return any(phrase in q for phrase in _PERSONAL_EMPLOYEE_PHRASES)


# ---------------------------------------------------------------------------
# Team-management phrases for TEAM_LEADER
# ---------------------------------------------------------------------------

_TEAM_MANAGEMENT_PHRASES: frozenset[str] = frozenset([
    "team request",
    "team leave",
    "Team Leave Calendar",
    "team member",
    "my team",
    "team task",
    "team workload",
    "pending leave",   # team-leader context: team's pending leaves
    "team project",
    "team approval",
    "approvals waiting",
    "approvals pending",
    "how many approvals",
    "pending approvals",
    "team pending",
])


def _is_team_management_question(q: str) -> bool:
    """Return True if the question is about managing the team (TEAM_LEADER scope)."""
    return any(phrase in q for phrase in _TEAM_MANAGEMENT_PHRASES)


# ---------------------------------------------------------------------------
# HR management phrases (counts / admin dashboard)
# ---------------------------------------------------------------------------

_HR_DASHBOARD_PHRASES: frozenset[str] = frozenset([
    "how many hr",
    "hr pending",
    "pending actions",
    "total pending",
    "platform pending",
    "new user",
    "new users",
    "users pending",
    "users waiting",
    "pending approval",
    "onboarding",
    "how many actions",
    "platform actions",
])


def _is_hr_dashboard_question(q: str) -> bool:
    """Return True if the question is about HR-level platform counts."""
    return any(phrase in q for phrase in _HR_DASHBOARD_PHRASES)


# ---------------------------------------------------------------------------
# Safe context accessor helpers
# ---------------------------------------------------------------------------

def _get_annual_days(request: ChatRequest) -> Optional[int]:
    """Return the EMPLOYEE annual leave available days from context, or None."""
    ctx = request.context
    if ctx is None:
        return None
    emp = ctx.employee
    if emp is None:
        return None
    return emp.annualAvailableDays  # may itself be None


def _get_sick_days(request: ChatRequest) -> Optional[int]:
    """Return the EMPLOYEE sick leave available days from context, or None."""
    ctx = request.context
    if ctx is None:
        return None
    emp = ctx.employee
    if emp is None:
        return None
    return emp.sickAvailableDays  # may itself be None


def _get_total_pending(request: ChatRequest) -> Optional[int]:
    """Return totalPendingRequests from employee context, or None."""
    ctx = request.context
    if ctx is None:
        return None
    emp = ctx.employee
    if emp is None:
        return None
    return emp.totalPendingRequests  # may itself be None


def _get_leaves_pending(request: ChatRequest) -> Optional[int]:
    """Return leavesPending from employee context, or None."""
    ctx = request.context
    if ctx is None:
        return None
    emp = ctx.employee
    if emp is None:
        return None
    return emp.leavesPending


def _get_tl_pending_approvals(request: ChatRequest) -> Optional[int]:
    """
    Return pendingTeamLeaderApprovals from team context, or None.

    Three-way return contract (mirrors TeamContext default=None):
      None  -> key was absent or team context itself is missing; value unknown.
      0     -> Spring Boot confirmed zero pending approvals.
      N > 0 -> Spring Boot confirmed exact count.
    """
    ctx = request.context
    if ctx is None:
        return None
    team = ctx.team
    if team is None:
        return None
    return team.pendingTeamLeaderApprovals


def _get_hr_total_pending(request: ChatRequest) -> Optional[int]:
    """Return totalPendingActions from hr context, or None."""
    ctx = request.context
    if ctx is None:
        return None
    hr = ctx.hr
    if hr is None:
        return None
    return hr.totalPendingActions


def _get_hr_new_users(request: ChatRequest) -> Optional[int]:
    """Return newUsersPendingApproval from hr context, or None."""
    ctx = request.context
    if ctx is None:
        return None
    hr = ctx.hr
    if hr is None:
        return None
    return hr.newUsersPendingApproval


# ---------------------------------------------------------------------------
# "Cannot see" helper — used when a context value is missing
# ---------------------------------------------------------------------------

def _cannot_see(field_description: str, page_label: str, page_route: str) -> ChatResponse:
    """
    Return a safe answer when the requested context value is absent.
    Never invents a number.
    """
    return ChatResponse(
        answer=(
            f"I can see you are asking about {field_description}, but I do not have access "
            f"to that specific value right now. You can check it directly on the platform."
        ),
        relatedPages=[RelatedPage(label=page_label, route=page_route)],
    )


# ---------------------------------------------------------------------------
# Public dispatcher
# ---------------------------------------------------------------------------

def get_platform_help(request: ChatRequest) -> Optional[ChatResponse]:
    """
    Try each role-aware handler in order.
    Return a ChatResponse on the first match, or None if nothing matches.
    """
    q = request.question.lower()

    # Defensive role normalisation:
    #   1. Upper-case the raw value.
    #   2. Strip the Spring Security "ROLE_" prefix if present.
    role = request.role.upper()
    if role.startswith("ROLE_"):
        role = role[len("ROLE_"):]

    # IMPORTANT: handler order is load-bearing.
    #   1. _handle_hr_manager_personal_redirect  MUST stay first —
    #      it blocks HR_MANAGER from reaching employee handlers.
    #   2. _handle_hr_dashboard_counts           MUST come second for HR_MANAGER —
    #      catches count/pending questions before generic handlers fire.
    #   3. _handle_team_leader_approvals         TEAM_LEADER approval count questions.
    #   3b._handle_team_leader_team_availability  TEAM_LEADER team availability / absences.
    #      MUST be before _handle_team_leader_team_leave so availability questions
    #      return Team Leave Calendar first without being caught by the leave handler.
    #   3c._handle_team_leader_team_leave         TEAM_LEADER team leave request questions.
    #      MUST be before _handle_team_leader_team so specific team-leave questions
    #      get the precise Team Requests vs Team Leave Calendar answer.
    #   4. _handle_team_leader_team              general team-management guidance.
    #   5. _handle_pending_requests              EMPLOYEE / TEAM_LEADER pending count.
    #   6. _handle_annual_leave_balance          annual leave balance (context-aware).
    #   7. _handle_sick_leave_balance            sick leave balance (context-aware).
    #   8. _handle_document_notification         document-ready / notification / email Q.
    #   9. _handle_working_time                  working hours / weekends / public holidays.
    #  10. _handle_request_status               request status / tracking.
    #  11. _handle_platform_overview            "what can I do" overview.
    #  12. _handle_loan                          loan navigation.
    #  13. _handle_leave_request                 leave submission guidance.
    #      NOTE: _handle_working_time MUST be before _handle_leave_request so that
    #      "weekends count in leave" is caught by working-time, not leave-request.
    #  14. _handle_hr_user_setup                 HR new-user workflow.
    #  15. _handle_profile                       profile navigation.
    handlers = [
        _handle_hr_manager_personal_redirect,   # guard: MUST be first
        _handle_hr_dashboard_counts,            # HR count questions
        _handle_team_leader_approvals,          # TL pending approvals count
        _handle_team_leader_team_availability,  # TL team availability / absences (MUST precede team-leave handler)
        _handle_team_leader_team_leave,         # TL team leave questions (MUST precede general team handler)
        _handle_team_leader_team,               # TL team-management (general)
        _handle_pending_requests,               # employee pending request count
        _handle_annual_leave_balance,           # annual leave balance
        _handle_sick_leave_balance,             # sick leave balance
        _handle_document_notification,          # document readiness / notifications / email
        _handle_working_time,                   # working hours / weekends / public holidays
        _handle_request_status,                 # request status tracking
        _handle_platform_overview,              # "what can I do" platform overview
        _handle_loan,
        _handle_leave_request,                  # MUST stay after _handle_working_time
        _handle_hr_user_setup,
        _handle_profile,
    ]

    for handler in handlers:
        result = handler(q, role, request)
        if result is not None:
            return result

    return None


# ---------------------------------------------------------------------------
# HR_MANAGER personal-flow redirect  (MUST stay first in handlers list)
# ---------------------------------------------------------------------------

def _handle_hr_manager_personal_redirect(
    q: str, role: str, request: ChatRequest
) -> Optional[ChatResponse]:
    """
    Redirect HR_MANAGER away from personal employee-flow questions.
    Only fires when role == HR_MANAGER AND the question looks like a personal
    employee action (leave balance, loan request, leave submission, personal requests).
    Profile questions are intentionally excluded — HR_MANAGER may still ask
    about their own profile.
    """
    if role != "HR_MANAGER":
        return None
    if not _is_personal_employee_question(q):
        return None

    return ChatResponse(
        answer=(
            "Your account is configured as an HR Manager account, which is treated as a "
            "management account by the assistant — not a personal employee account. "
            "Personal actions such as checking your own leave balance, requesting a loan, "
            "or submitting a personal leave request are not available through this account type. \n\n"
            "As an HR Manager you can: review and manage employee leave requests, oversee "
            "loan and administrative requests, administer user accounts, monitor approvals, and access HR reports. "
            "Please use the HR management sections below."
        ),
        relatedPages=[
            RelatedPage(label="HR Dashboard",    route="/hr/dashboard"),
            RelatedPage(label="Leave Approvals",  route="/hr/approvals"),
            RelatedPage(label="All HR Requests",  route="/hr/requests"),
            RelatedPage(label="User Management",  route="/hr/users"),
            RelatedPage(label="HR Reports",       route="/hr/reports"),
        ],
    )


# ---------------------------------------------------------------------------
# HR_MANAGER dashboard / count questions
# ---------------------------------------------------------------------------

def _handle_hr_dashboard_counts(
    q: str, role: str, request: ChatRequest
) -> Optional[ChatResponse]:
    """
    Answer HR_MANAGER questions about platform-wide pending action counts
    and new-user onboarding backlog, reading from context.hr.

    Fires only for HR_MANAGER role.  Never fires for EMPLOYEE or TEAM_LEADER.
    """
    if role != "HR_MANAGER":
        return None

    # New-user pending questions (more specific — checked first)
    _new_user_phrases = ("new user", "users waiting", "users pending", "onboarding",
                         "pending approval", "new users")
    is_new_user_q = any(phrase in q for phrase in _new_user_phrases)

    # General pending action count questions
    _action_phrases = ("how many hr", "hr pending", "pending actions", "total pending",
                       "platform pending", "how many actions", "platform actions")
    is_action_q = any(phrase in q for phrase in _action_phrases)

    if not (is_new_user_q or is_action_q):
        return None

    # --- New-user backlog question ---
    if is_new_user_q:
        new_users = _get_hr_new_users(request)
        if new_users is not None:
            if new_users == 0:
                answer = (
                    "There are currently no new users waiting for HR approval — "
                    "your onboarding queue is clear."
                )
            elif new_users == 1:
                answer = (
                    "There is 1 new user currently waiting for HR approval and role assignment. "
                    "You can review and onboard them from the User Management section."
                )
            else:
                answer = (
                    f"There are {new_users} new users currently waiting for HR approval "
                    f"and role assignment. You can review and onboard them from the "
                    f"User Management section."
                )
        else:
            answer = (
                "I do not have the current new-user onboarding count available right now. "
                "You can check the full list of pending users directly on the platform."
            )
        return ChatResponse(
            answer=answer,
            relatedPages=[
                RelatedPage(label="User Management", route="/hr/users"),
                RelatedPage(label="HR Dashboard",    route="/hr/dashboard"),
            ],
        )

    # --- Total pending actions question ---
    total = _get_hr_total_pending(request)
    if total is not None:
        if total == 0:
            answer = (
                "There are currently no pending actions on the platform — everything is up to date."
            )
        elif total == 1:
            answer = (
                "There is 1 pending action waiting for your attention on the platform. "
                "Check the HR Dashboard and Requests section for details."
            )
        else:
            answer = (
                f"There are {total} pending actions currently waiting for HR attention across "
                f"the platform. Check the HR Dashboard and Requests section for the full breakdown."
            )
    else:
        answer = (
            "I do not have the current pending actions count available right now. "
            "You can see the full breakdown directly on the HR Dashboard."
        )

    return ChatResponse(
        answer=answer,
        relatedPages=[
            RelatedPage(label="HR Dashboard",    route="/hr/dashboard"),
            RelatedPage(label="All HR Requests", route="/hr/requests"),
        ],
    )


# ---------------------------------------------------------------------------
# TEAM_LEADER pending approvals count
# ---------------------------------------------------------------------------

def _handle_team_leader_approvals(
    q: str, role: str, request: ChatRequest
) -> Optional[ChatResponse]:
    """
    Answer TEAM_LEADER questions about how many team leave approvals are waiting,
    reading from context.team.pendingTeamLeaderApprovals.

    Three-way value contract:
      None  -> value unknown (key absent or service failed during context assembly);
               tell the user and send them to the page.  Never invent a count.
      0     -> Spring Boot confirmed no pending approvals; say the queue is clear.
      N > 0 -> Spring Boot confirmed N pending approvals; state the exact number.
    """
    if role != "TEAM_LEADER":
        return None

    _approval_phrases = ("team approval", "approvals waiting", "approvals pending",
                         "how many approvals", "pending approvals", "team pending")
    if not any(phrase in q for phrase in _approval_phrases):
        return None

    pending = _get_tl_pending_approvals(request)
    if pending is not None:
        if pending == 0:
            answer = (
                "Your team has no leave requests currently waiting for your approval — "
                "you are all caught up."
            )
        elif pending == 1:
            answer = (
                "There is 1 leave request from your team currently waiting for your approval. "
                "You can review it in the Team Requests section."
            )
        else:
            answer = (
                f"There are {pending} leave requests from your team currently waiting for "
                f"your approval. You can review them in the Team Requests section."
            )
    else:
        answer = (
            "I do not have your team's current pending approval count available right now. "
            "You can check it directly in the Team Requests section."
        )

    return ChatResponse(
        answer=answer,
        relatedPages=[
            RelatedPage(label="Team Requests",      route="/team/requests"),
            RelatedPage(label="Team Leave Calendar", route="/team/calendar"),
        ],
    )


# ---------------------------------------------------------------------------
# TEAM_LEADER team leave request handler  (v1.6 — inserted before general team handler)
# ---------------------------------------------------------------------------

# Phrases that indicate a TEAM_LEADER question specifically about reviewing
# or finding their team's leave requests / approvals (not personal leave).
_TEAM_LEAVE_REQUEST_PHRASES: tuple[str, ...] = (
    "team leave request",
    "team's leave request",
    "check team leave",
    "review team leave",
    "team leave approval",
    "team leave approvals",
    "where are team leave",
    "where can i check team leave",
    "where do i review team leave",
    "how do i check my team",
    "check my team's leave",
    "team member leave",
    "team members leave",
)

# Phrases that indicate a TEAM_LEADER question about team availability,
# absences, or the Team Leave Calendar specifically — NOT about reviewing
# pending leave requests.
_TEAM_AVAILABILITY_PHRASES: tuple[str, ...] = (
    "team availability",
    "team's availability",
    "team absences",
    "team's absences",
    "team absence",
    "who is absent",
    "who is off",
    "overlapping absences",
    "overlapping leaves",
    "leave overlap",
    "absences overlap",
    "team leave calendar",
    "see team leave",
    "view team leave",
    "team calendar",
)


def _handle_team_leader_team_availability(
    q: str, role: str, request: ChatRequest
) -> Optional[ChatResponse]:
    """
    Answer TEAM_LEADER questions about team availability, absences, and
    the Team Leave Calendar.

    This handler is more specific than _handle_team_leader_team_leave:
    it fires when the user is asking *where to see* team availability or
    absences, rather than asking about pending leave requests to act on.

    Must appear BEFORE _handle_team_leader_team_leave in the dispatcher
    so that availability questions are caught here first.

    Related pages: Team Leave Calendar first (primary), Team Requests second.
    My Leave Requests is NOT included — this is a team context, not personal.
    """
    if role != "TEAM_LEADER":
        return None
    if not any(phrase in q for phrase in _TEAM_AVAILABILITY_PHRASES):
        return None

    return ChatResponse(
        answer=(
            "Use Team Leave Calendar to see your team's approved and pending absences, "
            "check availability, and spot overlapping leave before making a decision.\n"
            "Use Team Requests to review and act on pending leave requests from your team members."
        ),
        relatedPages=[
            RelatedPage(label="Team Leave Calendar", route="/team/calendar"),
            RelatedPage(label="Team Requests",       route="/team/requests"),
        ],
    )


def _handle_team_leader_team_leave(
    q: str, role: str, request: ChatRequest
) -> Optional[ChatResponse]:
    """
    Answer TEAM_LEADER questions specifically about reviewing team leave requests.

    Distinguishes the two dedicated pages:
      - Team Requests       : where pending leave requests are reviewed and acted on.
      - Team Leave Calendar : where overlapping absences and team availability
                              can be checked before making a decision.

    My Leave Requests is intentionally not included as a chip here (team context
    only). The text still mentions personal leave is separate.

    Must appear BEFORE _handle_team_leader_team in the dispatcher.
    """
    if role != "TEAM_LEADER":
        return None
    if not any(phrase in q for phrase in _TEAM_LEAVE_REQUEST_PHRASES):
        return None

    return ChatResponse(
        answer=(
            "Use Team Requests to review and act on pending leave requests from your team members.\n"
            "Use Team Leave Calendar to check team availability and spot overlapping absences "
            "before making your decision.\n"
            "Your own personal leave requests are in My Leave Requests, not here."
        ),
        relatedPages=[
            RelatedPage(label="Team Requests",       route="/team/requests"),
            RelatedPage(label="Team Leave Calendar", route="/team/calendar"),
        ],
    )


# ---------------------------------------------------------------------------
# TEAM_LEADER team-management handler (general)
# ---------------------------------------------------------------------------

def _handle_team_leader_team(
    q: str, role: str, request: ChatRequest
) -> Optional[ChatResponse]:
    """
    Answer TEAM_LEADER questions about managing their team.
    Routes to real /team/* pages confirmed in React App.jsx.
    """
    if role != "TEAM_LEADER":
        return None
    if not _is_team_management_question(q):
        return None

    return ChatResponse(
        answer=(
            "As a Team Leader you can manage your team from the dedicated team sections. "
            "Review pending leave requests from your team members, check the team leave calendar, "
            "manage team members, and track projects and tasks."
        ),
        relatedPages=[
            RelatedPage(label="Team Requests",      route="/team/requests"),
            RelatedPage(label="Team Leave Calendar", route="/team/calendar"),
            RelatedPage(label="Team Members",        route="/team/members"),
            RelatedPage(label="Projects & Tasks",    route="/team/tasks"),
        ],
    )


# ---------------------------------------------------------------------------
# EMPLOYEE / TEAM_LEADER: pending request count
# ---------------------------------------------------------------------------

def _handle_pending_requests(
    q: str, role: str, request: ChatRequest
) -> Optional[ChatResponse]:
    """
    Answer EMPLOYEE / TEAM_LEADER questions about how many personal pending
    requests they have, reading from context.employee.
    """
    if role not in ("EMPLOYEE", "TEAM_LEADER"):
        return None

    _pending_phrases = ("how many pending requests", "do i have pending",
                        "total pending", "open requests", "my pending",
                        "my open requests", "how many requests do i")
    if not any(phrase in q for phrase in _pending_phrases):
        return None

    total = _get_total_pending(request)
    leaves = _get_leaves_pending(request)

    if total is not None:
        if total == 0:
            answer = (
                "You currently have no open pending requests — all your requests have been processed."
            )
        else:
            parts = []
            if leaves and leaves > 0:
                parts.append(f"{leaves} leave request{'s' if leaves != 1 else ''}")
            emp = (request.context.employee if request.context else None)
            if emp:
                if emp.documentsPending:
                    parts.append(f"{emp.documentsPending} document request{'s' if emp.documentsPending != 1 else ''}")
                if emp.loansPending:
                    parts.append(f"{emp.loansPending} loan request{'s' if emp.loansPending != 1 else ''}")
                if emp.authorizationsPending:
                    parts.append(f"{emp.authorizationsPending} authorization request{'s' if emp.authorizationsPending != 1 else ''}")

            if parts:
                breakdown = ", ".join(parts)
                answer = (
                    f"You currently have {total} open pending request{'s' if total != 1 else ''}: "
                    f"{breakdown}. You can view them in the relevant sections below."
                )
            else:
                answer = (
                    f"You currently have {total} open pending request{'s' if total != 1 else ''}. "
                    f"You can view them in the relevant sections below."
                )
    else:
        answer = (
            "I do not have your current pending request count available right now. "
            "You can check all your open requests directly on the platform."
        )

    return ChatResponse(
        answer=answer,
        relatedPages=[
            RelatedPage(label="My Leave Requests",    route="/employee/leave"),
            RelatedPage(label="My Loans",             route="/employee/loans"),
        ],
    )


# ---------------------------------------------------------------------------
# Annual leave balance handler (context-aware)
# ---------------------------------------------------------------------------

def _handle_annual_leave_balance(
    q: str, role: str, request: ChatRequest
) -> Optional[ChatResponse]:
    """
    Personal annual leave balance for EMPLOYEE and TEAM_LEADER.
    Reads from context.employee.annualAvailableDays.
    HR_MANAGER questions are caught earlier by the redirect guard.
    """
    # Exclude questions specifically about sick leave only
    is_sick_only = "sick" in q and "annual" not in q
    if is_sick_only:
        return None  # handled by _handle_sick_leave_balance

    is_annual = (
        "annual" in q
        or ("leave" in q and ("balance" in q or "remaining" in q or "how many days" in q or "how much" in q))
        or ("leave balance" in q)
    )
    if not is_annual:
        return None

    annual_days = _get_annual_days(request)

    if annual_days is not None:
        answer = (
            f"Your current annual leave balance is {annual_days} day{'s' if annual_days != 1 else ''}. "
            "You can view the full breakdown and submit new requests in the Leave section."
        )
    else:
        # Legacy flat context fallback (old test fixtures send {"leave": {"balance": N}})
        leave_ctx = (request.context.leave or {}) if request.context else {}
        legacy_balance = leave_ctx.get("balance")
        if legacy_balance is not None:
            answer = (
                f"Your current annual leave balance is {legacy_balance} day(s). "
                "You can view the full breakdown and submit new requests in the Leave section."
            )
        else:
            answer = (
                "You can check your current leave balance in the Leave section. "
                "It shows your annual, sick, and any special leave days remaining."
            )

    return ChatResponse(
        answer=answer,
        relatedPages=[
            RelatedPage(label="My Leave Requests", route="/employee/leave"),
        ],
    )


# ---------------------------------------------------------------------------
# Sick leave balance handler (context-aware)
# ---------------------------------------------------------------------------

def _handle_sick_leave_balance(
    q: str, role: str, request: ChatRequest
) -> Optional[ChatResponse]:
    """
    Personal sick leave balance for EMPLOYEE and TEAM_LEADER.
    Reads from context.employee.sickAvailableDays.
    Only fires when the question is specifically about sick leave.
    """
    if "sick" not in q:
        return None
    if "leave" not in q and "balance" not in q and "days" not in q:
        return None
    # HR_MANAGER questions already caught by redirect guard
    if role == "HR_MANAGER":
        return None

    sick_days = _get_sick_days(request)

    if sick_days is not None:
        answer = (
            f"Your current sick leave balance is {sick_days} day{'s' if sick_days != 1 else ''}. "
            "You can view the full breakdown in the Leave section."
        )
    else:
        answer = (
            "I do not have your current sick leave balance available right now. "
            "You can check it in the Leave section."
        )

    return ChatResponse(
        answer=answer,
        relatedPages=[
            RelatedPage(label="My Leave Requests", route="/employee/leave"),
        ],
    )


# ---------------------------------------------------------------------------
# Document readiness / notification / email handler  (v1.5 polish)
# ---------------------------------------------------------------------------

# Notification/email intent: user wants to know IF/HOW they will be told
_DOCUMENT_NOTIFY_PHRASES: tuple[str, ...] = (
    "notified when",
    "get notified",
    "will i be notified",
    "will i get notified",
    "receive a notification",
    "receive notification",
    "get a notification",
    "send me a notification",
    "notify me",
    "do i get an email",
    "receive an email",
    "get an email",
    "send me an email",
    "email me",
    "email notification",
    "document ready",
    "document is ready",
    "certificate ready",
    "certificate is ready",
    "document uploaded",
    "hr uploads",
    "hr upload",
    "document prepared",
    "how do i know",
    "how will i know",
)

# Access/download intent: user wants to find or download their document
_DOCUMENT_ACCESS_PHRASES: tuple[str, ...] = (
    "document available",
    "my document",
    "my certificate",
    "access my document",
    "access my certificate",
    "download my document",
    "download my certificate",
    "where is my document",
    "where is my certificate",
    "find my document",
    "where can i find my document",
    "where can i find my certificate",
)


def _handle_document_notification(
    q: str, role: str, request: ChatRequest
) -> Optional[ChatResponse]:
    """
    Two sub-cases:
      A. Notification/email intent — how will the user be informed?
      B. Access/download intent   — where can the user find the document?

    HR_MANAGER is excluded — personal employee-flow question.
    Never claims a specific document is ready.
    """
    if role == "HR_MANAGER":
        return None

    is_notify = any(phrase in q for phrase in _DOCUMENT_NOTIFY_PHRASES)
    is_access = any(phrase in q for phrase in _DOCUMENT_ACCESS_PHRASES)

    if not (is_notify or is_access):
        return None

    if is_notify:
        # Case A: the user wants to know when/how HR will tell them
        return ChatResponse(
            answer=(
                "When HR prepares your document, you'll receive an in-app notification. "
                "An email may also be sent depending on your notification settings. "
                "Check My Documents to download ready documents and Notifications for updates."
            ),
            relatedPages=[
                RelatedPage(label="My Documents",  route="/employee/documents"),
                RelatedPage(label="Notifications", route="/employee/notifications"),
            ],
        )
    else:
        # Case B: the user wants to find or download their document
        return ChatResponse(
            answer=(
                "Your requested documents are in My Documents. "
                "Ready documents can be downloaded directly from there. "
                "You'll get an in-app notification (and possibly an email) when HR finishes preparation."
            ),
            relatedPages=[
                RelatedPage(label="My Documents",  route="/employee/documents"),
                RelatedPage(label="Notifications", route="/employee/notifications"),
            ],
        )


# ---------------------------------------------------------------------------
# Working time / working days / weekends / public holidays handler  (NEW in v1.4)
# ---------------------------------------------------------------------------

# Phrases that indicate a question about working hours, working days, or how
# weekends and public holidays affect leave and loan meeting scheduling.
_WORKING_TIME_PHRASES: tuple[str, ...] = (
    "working time",
    "work time",
    "working hours",
    "work hours",
    "business hours",
    "working days",
    "work days",
    "workdays",
    "weekends count",
    "weekend count",
    "do weekends",
    "are weekends",
    "weekends included",
    "weekends excluded",
    "saturday",
    "sunday",
    "public holiday",
    "public holidays",
    "national holiday",
    "national holidays",
    "tunisian holiday",
    "bank holiday",
    "holidays count",
    "holidays excluded",
    "leave deduction",
    "deducted from leave",
    "count toward leave",
    "count as leave",
    "loan meeting",
    "meeting slot",
    "meeting time",
    "meeting hour",
    "available slot",
)


def _handle_working_time(
    q: str, role: str, request: ChatRequest
) -> Optional[ChatResponse]:
    """
    Explain working hours, working days, leave deduction rules, and loan
    meeting time slots.

    Fires for all roles (factual platform information, no personal-flow risk).
    IMPORTANT: must appear BEFORE _handle_leave_request in the dispatcher.
    """
    if not any(phrase in q for phrase in _WORKING_TIME_PHRASES):
        return None

    return ChatResponse(
        answer=(
            "Working days are Monday to Friday.\n"
            "Weekends and Tunisian public holidays are excluded from leave deductions — only working days are counted.\n"
            "Loan meeting slots: 08:00, 09:00, 10:00, 11:00, 13:00, 14:00, 15:00, 16:00 (12:00 excluded for lunch)."
        ),
        relatedPages=[
            RelatedPage(label="My Leave Requests", route="/employee/leave"),
            RelatedPage(label="My Loans",          route="/employee/loans"),
        ],
    )


# ---------------------------------------------------------------------------
# Request status / tracking handler  (NEW in v1.4)
# ---------------------------------------------------------------------------

# Phrases that indicate a question about checking the current status of a
# submitted request (leave, document, loan, authorization).
_REQUEST_STATUS_PHRASES: tuple[str, ...] = (
    "check my request",
    "check request status",
    "check the status",
    "track my request",
    "track my leave",          # catches "how do i track my leave?" before leave handler
    "track request",
    "request status",
    "status of my request",
    "where is my request",
    "where can i check",
    "where can i see",
    "where can i track",
    "where can i find my request",
    "see my request",
    "view my request",
    "status of my leave",
    "my leave status",
    "my loan status",
    "my document status",
    "authorization status",
    "request approved",
    "request rejected",
    "request pending",
    "is my request",
    "has my request",
    "did my request",
    "what happened to my request",
    "follow up on my request",
    "follow up my request",
    "follow my leave",          # catches "how can i follow my leave request?"
)


def _handle_request_status(
    q: str, role: str, request: ChatRequest
) -> Optional[ChatResponse]:
    """
    Explain where users can check the status of each type of request and
    what the common status values mean in user-friendly language.

    Fires for EMPLOYEE and TEAM_LEADER.
    HR_MANAGER: management-facing answer (they track employee requests,
    not personal ones).

    Rules:
    - Do not expose raw enum names; use readable labels.
    - Guide to the relevant page(s) depending on the question specificity.
    - Do not claim a specific request is approved or rejected without context.
    """
    if not any(phrase in q for phrase in _REQUEST_STATUS_PHRASES):
        return None

    # HR_MANAGER gets a management-appropriate answer
    if role == "HR_MANAGER":
        return ChatResponse(
            answer=(
                "You can track and manage all employee requests from the HR Requests section. "
                "Each request shows its current status: pending, waiting for a decision, "
                "approved, rejected, or cancelled. You can filter by type and take action "
                "directly from the list."
            ),
            relatedPages=[
                RelatedPage(label="All HR Requests", route="/hr/requests"),
                RelatedPage(label="HR Dashboard",    route="/hr/dashboard"),
            ],
        )

    # Determine if the question is narrowed to a specific request type
    is_leave  = "leave" in q
    is_loan   = "loan" in q
    is_doc    = "document" in q or "certificate" in q
    is_auth   = "authorization" in q or "authoris" in q

    # Build a focused answer if the question targets one type
    specific = sum([is_leave, is_loan, is_doc, is_auth])

    if specific == 1:
        if is_leave:
            answer = (
                "Check My Leave Requests for the status of your leave. "
                "Statuses: pending (waiting for Team Leader), waiting for HR, approved, rejected, or cancelled."
            )
            pages = [RelatedPage(label="My Leave Requests", route="/employee/leave")]
        elif is_loan:
            answer = (
                "Check My Loans for the status of your loan request. "
                "Statuses: pending, meeting scheduled, approved, or rejected."
            )
            pages = [RelatedPage(label="My Loans", route="/employee/loans")]
        elif is_doc:
            answer = (
                "Check My Documents for the status of your document request. "
                "Statuses: pending (HR is processing), ready to download, or rejected."
            )
            pages = [
                RelatedPage(label="My Documents",  route="/employee/documents"),
                RelatedPage(label="Notifications", route="/employee/notifications"),
            ]
        else:  # authorization
            answer = (
                "Check Authorizations for the status of your authorization request. "
                "Statuses: pending, approved, or rejected."
            )
            pages = [RelatedPage(label="Authorizations", route="/employee/authorizations")]
    else:
        # General "where can I check my requests" answer
        answer = (
            "Track each request in its own section:\n"
            "\u2022 Leave — My Leave Requests (pending, waiting for HR, approved, rejected, cancelled)\n"
            "\u2022 Documents — My Documents (pending, ready, rejected)\n"
            "\u2022 Loans — My Loans (pending, meeting scheduled, approved, rejected)\n"
            "\u2022 Authorizations — Authorizations (pending, approved, rejected)"
        )
        pages = [
            RelatedPage(label="My Leave Requests", route="/employee/leave"),
            RelatedPage(label="My Documents",      route="/employee/documents"),
            RelatedPage(label="My Loans",          route="/employee/loans"),
            RelatedPage(label="Authorizations",    route="/employee/authorizations"),
        ]

    return ChatResponse(answer=answer, relatedPages=pages)


# ---------------------------------------------------------------------------
# Platform overview / "what can I do" handler  (NEW in v1.4)
# ---------------------------------------------------------------------------

# Phrases that trigger a role-aware overview of what the platform offers.
_PLATFORM_OVERVIEW_PHRASES: tuple[str, ...] = (
    "what can i do",
    "what can i use",
    "what features",
    "what is this platform",
    "what does this platform",
    "what does this website",
    "what can this platform",
    "what can this website",
    "what is this website",
    "what is this app",
    "what can i do here",
    "what can i do on this",
    "how do i use this",
    "how do i use the platform",
    "overview of the platform",
    "platform overview",
    "what is available",
    "what are my options",
)


def _handle_platform_overview(
    q: str, role: str, request: ChatRequest
) -> Optional[ChatResponse]:
    """
    Return a concise, role-aware overview of platform features.

    EMPLOYEE / TEAM_LEADER: personal self-service features.
    HR_MANAGER: management / administrative features.

    Kept short — the old generic paragraph was too long for a chat answer.
    """
    if not any(phrase in q for phrase in _PLATFORM_OVERVIEW_PHRASES):
        return None

    if role == "HR_MANAGER":
        return ChatResponse(
            answer=(
                "As an HR Manager you have access to the following administrative sections:\n"
                "• User Management — create, configure, and activate employee accounts\n"
                "• Leave Approvals — review and process employee leave requests\n"
                "• HR Requests — manage document, loan, and authorization requests\n"
                "• HR Calendar — view team availability and approved leave schedules\n"
                "• Statistics & Reports — platform usage and leave analytics\n"
                "• Teams — manage team composition and assignments"
            ),
            relatedPages=[
                RelatedPage(label="HR Dashboard",    route="/hr/dashboard"),
                RelatedPage(label="User Management", route="/hr/users"),
                RelatedPage(label="All HR Requests", route="/hr/requests"),
                RelatedPage(label="HR Reports",      route="/hr/reports"),
            ],
        )

    if role == "TEAM_LEADER":
        return ChatResponse(
            answer=(
                "As a Team Leader you have two areas:\n"
                "Personal: request and track leave, loans, documents, and authorizations; "
                "view your profile and situation summary.\n"
                "Team: review team leave requests, check team availability in the team calendar, "
                "follow projects, tasks, and team notifications."
            ),
            relatedPages=[
                RelatedPage(label="My Leave Requests",   route="/employee/leave"),
                RelatedPage(label="Team Requests",       route="/team/requests"),
                RelatedPage(label="Team Leave Calendar", route="/team/calendar"),
                RelatedPage(label="Projects & Tasks",    route="/team/tasks"),
                RelatedPage(label="Notifications",       route="/employee/notifications"),
            ],
        )

    # EMPLOYEE (and defensively: NEW_USER)
    return ChatResponse(
        answer=(
            "As an employee you can:\n"
            "\u2022 Request and track leave (annual, sick, and other types)\n"
            "\u2022 Request official documents and certificates\n"
            "\u2022 Apply for a personal loan\n"
            "\u2022 Submit authorizations (time-off, equipment)\n"
            "\u2022 View tasks and notifications\n"
            "\u2022 Update your profile and view your situation summary"
        ),
        relatedPages=[
            RelatedPage(label="My Leave Requests", route="/employee/leave"),
            RelatedPage(label="My Documents",      route="/employee/documents"),
            RelatedPage(label="My Loans",          route="/employee/loans"),
            RelatedPage(label="Notifications",     route="/employee/notifications"),
        ],
    )


# ---------------------------------------------------------------------------
# Loan handler
# ---------------------------------------------------------------------------

def _handle_loan(q: str, role: str, request: ChatRequest) -> Optional[ChatResponse]:
    """
    EMPLOYEE / TEAM_LEADER: personal loan guidance -> /employee/loans
    HR_MANAGER: loan management guidance           -> /hr/requests
    """
    if "loan" not in q:
        return None

    if role in ("EMPLOYEE", "TEAM_LEADER"):
        return ChatResponse(
            answer=(
                "To request a loan, go to the Loans section under your personal requests. "
                "Fill in the loan request form and submit — HR will review and process it."
            ),
            relatedPages=[
                RelatedPage(label="My Loans", route="/employee/loans"),
            ],
        )

    if role == "HR_MANAGER":
        return ChatResponse(
            answer=(
                "You can review and action employee loan requests in the HR Requests section. "
                "Each request shows the employee details, requested amount, and supporting documents."
            ),
            relatedPages=[
                RelatedPage(label="All HR Requests", route="/hr/requests"),
            ],
        )

    return None


# ---------------------------------------------------------------------------
# Leave request submission handler
# ---------------------------------------------------------------------------

def _handle_leave_request(q: str, role: str, request: ChatRequest) -> Optional[ChatResponse]:
    """
    Personal leave request submission for EMPLOYEE and TEAM_LEADER -> /employee/leave
    HR_MANAGER personal questions are caught earlier by the redirect guard.

    NOTE: this handler must stay AFTER _handle_working_time so that questions
    like "do weekends count in leave?" are caught by working-time first.
    """
    if "leave" not in q:
        return None
    if any(kw in q for kw in ["balance", "remaining", "how many days", "how much", "annual", "sick"]):
        return None  # handled by _handle_annual_leave_balance or _handle_sick_leave_balance

    return ChatResponse(
        answer=(
            "To submit a leave request, go to the Leave section and click 'New Request'. "
            "Select the leave type, choose your dates, add any notes, and submit for approval."
        ),
        relatedPages=[
            RelatedPage(label="My Leave Requests", route="/employee/leave"),
        ],
    )


# ---------------------------------------------------------------------------
# HR user-setup handler
# ---------------------------------------------------------------------------

def _handle_hr_user_setup(q: str, role: str, request: ChatRequest) -> Optional[ChatResponse]:
    """
    HR_MANAGER: create/configure new user accounts -> /hr/users
    """
    if role != "HR_MANAGER":
        return None
    if not ("setup" in q or "new user" in q or "create user" in q or "add user" in q):
        return None

    return ChatResponse(
        answer=(
            "You can create and configure new user accounts in the User Management section. "
            "Fill in the employee details, assign the appropriate role, and the user will "
            "receive an activation email to complete their account setup."
        ),
        relatedPages=[
            RelatedPage(label="User Management", route="/hr/users"),
        ],
    )


# ---------------------------------------------------------------------------
# Profile handler
# ---------------------------------------------------------------------------

def _handle_profile(q: str, role: str, request: ChatRequest) -> Optional[ChatResponse]:
    """
    Profile guidance.
    EMPLOYEE and TEAM_LEADER -> /employee/profile  (confirmed in React App.jsx and Sidebar.jsx)
    HR_MANAGER               -> no dedicated profile route exists; answer without a link.
    """
    if "profile" not in q and "my information" not in q and "my details" not in q:
        return None

    if role == "HR_MANAGER":
        return ChatResponse(
            answer=(
                "You can view and update your personal account information through your profile settings. "
                "Contact a system administrator if you need to change details that require verification."
            ),
            relatedPages=[],
        )

    # EMPLOYEE and TEAM_LEADER both use /employee/profile
    return ChatResponse(
        answer=(
            "You can view and update your personal information in your Profile section. "
            "Contact HR if you need to change details that require verification."
        ),
        relatedPages=[
            RelatedPage(label="My Profile", route="/employee/profile"),
        ],
    )
