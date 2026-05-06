"""
platform_help_service.py
------------------------
Role-aware, rule-based platform guidance for ArabSoft users (v1.3 / local mode).

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
    "team calendar",
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
    #   4. _handle_team_leader_team              general team-management guidance.
    #   5. _handle_pending_requests              EMPLOYEE / TEAM_LEADER pending count.
    #   6. _handle_annual_leave_balance          annual leave balance (context-aware).
    #   7. _handle_sick_leave_balance            sick leave balance (context-aware).
    #   8. _handle_loan                          loan navigation.
    #   9. _handle_leave_request                 leave submission guidance.
    #  10. _handle_hr_user_setup                 HR new-user workflow.
    #  11. _handle_profile                       profile navigation.
    handlers = [
        _handle_hr_manager_personal_redirect,   # guard: MUST be first
        _handle_hr_dashboard_counts,            # HR count questions
        _handle_team_leader_approvals,          # TL pending approvals count
        _handle_team_leader_team,               # TL team-management (general)
        _handle_pending_requests,               # employee pending request count
        _handle_annual_leave_balance,           # annual leave balance
        _handle_sick_leave_balance,             # sick leave balance
        _handle_loan,
        _handle_leave_request,
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
    # Match: "leave balance", "annual leave", "how much leave", "annual balance"
    # but NOT questions specifically about sick leave only
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
