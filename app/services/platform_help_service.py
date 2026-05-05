"""
platform_help_service.py
------------------------
Role-aware, rule-based platform guidance for ArabSoft users (v1.2 / local mode).

Each handler receives the lowercased question and the original ChatRequest,
and returns a ChatResponse when it matches, or None to fall through.

Valid roles (Spring Boot TypeRole enum):
  EMPLOYEE    — personal employee self-service  → /employee/*
  TEAM_LEADER — team management + personal flow → /team/* and /employee/*
  HR_MANAGER  — HR administration only          → /hr/*

Role normalization
------------------
Spring Boot sends the role as the enum name (e.g. "HR_MANAGER").
As a defensive measure the ROLE_ Spring Security prefix is also stripped,
so "ROLE_HR_MANAGER" is treated identically to "HR_MANAGER".

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
    # personal requests
    "my personal requests",
    "show my requests",
    "my requests",
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
])


def _is_team_management_question(q: str) -> bool:
    """Return True if the question is about managing the team (TEAM_LEADER scope)."""
    return any(phrase in q for phrase in _TEAM_MANAGEMENT_PHRASES)


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
    # This means "ROLE_HR_MANAGER" and "HR_MANAGER" are treated identically.
    role = request.role.upper()
    if role.startswith("ROLE_"):
        role = role[len("ROLE_"):]

    # IMPORTANT: handler order is load-bearing.
    #   1. _handle_hr_manager_personal_redirect  MUST stay first —
    #      it blocks HR_MANAGER from reaching employee handlers.
    #   2. _handle_team_leader_team              comes before generic leave/loan
    #      handlers so team-management questions are caught early for TEAM_LEADER.
    handlers = [
        _handle_hr_manager_personal_redirect,   # guard: must be first
        _handle_team_leader_team,               # TEAM_LEADER team-management
        _handle_loan,
        _handle_leave_balance,
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
# TEAM_LEADER team-management handler
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
# Loan handler
# ---------------------------------------------------------------------------

def _handle_loan(q: str, role: str, request: ChatRequest) -> Optional[ChatResponse]:
    """
    EMPLOYEE / TEAM_LEADER: personal loan guidance → /employee/loans
    HR_MANAGER: loan management guidance           → /hr/requests
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
# Leave balance handler
# ---------------------------------------------------------------------------

def _handle_leave_balance(q: str, role: str, request: ChatRequest) -> Optional[ChatResponse]:
    """
    Personal leave balance for EMPLOYEE and TEAM_LEADER → /employee/leave
    """
    if "leave balance" not in q and not ("leave" in q and "balance" in q):
        return None

    leave_ctx = (request.context.leave or {}) if request.context else {}
    balance = leave_ctx.get("balance")

    if balance is not None:
        answer = (
            f"Your current annual leave balance is {balance} day(s). "
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
# Leave request submission handler
# ---------------------------------------------------------------------------

def _handle_leave_request(q: str, role: str, request: ChatRequest) -> Optional[ChatResponse]:
    """
    Personal leave request submission for EMPLOYEE and TEAM_LEADER → /employee/leave
    HR_MANAGER personal questions are caught earlier by the redirect guard.
    """
    if "leave" not in q:
        return None
    if any(kw in q for kw in ["balance", "remaining", "how many days"]):
        return None  # handled by _handle_leave_balance

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
    HR_MANAGER: create/configure new user accounts → /hr/users
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
    EMPLOYEE and TEAM_LEADER → /employee/profile  (confirmed in React App.jsx and Sidebar.jsx)
    HR_MANAGER               → no dedicated profile route exists; answer without a link.
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
