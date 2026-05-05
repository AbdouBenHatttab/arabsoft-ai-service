"""
trusted_routes.py
-----------------
Single source of truth for every valid React route in the ArabSoft platform.

Rules:
  - Routes are derived from the real React Router table (App.jsx).
  - No invented or placeholder routes are allowed here.
  - The sanitizer and the external-agent system prompt both import from this file.

Exports:
  TRUSTED_ROUTES          frozenset of every valid route across all roles.
  ROLE_ROUTES             dict mapping normalised role name -> frozenset of allowed routes.
  normalize_role()        strips the Spring Security "ROLE_" prefix and upper-cases.
  filter_related_pages_for_role()  removes pages whose route is not allowed for the given role.
"""

from typing import List
from app.schemas import RelatedPage


# ---------------------------------------------------------------------------
# Per-role allowed routes (verified against React App.jsx)
# ---------------------------------------------------------------------------

_EMPLOYEE_ROUTES: frozenset[str] = frozenset([
    "/employee/dashboard",
    "/employee/leave",
    "/employee/calendar",
    "/employee/tasks",
    "/employee/loans",
    "/employee/documents",
    "/employee/authorizations",
    "/employee/notifications",
    "/employee/profile",
    "/employee/situation",
])

_TEAM_LEADER_ROUTES: frozenset[str] = frozenset([
    # Personal employee flows
    "/employee/leave",
    "/employee/loans",
    "/employee/profile",
    # Team management flows
    "/team/dashboard",
    "/team/requests",
    "/team/calendar",
    "/team/tasks",
    "/team/members",
    "/team/notifications",
])

_HR_MANAGER_ROUTES: frozenset[str] = frozenset([
    "/hr/dashboard",
    "/hr/users",
    "/hr/approvals",
    "/hr/statistics",
    "/hr/teams",
    "/hr/requests",
    "/hr/calendar",
    "/hr/calendar/view",
    "/hr/reports",
    "/hr/settings",
])


# ---------------------------------------------------------------------------
# Public exports
# ---------------------------------------------------------------------------

ROLE_ROUTES: dict[str, frozenset[str]] = {
    "EMPLOYEE":    _EMPLOYEE_ROUTES,
    "TEAM_LEADER": _TEAM_LEADER_ROUTES,
    "HR_MANAGER":  _HR_MANAGER_ROUTES,
}

TRUSTED_ROUTES: frozenset[str] = (
    _EMPLOYEE_ROUTES | _TEAM_LEADER_ROUTES | _HR_MANAGER_ROUTES
)


def normalize_role(role: str) -> str:
    """
    Upper-case the role string and strip the Spring Security 'ROLE_' prefix.
    'ROLE_HR_MANAGER' and 'HR_MANAGER' both return 'HR_MANAGER'.
    """
    r = role.upper()
    if r.startswith("ROLE_"):
        r = r[len("ROLE_"):]
    return r


def filter_related_pages_for_role(pages: List[RelatedPage], role: str) -> tuple[List[RelatedPage], List[str]]:
    """
    Return (kept_pages, removed_routes).

    Removes any RelatedPage whose route:
      - is not in TRUSTED_ROUTES at all (invented route), OR
      - is not in the allowed set for the given role (wrong-role route).

    Never invents replacement pages.
    """
    normalised = normalize_role(role)
    allowed = ROLE_ROUTES.get(normalised, frozenset())

    kept: List[RelatedPage] = []
    removed: List[str] = []

    for page in pages:
        if page.route in TRUSTED_ROUTES and page.route in allowed:
            kept.append(page)
        else:
            removed.append(page.route)

    return kept, removed
