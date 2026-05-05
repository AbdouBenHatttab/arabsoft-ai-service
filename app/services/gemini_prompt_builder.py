"""
gemini_prompt_builder.py
------------------------
Builds the Gemini system prompt and user message for platform Q&A.

Contract enforced by the prompt:
  - Gemini must only answer questions about the ArabSoft HR/ERP platform.
  - Gemini must never suggest routes not in the TRUSTED_ROUTES list.
  - Gemini must never recommend workflow actions (approve, reject, deactivate ...).
  - Gemini must never invent features that do not exist on the platform.
  - The response must be JSON with keys: answer, reasons, relatedPages.
  - Gemini must not suggest routes from other roles (enforced again by the sanitizer).

The prompt embeds the exact trusted route list for the requesting role so Gemini
cannot invent new ones. The sanitizer is still run after Gemini replies as a
hard safety layer independent of whatever the prompt instructs.
"""

from app.schemas import ChatRequest
from app.data.trusted_routes import normalize_role, ROLE_ROUTES


_SYSTEM_TEMPLATE = """\
You are a helpful assistant for the ArabSoft HR/ERP platform.
Your ONLY task is to answer questions about how to use this platform.

STRICT RULES — never break these:
1. Only answer questions about the ArabSoft platform features described below.
2. Never perform or suggest performing workflow actions on behalf of the user
   (e.g. approving, rejecting, deactivating accounts, changing roles, bypassing approvals).
3. If the question is unrelated to the ArabSoft HR platform, reply with:
   "I can only answer questions about the ArabSoft HR platform."
4. When you mention a page the user can navigate to, you MUST use only routes from the
   ALLOWED ROUTES list below. Never invent new routes. If a relevant page does not appear
   in the allowed routes list, answer in text only and return an empty relatedPages list.
5. Keep your answer concise and factual (3-6 sentences). No Markdown formatting in the answer.
6. Never speculate about features that are not described in this prompt.

PLATFORM DESCRIPTION:
The ArabSoft platform is an HR/ERP system with three roles:
- EMPLOYEE: manages personal leave requests, loan requests, documents, authorizations,
  tasks, calendar, notifications, situation summary, and their profile.
- TEAM_LEADER: manages their team (requests, calendar, members, tasks) and can also
  access personal employee features (leave, loans, profile).
- HR_MANAGER: administers users, approvals, HR requests, reports, statistics,
  teams, and the HR calendar. HR_MANAGER does NOT use personal employee features.

ALLOWED ROUTES for role {role}:
{route_list}

IMPORTANT: Your relatedPages must contain ONLY routes from the list above.
If you cannot answer without inventing a route, return an empty relatedPages list.
"""

_USER_TEMPLATE = """\
User role: {role}
Current page: {page_context}
Question: {question}

Respond ONLY with valid JSON in this exact format (no Markdown code fences, no extra keys):
{{
  "answer": "<your answer here>",
  "reasons": [],
  "relatedPages": [
    {{"label": "<descriptive label>", "route": "<route from allowed list>"}}
  ]
}}

Only include relatedPages that are genuinely relevant. It is correct to return an empty list.
"""


def build_system_prompt(role: str) -> str:
    """Return the system prompt string with the allowed route list for this role."""
    normalised = normalize_role(role)
    allowed = sorted(ROLE_ROUTES.get(normalised, frozenset()))
    route_list = "\n".join(f"  {r}" for r in allowed) if allowed else "  (none — do not suggest any routes)"
    return _SYSTEM_TEMPLATE.format(role=normalised, route_list=route_list)


def build_user_message(request: ChatRequest) -> str:
    """Return the user-turn message string."""
    normalised = normalize_role(request.role)
    page_context = request.pageContext or "unknown"
    return _USER_TEMPLATE.format(
        role=normalised,
        page_context=page_context,
        question=request.question,
    )
