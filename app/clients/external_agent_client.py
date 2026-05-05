"""
external_agent_client.py
------------------------
Safe skeleton for calling an external agentic AI provider.

V2 foundation behaviour:
  - If EXTERNAL_AGENT_ENABLED=false (the default), returns None immediately.
    No network call is ever made in local/test mode.
  - If enabled, sends a POST request to the configured provider endpoint.
  - On timeout, connection error, or any unexpected exception: logs the error
    and returns None so the caller falls back gracefully.
  - Never propagates provider errors to the API caller.

The caller (assistant_service.py) is responsible for:
  - Checking the return value (None -> fall through to fallback)
  - Running the sanitizer on any returned ChatResponse

Usage:
  from app.clients.external_agent_client import call_external_agent
  result = call_external_agent(request)   # returns ChatResponse | None
"""

import logging
from typing import Optional

import httpx

from app.config import settings
from app.schemas import ChatRequest, ChatResponse

logger = logging.getLogger(__name__)


def call_external_agent(request: ChatRequest) -> Optional[ChatResponse]:
    """
    Call the configured external AI provider and return a ChatResponse,
    or None if the agent is disabled or the call fails.
    """
    if not settings.external_agent_enabled:
        logger.debug("External agent is disabled. Skipping provider call.")
        return None

    payload = {
        "role": request.role,
        "question": request.question,
        "pageContext": request.pageContext,
        "context": request.context.model_dump() if request.context else {},
    }

    try:
        with httpx.Client(timeout=settings.external_agent_timeout_seconds) as client:
            resp = client.post(
                f"{settings.external_agent_base_url}/agent/chat",
                json=payload,
                headers={
                    "Authorization": f"Bearer {settings.external_agent_api_key}",
                    "Content-Type": "application/json",
                },
            )
            resp.raise_for_status()
            data = resp.json()

        return ChatResponse(
            answer=data.get("answer", ""),
            reasons=data.get("reasons", []),
            warnings=data.get("warnings", []),
            relatedPages=data.get("relatedPages", []),
            disclaimer=data.get(
                "disclaimer",
                "The assistant provides guidance only. Final decisions remain with authorized users.",
            ),
            aiGenerated=True,
            source="external_ai",
        )

    except httpx.TimeoutException:
        logger.warning("External agent call timed out after %ss.", settings.external_agent_timeout_seconds)
        return None

    except httpx.HTTPStatusError as exc:
        logger.warning("External agent returned HTTP %s.", exc.response.status_code)
        return None

    except Exception as exc:  # noqa: BLE001
        logger.warning("External agent call failed: %s", exc)
        return None
