"""
gemini_client.py
----------------
Calls the Google Gemini generateContent REST API for platform Q&A.

Behaviour:
  - If GEMINI_ENABLED=false (the default), returns None immediately.
    No network call is ever made in local/test mode.
  - Sends a single-turn request: system instruction + one user message.
  - Expects Gemini to reply with JSON {answer, reasons, relatedPages}.
  - On any failure (timeout, HTTP error, JSON parse error): logs and returns None.
  - Never propagates errors to the caller; caller falls back gracefully.

The caller (assistant_service.py) MUST run sanitize_response on the result
before returning it to the API consumer.

API endpoint used:
  POST https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent
  Headers:
    x-goog-api-key: <GEMINI_API_KEY>   <- API key goes in header, NOT query param
    Content-Type: application/json

Gemini REST docs:
  https://ai.google.dev/api/rest/v1beta/models/generateContent
"""

import json
import logging
from typing import Optional

import httpx

from app.config import settings
from app.schemas import ChatRequest, ChatResponse, RelatedPage
from app.services.gemini_prompt_builder import build_system_prompt, build_user_message

logger = logging.getLogger(__name__)

_GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta/models"


def call_gemini(request: ChatRequest) -> Optional[ChatResponse]:
    """
    Call Gemini and return a ChatResponse, or None if disabled or on any error.
    """
    if not settings.gemini_enabled:
        logger.debug("Gemini is disabled. Skipping call.")
        return None

    if not settings.gemini_api_key:
        logger.warning("GEMINI_ENABLED=true but GEMINI_API_KEY is empty. Skipping call.")
        return None

    system_prompt = build_system_prompt(request.role)
    user_message = build_user_message(request)

    # Correct URL: .../v1beta/models/{model}:generateContent
    # API key is sent as a request header, NOT as a ?key= query parameter.
    url = f"{_GEMINI_BASE}/{settings.gemini_model}:generateContent"

    headers = {
        "x-goog-api-key": settings.gemini_api_key,
        "Content-Type": "application/json",
    }

    payload = {
        "system_instruction": {
            "parts": [{"text": system_prompt}]
        },
        "contents": [
            {
                "role": "user",
                "parts": [{"text": user_message}],
            }
        ],
        "generationConfig": {
            "temperature": 0.2,
            "maxOutputTokens": 512,
        },
    }

    try:
        with httpx.Client(timeout=settings.gemini_timeout_seconds) as client:
            resp = client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()

        raw_text = (
            data.get("candidates", [{}])[0]
            .get("content", {})
            .get("parts", [{}])[0]
            .get("text", "")
            .strip()
        )

        # Strip Markdown code fences if Gemini wrapped the JSON anyway
        if raw_text.startswith("```"):
            raw_text = raw_text.split("```")[1]
            if raw_text.startswith("json"):
                raw_text = raw_text[4:]
            raw_text = raw_text.strip()

        parsed = json.loads(raw_text)

        related_pages = [
            RelatedPage(label=p["label"], route=p["route"])
            for p in parsed.get("relatedPages", [])
            if isinstance(p, dict) and "label" in p and "route" in p
        ]

        return ChatResponse(
            answer=parsed.get("answer", ""),
            reasons=parsed.get("reasons", []),
            warnings=[],
            relatedPages=related_pages,
            aiGenerated=True,
            source="external_ai",
        )

    except httpx.TimeoutException:
        logger.warning(
            "Gemini call timed out after %ss.", settings.gemini_timeout_seconds
        )
        return None

    except httpx.HTTPStatusError as exc:
        logger.warning("Gemini returned HTTP %s.", exc.response.status_code)
        return None

    except (json.JSONDecodeError, KeyError, IndexError) as exc:
        logger.warning("Gemini response could not be parsed: %s", exc)
        return None

    except Exception as exc:  # noqa: BLE001
        logger.warning("Gemini call failed: %s", exc)
        return None
