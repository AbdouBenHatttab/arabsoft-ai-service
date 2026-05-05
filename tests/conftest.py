"""
tests/conftest.py
-----------------
Session-scoped test isolation for provider settings.

Problem this solves
-------------------
Both gemini_client and external_agent_client read the module-level `settings`
singleton which is loaded once at import time from the real .env file.
If a developer has GEMINI_ENABLED=true in their .env for manual testing,
every test that does NOT explicitly patch settings will see Gemini as enabled
and may make real or unexpected mock calls.

Solution
--------
An autouse fixture patches BOTH providers to disabled+safe defaults before
every single test.  Individual tests that need a provider enabled must patch
it themselves (which they already do via _mock_gemini_settings /
_mock_external_agent_settings helpers).  Those per-test patches take
precedence because they nest inside this outer patch.

Isolation contract
------------------
- No test reads the real .env.
- No real API key is ever used inside a test.
- Provider-disabled tests pass whether .env has ENABLED=true or false.
- Provider-enabled tests always use fake keys ("test-key").
- Pipeline order is unchanged: refusal -> local_rules -> gemini ->
  external_agent -> fallback.
"""

import pytest
from unittest.mock import patch, MagicMock


# ---------------------------------------------------------------------------
# Safe baseline settings objects
# All fields required by either client are present.
# ---------------------------------------------------------------------------

class _SafeGeminiSettings:
    gemini_enabled = False
    gemini_api_key = ""                  # empty — never sent to network
    gemini_model = "gemini-2.5-flash"
    gemini_timeout_seconds = 10
    # external_agent fields present so patching gemini settings object
    # does not break any code that reads settings.external_agent_* inside
    # the same request path (assistant_service imports both clients).
    external_agent_enabled = False
    external_agent_base_url = "http://localhost:9999"
    external_agent_api_key = ""
    external_agent_model = ""
    external_agent_timeout_seconds = 8


class _SafeExternalAgentSettings:
    external_agent_enabled = False
    external_agent_base_url = "http://localhost:9999"
    external_agent_api_key = ""          # empty — never sent to network
    external_agent_model = ""
    external_agent_timeout_seconds = 8
    # gemini fields present for symmetry
    gemini_enabled = False
    gemini_api_key = ""
    gemini_model = "gemini-2.5-flash"
    gemini_timeout_seconds = 10


# ---------------------------------------------------------------------------
# Autouse fixture — runs for every test in the suite
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _isolate_provider_settings():
    """
    Force both external providers OFF for every test.

    Tests that need a provider enabled patch settings themselves inside
    their own `with patch(...) as ms:` block, which nests inside this
    outer patch and overrides it for the duration of that test only.
    """
    with patch(
        "app.clients.gemini_client.settings",
        new=_SafeGeminiSettings(),
    ), patch(
        "app.clients.external_agent_client.settings",
        new=_SafeExternalAgentSettings(),
    ), patch(
        "app.services.drafting_service.settings",
        new=_SafeGeminiSettings(),
    ):
        yield
