"""
config.py
---------
Application settings loaded from environment variables or a .env file.

All external-agent settings default to safe values:
  EXTERNAL_AGENT_ENABLED=false  ->  no network calls are made.

Usage:
  from app.config import settings
  if settings.external_agent_enabled: ...
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # External agent — disabled by default so local tests never touch a real provider
    external_agent_enabled: bool = Field(default=False)
    external_agent_base_url: str = Field(default="http://localhost:9999")
    external_agent_api_key: str = Field(default="")
    external_agent_model: str = Field(default="")
    external_agent_timeout_seconds: int = Field(default=8)

    # Gemini — disabled by default; set GEMINI_ENABLED=true + GEMINI_API_KEY to activate
    gemini_enabled: bool = Field(default=False)
    gemini_api_key: str = Field(default="")
    gemini_model: str = Field(default="gemini-2.5-flash")
    gemini_timeout_seconds: int = Field(default=10)


settings = Settings()
