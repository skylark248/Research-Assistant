"""Provider availability for the UI toggle.

Cloud providers are "available" when their API key is set (no network call —
a wrong key still fails loudly at chat time, same policy as everywhere else).
Local means Ollama answers /v1/models within 1.5s.
"""

import requests
from pydantic import BaseModel

from config import settings


class ProviderStatus(BaseModel):
    provider: str
    available: bool
    detail: str = ""
    model: str
    is_default: bool = False


def _probe_local() -> tuple[bool, str]:
    url = settings.local_base_url.rstrip("/") + "/models"
    try:
        resp = requests.get(url, timeout=1.5)
        resp.raise_for_status()
        return True, ""
    except requests.RequestException:
        return False, f"Ollama unreachable at {settings.local_base_url}"


def check_provider(name: str) -> ProviderStatus:
    if name == "anthropic":
        available = bool(settings.anthropic_api_key)
        return ProviderStatus(
            provider=name, available=available,
            detail="" if available else "no API key set",
            model=settings.anthropic_model, is_default=settings.llm_provider == name,
        )
    if name == "openai":
        available = bool(settings.openai_api_key)
        return ProviderStatus(
            provider=name, available=available,
            detail="" if available else "no API key set",
            model=settings.openai_model, is_default=settings.llm_provider == name,
        )
    if name == "local":
        available, detail = _probe_local()
        return ProviderStatus(
            provider=name, available=available, detail=detail,
            model=settings.local_model, is_default=settings.llm_provider == name,
        )
    raise ValueError(f"Unknown provider: {name}")


def check_providers() -> list[ProviderStatus]:
    return [check_provider(name) for name in ("anthropic", "openai", "local")]
