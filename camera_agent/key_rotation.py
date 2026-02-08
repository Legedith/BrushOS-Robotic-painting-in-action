from __future__ import annotations

import os
import re
from typing import Callable, Iterable, TypeVar

T = TypeVar("T")


def _split_keys(raw: str) -> list[str]:
    """Split a raw key list into individual keys."""
    parts = re.split(r"[\s,;]+", raw.strip())
    return [part for part in parts if part]


def get_api_keys() -> list[str]:
    """Get API keys from environment."""
    raw_keys = os.environ.get("GOOGLE_API_KEYS", "").strip()
    if raw_keys:
        keys = _split_keys(raw_keys)
        if keys:
            return keys
    single = os.environ.get("GOOGLE_API_KEY", "").strip()
    if single:
        return [single]
    raise RuntimeError("GOOGLE_API_KEYS or GOOGLE_API_KEY is required.")


def is_quota_error(exc: Exception) -> bool:
    """Detect quota exhaustion errors."""
    message = str(exc)
    if "RESOURCE_EXHAUSTED" in message or "Quota exceeded" in message:
        return True
    status = getattr(exc, "status_code", None)
    if status == 429:
        return True
    return False


def with_key_rotation(call: Callable[[str], T], keys: Iterable[str] | None = None) -> T:
    """Execute a call with key rotation on quota errors."""
    key_list = list(keys) if keys is not None else get_api_keys()
    last_error: Exception | None = None
    for key in key_list:
        try:
            return call(key)
        except Exception as exc:  # noqa: BLE001
            if not is_quota_error(exc):
                raise
            last_error = exc
    raise RuntimeError("All API keys are exhausted.") from last_error
