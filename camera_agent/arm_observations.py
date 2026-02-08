from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_ARM_OBSERVATIONS_PATH = (
    Path("captures") / "agent_state" / "arm_observations.jsonl"
)


def _ensure_parent(path: Path) -> None:
    """Ensure the parent directory exists."""
    path.parent.mkdir(parents=True, exist_ok=True)


def _ensure_json_value(value: Any, path: str) -> None:
    """Validate a JSON-serializable value."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _ensure_json_value(item, f"{path}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"Observation key at {path} must be a string.")
            _ensure_json_value(item, f"{path}.{key}" if path else key)
        return
    raise ValueError(f"Observation value at {path} is not JSON-serializable.")


def _ensure_json_object(payload: dict[str, Any]) -> None:
    """Validate that a dict is JSON-serializable."""
    for key, value in payload.items():
        if not isinstance(key, str):
            raise ValueError("Observation keys must be strings.")
        _ensure_json_value(value, key)


def _resolve_path(output_path: str | None) -> Path:
    """Resolve the observation log path."""
    return Path(output_path) if output_path else DEFAULT_ARM_OBSERVATIONS_PATH


def append_observation(
    payload: dict[str, Any],
    *,
    output_path: str | None = None,
) -> Path:
    """Append a single observation record to the JSONL file."""
    _ensure_json_object(payload)
    if "timestamp" not in payload:
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **payload,
        }
    path = _resolve_path(output_path)
    _ensure_parent(path)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True) + "\n")
    return path


def read_observations(
    *,
    limit: int = 50,
    output_path: str | None = None,
) -> list[dict[str, Any]]:
    """Read the most recent observation records."""
    if limit <= 0:
        raise ValueError("limit must be > 0.")
    path = _resolve_path(output_path)
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    records: list[dict[str, Any]] = []
    for line in lines[-limit:]:
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError("Observation records must be JSON objects.")
        records.append(payload)
    return records
