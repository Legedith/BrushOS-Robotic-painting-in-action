from __future__ import annotations

import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Iterable, TYPE_CHECKING

from google.adk.memory.base_memory_service import BaseMemoryService, SearchMemoryResponse
from google.adk.memory.memory_entry import MemoryEntry
from google.adk.sessions.sqlite_session_service import SqliteSessionService
from google.genai import types

DEFAULT_SESSION_DB_PATH = Path("captures") / "agent_state" / "sessions.sqlite"
DEFAULT_MEMORY_PATH = Path("captures") / "agent_state" / "memory.jsonl"

_WORD_RE = re.compile(r"[A-Za-z]+")


def _ensure_parent(path: Path) -> None:
    """Ensure the parent directory exists."""
    path.parent.mkdir(parents=True, exist_ok=True)


def resolve_session_db_path() -> Path:
    """Resolve the session database path."""
    value = os.environ.get("CAMERA_AGENT_SESSION_DB")
    return Path(value) if value else DEFAULT_SESSION_DB_PATH


def resolve_memory_path() -> Path:
    """Resolve the memory JSONL path."""
    value = os.environ.get("CAMERA_AGENT_MEMORY_PATH")
    return Path(value) if value else DEFAULT_MEMORY_PATH


def create_session_service() -> SqliteSessionService:
    """Create a SQLite-backed session service."""
    db_path = resolve_session_db_path()
    _ensure_parent(db_path)
    return SqliteSessionService(str(db_path))


def create_memory_service() -> "JsonlMemoryService":
    """Create a JSONL-backed memory service."""
    memory_path = resolve_memory_path()
    return JsonlMemoryService(memory_path)


def _extract_words_lower(text: str) -> set[str]:
    """Extract lowercase words for keyword search."""
    return {match.group(0).lower() for match in _WORD_RE.finditer(text)}


def _content_text(parts: Iterable[types.Part]) -> str:
    """Extract concatenated text from content parts."""
    texts = [part.text for part in parts if part.text]
    return "\n".join(texts).strip()


class JsonlMemoryService(BaseMemoryService):
    """A JSONL-backed memory service for persistent local memory."""

    def __init__(self, path: Path):
        self._path = Path(path)
        _ensure_parent(self._path)

    async def add_session_to_memory(self, session: "Session") -> None:
        """Persist session events to the memory file."""
        records: list[dict[str, object]] = []
        for event in session.events:
            if not event.content or not event.content.parts:
                continue
            text = _content_text(event.content.parts)
            if not text:
                continue
            record = {
                "app_name": session.app_name,
                "user_id": session.user_id,
                "session_id": session.id,
                "event_id": event.id,
                "author": event.author,
                "timestamp": datetime.fromtimestamp(event.timestamp).isoformat(),
                "role": event.content.role,
                "text": text,
            }
            records.append(record)

        if not records:
            return

        with self._path.open("a", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=True) + "\n")

    async def search_memory(
        self,
        *,
        app_name: str,
        user_id: str,
        query: str,
    ) -> SearchMemoryResponse:
        """Search memory entries by keyword match."""
        response = SearchMemoryResponse()
        if not self._path.is_file():
            return response

        words_in_query = _extract_words_lower(query)
        if not words_in_query:
            return response

        with self._path.open("r", encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                if record.get("app_name") != app_name or record.get("user_id") != user_id:
                    continue
                text = str(record.get("text", "")).strip()
                if not text:
                    continue
                words_in_event = _extract_words_lower(text)
                if not words_in_event:
                    continue
                if not any(word in words_in_event for word in words_in_query):
                    continue

                content = types.Content(
                    role=str(record.get("role") or "user"),
                    parts=[types.Part.from_text(text=text)],
                )
                response.memories.append(
                    MemoryEntry(
                        content=content,
                        author=record.get("author"),
                        timestamp=record.get("timestamp"),
                        custom_metadata={
                            "session_id": record.get("session_id"),
                            "event_id": record.get("event_id"),
                        },
                        id=record.get("event_id"),
                    )
                )

        return response


if TYPE_CHECKING:
    from google.adk.sessions.session import Session
