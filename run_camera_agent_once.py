from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from google.adk.artifacts.in_memory_artifact_service import InMemoryArtifactService
from google.adk.runners import Runner
from google.genai import types

from camera_agent.agent import root_agent
from camera_agent.persistence import create_memory_service, create_session_service


def load_env_from_file(env_path: Path) -> None:
    """Load environment variables from a simple KEY=VALUE file."""
    if not env_path.is_file():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def content_from_text(text: str) -> types.Content:
    """Create a content object from text."""
    return types.Content(role="user", parts=[types.Part.from_text(text=text)])


def main() -> int:
    prompt = " ".join(sys.argv[1:]).strip()
    if not prompt:
        print("Usage: uv run run_camera_agent_once.py \"your prompt\"")
        return 1

    load_env_from_file(Path("camera_agent/.env"))

    session_service = create_session_service()
    memory_service = create_memory_service()
    runner = Runner(
        agent=root_agent,
        app_name="camera_agent",
        session_service=session_service,
        memory_service=memory_service,
        artifact_service=InMemoryArtifactService(),
        auto_create_session=True,
    )
    session_id = os.environ.get("CAMERA_AGENT_SESSION_ID", "session")
    user_id = os.environ.get("CAMERA_AGENT_USER_ID", "user")

    final_text = ""
    function_responses: list[dict[str, object]] = []
    function_calls: list[dict[str, object]] = []

    for event in runner.run(
        user_id=user_id,
        session_id=session_id,
        new_message=content_from_text(prompt),
    ):
        print(f"Event: author={event.author} final={event.is_final_response()}")
        for call in event.get_function_calls():
            function_calls.append(
                {
                    "name": call.name,
                    "args": call.args,
                }
            )
        for response in event.get_function_responses():
            function_responses.append(
                {"name": response.name, "response": response.response}
            )
        if event.content and event.content.parts:
            texts = [part.text for part in event.content.parts if part.text]
            if texts and event.is_final_response():
                final_text = "\n".join(texts)

    if final_text:
        print(final_text)
    if function_calls:
        print("\nTool calls:")
        print(json.dumps(function_calls, indent=2))
    if function_responses:
        print("\nTool outputs:")
        print(json.dumps(function_responses, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
