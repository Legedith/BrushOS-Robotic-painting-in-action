from __future__ import annotations

from _bootstrap import add_repo_root
add_repo_root()


import asyncio
import json
import os
import sys
from pathlib import Path

from google.adk.artifacts.in_memory_artifact_service import InMemoryArtifactService
from google.adk.runners import Runner
from google.genai import types

from camera_agent.agent import root_agent
from camera_agent.key_rotation import get_api_keys, is_retryable_error
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


async def run_once(
    *,
    prompt: str,
    session_id: str,
    user_id: str,
) -> tuple[str, list[dict[str, object]], list[dict[str, object]]]:
    """Run the agent once and capture results."""
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

    final_text = ""
    function_responses: list[dict[str, object]] = []
    function_calls: list[dict[str, object]] = []

    async for event in runner.run_async(
        user_id=user_id,
        session_id=session_id,
        new_message=content_from_text(prompt),
    ):
        print(f"Event: author={event.author} final={event.is_final_response()}")
        for call in event.get_function_calls():
            function_calls.append({"name": call.name, "args": call.args})
            print(f"Tool call: {call.name} {json.dumps(call.args, ensure_ascii=True)}")
        for response in event.get_function_responses():
            function_responses.append(
                {"name": response.name, "response": response.response}
            )
            print(
                f"Tool output: {response.name} "
                f"{json.dumps(response.response, ensure_ascii=True)}"
            )
        if event.content and event.content.parts:
            texts = [part.text for part in event.content.parts if part.text]
            if texts and event.is_final_response():
                final_text = "\n".join(texts)

    return final_text, function_calls, function_responses


def main() -> int:
    prompt = " ".join(sys.argv[1:]).strip()
    if not prompt:
        print("Usage: uv run run_camera_agent_once.py \"your prompt\"")
        return 1

    load_env_from_file(Path("camera_agent/.env"))

    model_override = os.environ.get("CAMERA_AGENT_MODEL")
    if model_override:
        root_agent.model = model_override

    session_id = os.environ.get("CAMERA_AGENT_SESSION_ID", "session")
    user_id = os.environ.get("CAMERA_AGENT_USER_ID", "user")

    keys = get_api_keys()
    final_text = ""
    function_calls: list[dict[str, object]] = []
    function_responses: list[dict[str, object]] = []
    last_error: Exception | None = None

    for key in keys:
        os.environ["GOOGLE_API_KEY"] = key
        try:
            final_text, function_calls, function_responses = asyncio.run(
                run_once(prompt=prompt, session_id=session_id, user_id=user_id)
            )
            last_error = None
            break
        except Exception as exc:  # noqa: BLE001
            if is_retryable_error(exc):
                last_error = exc
                continue
            raise

    if last_error is not None:
        raise last_error

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
