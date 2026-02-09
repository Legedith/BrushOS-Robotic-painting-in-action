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


async def run_prompt(
    runner: Runner,
    *,
    prompt: str,
    session_id: str,
    user_id: str,
) -> None:
    """Run a single prompt and stream tool output."""
    final_text = ""
    async for event in runner.run_async(
        user_id=user_id,
        session_id=session_id,
        new_message=content_from_text(prompt),
    ):
        print(f"Event: author={event.author} final={event.is_final_response()}")
        for call in event.get_function_calls():
            print(f"Tool call: {call.name} {json.dumps(call.args, ensure_ascii=True)}")
        for response in event.get_function_responses():
            print(
                f"Tool output: {response.name} "
                f"{json.dumps(response.response, ensure_ascii=True)}"
            )
        if event.content and event.content.parts and event.is_final_response():
            texts = [part.text for part in event.content.parts if part.text]
            if texts:
                final_text = "\n".join(texts)

    if final_text:
        print(final_text)


def main() -> int:
    load_env_from_file(Path("camera_agent/.env"))

    model_override = os.environ.get("CAMERA_AGENT_MODEL")
    if model_override:
        root_agent.model = model_override

    session_id = os.environ.get("CAMERA_AGENT_SESSION_ID", "session")
    user_id = os.environ.get("CAMERA_AGENT_USER_ID", "user")

    runner = Runner(
        agent=root_agent,
        app_name="camera_agent",
        session_service=create_session_service(),
        memory_service=create_memory_service(),
        artifact_service=InMemoryArtifactService(),
        auto_create_session=True,
    )

    print("Interactive mode. Type /exit to quit.")
    while True:
        try:
            prompt = input("You> ").strip()
        except EOFError:
            break
        if not prompt:
            continue
        if prompt.lower() in {"/exit", "exit", "quit"}:
            break
        try:
            asyncio.run(run_prompt(runner, prompt=prompt, session_id=session_id, user_id=user_id))
        except Exception as exc:  # noqa: BLE001
            print(f"Run failed: {type(exc).__name__}: {exc}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
