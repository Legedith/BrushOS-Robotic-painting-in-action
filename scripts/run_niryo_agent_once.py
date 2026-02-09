from __future__ import annotations

from _bootstrap import add_repo_root
add_repo_root()


import json
import os
import sys

from google.adk.runners import InMemoryRunner
from google.genai import types

from niryo_agent.agent import root_agent


def content_from_text(text: str) -> types.Content:
    """Create a content object from text."""
    return types.Content(role="user", parts=[types.Part.from_text(text=text)])


def main() -> int:
    prompt = " ".join(sys.argv[1:]).strip()
    if not prompt:
        print("Usage: uv run run_niryo_agent_once.py \"your prompt\"")
        return 1

    runner = InMemoryRunner(agent=root_agent, app_name="niryo_arm_agent")
    runner.auto_create_session = True
    session_id = os.environ.get("NIRYO_AGENT_SESSION_ID", "session")
    user_id = os.environ.get("NIRYO_AGENT_USER_ID", "user")

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
            function_calls.append({"name": call.name, "args": call.args})
        for response in event.get_function_responses():
            function_responses.append({"name": response.name, "response": response.response})
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
