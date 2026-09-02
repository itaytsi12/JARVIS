import json
import os

from openai import OpenAI
from config.events import model_activity

from brain.intent_router import TOOLS


# `.env` is loaded exactly once, from the project root, by
# `config/settings.py` -- importing it here is what guarantees that has
# happened before the environment is read below. A local `load_dotenv()`
# call would search upward from the CURRENT WORKING DIRECTORY instead,
# which is how the runtime silently ended up with no API key when it was
# started from anywhere but the repository root.
import config  # noqa: F401  -- imported for its .env loading side effect

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)


def create_plan(message: str) -> list[dict]:
    from brain.intent_router import cloud_intent_available

    if not cloud_intent_available():
        # An empty plan is this function's existing "I could not plan
        # that" answer, and every caller already handles it. Raising a
        # 401 instead would turn a missing credential into a crash.
        return []
    # UI/status hook: brackets THIS real request with
    # started/succeeded/failed events (config/events.py). It only
    # observes -- the request below is unchanged, and a subscriber
    # that raises can never reach this call site.
    with model_activity("openai"):
        response = client.responses.create(
            model="gpt-5-mini",
            input=[
                {
                    "role": "system",
                    "content": (
                        "You are the action planner for Jarvis, a Windows desktop assistant. "
                        "Break the user's request into the minimum number of available tool calls. "
                        "Call the tools in the exact order they should be executed. "
                        "Do not add unnecessary actions. "
                        "Every requested step must be represented and achievable with the available tools. "
                        "If the available tools cannot complete the entire request, return no tool calls; never return a partial plan or silently omit a step. "
                        "Do not answer conversationally."
                    )
                },
                {
                    "role": "user",
                    "content": message
                }
            ],
            tools=TOOLS
        )

    actions = []

    for item in response.output:
        if item.type == "function_call":
            try:arguments=json.loads(item.arguments)
            except (TypeError,json.JSONDecodeError):arguments=None
            actions.append({
                "tool": item.name,
                "arguments": arguments
            })

    return actions
