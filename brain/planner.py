import json
import os

from dotenv import load_dotenv
from openai import OpenAI

from brain.intent_router import TOOLS


load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)


def create_plan(message: str) -> list[dict]:
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
