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
            actions.append({
                "tool": item.name,
                "arguments": json.loads(item.arguments)
            })

    return actions