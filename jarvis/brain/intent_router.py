import json
import os

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)


TOOLS = [
    {
        "type": "function",
        "name": "open_application",
        "description": "Open a Windows application.",
        "parameters": {
            "type": "object",
            "properties": {
                "app_name": {
                    "type": "string"
                }
            },
            "required": ["app_name"],
            "additionalProperties": False
        },
        "strict": True
    },

    {
        "type": "function",
        "name": "close_application",
        "description": "Close a Windows application.",
        "parameters": {
            "type": "object",
            "properties": {
                "app_name": {
                    "type": "string"
                }
            },
            "required": ["app_name"],
            "additionalProperties": False
        },
        "strict": True
    },

    {
        "type": "function",
        "name": "open_website",
        "description": "Open a website in the browser.",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string"
                }
            },
            "required": ["url"],
            "additionalProperties": False
        },
        "strict": True
    },

    {
        "type": "function",
        "name": "volume_up",
        "description": "Increase the computer volume.",
        "parameters": {
            "type": "object",
            "properties": {
                "amount": {
                    "type": "integer"
                }
            },
            "required": ["amount"],
            "additionalProperties": False
        },
        "strict": True
    },

    {
        "type": "function",
        "name": "volume_down",
        "description": "Decrease the computer volume.",
        "parameters": {
            "type": "object",
            "properties": {
                "amount": {
                    "type": "integer"
                }
            },
            "required": ["amount"],
            "additionalProperties": False
        },
        "strict": True
    },

    {
        "type": "function",
        "name": "mute_volume",
        "description": "Mute or unmute the computer volume.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False
        },
        "strict": True
    },

     {
        "type": "function",
        "name": "take_screenshot",
        "description": "Take a screenshot of the current screen.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False
        },
        "strict": True
    },

    {
        "type": "function",
        "name": "type_text",
        "description": "Type text into the currently focused application.",
        "parameters": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string"
                },
                "delay": {
                    "type": "number"
                }
            },
            "required": ["text", "delay"],
            "additionalProperties": False
        },
        "strict": True
    }
]


def classify_intent(message: str):
    response = client.responses.create(
        model="gpt-5-mini",
        input=[
            {
                "role": "system",
                "content": (
                    "You are the intent router for a Windows desktop assistant "
                    "called Jarvis. "
                    "If the user wants one or more available computer actions, "
                    "call all required tools in the correct order. "
                    "Do not call tools for normal conversation."
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

    if actions:
        return {
            "type": "tools",
            "actions": actions
        }

    return {
        "type": "ai",
        "message": message
    }

    return {
        "type": "ai",
        "message": message
    }