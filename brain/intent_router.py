import json
import os

from openai import OpenAI
from config.events import model_activity


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


TOOLS = [
    {
        "type":"function","name":"wait_for_window","description":"Wait for a previously opened Windows application to expose a visible responsive window.",
        "parameters":{"type":"object","properties":{"app_name":{"type":"string"}},"required":["app_name"],"additionalProperties":False},"strict":True
    },
    {
        "type":"function","name":"click_ui_element","description":"Click one uniquely named visible UI element in a verified open application.",
        "parameters":{"type":"object","properties":{"app_name":{"type":"string"},"name":{"type":"string"},"control_type":{"type":"string"}},"required":["app_name","name","control_type"],"additionalProperties":False},"strict":True
    },
    {
        "type":"function","name":"press_key","description":"Press a key or keyboard shortcut in the verified active application.",
        "parameters":{"type":"object","properties":{"key":{"type":"string"}},"required":["key"],"additionalProperties":False},"strict":True
    },
    {
        "type":"function",
        "name":"inspect_window",
        "description":"Read a bounded list of accessibility controls from an open Windows application. This does not click or type.",
        "parameters":{
            "type":"object",
            "properties":{"app_name":{"type":"string"},"limit":{"type":"integer","minimum":1,"maximum":100}},
            "required":["app_name","limit"],
            "additionalProperties":False
        },
        "strict":True
    },
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
    usage=getattr(response,"usage",None);usage_metadata={"input_tokens":getattr(usage,"input_tokens",0) or 0,"output_tokens":getattr(usage,"output_tokens",0) or 0}

    actions = []

    for item in response.output:
        if item.type == "function_call":
            try:arguments=json.loads(item.arguments)
            except (TypeError,json.JSONDecodeError):arguments=None
            actions.append({
                "tool": item.name,
                "arguments": arguments
            })

    if actions:
        return {
            "type": "tools",
            "actions": actions,
            "route_source": "cloud_intent_router",
            "model": "gpt-5-mini",
            "model_calls": 1,
            **usage_metadata,
        }

    return {
        "type": "ai",
        "message": message,
        "route_source": "cloud_intent_router",
        "model": "gpt-5-mini",
        "model_calls": 1,
        **usage_metadata,
    }

    return {
        "type": "ai",
        "message": message
    }
