import os

from dotenv import load_dotenv
from openai import OpenAI

from brain.router import route_command
from brain.tool_router import execute_tool
from brain.planner import create_plan
from brain.executor import Executor
from brain.models import Action


load_dotenv()


client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)


executor = Executor()


def ask_ai(message: str) -> str:
    response = client.responses.create(
        model="gpt-5-mini",
        input=message
    )

    return response.output_text


def run_agent(command: str) -> str:
    route = route_command(
        command
    )

    # -------------------------
    # Single local tool
    # -------------------------

    if route["type"] == "tool":
        action = Action(
            tool=route["tool"],
            args=route["arguments"]
        )

        result = executor.execute_action(
            action
        )

        if result.success:
            return result.message

        return (
            f"{result.message}\n"
            f"Error: {result.error}"
        )

    # -------------------------
    # Local multi-step plan
    # -------------------------

    if route["type"] == "local_plan":
        results = executor.execute_plan(
            route["actions"]
        )

        return format_results(
            results
        )

    # -------------------------
    # Multi-step AI plan
    # -------------------------

    if route["type"] == "plan":
        planned_actions = create_plan(
            route["message"]
        )

        if not planned_actions:
            return "I couldn't create a plan for that."

        actions = [
            Action(
                tool=action["tool"],
                args=action["arguments"]
            )
            for action in planned_actions
        ]

        results = executor.execute_plan(
            actions
        )

        return format_results(
            results
        )

    # -------------------------
    # Intent router tools
    # -------------------------

    if route["type"] == "tools":
        actions = [
            Action(
                tool=action["tool"],
                args=action["arguments"]
            )
            for action in route["actions"]
        ]

        results = executor.execute_plan(
            actions
        )

        return format_results(
            results
        )

    # -------------------------
    # Normal AI question
    # -------------------------

    if route["type"] == "ai":
        return ask_ai(
            route["message"]
        )

    return "I couldn't understand what to do."


def format_results(results) -> str:
    messages = []

    for result in results:
        if result.success:
            messages.append(
                result.message
            )

        else:
            message = result.message

            if result.error:
                message += (
                    f"\nError: {result.error}"
                )

            messages.append(
                message
            )

    return "\n".join(
        messages
    )