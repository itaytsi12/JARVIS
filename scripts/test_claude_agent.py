"""Optional REAL Claude smoke test -- makes a genuine, paid API call.

Never run by the automated suite (`pytest.ini` limits discovery to
`tests/`, and this refuses to do anything without an explicit `--run`).
Use it once after adding `ANTHROPIC_API_KEY` to confirm the provider,
the tool loop, and cost accounting all work end to end against the real
service.

    python scripts/test_claude_agent.py --run
    python scripts/test_claude_agent.py --run --goal "read main.py and tell me what it does"

The default goal is deliberately trivial and read-only: it asks for the
time, which exercises the full tool-use round trip for a few hundred
tokens.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DEFAULT_GOAL = "Tell me the current time using your tools, then say it in one short sentence."


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", action="store_true", help="Actually make the paid API call")
    parser.add_argument("--goal", default=DEFAULT_GOAL, help="What to ask the agent to do")
    arguments = parser.parse_args()

    if not arguments.run:
        print(__doc__)
        print("Refusing to make a paid API call without --run.")
        return 0

    from config import configure_logging, get_config
    from providers.registry import get_agent_provider, provider_status

    configure_logging()
    config = get_config()
    if not config.has_anthropic_credentials:
        print("ANTHROPIC_API_KEY is not set. Add it to your .env and try again.")
        print(json.dumps(provider_status(), indent=2))
        return 1

    provider = get_agent_provider()
    if provider is None:
        print("No agent provider is available even though a key is set:")
        print(json.dumps(provider_status(), indent=2))
        return 1

    print(f"Provider: {provider.name}  Model: {provider.model}")
    print(f"Goal: {arguments.goal}\n")

    from brain.agent_service import run_agent_task

    def progress(stage: str, payload: dict) -> None:
        if stage == "tool_result":
            mark = "ok " if payload.get("success") else "FAIL"
            print(f"  [{mark}] {payload.get('tool')}")

    outcome = run_agent_task(arguments.goal, progress=progress)

    print(f"\nAnswer: {outcome.answer}\n")
    print(json.dumps(outcome.describe(), indent=2))

    from providers.usage import get_usage_store

    summary = get_usage_store().for_task(outcome.task_id)
    print("\nUsage for this task:")
    print(json.dumps(summary.to_dict(), indent=2))
    if not summary.cost_is_complete:
        print("(Some calls used a model with no configured price -- the cost above is a floor.)")
    return 0 if outcome.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
