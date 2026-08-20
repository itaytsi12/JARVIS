"""Where does each request actually go? -- the no-voice-needed routing check.

Two modes, same command list:

    python scripts/verify_agent_routing.py
        Dry run. Resolves every request through the REAL `brain.router` and
        prints the route, the route source and the complexity signals. Makes
        no model call of any kind, so it is free and safe to run any time.

    python scripts/verify_agent_routing.py --run
        Additionally executes the agent-bound requests through the REAL
        `brain.agent.run_agent`, which means REAL, PAID Anthropic calls.
        Prints what actually happened -- provider, model, steps, which tools
        ran, tokens, cost -- because "the route says agent_task" is not
        evidence that the runtime did anything.

    python scripts/verify_agent_routing.py --run --only 5
        Just one case, by its number.

This exists because the live failures it checks were invisible from the route
alone: the request reached the legacy planner instead, spent an OpenAI call,
and answered "I couldn't create a safe local plan for that task." Never run by
the automated suite (`pytest.ini` limits discovery to `tests/`).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def show(text: str) -> None:
    """Print `text` without ever crashing on the console encoding.

    A real agent answer legitimately contains characters the Windows cp1252
    console cannot encode (an arrow, a Hebrew song title). A verification
    tool that raises UnicodeEncodeError on a SUCCESSFUL run reports the
    wrong thing entirely, so unencodable characters are replaced rather
    than allowed to fail the check.
    """
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    print(text.encode(encoding, errors="replace").decode(encoding, errors="replace"))

#: (label, request, what it must resolve to). The expectation is a route TYPE,
#: matching the behaviours asked for after the live voice testing session.
CASES = [
    ("simple app", "Open Spotify", "tool"),
    ("simple audio", "Volume down", "tool"),
    ("browser plan", "Open YouTube and search for Iron Man.", "plan"),
    ("agent / filesystem", "Tell me what files are in the JARVIS project folder. Do not modify anything.", "agent_task"),
    ("agent / read a file", "Read main.py and tell me what it does. Do not modify anything.", "agent_task"),
    ("agent / architecture", "Inspect the JARVIS project and explain how the main components are connected. Do not modify anything.", "agent_task"),
    ("agent / terminal", "Run git status in the JARVIS project and tell me what changed. Do not modify anything.", "agent_task"),
    ("music", "Open Music.", "tool"),
    ("music", "Play Israeli playlist.", "tool"),
]


def dry_run(cases) -> int:
    from brain.router import route_command
    from providers.registry import agent_escalation_available
    from voice.text_normalizer import normalize_transcript

    print(f"agent available: {agent_escalation_available()}\n")
    failures = 0
    for index, (label, request, expected) in enumerate(cases, start=1):
        # Both entry points: `main.py` types the request straight in, the
        # voice loop normalizes the transcript first. They must agree.
        spoken, _ = normalize_transcript(f"Hey Jarvis, {request}")
        typed = route_command(request)
        voice = route_command(spoken)
        ok = typed["type"] == expected and voice["type"] == expected and typed.get("tool") == voice.get("tool")
        failures += not ok
        show(f"{index}. {label}: {request!r}")
        print(f"   typed -> type={typed['type']} tool={typed.get('tool')} source={typed.get('route_source') or '-'}")
        print(f"   voice -> type={voice['type']} tool={voice.get('tool')} source={voice.get('route_source') or '-'}")
        if typed.get("complexity"):
            print(f"   signals: {typed['complexity']['signals']} reason={typed['complexity']['reason']}")
        print(f"   expected={expected}  {'OK' if ok else 'MISMATCH'}\n")
    print(f"{len(cases) - failures}/{len(cases)} routed as expected")
    return 1 if failures else 0


def live_run(cases) -> int:
    from config import configure_logging, get_config
    from providers.registry import get_agent_provider, provider_status

    configure_logging()
    if not get_config().has_anthropic_credentials:
        print("ANTHROPIC_API_KEY is not set; there is nothing to verify live.")
        print(provider_status())
        return 1
    provider = get_agent_provider()
    if provider is None:
        print("No agent provider is available even though a key is set:")
        print(provider_status())
        return 1
    print(f"Provider: {provider.name}  Model: {provider.model}\n")

    from brain.agent import run_agent

    failures = 0
    for index, (label, request, expected) in enumerate(cases, start=1):
        if expected != "agent_task":
            continue
        show(f"--- {index}. {label}: {request!r}")
        outcome: dict = {}
        started = time.perf_counter()
        try:
            answer = run_agent(request, execution_outcome=outcome)
        except Exception as exc:
            failures += 1
            show(f"    RAISED {type(exc).__name__}: {exc}\n")
            continue
        elapsed_ms = (time.perf_counter() - started) * 1000
        tools = outcome.get("agent_tools") or [action.get("tool") for action in (outcome.get("actions") or [])]
        reached_the_runtime = outcome.get("route_type") == "agent_task"
        used_the_model = (outcome.get("model_calls") or 0) > 0
        failures += not (reached_the_runtime and used_the_model and outcome.get("success"))
        print(f"    route_type   : {outcome.get('route_type')} (source={outcome.get('route_source')})")
        print(f"    model_calls  : {outcome.get('model_calls')}")
        show(f"    tools called : {tools or '(none -- the model answered without acting)'}")
        print(f"    success      : {outcome.get('success')}  verified: {outcome.get('verified')}")
        print(f"    elapsed      : {elapsed_ms:.0f} ms")
        show(f"    answer       : {answer}\n")

    from providers.usage import get_usage_store

    # Recorded usage, read back from the store rather than from the run --
    # "token/cost usage is recorded" is only proven by what persisted.
    print("recorded usage:", get_usage_store().total().to_dict())
    return 1 if failures else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", action="store_true", help="Actually execute the agent cases (REAL, PAID API calls)")
    parser.add_argument("--only", type=int, help="Run just one case, by its number in the dry-run listing")
    arguments = parser.parse_args()

    cases = CASES if arguments.only is None else [CASES[arguments.only - 1]]
    status = dry_run(cases)
    if not arguments.run:
        print("\n(dry run -- pass --run to execute the agent cases against the real API)")
        return status
    print("\n=== live agent execution (real, paid) ===\n")
    return live_run(cases) or status


if __name__ == "__main__":
    raise SystemExit(main())
