import argparse
import json


def typed_mode():
    from brain.agent import run_agent

    print("\nJARVIS ONLINE")
    print("Type 'exit' to stop, 'status' for runtime/task state, 'tasks' for running tasks.\n")

    while True:
        command = input("You: ").strip()

        if command.lower() == "exit":
            print("Jarvis: Goodbye.")
            break

        if not command:
            continue

        if command.lower() == "status":
            print(json.dumps(runtime_status(), indent=2))
            continue

        if command.lower() == "tasks":
            from tasks.manager import get_task_manager

            print(f"Jarvis: {get_task_manager().describe_active()}")
            continue

        try:
            response = run_agent(command)
            print(f"\nJarvis: {response}\n")

        except Exception as e:
            print(f"\nJarvis error: {e}\n")


def voice_mode():
    # Import voice modules lazily so typed mode remains functional if voice deps missing.
    try:
        from voice.voice_controller import run_voice_loop
    except Exception as e:
        print(f"Voice mode unavailable: {e}")
        return

    run_voice_loop()


def runtime_status() -> dict:
    """Configuration, provider availability, memory and task state.

    The first thing to check when Claude "isn't being used": it says
    exactly which provider is active, or why none is.
    """
    from config.logging_setup import describe_runtime
    from memory.agent_memory import get_agent_memory
    from providers.usage import get_usage_store
    from tasks.manager import get_task_manager

    status = describe_runtime()
    status["memory"] = get_agent_memory().statistics()
    status["tasks"] = get_task_manager().snapshot()
    status["usage"] = get_usage_store().total().to_dict()
    return status


def agent_mode(goal: str, background: bool = False) -> int:
    """Run one goal through the agent runtime and report honestly."""
    from brain.agent_service import run_agent_task, submit_agent_task

    def progress(stage: str, payload: dict) -> None:
        if stage == "tool_result":
            mark = "ok " if payload.get("success") else "FAIL"
            print(f"  [{mark}] {payload.get('tool')}" + (f" -- {payload.get('error')}" if payload.get("error") else ""))

    if background:
        # Submitting returns immediately -- that is the point of the flag,
        # and what a voice command does. A CLI process still has to stay
        # alive for the task to finish, so wait and report rather than
        # exiting and silently killing it.
        handle = submit_agent_task(goal)
        print(f"Task {handle.task_id} submitted; the prompt was never blocked. Waiting for it to finish...")
        handle.wait()
        task = handle.task
        print(f"\nJarvis: {task.result or task.error or 'No result.'}\n")
        print(json.dumps(task.to_dict(include_observations=False), indent=2))
        return 0 if task.status.value == "completed" else 1

    outcome = run_agent_task(goal, progress=progress)
    print(f"\nJarvis: {outcome.answer}\n")
    print(json.dumps(outcome.describe(), indent=2))
    return 0 if outcome.success else 1


def main():
    parser = argparse.ArgumentParser(description="JARVIS -- Windows desktop AI agent")
    parser.add_argument("--voice", action="store_true", help="Start push-to-talk voice mode")
    parser.add_argument("--dry-run", metavar="GOAL", help="Print a local agent plan without executing it")
    parser.add_argument("--tray", action="store_true", help="Run the always-on system tray assistant")
    parser.add_argument("--agent", metavar="GOAL", help="Run one goal through the agent runtime (needs ANTHROPIC_API_KEY)")
    parser.add_argument("--background", action="store_true", help="With --agent: submit the goal as a background task")
    parser.add_argument("--status", action="store_true", help="Print configuration, provider, memory and task status")
    args = parser.parse_args()

    from config import configure_logging, log_startup_status

    configure_logging()
    # Whether the agent provider is usable -- and, when it is not, WHY --
    # is reported here, before any mode starts. Every mode below goes
    # through this one call, so the typed, voice, agent and tray runtimes
    # all use identical provider configuration and identical reporting.
    log_startup_status()

    if args.status:
        print(json.dumps(runtime_status(), indent=2))
        raise SystemExit(0)
    if args.agent:
        raise SystemExit(agent_mode(args.agent, background=args.background))
    if args.tray:
        from voice.tray_app import run_tray
        raise SystemExit(run_tray())
    elif args.dry_run:
        from brain.task_planner import create_task_plan, format_plan

        plan = create_task_plan(args.dry_run)
        print(format_plan(plan) if plan else "No deterministic plan could be created.")
    elif args.voice:
        voice_mode()
    else:
        typed_mode()


if __name__ == "__main__":
    main()
