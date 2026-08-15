import argparse


def typed_mode():
    from brain.agent import run_agent

    print("\nJARVIS ONLINE")
    print("Type 'exit' to stop.\n")

    while True:
        command = input("You: ").strip()

        if command.lower() == "exit":
            print("Jarvis: Goodbye.")
            break

        if not command:
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--voice", action="store_true", help="Start push-to-talk voice mode")
    parser.add_argument("--dry-run", metavar="GOAL", help="Print a local agent plan without executing it")
    parser.add_argument("--tray", action="store_true", help="Run the always-on system tray assistant")
    args = parser.parse_args()

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
