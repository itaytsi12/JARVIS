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
    args = parser.parse_args()

    if args.voice:
        voice_mode()
    else:
        typed_mode()


if __name__ == "__main__":
    main()