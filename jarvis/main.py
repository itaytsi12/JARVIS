from brain.agent import run_agent


def main():
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


if __name__ == "__main__":
    main()