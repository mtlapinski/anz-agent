import os
import sys
import anthropic
from dotenv import load_dotenv
from agent import chat

REQUIRED_KEYS = [
    "ANTHROPIC_API_KEY",
    "SERPAPI_KEY",
    "LANGFUSE_PUBLIC_KEY",
    "LANGFUSE_SECRET_KEY",
]


def check_credentials() -> None:
    missing = [k for k in REQUIRED_KEYS if not os.environ.get(k)]
    if missing:
        print(f"Error: missing environment variables: {', '.join(missing)}")
        print("Copy .env.example to .env and fill in your keys.")
        sys.exit(1)


def main() -> None:
    load_dotenv()
    check_credentials()

    client = anthropic.Anthropic()
    history: list = []

    print("Amazon Shopping Assistant")
    print("Type 'quit' or 'exit' to stop.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        if not user_input:
            continue

        try:
            response = chat(client, history, user_input)
            print(f"\nAssistant: {response}\n")
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}. Please try again.\n")

    # flush Langfuse event queue so last trace is not lost on exit
    try:
        from agent import _get_langfuse
        _get_langfuse().flush()
    except Exception:
        pass


if __name__ == "__main__":
    main()
