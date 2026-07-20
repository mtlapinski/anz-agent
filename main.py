import os
import sys
from dotenv import load_dotenv
from llm import ModelConfig, create_client
from agent import chat

DEFAULT_PROVIDER = "google"
DEFAULT_MODEL = "gemini-2.0-flash-lite"
ANTHROPIC_DEFAULT_MODEL = "claude-haiku-4-5-20251001"

PROVIDER_KEYS = {
    "google": "GOOGLE_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
}


def select_model() -> ModelConfig:
    print("Provider? [1] Google (default)  [2] Anthropic : ", end="", flush=True)
    choice = input().strip()
    if choice == "2":
        provider = "anthropic"
        default_model = ANTHROPIC_DEFAULT_MODEL
    else:
        provider = "google"
        default_model = DEFAULT_MODEL
    print(f"Model? [{default_model}] : ", end="", flush=True)
    model = input().strip() or default_model
    return ModelConfig(provider=provider, model=model)


def check_credentials(config: ModelConfig) -> None:
    required = ["SERPAPI_KEY", PROVIDER_KEYS[config.provider]]
    missing = [k for k in required if not os.environ.get(k)]
    if missing:
        print(f"Error: missing environment variables: {', '.join(missing)}")
        print("Copy .env.example to .env and fill in your keys.")
        sys.exit(1)


def _has_tool_turns(history: list) -> bool:
    return any(
        isinstance(msg.get("content"), list) and
        any(b.get("type") in ("tool_use", "tool_result") for b in msg["content"])
        for msg in history
    )


def handle_model_command(args: str, current_config: ModelConfig, history: list) -> tuple[ModelConfig, list]:
    """Handle /model command. Returns (config, history) — unchanged on error or cancel."""
    if args:
        parts = args.strip().split(None, 1)
        if len(parts) != 2:
            print("Usage: /model <provider> <model>  (e.g. /model google gemini-2.0-flash-lite)")
            return current_config, history
        provider, model = parts
        if provider not in PROVIDER_KEYS:
            print(f"Unknown provider {provider!r}. Choose 'google' or 'anthropic'.")
            return current_config, history
        new_config = ModelConfig(provider=provider, model=model)
    else:
        new_config = select_model()

    key = PROVIDER_KEYS[new_config.provider]
    if not os.environ.get(key):
        print(f"Error: {key} is not set. Keeping current model.")
        return current_config, history

    if _has_tool_turns(history):
        print("Warning: history contains tool call turns that may not transfer cleanly.")

    if history:
        print("Start fresh conversation? [y/N] : ", end="", flush=True)
        if input().strip().lower() == "y":
            history = []

    print(f"Using {new_config.provider} / {new_config.model}")
    return new_config, history


def main() -> None:
    load_dotenv()

    model_config = select_model()
    check_credentials(model_config)
    print(f"Using {model_config.provider} / {model_config.model}\n")

    client = create_client(model_config)
    history: list = []

    print("Amazon Shopping Assistant")
    print("Type 'quit', 'exit', or '/model' to change the model.\n")

    while True:
        try:
            user_input = input("What can I help you find? : ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        if not user_input:
            continue

        if user_input == "/model" or user_input.startswith("/model "):
            args = user_input[len("/model"):].strip()
            old_config = model_config
            model_config, history = handle_model_command(args, model_config, history)
            if model_config != old_config:
                client = create_client(model_config)
            continue

        try:
            response = chat(client, history, user_input, model_config)
            print(f"\nAssistant: {response}\n")
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            if history and history[-1]["role"] == "user":
                history.pop()
            print(f"\nError: {e}. Please try again.\n")

    try:
        from agent import _get_langfuse
        _get_langfuse().flush()
    except Exception:
        pass


if __name__ == "__main__":
    main()
