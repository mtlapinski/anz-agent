from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Any

import anthropic


@dataclass
class ModelConfig:
    provider: str  # "anthropic" | "google"
    model: str


@dataclass
class LLMResponse:
    text: str | None
    tool_calls: list[dict] | None  # [{name, id, input}]
    input_tokens: int
    output_tokens: int


def create_client(config: ModelConfig) -> Any:
    if config.provider == "anthropic":
        return anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    if config.provider == "google":
        import google.generativeai as genai
        genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
        return genai
    raise ValueError(f"Unknown provider: {config.provider!r}")


def complete(
    client: Any,
    config: ModelConfig,
    system: str,
    tools: list[dict],
    messages: list[dict],
) -> LLMResponse:
    if config.provider == "anthropic":
        return _complete_anthropic(client, config.model, system, tools, messages)
    if config.provider == "google":
        return _complete_google(config.model, system, tools, messages)
    raise ValueError(f"Unknown provider: {config.provider!r}")


def _complete_anthropic(client, model: str, system: str, tools: list[dict], messages: list[dict]) -> LLMResponse:
    response = client.messages.create(
        model=model,
        max_tokens=2048,
        system=system,
        tools=tools,
        messages=messages,
    )
    if response.stop_reason == "tool_use":
        tool_calls = [
            {"name": b.name, "id": b.id, "input": b.input}
            for b in response.content
            if b.type == "tool_use"
        ]
        return LLMResponse(
            text=None,
            tool_calls=tool_calls,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )
    if response.stop_reason not in ("end_turn", "stop_sequence"):
        raise RuntimeError(f"Unexpected stop_reason: {response.stop_reason!r}")
    text = next((b.text for b in response.content if hasattr(b, "text")), "")
    return LLMResponse(
        text=text,
        tool_calls=None,
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
    )


def _complete_google(model_name: str, system: str, tools: list[dict], messages: list[dict]) -> LLMResponse:
    # Implemented in Task 2
    raise NotImplementedError("Google provider not yet implemented")
