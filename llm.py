from __future__ import annotations
import json
import os
from dataclasses import dataclass
from typing import Any

import anthropic

try:
    import google.generativeai as genai
except ImportError:
    genai = None  # type: ignore


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
    if genai is None:
        raise ImportError("google-generativeai is not installed. Run: pip install google-generativeai")
    google_tools = _anthropic_tools_to_google(tools)
    google_messages = _anthropic_messages_to_google(messages)

    model = genai.GenerativeModel(
        model_name=model_name,
        system_instruction=system,
        tools=google_tools,
    )
    response = model.generate_content(google_messages)

    usage = response.usage_metadata
    input_tokens = usage.prompt_token_count if usage else 0
    output_tokens = usage.candidates_token_count if usage else 0

    parts = response.candidates[0].content.parts
    tool_calls = []
    for part in parts:
        if hasattr(part, "function_call") and part.function_call.name:
            tool_calls.append({
                "name": part.function_call.name,
                "id": f"google_{part.function_call.name}",
                "input": dict(part.function_call.args),
            })

    if tool_calls:
        return LLMResponse(text=None, tool_calls=tool_calls, input_tokens=input_tokens, output_tokens=output_tokens)

    text = "".join(part.text for part in parts if hasattr(part, "text"))
    return LLMResponse(text=text, tool_calls=None, input_tokens=input_tokens, output_tokens=output_tokens)


def _anthropic_tools_to_google(tools: list[dict]) -> list[dict]:
    if not tools:
        return []
    return [{
        "function_declarations": [
            {"name": t["name"], "description": t["description"], "parameters": t["input_schema"]}
            for t in tools
        ]
    }]


def _anthropic_messages_to_google(messages: list[dict]) -> list[dict]:
    result = []
    for msg in messages:
        role = "model" if msg["role"] == "assistant" else "user"
        content = msg["content"]
        if isinstance(content, str):
            result.append({"role": role, "parts": [{"text": content}]})
        elif isinstance(content, list):
            parts = []
            for block in content:
                btype = block.get("type")
                if btype == "text":
                    parts.append({"text": block["text"]})
                elif btype == "tool_use":
                    parts.append({"function_call": {"name": block["name"], "args": block["input"]}})
                elif btype == "tool_result":
                    raw = block["content"]
                    if isinstance(raw, str):
                        try:
                            raw = json.loads(raw)
                        except Exception:
                            raw = {"result": raw}
                    tool_name = _find_tool_name(messages, block["tool_use_id"])
                    parts.append({"function_response": {"name": tool_name, "response": raw}})
            if parts:
                result.append({"role": role, "parts": parts})
    return result


def _find_tool_name(messages: list[dict], tool_use_id: str) -> str:
    for msg in messages:
        content = msg.get("content", [])
        if isinstance(content, list):
            for block in content:
                if block.get("type") == "tool_use" and block.get("id") == tool_use_id:
                    return block["name"]
    return "unknown_tool"
