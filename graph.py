from __future__ import annotations
from dataclasses import dataclass
from typing import Annotated, Any, TypedDict

import agent
import llm
from agent import SYSTEM_PROMPT, TOOLS
from llm import ModelConfig


def add_to_history(existing: list, new: list) -> list:
    return existing + new


class GraphState(TypedDict):
    history: Annotated[list, add_to_history]
    new_message: str | None
    made_tool_call_this_turn: bool
    pending_tool_calls: list | None
    last_search_input: dict | None
    trace_id: str | None
    response: str | None


@dataclass
class GraphContext:
    client: Any
    model_config: ModelConfig


def agent_node(state: GraphState, runtime) -> dict:
    client = runtime.context.client
    model_config = runtime.context.model_config

    delta = []
    trace_id = state.get("trace_id")

    if state.get("new_message"):
        delta.append({"role": "user", "content": state["new_message"]})
        try:
            trace_id = agent._get_langfuse().create_trace_id()
        except Exception:
            trace_id = None

    history = state["history"] + delta

    generation = None
    if trace_id:
        try:
            generation = agent._get_langfuse().start_observation(
                trace_context={"trace_id": trace_id},
                name="llm",
                as_type="generation",
                input={"system": SYSTEM_PROMPT, "messages": history},
                model=f"{model_config.provider}/{model_config.model}",
            )
        except Exception:
            generation = None

    llm_response = llm.complete(client, model_config, SYSTEM_PROMPT, TOOLS, history)
    print(f"[tokens: {llm_response.input_tokens} in / {llm_response.output_tokens} out]")

    if generation:
        try:
            generation.update(
                output=str(llm_response.text or llm_response.tool_calls),
                usage_details={"input": llm_response.input_tokens, "output": llm_response.output_tokens},
            )
            generation.end()
        except Exception:
            pass

    if llm_response.tool_calls:
        delta.append({"role": "assistant", "content": [
            {"type": "tool_use", "id": tc["id"], "name": tc["name"], "input": tc["input"]}
            for tc in llm_response.tool_calls
        ]})
        search_call = next((tc for tc in llm_response.tool_calls if tc["name"] == "search_amazon"), None)
        return {
            "history": delta,
            "new_message": None,
            "made_tool_call_this_turn": True,
            "pending_tool_calls": llm_response.tool_calls,
            "last_search_input": search_call["input"] if search_call else state.get("last_search_input"),
            "trace_id": trace_id,
        }

    text = llm_response.text or ""
    delta.append({"role": "assistant", "content": [{"type": "text", "text": text}]})
    return {
        "history": delta,
        "new_message": None,
        "made_tool_call_this_turn": False,
        "pending_tool_calls": None,
        "trace_id": trace_id,
        "response": text,
    }
