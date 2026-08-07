from __future__ import annotations
from dataclasses import asdict, dataclass
from typing import Annotated, Any, TypedDict
import json
import os

import agent
import llm
from agent import SYSTEM_PROMPT, TOOLS
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.graph import END, StateGraph, START
from langgraph.types import interrupt
from llm import ModelConfig

def add_to_history(existing: list, new: list) -> list:
    return existing + new


class GraphState(TypedDict):
    history: Annotated[list, add_to_history]
    new_message: str | None
    made_tool_call_this_turn: bool
    pending_tool_calls: list | None
    last_search_input: dict | None
    last_search_results: dict | None
    trace_id: str | None
    response: str | None
    judge_score: "dict | None"


@dataclass
class GraphContext:
    client: Any
    model_config: ModelConfig


def agent_node(state: GraphState, runtime) -> dict:
    client = runtime.context.client
    model_config = runtime.context.model_config

    delta = []
    trace_id = state.get("trace_id")
    made_tool_call_this_turn = state.get("made_tool_call_this_turn", False)
    last_search_input = state.get("last_search_input")

    if state.get("new_message"):
        delta.append({"role": "user", "content": state["new_message"]})
        made_tool_call_this_turn = False
        last_search_input = None
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
            {
                "type": "tool_use",
                "id": tc["id"],
                "name": tc["name"],
                "input": tc["input"],
                "thought_signature": tc.get("thought_signature"),
            }
            for tc in llm_response.tool_calls
        ]})
        search_call = next((tc for tc in llm_response.tool_calls if tc["name"] == "search_amazon"), None)
        return {
            "history": delta,
            "new_message": None,
            "made_tool_call_this_turn": True,
            "pending_tool_calls": llm_response.tool_calls,
            "last_search_input": search_call["input"] if search_call else last_search_input,
            "trace_id": trace_id,
        }

    text = llm_response.text or ""
    delta.append({"role": "assistant", "content": [{"type": "text", "text": text}]})
    return {
        "history": delta,
        "new_message": None,
        "made_tool_call_this_turn": made_tool_call_this_turn,
        "pending_tool_calls": None,
        "last_search_input": last_search_input,
        "trace_id": trace_id,
        "response": text,
    }


def tools_node(state: GraphState) -> dict:
    trace_id = state.get("trace_id")
    tool_results = []
    last_search_results = state.get("last_search_results")
    for tc in state["pending_tool_calls"]:
        result = agent.run_tool(tc["name"], tc["input"], trace_id=trace_id)
        content = result
        if tc["name"] == "search_amazon":
            last_search_results = json.loads(result)
            llm_view = {
                **last_search_results,
                "products": agent.strip_image_for_llm(last_search_results.get("products", [])),
            }
            content = json.dumps(llm_view)
        tool_results.append({"type": "tool_result", "tool_use_id": tc["id"], "content": content})
    return {
        "history": [{"role": "user", "content": tool_results}],
        "pending_tool_calls": None,
        "last_search_results": last_search_results,
    }


def route_after_agent(state: GraphState) -> str:
    if state.get("pending_tool_calls"):
        return "tools"
    if state.get("made_tool_call_this_turn"):
        return "judge"
    return END


def _eval_context(state: GraphState) -> dict:
    search_input = state.get("last_search_input") or {}
    return {
        "query": search_input.get("query", ""),
        "optimize_for": search_input.get("optimize_for", ""),
        "recommendation": state.get("response", ""),
    }


def judge_node(state: GraphState, runtime) -> dict:
    context = _eval_context(state)
    try:
        judge_score = agent.judge_recommendation(runtime.context.client, runtime.context.model_config, context)
    except Exception:
        judge_score = None
    if judge_score is None:
        print("[judge scoring failed after retry, continuing without judge score]")
    return {"judge_score": asdict(judge_score) if judge_score is not None else None}


def eval_node(state: GraphState) -> dict:
    context = _eval_context(state)
    human_score = interrupt(context)
    judge_dict = state.get("judge_score")
    judge_score = agent.JudgeScore(**judge_dict) if judge_dict is not None else None
    agent.record_score(state.get("trace_id"), context, human_score, judge_score)
    return {}


def build_graph():
    builder = StateGraph(GraphState, context_schema=GraphContext)
    builder.add_node("agent", agent_node)
    builder.add_node("tools", tools_node)
    builder.add_node("judge", judge_node)
    builder.add_node("eval", eval_node)
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", route_after_agent, {"tools": "tools", "judge": "judge", END: END})
    builder.add_edge("tools", "agent")
    builder.add_edge("judge", "eval")
    builder.add_edge("eval", END)
    database_url = os.environ.get("DATABASE_URL")
    if database_url:
        # Kept open for the process lifetime — this app builds one graph at
        # import time (see server.py/main.py) and reuses it for every request.
        # Store the context manager on the compiled graph to keep the connection
        # alive independently of other graph instances.
        pg_context = PostgresSaver.from_conn_string(database_url)
        checkpointer = pg_context.__enter__()
        checkpointer.setup()
    else:
        pg_context = None
        checkpointer = MemorySaver()
    compiled = builder.compile(checkpointer=checkpointer)
    # Keep the PostgresSaver context manager alive for this specific graph instance
    compiled._pg_context = pg_context
    return compiled
