import os
import uuid
from typing import Literal

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langgraph.types import Command

from agent import EvalScore
from graph import build_graph, GraphContext
from llm import ModelConfig, create_client
from main import PROVIDER_KEYS

load_dotenv()

app = FastAPI()

_graph = build_graph()
_sessions: dict[str, GraphContext] = {}


class SessionRequest(BaseModel):
    provider: Literal["google", "anthropic"]
    model: str


class SessionResponse(BaseModel):
    thread_id: str


class ChatRequest(BaseModel):
    thread_id: str
    message: str


class ResumeRequest(BaseModel):
    thread_id: str
    score: int
    note: str | None = None


@app.post("/session", response_model=SessionResponse)
def create_session(req: SessionRequest) -> SessionResponse:
    missing = [k for k in ["SERPAPI_KEY", PROVIDER_KEYS[req.provider]] if not os.environ.get(k)]
    if missing:
        raise HTTPException(status_code=400, detail=f"missing environment variables: {', '.join(missing)}")

    config = ModelConfig(provider=req.provider, model=req.model)
    thread_id = str(uuid.uuid4())
    _sessions[thread_id] = GraphContext(client=create_client(config), model_config=config)
    return SessionResponse(thread_id=thread_id)


def _graph_config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}


def _get_context(thread_id: str) -> GraphContext:
    context = _sessions.get(thread_id)
    if context is None:
        raise HTTPException(status_code=404, detail="unknown thread_id")
    return context


def _format_chat_result(result: dict) -> dict:
    made_tool_call = result.get("made_tool_call_this_turn", False)
    last_results = result.get("last_search_results") if made_tool_call else None
    last_input = result.get("last_search_input") or {}
    products = last_results["products"] if last_results else None
    view = last_input.get("view") if last_results else None

    if result.get("__interrupt__"):
        ctx = result["__interrupt__"][0].value
        return {
            "type": "eval_request",
            "query": ctx["query"],
            "optimize_for": ctx["optimize_for"],
            "recommendation": ctx["recommendation"],
            "products": products,
            "view": view,
        }
    return {
        "type": "message",
        "text": result.get("response") or "",
        "products": products,
        "view": view,
    }


@app.post("/chat")
def chat(req: ChatRequest) -> dict:
    context = _get_context(req.thread_id)
    try:
        result = _graph.invoke({"new_message": req.message}, config=_graph_config(req.thread_id), context=context)
    except Exception as e:
        return {"type": "error", "message": str(e)}
    return _format_chat_result(result)


@app.post("/resume")
def resume(req: ResumeRequest) -> dict:
    context = _get_context(req.thread_id)
    score = EvalScore(overall=req.score, note=req.note)
    try:
        _graph.invoke(Command(resume=score), config=_graph_config(req.thread_id), context=context)
    except Exception as e:
        return {"type": "error", "message": str(e)}
    return {"type": "message", "text": "Thanks for the rating!", "products": None, "view": None}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
