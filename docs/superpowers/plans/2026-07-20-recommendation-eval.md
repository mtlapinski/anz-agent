# Recommendation Eval Scoring Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a human-in-the-loop evaluation step that scores whether the agent's Amazon recommendations are useful, restructuring `agent.py`'s conversation loop as a LangGraph `StateGraph` so the eval step can pause execution via `interrupt()`.

**Architecture:** Two phases. Phase 1 (Task 1) fixes a pre-existing bug — `agent.py`'s Langfuse tracing calls a v2/v3 API (`lf.trace()`) that doesn't exist in the installed v4 client, so it's been silently failing — fixed and verified *in place*, inside the current synchronous `chat()` loop, before any structural change. Phase 2 (Tasks 2-7) restructures `agent.py`'s loop as a 3-node LangGraph `StateGraph` (`agent` → `tools` → loop, or `agent` → `eval` when a recommendation is ready), porting the now-verified v4 tracing pattern into the new node functions. `eval` calls `interrupt()` to pause for a human 1-5 rating, then logs it to Langfuse and a local JSONL file. `llm.py` and `tools/amazon.py` are untouched — LangGraph is orchestration only.

**Tech Stack:** `langgraph` (new dependency, `StateGraph`/`interrupt`/`Command`/`MemorySaver`), existing `llm.py` provider adapter, `langfuse` v4 client (already a dependency; Task 1 fixes the code calling it).

## Global Constraints

- `langgraph>=1.0.0` in `requirements.txt`. Verified against the actually-installed `1.2.9` — do not use APIs from older `0.2.x`-era tutorials (e.g. `config["configurable"]` for custom dependency injection); this version uses `context_schema` + a node parameter literally named `runtime`.
- **The installed `langfuse` is `4.5.1`.** `requirements.txt`'s floor is `langfuse>=2.0.0`, but v4 removed the `Langfuse().trace(...)` / `trace.generation(...)` builder API entirely — calling it raises `AttributeError`. This is caught by existing try/except-and-continue code, so it fails silently rather than crashing. Task 1 fixes this using the real v4 API: `Langfuse().create_trace_id()`, `Langfuse().start_observation(trace_context={"trace_id": ...}, name=..., as_type=..., ...)`, `<span>.update(...)`, `<span>.end()`, `Langfuse().create_score(trace_id=..., name=..., value=..., comment=...)`. Every later task that touches tracing builds on this same verified pattern.
- LangGraph node functions needing injected runtime dependencies (the LLM client, `ModelConfig`) must name their second parameter exactly `runtime` — verified experimentally; a differently-named or type-annotated-only parameter does not get the context injected and raises `TypeError: missing 1 required positional argument`.
- Every `graph.invoke(...)` call (including the resume call after an interrupt) must pass the same `context=GraphContext(...)` — it is not persisted across calls automatically.
- All existing error-handling conventions are preserved: Langfuse calls are optional and wrapped in try/except-and-continue; nothing about tracing/scoring failing should crash a conversation turn.

---

## File Structure

```
anz-agent/
├── agent.py             # Task 1: chat()/run_tool() fixed to v4 Langfuse API.
│                         # Task 3: EvalScore, record_score() added.
│                         # Task 6: chat() removed (superseded by graph.py).
├── graph.py              # new (Tasks 4-6) — GraphContext, GraphState, node functions, build_graph()
├── main.py               # modified (Task 7) — invokes graph instead of chat(), handles interrupt/resume, prompt_for_score()
├── requirements.txt      # modified (Task 2) — add langgraph
├── .gitignore             # modified (Task 3) — add evals/
└── evals/                 # new dir, created at runtime by record_score(), not committed
tests/
├── test_agent.py         # Task 1: chat()/run_tool() Langfuse tests updated to v4.
│                         # Task 3: EvalScore/record_score tests added.
│                         # Task 6: obsolete chat() tests removed.
├── test_graph.py         # new (Tasks 4-6) — node-level and full-graph tests
└── test_main.py           # new (Task 7) — prompt_for_score() and handle_model_command() tests
```

---

### Task 1: Fix `agent.py`'s existing Langfuse integration to the real v4 API

**Files:**
- Modify: `agent.py` (`chat()`, `run_tool()`)
- Test: `tests/test_agent.py`

**Interfaces:**
- Modifies `chat(client, history, user_message, model_config) -> str` — same signature, internal tracing calls rewritten to v4.
- Modifies `run_tool(tool_name: str, tool_input: dict, trace=None)` → `run_tool(tool_name: str, tool_input: dict, trace_id: str | None = None)` — signature changes from passing a `trace` *object* (which no longer has a `.span()` method in v4) to a `trace_id` *string*, since v4's `start_observation(trace_context={"trace_id": ...})` is how you attach a new observation to an existing trace without holding a reference to the original handle.

This is a pure bug fix — no new behavior, no LangGraph yet. It exists so the v4 tracing pattern is proven correct in the current, simple synchronous loop before Task 4 ports the same pattern into `graph.py`'s `agent_node`.

- [ ] **Step 1: Write the failing tests**

In `tests/test_agent.py`, delete the existing `test_chat_logs_token_usage` test (it hardcodes the old, broken v2-style mock chain: `mock_lf.return_value.trace.return_value` / `generation.end(usage=...)` — these methods don't exist on the real v4 client, and after this fix `chat()` won't call them). Replace it and add these tests (the existing `test_run_tool_search_amazon`, `test_run_tool_unknown_raises`, `test_chat_text_response_updates_history`, `test_chat_tool_call_then_text`, `test_chat_passes_system_prompt` stay as-is — they don't assert internal Langfuse call shapes and continue to pass against a plain `MagicMock()`-based `_get_langfuse` patch):

```python
@patch("agent._get_langfuse")
def test_chat_uses_v4_langfuse_api(mock_lf):
    from agent import chat, SYSTEM_PROMPT
    # spec= restricts the mock to real v4 methods; calling a nonexistent method
    # (e.g. the old .trace()) would raise AttributeError and fail this test.
    mock_client = MagicMock(spec=["create_trace_id", "start_observation", "create_score", "flush"])
    mock_client.create_trace_id.return_value = "trace-xyz"
    mock_generation = MagicMock()
    mock_client.start_observation.return_value = mock_generation
    mock_lf.return_value = mock_client

    with patch("agent.llm") as mock_llm:
        mock_llm.complete.return_value = LLMResponse(
            text="hi", tool_calls=None, input_tokens=10, output_tokens=5
        )
        result = chat(MagicMock(), [], "hello", default_config())

    assert result == "hi"
    mock_client.create_trace_id.assert_called_once()
    mock_client.start_observation.assert_called_once_with(
        trace_context={"trace_id": "trace-xyz"},
        name="llm",
        as_type="generation",
        input={"system": SYSTEM_PROMPT, "messages": [{"role": "user", "content": "hello"}]},
        model="anthropic/claude-haiku-4-5-20251001",
    )
    mock_generation.update.assert_called_once_with(
        output="hi", usage_details={"input": 10, "output": 5}
    )
    mock_generation.end.assert_called_once()


@patch("agent._get_langfuse")
def test_chat_tool_call_passes_trace_id_to_run_tool(mock_lf):
    from agent import chat
    mock_client = MagicMock(spec=["create_trace_id", "start_observation", "create_score", "flush"])
    mock_client.create_trace_id.return_value = "trace-xyz"
    mock_client.start_observation.return_value = MagicMock()
    mock_lf.return_value = mock_client

    with patch("agent.llm") as mock_llm, patch("agent.run_tool") as mock_run_tool:
        mock_llm.complete.side_effect = [
            LLMResponse(text=None, tool_calls=[{"name": "search_amazon", "id": "tu_1",
                "input": {"query": "laptop", "optimize_for": "price", "max_results": 5}}],
                input_tokens=80, output_tokens=20),
            LLMResponse(text="Here are the top laptops...", tool_calls=None, input_tokens=200, output_tokens=50),
        ]
        mock_run_tool.return_value = json.dumps({"products": []})
        chat(MagicMock(), [], "Find me a laptop", default_config())

    mock_run_tool.assert_called_once_with(
        "search_amazon", {"query": "laptop", "optimize_for": "price", "max_results": 5}, trace_id="trace-xyz"
    )


@patch("agent._get_langfuse")
def test_run_tool_creates_langfuse_span_with_trace_id(mock_lf):
    from agent import run_tool
    mock_client = MagicMock(spec=["create_trace_id", "start_observation", "create_score", "flush"])
    mock_span = MagicMock()
    mock_client.start_observation.return_value = mock_span
    mock_lf.return_value = mock_client

    with patch("agent.search_amazon") as mock_search:
        mock_search.return_value = {"products": []}
        run_tool("search_amazon", {"query": "laptop", "optimize_for": "price", "max_results": 5},
                  trace_id="trace-123")

    mock_client.start_observation.assert_called_once_with(
        trace_context={"trace_id": "trace-123"},
        name="search_amazon",
        as_type="tool",
        input={"query": "laptop", "optimize_for": "price", "max_results": 5},
    )
    mock_span.update.assert_called_once_with(output={"products": []})
    mock_span.end.assert_called_once()


@patch("agent._get_langfuse")
def test_run_tool_no_span_when_trace_id_none(mock_lf):
    from agent import run_tool
    mock_client = MagicMock(spec=["create_trace_id", "start_observation", "create_score", "flush"])
    mock_lf.return_value = mock_client

    with patch("agent.search_amazon") as mock_search:
        mock_search.return_value = {"products": []}
        run_tool("search_amazon", {"query": "laptop", "optimize_for": "price", "max_results": 5})

    mock_client.start_observation.assert_not_called()


@patch("agent._get_langfuse")
def test_run_tool_continues_when_langfuse_raises(mock_lf):
    from agent import run_tool
    mock_lf.side_effect = Exception("langfuse down")

    with patch("agent.search_amazon") as mock_search:
        mock_search.return_value = {"products": []}
        result = run_tool("search_amazon", {"query": "laptop", "optimize_for": "price", "max_results": 5},
                            trace_id="trace-123")

    assert json.loads(result) == {"products": []}


@patch("agent._get_langfuse")
def test_chat_continues_when_langfuse_raises(mock_lf):
    from agent import chat
    mock_lf.side_effect = Exception("langfuse down")

    with patch("agent.llm") as mock_llm:
        mock_llm.complete.return_value = LLMResponse(
            text="hi", tool_calls=None, input_tokens=10, output_tokens=5
        )
        result = chat(MagicMock(), [], "hello", default_config())

    assert result == "hi"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
source .venv/bin/activate && pytest tests/test_agent.py -v
```
Expected: the new tests FAIL — `test_chat_uses_v4_langfuse_api` and `test_run_tool_creates_langfuse_span_with_trace_id` fail because current code calls `lf.trace(...)` / `trace.span(...)`, which don't exist on the `spec=[...]`-restricted mocks (`AttributeError`), and `test_chat_tool_call_passes_trace_id_to_run_tool` fails because `run_tool` is currently called with `trace=` not `trace_id=`.

- [ ] **Step 3: Fix `chat()` in `agent.py`**

Replace the existing `chat()` function:
```python
def chat(client, history: list, user_message: str, model_config: ModelConfig) -> str:
    try:
        trace_id = _get_langfuse().create_trace_id()
    except Exception:
        trace_id = None

    history.append({"role": "user", "content": user_message})

    while True:
        generation = None
        if trace_id:
            try:
                generation = _get_langfuse().start_observation(
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
                    usage_details={
                        "input": llm_response.input_tokens,
                        "output": llm_response.output_tokens,
                    },
                )
                generation.end()
            except Exception:
                pass

        if llm_response.tool_calls:
            tool_results = []
            for tc in llm_response.tool_calls:
                result = run_tool(tc["name"], tc["input"], trace_id=trace_id)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tc["id"],
                    "content": result,
                })
            history.append({"role": "assistant", "content": [
                {"type": "tool_use", "id": tc["id"], "name": tc["name"], "input": tc["input"]}
                for tc in llm_response.tool_calls
            ]})
            history.append({"role": "user", "content": tool_results})
        else:
            text = llm_response.text or ""
            history.append({"role": "assistant", "content": [{"type": "text", "text": text}]})
            return text
```

- [ ] **Step 4: Fix `run_tool()` in `agent.py`**

Replace the existing `run_tool()` function:
```python
def run_tool(tool_name: str, tool_input: dict, trace_id: str | None = None) -> str:
    if tool_name == "search_amazon":
        span = None
        if trace_id:
            try:
                span = _get_langfuse().start_observation(
                    trace_context={"trace_id": trace_id},
                    name="search_amazon",
                    as_type="tool",
                    input=tool_input,
                )
            except Exception:
                span = None
        result = search_amazon(**tool_input)
        if span:
            try:
                span.update(output=result)
                span.end()
            except Exception:
                pass
        return json.dumps(result)
    raise ValueError(f"Unknown tool: {tool_name}")
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
source .venv/bin/activate && pytest tests/test_agent.py -v
```
Expected: all PASS.

- [ ] **Step 6: Verify the fix manually against the real Langfuse client class**

This confirms the fix works against the actual installed v4 client, not just against mocks:
```bash
source .venv/bin/activate && python3 -c "
from langfuse import Langfuse
lf = Langfuse(public_key='pk-lf-test', secret_key='sk-lf-test')
trace_id = lf.create_trace_id()
gen = lf.start_observation(trace_context={'trace_id': trace_id}, name='llm', as_type='generation', input={'x': 1}, model='anthropic/claude-haiku-4-5')
gen.update(output='hello', usage_details={'input': 10, 'output': 5})
gen.end()
lf.create_score(trace_id=trace_id, name='usefulness', value=4, comment='good')
print('v4 API calls succeeded, no AttributeError')
"
```
Expected: `v4 API calls succeeded, no AttributeError` (a `Failed to export span batch` warning printed to stderr is expected and harmless — that's the background exporter failing to reach Langfuse's servers with fake test credentials, not a code error).

- [ ] **Step 7: Commit**

```bash
git add agent.py tests/test_agent.py
git commit -m "fix: migrate agent.py Langfuse tracing from broken v2 API to real v4 API"
```

---

### Task 2: Add and verify the `langgraph` dependency

**Files:**
- Modify: `requirements.txt`

**Interfaces:**
- Produces: `langgraph` importable in the venv for all later tasks.

- [ ] **Step 1: Add the dependency**

Add this line to `requirements.txt` (after `langfuse>=2.0.0`):
```
langgraph>=1.0.0
```

- [ ] **Step 2: Install and verify**

```bash
source .venv/bin/activate && pip install -r requirements.txt
python3 -c "from langgraph.graph import StateGraph, END, START; from langgraph.types import interrupt, Command; from langgraph.checkpoint.memory import MemorySaver; from langgraph.runtime import Runtime; print('ok')"
```
Expected: `ok` printed, no `ImportError`.

- [ ] **Step 3: Commit**

```bash
git add requirements.txt
git commit -m "chore: add langgraph dependency"
```

---

### Task 3: `EvalScore` dataclass and `record_score()` in `agent.py`

**Files:**
- Modify: `agent.py`
- Modify: `.gitignore`
- Test: `tests/test_agent.py`

**Interfaces:**
- Consumes: `agent._get_langfuse()` (fixed in Task 1 — `create_score(trace_id=..., name=..., value=..., comment=...)` is real v4 API, verified in Task 1 Step 6)
- Produces:
  - `EvalScore` dataclass: `overall: int | None`, `note: str | None`, `criteria: dict[str, int] | None = None`
  - `record_score(trace_id: str | None, context: dict, score: EvalScore | None) -> None` — `context` is `{"query": str, "optimize_for": str, "recommendation": str}`. Writes to Langfuse (if `trace_id` given) and appends a JSONL line to `evals/scores.jsonl`. If `score is None` (user skipped rating), does nothing (no Langfuse call, no JSONL row).

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_agent.py` (new imports at top of file: `import os`):

```python
def test_eval_score_defaults():
    from agent import EvalScore
    score = EvalScore(overall=4, note="good pick")
    assert score.overall == 4
    assert score.note == "good pick"
    assert score.criteria is None


@patch("agent._get_langfuse")
def test_record_score_writes_jsonl(mock_lf, tmp_path, monkeypatch):
    from agent import record_score, EvalScore
    monkeypatch.chdir(tmp_path)
    score = EvalScore(overall=4, note="good pick")
    context = {"query": "laptop", "optimize_for": "price", "recommendation": "Here are 3 laptops..."}

    record_score("trace-123", context, score)

    jsonl_path = tmp_path / "evals" / "scores.jsonl"
    assert jsonl_path.exists()
    lines = jsonl_path.read_text().strip().split("\n")
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert row["query"] == "laptop"
    assert row["optimize_for"] == "price"
    assert row["recommendation"] == "Here are 3 laptops..."
    assert row["overall"] == 4
    assert row["note"] == "good pick"
    assert "timestamp" in row


@patch("agent._get_langfuse")
def test_record_score_calls_langfuse_create_score(mock_lf, tmp_path, monkeypatch):
    from agent import record_score, EvalScore
    monkeypatch.chdir(tmp_path)
    mock_client = MagicMock()
    mock_lf.return_value = mock_client
    score = EvalScore(overall=5, note=None)
    context = {"query": "mouse", "optimize_for": "quality", "recommendation": "The X mouse..."}

    record_score("trace-abc", context, score)

    mock_client.create_score.assert_called_once_with(
        trace_id="trace-abc", name="usefulness", value=5, comment=None
    )


@patch("agent._get_langfuse")
def test_record_score_skips_when_score_is_none(mock_lf, tmp_path, monkeypatch):
    from agent import record_score
    monkeypatch.chdir(tmp_path)
    mock_client = MagicMock()
    mock_lf.return_value = mock_client
    context = {"query": "mouse", "optimize_for": "quality", "recommendation": "The X mouse..."}

    record_score("trace-abc", context, None)

    mock_client.create_score.assert_not_called()
    assert not (tmp_path / "evals" / "scores.jsonl").exists()


@patch("agent._get_langfuse")
def test_record_score_skips_langfuse_when_trace_id_none(mock_lf, tmp_path, monkeypatch):
    from agent import record_score, EvalScore
    monkeypatch.chdir(tmp_path)
    mock_client = MagicMock()
    mock_lf.return_value = mock_client
    score = EvalScore(overall=3, note=None)

    record_score(None, {"query": "q", "optimize_for": "price", "recommendation": "r"}, score)

    mock_client.create_score.assert_not_called()
    assert (tmp_path / "evals" / "scores.jsonl").exists()


@patch("agent._get_langfuse")
def test_record_score_continues_when_langfuse_raises(mock_lf, tmp_path, monkeypatch):
    from agent import record_score, EvalScore
    monkeypatch.chdir(tmp_path)
    mock_lf.side_effect = Exception("langfuse down")
    score = EvalScore(overall=2, note=None)

    record_score("trace-x", {"query": "q", "optimize_for": "price", "recommendation": "r"}, score)

    assert (tmp_path / "evals" / "scores.jsonl").exists()


@patch("agent._get_langfuse")
def test_record_score_continues_when_jsonl_write_fails(mock_lf, tmp_path, monkeypatch):
    from agent import record_score, EvalScore
    monkeypatch.chdir(tmp_path)
    (tmp_path / "evals").mkdir()
    (tmp_path / "evals" / "scores.jsonl").mkdir()  # a directory where the file should go -> open() fails
    score = EvalScore(overall=2, note=None)

    record_score(None, {"query": "q", "optimize_for": "price", "recommendation": "r"}, score)  # should not raise
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
source .venv/bin/activate && pytest tests/test_agent.py -k "eval_score or record_score" -v
```
Expected: FAIL with `ImportError: cannot import name 'EvalScore'` or `'record_score'`.

- [ ] **Step 3: Implement `EvalScore` and `record_score()`**

Add to `agent.py`, near the top (after imports, before `SYSTEM_PROMPT`):
```python
from dataclasses import dataclass
from datetime import datetime, timezone
```

Add near the bottom of `agent.py`:
```python
@dataclass
class EvalScore:
    overall: int | None
    note: str | None
    criteria: dict[str, int] | None = None


def record_score(trace_id: str | None, context: dict, score: "EvalScore | None") -> None:
    if score is None:
        return

    if trace_id:
        try:
            _get_langfuse().create_score(
                trace_id=trace_id,
                name="usefulness",
                value=score.overall,
                comment=score.note,
            )
        except Exception:
            pass

    try:
        os.makedirs("evals", exist_ok=True)
        row = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "query": context.get("query", ""),
            "optimize_for": context.get("optimize_for", ""),
            "recommendation": context.get("recommendation", ""),
            "overall": score.overall,
            "note": score.note,
        }
        with open("evals/scores.jsonl", "a") as f:
            f.write(json.dumps(row) + "\n")
    except Exception as e:
        print(f"Warning: failed to write eval score: {e}")
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
source .venv/bin/activate && pytest tests/test_agent.py -k "eval_score or record_score" -v
```
Expected: all PASS.

- [ ] **Step 5: Add `evals/` to `.gitignore`**

Append to `.gitignore`:
```
evals/
```

- [ ] **Step 6: Run the full test suite**

```bash
source .venv/bin/activate && pytest tests/ -v
```
Expected: all PASS.

- [ ] **Step 7: Commit**

```bash
git add agent.py .gitignore tests/test_agent.py
git commit -m "feat: add EvalScore and record_score() with Langfuse v4 + JSONL storage"
```

---

### Task 4: `graph.py` — `GraphContext`, `GraphState`, `agent_node`

**Files:**
- Create: `graph.py`
- Test: `tests/test_graph.py`

**Interfaces:**
- Consumes: `llm.complete()` (existing), `agent.SYSTEM_PROMPT`, `agent.TOOLS`, `agent._get_langfuse()` (fixed in Task 1)
- Produces:
  - `GraphContext` dataclass: `client: Any`, `model_config: ModelConfig`
  - `GraphState` TypedDict: `history: Annotated[list, add_to_history]`, `new_message: str | None`, `made_tool_call_this_turn: bool`, `pending_tool_calls: list | None`, `last_search_input: dict | None`, `trace_id: str | None`, `response: str | None`
  - `agent_node(state: GraphState, runtime) -> dict` — this ports the exact v4 tracing pattern verified in Task 1's `chat()` fix into the new node.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_graph.py`:
```python
import json
from unittest.mock import MagicMock, patch
import pytest
from llm import ModelConfig, LLMResponse


def default_config():
    return ModelConfig(provider="anthropic", model="claude-haiku-4-5-20251001")


def empty_state(**overrides):
    base = {
        "history": [],
        "new_message": None,
        "made_tool_call_this_turn": False,
        "pending_tool_calls": None,
        "last_search_input": None,
        "trace_id": None,
        "response": None,
    }
    base.update(overrides)
    return base


class FakeRuntime:
    def __init__(self, context):
        self.context = context


@patch("graph.agent._get_langfuse")
@patch("graph.llm")
def test_agent_node_text_response_new_turn(mock_llm, mock_lf):
    from graph import agent_node, GraphContext
    mock_llm.complete.return_value = LLMResponse(
        text="What are you looking for?", tool_calls=None, input_tokens=10, output_tokens=5
    )
    mock_lf.return_value.create_trace_id.return_value = "trace-1"
    runtime = FakeRuntime(GraphContext(client=MagicMock(), model_config=default_config()))

    result = agent_node(empty_state(new_message="Hello"), runtime)

    assert result["response"] == "What are you looking for?"
    assert result["new_message"] is None
    assert result["made_tool_call_this_turn"] is False
    assert result["pending_tool_calls"] is None
    assert result["trace_id"] == "trace-1"
    assert result["history"][0] == {"role": "user", "content": "Hello"}
    assert result["history"][1]["role"] == "assistant"
    assert result["history"][1]["content"] == [{"type": "text", "text": "What are you looking for?"}]


@patch("graph.agent._get_langfuse")
@patch("graph.llm")
def test_agent_node_tool_call_response(mock_llm, mock_lf):
    from graph import agent_node, GraphContext
    mock_llm.complete.return_value = LLMResponse(
        text=None,
        tool_calls=[{"name": "search_amazon", "id": "tu_1",
                      "input": {"query": "laptop", "optimize_for": "price", "max_results": 5}}],
        input_tokens=80, output_tokens=20,
    )
    mock_lf.return_value.create_trace_id.return_value = "trace-2"
    runtime = FakeRuntime(GraphContext(client=MagicMock(), model_config=default_config()))

    result = agent_node(empty_state(new_message="Find me a laptop"), runtime)

    assert result["made_tool_call_this_turn"] is True
    assert result["pending_tool_calls"] == [{"name": "search_amazon", "id": "tu_1",
        "input": {"query": "laptop", "optimize_for": "price", "max_results": 5}}]
    assert result["last_search_input"] == {"query": "laptop", "optimize_for": "price", "max_results": 5}
    assert result["history"][-1]["content"][0]["type"] == "tool_use"
    assert "response" not in result


@patch("graph.agent._get_langfuse")
@patch("graph.llm")
def test_agent_node_continues_turn_without_new_message(mock_llm, mock_lf):
    from graph import agent_node, GraphContext
    mock_llm.complete.return_value = LLMResponse(
        text="Here are the top laptops...", tool_calls=None, input_tokens=200, output_tokens=50
    )
    runtime = FakeRuntime(GraphContext(client=MagicMock(), model_config=default_config()))
    prior_history = [
        {"role": "user", "content": "Find me a laptop"},
        {"role": "assistant", "content": [{"type": "tool_use", "id": "tu_1", "name": "search_amazon",
                                            "input": {"query": "laptop", "optimize_for": "price", "max_results": 5}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "tu_1", "content": "{}"}]},
    ]

    result = agent_node(
        empty_state(history=prior_history, new_message=None, made_tool_call_this_turn=True, trace_id="trace-2"),
        runtime,
    )

    assert result["response"] == "Here are the top laptops..."
    assert result["made_tool_call_this_turn"] is False  # must reset — no tool call this call
    assert result["pending_tool_calls"] is None
    assert result["trace_id"] == "trace-2"  # reused, not regenerated
    mock_lf.return_value.create_trace_id.assert_not_called()


@patch("graph.agent._get_langfuse")
@patch("graph.llm")
def test_agent_node_logs_token_usage(mock_llm, mock_lf, capsys):
    from graph import agent_node, GraphContext
    mock_llm.complete.return_value = LLMResponse(
        text="hi", tool_calls=None, input_tokens=100, output_tokens=50
    )
    runtime = FakeRuntime(GraphContext(client=MagicMock(), model_config=default_config()))

    agent_node(empty_state(new_message="hi"), runtime)

    captured = capsys.readouterr()
    assert "[tokens: 100 in / 50 out]" in captured.out
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
source .venv/bin/activate && pytest tests/test_graph.py -v
```
Expected: FAIL with `ModuleNotFoundError: No module named 'graph'`.

- [ ] **Step 3: Create `graph.py`**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
source .venv/bin/activate && pytest tests/test_graph.py -v
```
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add graph.py tests/test_graph.py
git commit -m "feat: add graph.py with GraphContext, GraphState, agent_node"
```

---

### Task 5: `graph.py` — `tools_node`, `route_after_agent`

**Files:**
- Modify: `graph.py`
- Test: `tests/test_graph.py`

**Interfaces:**
- Consumes: `agent.run_tool(tool_name, tool_input, trace_id=None)` (fixed in Task 1)
- Produces: `tools_node(state: GraphState) -> dict`, `route_after_agent(state: GraphState) -> str` (returns `"tools"`, `"eval"`, or `"__end__"`)

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_graph.py`:
```python
@patch("graph.agent.run_tool")
def test_tools_node_executes_and_appends_results(mock_run_tool):
    from graph import tools_node
    mock_run_tool.return_value = json.dumps({"products": []})
    state = empty_state(
        pending_tool_calls=[{"name": "search_amazon", "id": "tu_1",
                               "input": {"query": "laptop", "optimize_for": "price", "max_results": 5}}],
        trace_id="trace-1",
    )

    result = tools_node(state)

    mock_run_tool.assert_called_once_with(
        "search_amazon", {"query": "laptop", "optimize_for": "price", "max_results": 5}, trace_id="trace-1"
    )
    assert result["pending_tool_calls"] is None
    tool_result_msg = result["history"][0]
    assert tool_result_msg["role"] == "user"
    assert tool_result_msg["content"][0] == {
        "type": "tool_result", "tool_use_id": "tu_1", "content": json.dumps({"products": []})
    }


def test_route_after_agent_to_tools_when_pending():
    from graph import route_after_agent
    state = empty_state(pending_tool_calls=[{"name": "search_amazon", "id": "tu_1", "input": {}}])
    assert route_after_agent(state) == "tools"


def test_route_after_agent_to_eval_when_recommendation_made():
    from graph import route_after_agent
    state = empty_state(pending_tool_calls=None, made_tool_call_this_turn=True, response="Here are...")
    assert route_after_agent(state) == "eval"


def test_route_after_agent_to_end_when_no_tool_call():
    from langgraph.graph import END
    from graph import route_after_agent
    state = empty_state(pending_tool_calls=None, made_tool_call_this_turn=False, response="What are you looking for?")
    assert route_after_agent(state) == END
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
source .venv/bin/activate && pytest tests/test_graph.py -k "tools_node or route_after_agent" -v
```
Expected: FAIL with `ImportError: cannot import name 'tools_node'` (or `route_after_agent`).

- [ ] **Step 3: Add `tools_node` and `route_after_agent` to `graph.py`**

Add near the top of `graph.py`, alongside the other imports:
```python
from langgraph.graph import END
```

Add after `agent_node`:
```python
def tools_node(state: GraphState) -> dict:
    trace_id = state.get("trace_id")
    tool_results = []
    for tc in state["pending_tool_calls"]:
        result = agent.run_tool(tc["name"], tc["input"], trace_id=trace_id)
        tool_results.append({"type": "tool_result", "tool_use_id": tc["id"], "content": result})
    return {"history": [{"role": "user", "content": tool_results}], "pending_tool_calls": None}


def route_after_agent(state: GraphState) -> str:
    if state.get("pending_tool_calls"):
        return "tools"
    if state.get("made_tool_call_this_turn"):
        return "eval"
    return END
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
source .venv/bin/activate && pytest tests/test_graph.py -k "tools_node or route_after_agent" -v
```
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add graph.py tests/test_graph.py
git commit -m "feat: add tools_node and route_after_agent to graph.py"
```

---

### Task 6: `graph.py` — `eval_node`, `build_graph()`; remove `agent.chat()`

**Files:**
- Modify: `graph.py`
- Modify: `agent.py` (remove `chat()`)
- Modify: `tests/test_agent.py` (remove obsolete `chat()` tests)
- Test: `tests/test_graph.py`

**Interfaces:**
- Consumes: `agent.record_score()` (Task 3)
- Produces: `eval_node(state: GraphState) -> dict`, `build_graph() -> CompiledStateGraph` (compiled with `MemorySaver`, nodes `"agent"`, `"tools"`, `"eval"`)

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_graph.py`:
```python
@patch("graph.agent.record_score")
def test_eval_node_interrupts_then_records_score(mock_record_score):
    from graph import eval_node
    from unittest.mock import patch as mock_patch

    state = empty_state(
        last_search_input={"query": "laptop", "optimize_for": "price", "max_results": 5},
        response="Here are the top laptops...",
        trace_id="trace-1",
    )

    with mock_patch("graph.interrupt", return_value="fake-score") as mock_interrupt:
        result = eval_node(state)

    mock_interrupt.assert_called_once_with({
        "query": "laptop", "optimize_for": "price", "recommendation": "Here are the top laptops...",
    })
    mock_record_score.assert_called_once_with(
        "trace-1",
        {"query": "laptop", "optimize_for": "price", "recommendation": "Here are the top laptops..."},
        "fake-score",
    )
    assert result == {}


@patch("graph.agent._get_langfuse")
@patch("graph.llm")
def test_build_graph_full_flow_with_recommendation(mock_llm, mock_lf):
    from graph import build_graph, GraphContext
    from langgraph.types import Command

    mock_llm.complete.side_effect = [
        LLMResponse(text=None, tool_calls=[{"name": "search_amazon", "id": "tu_1",
            "input": {"query": "laptop", "optimize_for": "price", "max_results": 5}}],
            input_tokens=80, output_tokens=20),
        LLMResponse(text="Here are the top laptops...", tool_calls=None, input_tokens=200, output_tokens=50),
    ]
    mock_lf.return_value.create_trace_id.return_value = "trace-1"

    graph = build_graph()
    config = {"configurable": {"thread_id": "t-full-flow"}}
    context = GraphContext(client=MagicMock(), model_config=default_config())

    with patch("graph.agent.run_tool") as mock_run_tool, patch("graph.agent.record_score") as mock_record_score:
        mock_run_tool.return_value = json.dumps({"products": []})
        result = graph.invoke({"new_message": "Find me a laptop"}, config=config, context=context)

        assert "__interrupt__" in result
        interrupt_ctx = result["__interrupt__"][0].value
        assert interrupt_ctx["query"] == "laptop"
        assert interrupt_ctx["recommendation"] == "Here are the top laptops..."

        from agent import EvalScore
        score = EvalScore(overall=5, note="great")
        final = graph.invoke(Command(resume=score), config=config, context=context)

    assert final["response"] == "Here are the top laptops..."
    mock_record_score.assert_called_once()
    assert mock_record_score.call_args.args[2].overall == 5


@patch("graph.agent._get_langfuse")
@patch("graph.llm")
def test_build_graph_skips_eval_when_no_tool_call(mock_llm, mock_lf):
    from graph import build_graph, GraphContext

    mock_llm.complete.return_value = LLMResponse(
        text="What are you looking for?", tool_calls=None, input_tokens=10, output_tokens=5
    )
    mock_lf.return_value.create_trace_id.return_value = "trace-2"

    graph = build_graph()
    config = {"configurable": {"thread_id": "t-no-tool-call"}}
    context = GraphContext(client=MagicMock(), model_config=default_config())

    result = graph.invoke({"new_message": "Find me something"}, config=config, context=context)

    assert "__interrupt__" not in result
    assert result["response"] == "What are you looking for?"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
source .venv/bin/activate && pytest tests/test_graph.py -k "eval_node or build_graph" -v
```
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Add `eval_node` and `build_graph()` to `graph.py`**

Add to the imports at the top of `graph.py`:
```python
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START
from langgraph.types import interrupt
```

Add at the end of `graph.py`:
```python
def eval_node(state: GraphState) -> dict:
    search_input = state.get("last_search_input") or {}
    context = {
        "query": search_input.get("query", ""),
        "optimize_for": search_input.get("optimize_for", ""),
        "recommendation": state.get("response", ""),
    }
    score = interrupt(context)
    agent.record_score(state.get("trace_id"), context, score)
    return {}


def build_graph():
    builder = StateGraph(GraphState, context_schema=GraphContext)
    builder.add_node("agent", agent_node)
    builder.add_node("tools", tools_node)
    builder.add_node("eval", eval_node)
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", route_after_agent, {"tools": "tools", "eval": "eval", END: END})
    builder.add_edge("tools", "agent")
    builder.add_edge("eval", END)
    return builder.compile(checkpointer=MemorySaver())
```

- [ ] **Step 4: Run new tests to verify they pass**

```bash
source .venv/bin/activate && pytest tests/test_graph.py -v
```
Expected: all PASS.

- [ ] **Step 5: Remove `agent.chat()` and its obsolete tests**

In `agent.py`, delete the entire `chat()` function (Task 1 fixed it, and it's now fully superseded by `agent_node` + `tools_node` in `graph.py`).

In `tests/test_agent.py`, delete these now-obsolete tests (they test the removed `chat()` function): `test_chat_text_response_updates_history`, `test_chat_tool_call_then_text`, `test_chat_passes_system_prompt`, `test_chat_uses_v4_langfuse_api`, `test_chat_tool_call_passes_trace_id_to_run_tool`, `test_chat_continues_when_langfuse_raises`.

- [ ] **Step 6: Run the full test suite**

```bash
source .venv/bin/activate && pytest tests/ -v
```
Expected: all PASS, no leftover references to `agent.chat`.

- [ ] **Step 7: Commit**

```bash
git add graph.py agent.py tests/test_graph.py tests/test_agent.py
git commit -m "feat: add eval_node and build_graph(); remove superseded agent.chat()"
```

---

### Task 7: `main.py` — invoke the graph, handle the interrupt, rework `/model`

**Files:**
- Modify: `main.py`
- Test: `tests/test_main.py`

**Interfaces:**
- Consumes: `graph.build_graph()`, `graph.GraphContext` (Task 6), `agent.EvalScore` (Task 3)
- Produces: `prompt_for_score() -> EvalScore | None`, `handle_model_command(args, current_config, graph, thread_id) -> tuple[ModelConfig, str]` (signature changed: takes `graph` and `thread_id` instead of `history`)

- [ ] **Step 1: Write the failing tests**

Create `tests/test_main.py`:
```python
import builtins
from unittest.mock import MagicMock, patch
import pytest
from llm import ModelConfig


def default_config():
    return ModelConfig(provider="anthropic", model="claude-haiku-4-5-20251001")


def test_prompt_for_score_valid_input(monkeypatch):
    from main import prompt_for_score
    inputs = iter(["4", "worked well"])
    monkeypatch.setattr(builtins, "input", lambda: next(inputs))

    score = prompt_for_score()

    assert score.overall == 4
    assert score.note == "worked well"


def test_prompt_for_score_reprompts_on_invalid(monkeypatch, capsys):
    from main import prompt_for_score
    inputs = iter(["9", "not a number", "3", ""])
    monkeypatch.setattr(builtins, "input", lambda: next(inputs))

    score = prompt_for_score()

    assert score.overall == 3
    assert score.note is None
    assert "Please enter a number from 1 to 5." in capsys.readouterr().out


def test_prompt_for_score_ctrl_c_returns_none(monkeypatch):
    from main import prompt_for_score

    def raise_interrupt():
        raise KeyboardInterrupt()
    monkeypatch.setattr(builtins, "input", lambda: raise_interrupt())

    assert prompt_for_score() is None


@patch.dict("os.environ", {"ANTHROPIC_API_KEY": "fake"})
def test_handle_model_command_direct_args_no_history_no_prompt(monkeypatch):
    from main import handle_model_command
    mock_graph = MagicMock()
    mock_graph.get_state.return_value.values = {}

    config, thread_id = handle_model_command("anthropic claude-haiku-4-5-20251001", default_config(),
                                               mock_graph, "thread-1")

    assert config.provider == "anthropic"
    assert config.model == "claude-haiku-4-5-20251001"
    assert thread_id == "thread-1"  # no history -> no "start fresh" prompt, thread unchanged


@patch.dict("os.environ", {"ANTHROPIC_API_KEY": "fake"})
def test_handle_model_command_start_fresh_mints_new_thread_id(monkeypatch):
    from main import handle_model_command
    mock_graph = MagicMock()
    mock_graph.get_state.return_value.values = {"history": [{"role": "user", "content": "hi"}]}
    monkeypatch.setattr(builtins, "input", lambda: "y")

    config, thread_id = handle_model_command("anthropic claude-haiku-4-5-20251001", default_config(),
                                               mock_graph, "thread-1")

    assert thread_id != "thread-1"


@patch.dict("os.environ", {"ANTHROPIC_API_KEY": "fake"})
def test_handle_model_command_keep_history_same_thread(monkeypatch):
    from main import handle_model_command
    mock_graph = MagicMock()
    mock_graph.get_state.return_value.values = {"history": [{"role": "user", "content": "hi"}]}
    monkeypatch.setattr(builtins, "input", lambda: "n")

    config, thread_id = handle_model_command("anthropic claude-haiku-4-5-20251001", default_config(),
                                               mock_graph, "thread-1")

    assert thread_id == "thread-1"


def test_handle_model_command_missing_key_keeps_current(monkeypatch):
    from main import handle_model_command
    mock_graph = MagicMock()
    with patch.dict("os.environ", {}, clear=True):
        config, thread_id = handle_model_command("anthropic some-model", default_config(),
                                                   mock_graph, "thread-1")

    assert config == default_config()
    assert thread_id == "thread-1"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
source .venv/bin/activate && pytest tests/test_main.py -v
```
Expected: FAIL — `prompt_for_score` doesn't exist yet, `handle_model_command` has the old `history`-based signature.

- [ ] **Step 3: Rewrite `main.py`**

Replace the full contents of `main.py`:
```python
import os
import sys
import uuid
from dotenv import load_dotenv
from langgraph.types import Command
from llm import ModelConfig, create_client
from agent import EvalScore, _get_langfuse
from graph import build_graph, GraphContext

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


def prompt_for_score() -> EvalScore | None:
    while True:
        print("Rate usefulness (1-5): ", end="", flush=True)
        try:
            raw = input().strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return None
        if raw in ("1", "2", "3", "4", "5"):
            break
        print("Please enter a number from 1 to 5.")
    print("Note (optional): ", end="", flush=True)
    try:
        note = input().strip()
    except (EOFError, KeyboardInterrupt):
        note = ""
    return EvalScore(overall=int(raw), note=note or None)


def _has_tool_turns(history: list) -> bool:
    return any(
        isinstance(msg.get("content"), list) and
        any(b.get("type") in ("tool_use", "tool_result") for b in msg["content"])
        for msg in history
    )


def handle_model_command(args: str, current_config: ModelConfig, graph, thread_id: str) -> tuple[ModelConfig, str]:
    """Handle /model command. Returns (config, thread_id) — unchanged on error or cancel."""
    if args:
        parts = args.strip().split(None, 1)
        if len(parts) != 2:
            print("Usage: /model <provider> <model>  (e.g. /model google gemini-2.0-flash-lite)")
            return current_config, thread_id
        provider, model = parts
        if provider not in PROVIDER_KEYS:
            print(f"Unknown provider {provider!r}. Choose 'google' or 'anthropic'.")
            return current_config, thread_id
        new_config = ModelConfig(provider=provider, model=model)
    else:
        new_config = select_model()

    key = PROVIDER_KEYS[new_config.provider]
    if not os.environ.get(key):
        print(f"Error: {key} is not set. Keeping current model.")
        return current_config, thread_id

    state_values = graph.get_state({"configurable": {"thread_id": thread_id}}).values
    history = state_values.get("history", []) if state_values else []

    if history:
        if _has_tool_turns(history):
            print("Warning: history contains tool call turns that may not transfer cleanly.")
        print("Start fresh conversation? [y/N] : ", end="", flush=True)
        if input().strip().lower() == "y":
            thread_id = str(uuid.uuid4())

    print(f"Using {new_config.provider} / {new_config.model}")
    return new_config, thread_id


def main() -> None:
    load_dotenv()

    model_config = select_model()
    check_credentials(model_config)
    print(f"Using {model_config.provider} / {model_config.model}\n")

    client = create_client(model_config)
    graph = build_graph()
    thread_id = str(uuid.uuid4())

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
            model_config, thread_id = handle_model_command(args, model_config, graph, thread_id)
            if model_config != old_config:
                client = create_client(model_config)
            continue

        config = {"configurable": {"thread_id": thread_id}}
        context = GraphContext(client=client, model_config=model_config)
        try:
            result = graph.invoke({"new_message": user_input}, config=config, context=context)
            if result.get("__interrupt__"):
                interrupt_ctx = result["__interrupt__"][0].value
                print(f"\nRecommendation: {interrupt_ctx['recommendation']}\n")
                score = prompt_for_score()
                result = graph.invoke(Command(resume=score), config=config, context=context)
            print(f"\nAssistant: {result['response']}\n")
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}. Please try again.\n")

    try:
        _get_langfuse().flush()
    except Exception:
        pass


if __name__ == "__main__":
    main()
```

Note: the old `if history and history[-1]["role"] == "user": history.pop()` rollback-on-error logic is gone. It's no longer needed — LangGraph only commits a node's checkpoint on successful return, so a failed `graph.invoke()` call leaves the checkpointed `history` untouched automatically (verified experimentally during design: a raised exception inside a node does not commit that node's `history` delta).

- [ ] **Step 4: Run tests to verify they pass**

```bash
source .venv/bin/activate && pytest tests/test_main.py -v
```
Expected: all PASS.

- [ ] **Step 5: Run the full test suite**

```bash
source .venv/bin/activate && pytest tests/ -v
```
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add main.py tests/test_main.py
git commit -m "feat: invoke LangGraph from main.py, add eval score prompt, rework /model start-fresh"
```

---

### Task 8: Manual end-to-end verification

**Files:** none (verification only)

- [ ] **Step 1: Run the CLI with real or dummy credentials**

```bash
source .venv/bin/activate && python main.py
```

- [ ] **Step 2: Walk the golden path**

1. Select a provider/model at startup.
2. Ask for a product (e.g. "wireless mouse").
3. Answer the agent's clarifying questions (product + optimize_for) until it calls `search_amazon`.
4. Confirm the flow pauses after the recommendation is shown and prompts `Rate usefulness (1-5):`.
5. Enter a score (e.g. `4`) and an optional note.
6. Confirm the conversation continues normally afterward (can ask a follow-up).

- [ ] **Step 3: Confirm the score was recorded**

```bash
cat evals/scores.jsonl
```
Expected: one JSON line per rating given, with `query`, `optimize_for`, `recommendation`, `overall`, `note`, `timestamp` populated.

- [ ] **Step 4: Verify a non-recommendation turn doesn't prompt for a score**

Start a fresh run, type a vague request (e.g. "find me something") so the agent asks a clarifying question without searching yet — confirm no `Rate usefulness` prompt appears until an actual search/recommendation happens.

- [ ] **Step 5: Verify `/model` "start fresh"**

Mid-conversation (after at least one search), run `/model`, switch provider, and answer `y` to "Start fresh conversation?". Confirm the next turn has no memory of the prior conversation (a graph invoke on the new thread starts with empty history).

- [ ] **Step 6: Report results**

No commit for this task — it's verification only. If any step fails, return to the relevant earlier task and fix before considering the plan complete.
