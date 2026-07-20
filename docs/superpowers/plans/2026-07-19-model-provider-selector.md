# Model Provider Selector Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Google Gemini and Anthropic Claude provider selection with interactive startup prompt and `/model` slash command.

**Architecture:** A new `llm.py` module exposes `create_client()` and `complete()` that normalize provider differences. `agent.py` calls `llm.complete()` instead of the Anthropic SDK directly. `main.py` handles startup model selection and `/model` mid-session switching.

**Tech Stack:** Python 3.11+, `anthropic>=0.40.0`, `google-generativeai>=0.8.0`, `pytest`, `unittest.mock`

## Global Constraints

- Default provider: `google`, default model: `gemini-2.0-flash-lite`
- Default Anthropic model: `claude-haiku-4-5-20251001`
- History stored internally in Anthropic format; `llm.complete()` translates to Google format on each call
- All Langfuse calls wrapped in `try/except` (graceful degradation)
- `SERPAPI_KEY` always required; provider API key checked at startup for selected provider only
- No new globals — pass `ModelConfig` as a parameter

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `llm.py` | Create | `ModelConfig`, `LLMResponse`, `create_client()`, `complete()` |
| `tests/test_llm.py` | Create | Unit tests for `llm.py` |
| `agent.py` | Modify | Accept `ModelConfig`, call `llm.complete()` |
| `tests/test_agent.py` | Modify | Update tests for new `chat()` signature |
| `main.py` | Modify | Startup selection, `/model` command, provider credential check |
| `requirements.txt` | Modify | Add `google-generativeai>=0.8.0` |
| `.env.example` | Modify | Add `GOOGLE_API_KEY` |
| `README.md` | Modify | Document provider selection |

---

### Task 1: `llm.py` — ModelConfig, LLMResponse, and Anthropic provider

**Files:**
- Create: `llm.py`
- Create: `tests/test_llm.py`

**Interfaces:**
- Produces:
  - `ModelConfig(provider: str, model: str)` dataclass
  - `LLMResponse(text: str | None, tool_calls: list[dict] | None, input_tokens: int, output_tokens: int)` dataclass
  - `create_client(config: ModelConfig) -> Any`
  - `complete(client, config: ModelConfig, system: str, tools: list[dict], messages: list[dict]) -> LLMResponse`

- [ ] **Step 1: Write failing tests for Anthropic path**

Create `tests/test_llm.py`:

```python
import pytest
from unittest.mock import MagicMock, patch
from llm import ModelConfig, LLMResponse, create_client, complete


def make_anthropic_text_response(text="Here are results"):
    block = MagicMock()
    block.type = "text"
    block.text = text
    response = MagicMock()
    response.stop_reason = "end_turn"
    response.content = [block]
    response.usage.input_tokens = 100
    response.usage.output_tokens = 50
    return response


def make_anthropic_tool_response(name="search_amazon", tool_id="tu_1", input=None):
    if input is None:
        input = {"query": "blender", "optimize_for": "price", "max_results": 5}
    block = MagicMock()
    block.type = "tool_use"
    block.name = name
    block.id = tool_id
    block.input = input
    response = MagicMock()
    response.stop_reason = "tool_use"
    response.content = [block]
    response.usage.input_tokens = 80
    response.usage.output_tokens = 20
    return response


def test_create_client_anthropic():
    config = ModelConfig(provider="anthropic", model="claude-haiku-4-5-20251001")
    with patch("llm.anthropic.Anthropic") as mock_cls:
        mock_cls.return_value = MagicMock()
        client = create_client(config)
    mock_cls.assert_called_once()
    assert client is not None


def test_create_client_unknown_provider_raises():
    config = ModelConfig(provider="openai", model="gpt-4")
    with pytest.raises(ValueError, match="Unknown provider"):
        create_client(config)


def test_complete_anthropic_text_response():
    config = ModelConfig(provider="anthropic", model="claude-haiku-4-5-20251001")
    mock_client = MagicMock()
    mock_client.messages.create.return_value = make_anthropic_text_response("Found it!")

    result = complete(mock_client, config, "system", [], [{"role": "user", "content": "hi"}])

    assert isinstance(result, LLMResponse)
    assert result.text == "Found it!"
    assert result.tool_calls is None
    assert result.input_tokens == 100
    assert result.output_tokens == 50


def test_complete_anthropic_tool_response():
    config = ModelConfig(provider="anthropic", model="claude-haiku-4-5-20251001")
    mock_client = MagicMock()
    mock_client.messages.create.return_value = make_anthropic_tool_response()

    result = complete(mock_client, config, "system", [], [{"role": "user", "content": "find blender"}])

    assert result.text is None
    assert result.tool_calls is not None
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0]["name"] == "search_amazon"
    assert result.tool_calls[0]["id"] == "tu_1"
    assert result.tool_calls[0]["input"] == {"query": "blender", "optimize_for": "price", "max_results": 5}
    assert result.input_tokens == 80
    assert result.output_tokens == 20


def test_complete_anthropic_unexpected_stop_reason_raises():
    config = ModelConfig(provider="anthropic", model="claude-haiku-4-5-20251001")
    mock_client = MagicMock()
    response = MagicMock()
    response.stop_reason = "max_tokens"
    response.content = []
    response.usage.input_tokens = 10
    response.usage.output_tokens = 5
    mock_client.messages.create.return_value = response

    with pytest.raises(RuntimeError, match="Unexpected stop_reason"):
        complete(mock_client, config, "system", [], [{"role": "user", "content": "hi"}])
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
.venv/bin/python -m pytest tests/test_llm.py -v
```
Expected: `ModuleNotFoundError: No module named 'llm'`

- [ ] **Step 3: Create `llm.py` with Anthropic support**

```python
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
```

- [ ] **Step 4: Run tests — verify Anthropic tests pass**

```bash
.venv/bin/python -m pytest tests/test_llm.py -v -k "not google"
```
Expected: 5 tests pass, Google tests skip/not present yet.

- [ ] **Step 5: Commit**

```bash
git add llm.py tests/test_llm.py
git commit -m "feat: add llm.py adapter with Anthropic provider support"
```

---

### Task 2: Google Gemini support in `llm.py`

**Files:**
- Modify: `llm.py` (implement `_complete_google` and helpers)
- Modify: `tests/test_llm.py` (add Google tests)
- Modify: `requirements.txt`

**Interfaces:**
- Consumes: `ModelConfig`, `LLMResponse` from Task 1
- Produces: `_complete_google()`, `_anthropic_tools_to_google()`, `_anthropic_messages_to_google()`

- [ ] **Step 1: Add `google-generativeai` to requirements.txt**

Edit `requirements.txt` to add after the `anthropic` line:
```
google-generativeai>=0.8.0
```

Install it:
```bash
.venv/bin/pip install google-generativeai>=0.8.0
```

- [ ] **Step 2: Write failing Google tests**

Append to `tests/test_llm.py`:

```python
def make_google_text_response(text="Here are results"):
    part = MagicMock()
    part.text = text
    del part.function_call  # ensure hasattr check fails
    candidate = MagicMock()
    candidate.content.parts = [part]
    response = MagicMock()
    response.candidates = [candidate]
    response.usage_metadata.prompt_token_count = 200
    response.usage_metadata.candidates_token_count = 60
    return response


def make_google_tool_response(name="search_amazon", args=None):
    if args is None:
        args = {"query": "blender", "optimize_for": "price", "max_results": 5}
    fc = MagicMock()
    fc.name = name
    fc.args = args
    part = MagicMock()
    part.function_call = fc
    candidate = MagicMock()
    candidate.content.parts = [part]
    response = MagicMock()
    response.candidates = [candidate]
    response.usage_metadata.prompt_token_count = 150
    response.usage_metadata.candidates_token_count = 30
    return response


def test_complete_google_text_response():
    import google.generativeai as genai
    config = ModelConfig(provider="google", model="gemini-2.0-flash-lite")

    mock_model = MagicMock()
    mock_model.generate_content.return_value = make_google_text_response("Great choice!")

    with patch("llm.genai") as mock_genai:
        mock_genai.GenerativeModel.return_value = mock_model
        result = complete(None, config, "system", [], [{"role": "user", "content": "hi"}])

    assert result.text == "Great choice!"
    assert result.tool_calls is None
    assert result.input_tokens == 200
    assert result.output_tokens == 60


def test_complete_google_tool_response():
    config = ModelConfig(provider="google", model="gemini-2.0-flash-lite")

    mock_model = MagicMock()
    mock_model.generate_content.return_value = make_google_tool_response()

    with patch("llm.genai") as mock_genai:
        mock_genai.GenerativeModel.return_value = mock_model
        result = complete(None, config, "system", [], [{"role": "user", "content": "find blender"}])

    assert result.text is None
    assert result.tool_calls is not None
    assert result.tool_calls[0]["name"] == "search_amazon"
    assert result.tool_calls[0]["input"] == {"query": "blender", "optimize_for": "price", "max_results": 5}
    assert result.input_tokens == 150
    assert result.output_tokens == 30


def test_anthropic_tools_to_google():
    from llm import _anthropic_tools_to_google
    anthropic_tools = [{
        "name": "search_amazon",
        "description": "Search Amazon",
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    }]
    result = _anthropic_tools_to_google(anthropic_tools)
    assert len(result) == 1
    decl = result[0]["function_declarations"][0]
    assert decl["name"] == "search_amazon"
    assert decl["description"] == "Search Amazon"
    assert decl["parameters"]["properties"]["query"]["type"] == "string"


def test_anthropic_messages_to_google_plain_text():
    from llm import _anthropic_messages_to_google
    messages = [
        {"role": "user", "content": "Find me a blender"},
        {"role": "assistant", "content": [{"type": "text", "text": "What's your budget?"}]},
        {"role": "user", "content": "Under $50"},
    ]
    result = _anthropic_messages_to_google(messages)
    assert result[0] == {"role": "user", "parts": [{"text": "Find me a blender"}]}
    assert result[1] == {"role": "model", "parts": [{"text": "What's your budget?"}]}
    assert result[2] == {"role": "user", "parts": [{"text": "Under $50"}]}
```

- [ ] **Step 3: Run Google tests — verify they fail**

```bash
.venv/bin/python -m pytest tests/test_llm.py -v -k "google"
```
Expected: FAIL — `_complete_google` raises `NotImplementedError`

- [ ] **Step 4: Implement Google support in `llm.py`**

Replace the `_complete_google` stub and add helpers. Add `import google.generativeai as genai` at the top of `llm.py` after the `anthropic` import (inside a try/except so tests without the package still import):

Add at the top of `llm.py` after `import anthropic`:
```python
try:
    import google.generativeai as genai
except ImportError:
    genai = None  # type: ignore
```

Replace `_complete_google`:
```python
def _complete_google(model_name: str, system: str, tools: list[dict], messages: list[dict]) -> LLMResponse:
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
    return [{
        "function_declarations": [
            {"name": t["name"], "description": t["description"], "parameters": t["input_schema"]}
            for t in tools
        ]
    }]


def _anthropic_messages_to_google(messages: list[dict]) -> list[dict]:
    import json
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
```

- [ ] **Step 5: Run all `llm` tests — verify they pass**

```bash
.venv/bin/python -m pytest tests/test_llm.py -v
```
Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add llm.py tests/test_llm.py requirements.txt
git commit -m "feat: add Google Gemini provider support to llm.py"
```

---

### Task 3: Update `agent.py` to use `llm.complete()`

**Files:**
- Modify: `agent.py`
- Modify: `tests/test_agent.py`

**Interfaces:**
- Consumes: `ModelConfig`, `LLMResponse`, `complete()` from `llm.py` (Task 1–2)
- Produces: `chat(client, history, user_message, model_config) -> str` (new signature)

- [ ] **Step 1: Update `test_agent.py` for new `chat()` signature**

At the top of `tests/test_agent.py`, add the import:
```python
from llm import ModelConfig
```

Add a helper at the top of the file (after the existing `make_*` helpers):
```python
def default_config():
    return ModelConfig(provider="anthropic", model="claude-haiku-4-5-20251001")
```

Update every `chat(mock_client, ...)` call to pass `model_config=default_config()` as the last argument. Also update the test that asserts the model name — replace the assertion on `call_kwargs["model"]` with a check that `llm.complete` was called:

Replace `test_chat_text_response_updates_history`:
```python
@patch("agent._get_langfuse")
@patch("agent.llm")
def test_chat_text_response_updates_history(mock_llm, mock_lf):
    from agent import chat
    mock_llm.complete.return_value = LLMResponse(
        text="What are you looking for?", tool_calls=None, input_tokens=10, output_tokens=5
    )
    history = []
    result = chat(MagicMock(), history, "Hello", default_config())
    assert result == "What are you looking for?"
    assert history[0] == {"role": "user", "content": "Hello"}
    assert history[1]["role"] == "assistant"
```

Add `from llm import LLMResponse` to the imports in `test_agent.py`.

Replace `test_chat_tool_call_then_text`:
```python
@patch("agent._get_langfuse")
@patch("agent.llm")
def test_chat_tool_call_then_text(mock_llm, mock_lf):
    from agent import chat
    mock_llm.complete.side_effect = [
        LLMResponse(text=None, tool_calls=[{"name": "search_amazon", "id": "tu_123",
            "input": {"query": "laptop", "optimize_for": "price", "max_results": 5}}],
            input_tokens=80, output_tokens=20),
        LLMResponse(text="Here are the top laptops...", tool_calls=None, input_tokens=200, output_tokens=50),
    ]
    with patch("agent.search_amazon") as mock_search:
        mock_search.return_value = {"products": []}
        history = []
        result = chat(MagicMock(), history, "Find me a laptop", default_config())
    assert result == "Here are the top laptops..."
    assert mock_llm.complete.call_count == 2
    assert len(history) == 4
    tool_result_msg = history[2]
    assert tool_result_msg["role"] == "user"
    assert tool_result_msg["content"][0]["type"] == "tool_result"
    assert tool_result_msg["content"][0]["tool_use_id"] == "tu_123"
```

Replace `test_chat_passes_system_prompt`:
```python
@patch("agent._get_langfuse")
@patch("agent.llm")
def test_chat_passes_system_prompt(mock_llm, mock_lf):
    from agent import chat, SYSTEM_PROMPT
    mock_llm.complete.return_value = LLMResponse(
        text="hi", tool_calls=None, input_tokens=10, output_tokens=5
    )
    config = default_config()
    chat(MagicMock(), [], "hi", config)
    call_kwargs = mock_llm.complete.call_args
    assert call_kwargs.args[2] == SYSTEM_PROMPT   # system is 3rd positional arg
```

Replace `test_chat_logs_token_usage`:
```python
@patch("agent._get_langfuse")
@patch("agent.llm")
def test_chat_logs_token_usage(mock_llm, mock_lf):
    from agent import chat
    mock_llm.complete.return_value = LLMResponse(
        text="Here are results...", tool_calls=None, input_tokens=100, output_tokens=50
    )
    mock_trace = MagicMock()
    mock_lf.return_value.trace.return_value = mock_trace
    mock_generation = MagicMock()
    mock_trace.generation.return_value = mock_generation

    chat(MagicMock(), [], "find me a blender", default_config())

    mock_generation.end.assert_called_once()
    call_kwargs = mock_generation.end.call_args.kwargs
    assert call_kwargs["usage"]["input"] == 100
    assert call_kwargs["usage"]["output"] == 50
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
.venv/bin/python -m pytest tests/test_agent.py -v
```
Expected: failures because `chat()` doesn't yet accept `model_config`.

- [ ] **Step 3: Update `agent.py`**

Replace the entire file content:

```python
import json
import os
import anthropic
import llm
from langfuse import Langfuse
from llm import ModelConfig
from tools.amazon import search_amazon

SYSTEM_PROMPT = """You are a helpful Amazon shopping assistant. Your job is to help the user find the right product at the right price.

Before searching for products, you MUST ask the user:
1. What specific product they are looking for (if not already clear)
2. Whether they want to optimize for price, quality, balance, or something else

Once you have both pieces of information, call the search_amazon tool.

When presenting results, rank them according to the user's optimization goal:
- "price": cheapest options first
- "quality": highest-rated options first
- "balance": best combination of price and rating
- Any other goal: use your judgment

For each product, show: name, price, star rating, whether it ships with Prime, and the product URL.
Keep the presentation clear and scannable."""

TOOLS = [
    {
        "name": "search_amazon",
        "description": "Search Amazon for products matching the user's criteria.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search keywords for Amazon",
                },
                "optimize_for": {
                    "type": "string",
                    "description": "Optimization goal: 'price', 'quality', 'balance', or a custom description",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Number of results to return, between 3 and 5",
                },
                "max_price": {
                    "type": "number",
                    "description": "Optional maximum price in USD",
                },
            },
            "required": ["query", "optimize_for", "max_results"],
        },
    }
]

_langfuse: Langfuse | None = None


def _get_langfuse() -> Langfuse:
    global _langfuse
    if _langfuse is None:
        _langfuse = Langfuse(
            public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
            secret_key=os.environ["LANGFUSE_SECRET_KEY"],
            host=os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com"),
        )
    return _langfuse


def run_tool(tool_name: str, tool_input: dict, trace=None) -> str:
    if tool_name == "search_amazon":
        try:
            span = trace.span(name="search_amazon", input=tool_input) if trace else None
        except Exception:
            span = None
        result = search_amazon(**tool_input)
        try:
            if span:
                span.end(output=result)
        except Exception:
            pass
        return json.dumps(result)
    raise ValueError(f"Unknown tool: {tool_name}")


def chat(client, history: list, user_message: str, model_config: ModelConfig) -> str:
    try:
        lf = _get_langfuse()
        trace = lf.trace(name="shopping-turn", input={"message": user_message})
    except Exception:
        trace = None

    history.append({"role": "user", "content": user_message})

    while True:
        try:
            generation = trace.generation(
                name="llm",
                model=f"{model_config.provider}/{model_config.model}",
                input={"system": SYSTEM_PROMPT, "messages": history},
            ) if trace else None
        except Exception:
            generation = None

        llm_response = llm.complete(client, model_config, SYSTEM_PROMPT, TOOLS, history)

        print(f"[tokens: {llm_response.input_tokens} in / {llm_response.output_tokens} out]")

        try:
            if generation:
                generation.end(
                    output=str(llm_response.text or llm_response.tool_calls),
                    usage={
                        "input": llm_response.input_tokens,
                        "output": llm_response.output_tokens,
                    },
                )
        except Exception:
            pass

        if llm_response.tool_calls:
            tool_results = []
            for tc in llm_response.tool_calls:
                result = run_tool(tc["name"], tc["input"], trace=trace)
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
            try:
                if trace:
                    trace.update(output={"response": text})
            except Exception:
                pass
            return text
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
.venv/bin/python -m pytest tests/test_agent.py tests/test_llm.py -v
```
Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add agent.py tests/test_agent.py
git commit -m "refactor: update agent.py to use llm.complete() and ModelConfig"
```

---

### Task 4: Startup selection, `/model` command, and docs

**Files:**
- Modify: `main.py`
- Modify: `.env.example`
- Modify: `README.md`

**Interfaces:**
- Consumes: `ModelConfig`, `create_client()` from `llm.py`; `chat()` from `agent.py`
- No new public functions (all helpers are module-private)

- [ ] **Step 1: Update `main.py`**

Replace the entire file:

```python
import os
import sys
import anthropic
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

        if user_input.startswith("/model"):
            args = user_input[len("/model"):].strip()
            model_config, history = handle_model_command(args, model_config, history)
            client = create_client(model_config)
            continue

        try:
            response = chat(client, history, user_input, model_config)
            print(f"\nAssistant: {response}\n")
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}. Please try again.\n")

    try:
        from agent import _get_langfuse
        _get_langfuse().flush()
    except Exception:
        pass


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Update `.env.example`**

Replace contents of `.env.example`:
```
ANTHROPIC_API_KEY=    # https://console.anthropic.com
GOOGLE_API_KEY=       # https://aistudio.google.com/app/apikey
SERPAPI_KEY=          # https://serpapi.com — 100 free searches/month
LANGFUSE_PUBLIC_KEY=
LANGFUSE_SECRET_KEY=
LANGFUSE_HOST=https://cloud.langfuse.com
```

- [ ] **Step 3: Update `README.md`**

Replace the Setup and Configuration sections:

```markdown
## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# Fill in SERPAPI_KEY and either GOOGLE_API_KEY or ANTHROPIC_API_KEY
```

## Run

```bash
python main.py
```

At startup you'll be prompted to choose a provider and model:
```
Provider? [1] Google (default)  [2] Anthropic :
Model? [gemini-2.0-flash-lite] :
```

Press Enter to accept the defaults. Type `/model` at any time to switch mid-session.

Token usage is printed after each LLM call. Type `quit` or `exit` to stop.

## Configuration

| Variable | Required for |
|---|---|
| `GOOGLE_API_KEY` | Google provider — [aistudio.google.com](https://aistudio.google.com/app/apikey) |
| `ANTHROPIC_API_KEY` | Anthropic provider — [console.anthropic.com](https://console.anthropic.com) |
| `SERPAPI_KEY` | Always — [serpapi.com](https://serpapi.com) (100 free searches/month) |
| `LANGFUSE_PUBLIC_KEY` | No — optional observability |
| `LANGFUSE_SECRET_KEY` | No — optional observability |
| `LANGFUSE_HOST` | No — defaults to Langfuse cloud |

## Models

| Provider | Recommended model | Notes |
|---|---|---|
| Google | `gemini-2.0-flash-lite` | Default — cheapest |
| Google | `gemini-2.0-flash` | Better quality |
| Anthropic | `claude-haiku-4-5-20251001` | Fast and cheap |
| Anthropic | `claude-sonnet-4-6` | Higher quality |
```

- [ ] **Step 4: Run the full test suite**

```bash
.venv/bin/python -m pytest tests/ -v
```
Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add main.py .env.example README.md
git commit -m "feat: add startup model selection and /model slash command"
```
