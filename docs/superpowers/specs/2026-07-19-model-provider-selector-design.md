# Model Provider Selector — Design Spec

**Date:** 2026-07-19
**Status:** Approved

## Overview

Add support for multiple LLM providers (Google Gemini and Anthropic Claude) with interactive model selection at startup and a `/model` slash command to switch mid-session.

## Architecture

A new `llm.py` module abstracts all provider differences. `agent.py` calls `llm.complete()` instead of `client.messages.create()` directly. `main.py` handles startup selection and the `/model` command.

```
main.py  ──model_config──►  agent.py  ──llm.complete()──►  llm.py
                                                               ├── anthropic.Anthropic
                                                               └── google.generativeai
```

## Components

### `llm.py` — Provider adapter

Two public functions:

**`create_client(provider: str, model: str) -> LLMClient`**
Returns an initialized SDK client for the given provider. Reads `ANTHROPIC_API_KEY` or `GOOGLE_API_KEY` from environment.

**`complete(client, provider, model, system, tools, messages) -> LLMResponse`**
Normalizes the API call across providers. Returns a dataclass:
```python
@dataclass
class LLMResponse:
    text: str | None          # assistant text, if stop_reason is end_turn
    tool_calls: list | None   # list of {name, id, input} dicts, if tool_use
    input_tokens: int
    output_tokens: int
```

Internally branches on `provider` to handle:
- Tool definition schema (Anthropic `input_schema` vs Google `parameters`)
- Tool call parsing (Anthropic `tool_use` blocks vs Google `function_call` parts)
- Tool result format (Anthropic `tool_result` block vs Google `function_response` part)
- Role names (`assistant` vs `model`)
- System prompt parameter name

### `ModelConfig` dataclass

```python
@dataclass
class ModelConfig:
    provider: str   # "google" | "anthropic"
    model: str      # e.g. "gemini-2.0-flash-lite"
```

Defined in `llm.py`, passed from `main.py` into `agent.chat()`.

### `main.py` — Startup selection and `/model` command

**Startup flow:**
```
Provider? [1] Google (default)  [2] Anthropic : _
Model? [gemini-2.0-flash-lite] : _
Using google / gemini-2.0-flash-lite
```
Pressing Enter accepts the bracketed default.

**`/model` slash command** (detected in the input loop before passing to `chat()`):
- `/model` alone → re-runs the interactive selection prompt
- `/model google gemini-2.0-flash` → sets provider and model directly, prints confirmation
- On any model change → prompts: `Start fresh conversation? [y/N]`
  - Y: clears history
  - N: keeps history; if history contains tool call turns, prints a warning that results may not transfer cleanly

### `agent.py` — Changes

- `chat()` signature gains `model_config: ModelConfig` parameter
- `client.messages.create(...)` replaced with `llm.complete(client, model_config, ...)`
- Token usage read from `LLMResponse` instead of `response.usage`
- Langfuse generation span updated to log the active provider and model name
- History format stays as Anthropic-style dicts internally; `llm.complete()` translates to Google format on the way out and back in

## History Translation

Plain text turns are safe to carry across a provider switch (remap role names, rewrap content).

Tool call turns require translation and may not map cleanly. The `/model` switch prompts the user to start fresh when tool call turns are present in history.

## Default Model

`gemini-2.0-flash-lite` — Google's cheapest production model.

## Required Keys

| Variable | Required for |
|---|---|
| `ANTHROPIC_API_KEY` | Anthropic provider |
| `GOOGLE_API_KEY` | Google provider |
| `SERPAPI_KEY` | Always (Amazon search) |

Startup credential check validates only the key for the selected provider.

## Dependencies

Add to `requirements.txt`:
```
google-generativeai>=0.8.0
```

## Error Handling

| Scenario | Behavior |
|---|---|
| Selected provider key missing | Fail fast at startup with clear message |
| `/model` sets provider with missing key | Print error, keep current model |
| Google returns no tool call support | Surface error, suggest switching to Anthropic |

## Files Changed

```
anz-agent/
├── llm.py              # new — provider adapter
├── agent.py            # updated — use llm.complete(), accept ModelConfig
├── main.py             # updated — startup selection, /model command
├── requirements.txt    # updated — add google-generativeai
├── .env.example        # updated — add GOOGLE_API_KEY
└── README.md           # updated — document provider selection
```

## Out of Scope

- More than two providers
- Persisting model selection across sessions
- Per-turn provider switching (only on `/model` command)
