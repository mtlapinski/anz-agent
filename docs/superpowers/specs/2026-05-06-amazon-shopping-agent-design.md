# Amazon Shopping Agent — Design Spec

**Date:** 2026-05-06  
**Status:** Approved

## Overview

A command-line chat agent that helps users find products on Amazon at the right price. The user describes what they want in natural language; the agent asks clarifying questions (what to find, how to optimize), searches Amazon via the Product Advertising API, and returns the top 3–5 ranked results.

## Architecture

A single Python CLI process runs a conversational agent loop. Claude (via the Anthropic SDK) drives the conversation and decides when to invoke the `search_amazon` tool. All LLM and tool calls are traced through Langfuse for observability.

```
User (terminal)
    ↕ chat messages
agent.py (conversation loop)
    ↕ Anthropic SDK (tool use / multi-turn)
Claude (LLM)
    ↕ tool calls
tools/amazon.py → Amazon PA API v5
    ↕ traces
Langfuse (free cloud or localhost)
```

## Components

### `main.py` — CLI entry point
- Reads `.env` for API credentials
- Fails fast on startup if any required key is missing
- Initializes Langfuse client and Anthropic client
- Starts the agent conversation loop

### `agent.py` — Conversation loop
- Maintains message history as a list of dicts (standard Anthropic multi-turn format)
- Holds the system prompt instructing Claude to act as a shopping assistant
- On each turn: appends user message → calls Claude → handles tool call or text response
- If Claude returns a tool call: runs the tool, appends tool result, calls Claude again for final response
- If Claude returns text: prints to terminal, waits for next user input

**System prompt (paraphrase):**
> You are a shopping assistant. Before searching Amazon, ask the user what they're looking for and whether they want to optimize for price, quality, or something else. Once you have both pieces of information, use the search_amazon tool. Present the top 3–5 results clearly, ranked according to the user's optimization goal.

### `tools/amazon.py` — Amazon search tool
Implements the `search_amazon` tool that Claude can invoke.

**Tool input schema:**
```
query: str           # search terms derived from the conversation
optimize_for: str    # "price" | "quality" | "balance" | <custom string>
max_results: int     # 3–5
max_price: float     # optional budget cap
```

**Tool output (per product):**
```
title: str
price: float
currency: str
rating: float
review_count: int
prime: bool
url: str
```

Claude receives the list and narrates the top picks based on the optimization goal — leading with lowest price for "price", highest rating for "quality", best value for "balance".

## Data Flow

1. User types a message
2. Agent appends it to history and calls `claude.messages.create(tools=[...], messages=history)`
3. If response contains a tool call:
   a. Run `search_amazon` with the provided arguments
   b. Append tool result to history
   c. Call Claude again to generate the user-facing response
4. Print Claude's text response to terminal
5. Wait for next user input

## Error Handling

| Scenario | Behavior |
|---|---|
| Amazon PA API returns no results | Claude tells user, asks to refine the query |
| PA API rate limit hit | Surface error message, suggest retry |
| Missing API credentials at startup | Fail fast with clear message identifying which key |
| Anthropic API error | Surface error, allow user to retry |

## Observability

Langfuse wraps all API calls:
- Each conversation turn is a Langfuse trace
- Anthropic SDK call is a generation span within the trace, including input and output token counts
- SerpAPI call is a separate span within the trace
- Token usage (input tokens, output tokens) is logged per LLM call and surfaced in the Langfuse UI

Use the free Langfuse cloud tier or run locally via `docker compose up`.

## Configuration

All secrets in `.env` (gitignored). A `.env.example` is committed as a template.

Required keys:
- `ANTHROPIC_API_KEY`
- `AMAZON_ACCESS_KEY`
- `AMAZON_SECRET_KEY`
- `AMAZON_PARTNER_TAG`
- `LANGFUSE_PUBLIC_KEY`
- `LANGFUSE_SECRET_KEY`

## Project Structure

```
anz-agent/
├── main.py
├── agent.py
├── tools/
│   └── amazon.py
├── .env                  # gitignored
├── .env.example
├── requirements.txt
└── docs/
    └── superpowers/
        └── specs/
            └── 2026-05-06-amazon-shopping-agent-design.md
```

## Model

`claude-sonnet-4-6` — good balance of capability and cost for a conversational agent. Swap to `claude-opus-4-7` if response quality needs improvement.

## Dependencies

```
anthropic
python-dotenv
langfuse
amazon-paapi5
```

## Scope (v1)

- Single-shot search per conversation turn (no iterative refinement yet)
- No persistence — conversation history is in-memory only
- No payment or cart functionality (planned as future subagent)
- CLI only — no web UI
