# Strip UI-only fields from the LLM's tool_result

**Date:** 2026-08-06
**Status:** Approved

## Problem

`search_amazon` results flow through `tools_node` in [graph.py](../../../graph.py), which parses the tool output once into `last_search_results`. That same parsed object currently serves two different consumers:

1. The `tool_result` message appended to `history` — sent back to the LLM, and because the LangGraph checkpointer replays full history on every subsequent turn, this payload is resent verbatim on every future turn of the conversation, not just the turn it was produced.
2. `last_search_results`, served to the web UI as `products` in [server.py](../../../server.py) for rendering cards/table/chart views.

Every product dict includes an `image` (thumbnail URL) field. The web UI needs it for rendering. The LLM does not: the system prompt in [agent.py](../../../agent.py) only requires the assistant to surface title, price, rating, Prime status, and URL in its response text — `image` is never referenced. Today it's sent to the LLM anyway, and resent on every later turn, purely because both consumers currently share one representation.

Raising `max_results` (e.g. toward 100) multiplies this waste directly, but the problem exists at any result count.

## Fix

Split the two representations at the one point that already sees both: `tools_node`.

- `last_search_results` keeps the full product dicts, unchanged. The web UI is unaffected.
- The `tool_result` content appended to `history` (the thing actually sent to the LLM) uses a trimmed copy of the same products with `image` removed.

## Where the logic lives

A small helper function in `agent.py`:

```python
def strip_image_for_llm(products: list[dict]) -> list[dict]:
    return [{k: v for k, v in p.items() if k != "image"} for p in products]
```

`agent.py` already owns the LLM-facing contract (`SYSTEM_PROMPT`, `TOOLS`, `run_tool`), so this is a natural home for "what the LLM contract looks like."

`tools_node` in `graph.py` calls it only when the tool call was `search_amazon`, mirroring the existing special-case at the line that currently captures `last_search_results`. The `history` entry's `tool_result` content is built from the trimmed copy; `last_search_results` is built from the untouched original.

## Edge cases

- Error results (`{"error": ..., "products": []}`): no products to strip, no-op.
- Empty `products` list: no-op.
- Non-`search_amazon` tool results: untouched (no other tools exist today, but the split is scoped to `search_amazon` explicitly, not applied blindly to all tool results).

## Out of scope

- Any other product field (title, price, rating, prime, url, currency, review_count) — all are required by the system prompt for the LLM's response text, so none are trimmed.
- A general/configurable field-allowlist mechanism for future UI-only fields — YAGNI until a second such field exists.
- Compressing *stale* tool results from earlier turns (a separate mitigation for the same underlying token-growth problem) — not addressed here.
- Raising `max_results` itself — this spec only removes a fixed per-product cost; the `max_results` default is a separate decision.

## Testing

Extend the `tools_node` tests in `tests/test_graph.py`:
- The `history` entry's `tool_result` content (JSON) has no `image` key on any product.
- `result["last_search_results"]` still has `image` on each product, unchanged.
