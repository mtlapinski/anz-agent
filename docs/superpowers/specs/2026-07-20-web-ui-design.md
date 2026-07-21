# Web UI for the Amazon Shopping Agent

## Context

The agent currently runs only as a terminal CLI (`main.py`), driven by raw `input()`/`print()` and a LangGraph `StateGraph` (`graph.py`) that loops between an `agent` node (LLM call), a `tools` node (`search_amazon`), and an `eval` node (`interrupt()`-based 1-5 usefulness rating after a recommendation). This spec adds a local web UI as a second way to run the agent, focused on giving search results a richer, per-query visualization than the CLI's plain text list.

## Goals

- Chat interface in the browser, backed by the existing LangGraph graph unchanged in its core loop.
- A results panel next to the chat that renders the latest search's products using one of a small fixed set of views (cards / table / chart), chosen by the agent itself per query.
- Preserve the existing eval-rating step (1-5 + note), as an inline widget in the chat.
- Do not touch or retire the CLI — it keeps working exactly as it does today.

## Non-goals (v1)

- Streaming LLM output. `llm.complete()` (`llm.py`) calls both providers' non-streaming APIs today; adding streaming would mean building both a streaming LLM call and a push transport (WebSocket/SSE) at once for a benefit (partial chat text) that's UX polish, not correctness. Deferred — see rationale below.
- Mid-session provider/model switching in the web UI (the CLI's `/model` command has no web equivalent in v1).
- Persistence across server restarts. `MemorySaver` is in-memory, same as the CLI; a restarted server invalidates open `thread_id`s.
- Any new frontend test framework/tooling.

## Architecture

A new `server.py` runs a FastAPI app that wraps the same `build_graph()` / `GraphContext` the CLI uses, keyed by `thread_id` (one per browser session, generated client-side as a UUID, same pattern as `main.py`). Three endpoints:

- `POST /session` — `{provider, model}` → validates credentials via `check_credentials` (reused from `main.py`), creates the LLM client, returns `{thread_id}`.
- `POST /chat` — `{thread_id, message}` → runs the graph until it reaches `END` or the eval `interrupt()`, returns a JSON payload describing the outcome.
- `POST /resume` — `{thread_id, score, note}` → resumes the graph via `Command(resume=...)`, exactly like the CLI's `prompt_for_score()` flow.

A separate Vite-built React app is the frontend, dev-served on its own port, proxying API calls to FastAPI. Running the web UI means two processes: `python server.py` and `npm run dev` (documented in the README alongside the existing `python main.py` instructions). No process-management framework is introduced.

### Why not streaming (Approach B) or server-rendered (Approach C)

- **Streaming (WebSocket push of partial tokens)**: the current graph is fully synchronous end-to-end; nothing in the backend produces partial output today. Building the transport now means speculative infrastructure around a capability (streaming LLM calls) that doesn't exist yet. If the non-streaming pause proves annoying in practice, swapping `/chat` for an SSE/WebSocket endpoint later is additive — the graph, tool logic, and eval-interrupt flow are unaffected.
- **Server-rendered (FastAPI + Jinja + htmx)**: the whole point of this UI is a results panel that varies its visualization (cards/table/chart) per query — that's a much more natural fit for real React components, especially the chart view, than server-rendered partials.

## Components & Layout

Fixed two-pane layout: **chat on the left, results panel on the right**.

- **Chat pane** — scrolling message list (user/assistant turns) + text input. When a `/chat` response signals an eval interrupt, an inline rating widget (1-5 + optional note) renders as the next item in the chat; submitting posts to `/resume`.
- **Results panel** — renders the most recent search's products via one of three fixed React components, chosen by the agent itself:
  - `CardsView` — default browse view: product image (if available), title, price, star rating, Prime badge, link.
  - `TableView` — sortable columns (price, rating, reviews) for side-by-side comparison queries.
  - `ChartView` — price-vs-rating scatter, for "best value" / trade-off questions.
- **Session bar** (top) — provider/model dropdown shown before the first message, mirroring the CLI's startup prompt; locked once the session starts.

### How the agent picks a view

Add a `view` enum param (`"cards" | "table" | "chart"`) to the `search_amazon` tool schema in `agent.py`, with system-prompt guidance on when to use each (e.g. comparison queries → table or chart, general browsing → cards). This flows through the *existing* `last_search_input` state field in `graph.py` with no new plumbing. `tools_node` additionally captures the raw `products` list into a new `last_search_results` state field so `/chat` can return it directly instead of re-parsing history.

Since `CardsView` benefits from product images, `tools/amazon.py`'s `search_amazon` gets one added output field: `"image": item.get("thumbnail")`.

## Data Flow

```
Browser                          FastAPI (server.py)              LangGraph (graph.py)
   |  POST /session {provider,model}                                    |
   |--------------------------------->  check_credentials, create_client|
   |  <-- {thread_id}                                                   |
   |                                                                    |
   |  POST /chat {thread_id, message}                                  |
   |----------------------------------> graph.invoke(..., thread_id) -->|
   |                                       loops agent<->tools as needed|
   |                                     <-- END or interrupt(context) -|
   |  <-- {type: "message", text, products?, view?}                    |
   |      or {type: "eval_request", context}                           |
   |                                                                    |
   |  [if eval_request] POST /resume {thread_id, score, note}          |
   |----------------------------------> Command(resume=score) -------->|
   |  <-- {type: "message", text: "(scored)"}                          |
```

`/chat` response shape (one of):

```json
{ "type": "message", "text": "...", "products": [...] | null, "view": "cards" | "table" | "chart" | null }
```

```json
{ "type": "eval_request", "query": "...", "optimize_for": "...", "recommendation": "..." }
```

The frontend retains `products`/`view` from the last `message` response and re-renders the results panel only when a new non-null `products` array arrives — a plain conversational reply (e.g. a clarifying question) doesn't clear the panel.

## Error Handling

- Missing/invalid credentials at `/session` → 400, same message `check_credentials` produces today.
- Exceptions raised inside the graph during `/chat` (SerpAPI, LLM) → caught at the endpoint boundary, returned as `{"type": "error", "message": str(e)}`; the frontend renders this as a system message in the chat rather than crashing the pane. This is a new top-level catch — the FastAPI worker process must survive a single request's failure.
- Unknown/expired `thread_id` (server restarted, `MemorySaver` is in-memory) → 404; frontend prompts the user to start a new session.

## Testing

- `server.py` endpoint tests via FastAPI's `TestClient`, mocking the graph the same way `tests/test_main.py` already does: `/session` happy path and missing-credentials case, `/chat` happy path, `/chat` eval-interrupt path, `/resume`, and the error/404 cases above.
- Extend existing `tools/amazon.py` tests to cover the new `image` field.
- Extend the existing `search_amazon` tool-schema test in `agent.py`'s tests to cover the new `view` param.
- No new frontend test framework for v1 — manual verification in-browser (golden path: search → results panel renders correct view type → eval rating → resume) is the bar, consistent with this project's existing "mocked externals only, no integration tests" approach.
