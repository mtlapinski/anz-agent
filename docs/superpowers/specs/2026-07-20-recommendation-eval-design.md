# Recommendation Eval Scoring — Design Spec

**Date:** 2026-07-20
**Status:** Approved

## Overview

Add a human-in-the-loop evaluation step that scores whether the agent's recommendations are actually useful. The conversation loop in `agent.py` is restructured as a LangGraph `StateGraph`, and a new `eval` node uses LangGraph's `interrupt()` to pause after a recommendation and collect a usefulness score from the person running the CLI. Scores are logged to both Langfuse (for trace correlation) and a local JSONL file (as a seed dataset for a future automated judge).

LangGraph here is purely an orchestration layer, not a provider abstraction — `llm.py` (the Anthropic/Google adapter) and `tools/amazon.py` are unchanged. `langgraph` runs in-process as a regular Python dependency; no external infra or hosted service is required. The `MemorySaver` checkpointer is in-memory and resets each run.

## Architecture

```
        ┌────────┐   tool_calls    ┌─────────┐
   ───► │ agent  │ ──────────────► │  tools  │
        └────────┘                 └─────────┘
             │                          │
        final text                      │ (loop back)
             │                          ▼
             │                     ┌────────┐
             ├──(no tool calls  ◄──┤ agent  │
             │   this turn)        └────────┘
             ▼                          │
            END                    final text,
                                   preceded by
                                   ≥1 tool call
                                        │
                                        ▼
                                  ┌──────────┐
                                  │   eval   │ ◄── interrupt()
                                  └──────────┘     pauses for
                                        │           human score
                                        ▼
                                       END
```

`eval` only fires when the turn actually produced a recommendation — i.e. the final text response was preceded by at least one `search_amazon` call within that turn. A turn where the agent only asks a clarifying question (no tool call yet) routes straight to `END`, no score prompt.

## Components

### `graph.py` (new)

Builds and compiles the `StateGraph`. Owns:

- **State schema**: `history` (existing Anthropic-style message list, accumulated via the checkpointer — see below), `model_config`, `client`, `trace` (Langfuse trace handle), `made_tool_call_this_turn: bool` (reset at the start of each turn's invocation).
- **`agent_node`**: calls `llm.complete()` (unchanged from today's `agent.chat()` inner loop), appends the response to `history`. Sets `made_tool_call_this_turn = True` if the response includes tool calls.
- **`tools_node`**: runs `run_tool()` for each tool call (unchanged from today's `agent.py`), appends results to `history`, routes back to `agent_node`.
- **`eval_node`**: calls `interrupt()` with `{query, optimize_for, recommendation}`. On resume, receives `EvalScore` and calls `record_score()`.
- **Conditional edges**: `agent_node` → `tools_node` if tool calls present; else → `eval_node` if `made_tool_call_this_turn`; else → `END`. `eval_node` → `END`.
- Compiled with `MemorySaver` checkpointer, `thread_id` = one per conversation (see History Ownership below).

### History Ownership — checkpointer replaces the `history` list

Today `main.py` owns a `history` list and passes it into `chat()` on every call, mutating it in place. That model conflicts with the checkpointer, which needs to be the single source of truth once it's tracking state across the interrupt/resume boundary — otherwise there are two copies of history that can drift.

The checkpointer becomes authoritative for the whole conversation, not just the interrupt boundary:

- `main.py` no longer holds a `history` list. Each turn it invokes the graph with only the new user message; the graph reads/writes accumulated history via the checkpointer under the current `thread_id`.
- The `/model` command's "start fresh" flow (`main.py:handle_model_command`) changes from clearing a list to generating a new `thread_id` — the old thread's checkpointed state is simply abandoned.
- The existing `_has_tool_turns()` warning check reads the current thread's checkpointed state (`graph.get_state(config).values["history"]`) instead of a plain list.

### `agent.py` — Changes

- `SYSTEM_PROMPT`, `TOOLS`, `run_tool()`, `_get_langfuse()` stay as-is; imported by `graph.py`.
- The `chat()` while-loop is removed — its logic becomes `agent_node` + `tools_node` in `graph.py`.
- New `record_score(trace, context, score) -> None`: writes Langfuse score and JSONL row (see Storage below).

### `main.py` — Changes

The turn-handling block replaces the direct `chat()` call with a graph invocation that handles the interrupt:

```python
result = graph.invoke({"new_message": user_input, "model_config": model_config, ...},
                       config={"configurable": {"thread_id": thread_id}})
if "__interrupt__" in result:
    ctx = result["__interrupt__"][0].value
    print(f"\nRecommendation: {ctx['recommendation']}\n")
    score = prompt_for_score()  # loops until 1-5 given, Ctrl-C -> skip
    result = graph.invoke(Command(resume=score),
                           config={"configurable": {"thread_id": thread_id}})
print(f"\nAssistant: {result['response']}\n")
```

`prompt_for_score()`:
```
Rate usefulness (1-5): _
Note (optional): _
```
Reprompts on non-1-5 input. Ctrl-C during the prompt resumes the graph with `score=None` (treated as skipped, not an error).

## Eval Score

```python
@dataclass
class EvalScore:
    overall: int | None   # 1-5, None if skipped
    note: str | None
    criteria: dict[str, int] | None = None  # reserved for future rubric, unused now
```

## Storage — both Langfuse and local JSONL

- **Langfuse**: `trace.score(name="usefulness", value=score.overall, comment=score.note)` on the same trace already created for that turn. Wrapped in the existing try/except-and-continue pattern — Langfuse stays optional.
- **Local JSONL** (`evals/scores.jsonl`, new file, added to `.gitignore`): one line per rating —
  ```json
  {"timestamp": "...", "query": "...", "optimize_for": "...", "recommendation": "...", "overall": 4, "note": "..."}
  ```
  Durable independent of Langfuse configuration; becomes the seed dataset for a future automated judge.

## Error Handling

| Scenario | Behavior |
|---|---|
| Score input not 1-5 | Reprompt |
| Langfuse `trace.score()` fails | Catch, log warning, continue |
| JSONL write fails | Print warning, continue (non-fatal) |
| User hits Ctrl-C during score prompt | Resume graph with `score=None` ("skipped"), don't crash the turn |

## Dependencies

Add to `requirements.txt`:
```
langgraph>=0.2.0
```

## Files Changed

```
anz-agent/
├── graph.py             # new — StateGraph, node functions, compile + checkpointer
├── agent.py             # updated — chat() loop removed, node logic extracted; record_score() added
├── main.py              # updated — invoke graph, handle interrupt/resume, prompt for score
├── requirements.txt     # updated — add langgraph
├── .gitignore            # updated — add evals/scores.jsonl
└── evals/                # new dir — scores.jsonl written here at runtime
```

## Out of Scope

- Automated LLM-as-judge scoring (next step once the JSONL dataset has enough rows)
- Structured multi-criteria rubric (`criteria` field reserved but unused)
- Migrating `llm.py` to LangChain chat models
- Confirmation-before-search interrupt (same `interrupt()` pattern, different gate — future work)
- Search result caching to conserve SerpAPI quota (backlog — not a memory concern, see below)
- Long-term cross-session agent memory over prior searches (backlog — would use LangGraph's `Store`, distinct from the short-term checkpointer used here)
