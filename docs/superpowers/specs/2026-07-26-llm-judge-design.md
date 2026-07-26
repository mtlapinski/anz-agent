# LLM-as-Judge Scoring — Design Spec

**Date:** 2026-07-26
**Status:** Approved

## Overview

Add an automated LLM judge that scores each recommendation on three criteria (Relevance, Fit, Quality), running alongside — not replacing — the existing human `interrupt()` score in `eval_node`. The judge reuses the active session's LLM client/model config (no new provider setup). Judge results are written into the same `evals/scores.jsonl` row and the same Langfuse trace as the human score, so both are directly comparable per turn.

This supersedes the original plan to wait for a larger human-rated dataset before building a judge (see [2026-07-20-recommendation-eval-design.md](2026-07-20-recommendation-eval-design.md) Out of Scope) — with only 3 rows collected so far, and two of those rows flagging scale-direction confusion, building the judge now (with explicit anchors) is more useful than waiting.

## Architecture

**Revision (post-review):** the original design put the judge call inside `eval_node`, before `interrupt()`. LangGraph replays a node from the top on every resume — it doesn't resume mid-function — so anything before an `interrupt()` call re-executes on resume. That made the judge call run twice per turn (once before the human sees the prompt, once again after they answer), doubling judge cost/latency and risking a second-attempt failure silently discarding a first-attempt success. Fixed by splitting the single `eval_node` into two nodes, following the graph's existing one-node-one-responsibility pattern (`agent_node` / `tools_node`):

```
judge_node (new, no interrupt — runs exactly once, LangGraph checkpoints it as a completed step):
  context = {query, optimize_for, recommendation}
  judge_score = judge_recommendation(client, model_config, context)   # try, retry once on failure, else None + console message
  → { judge_score }

eval_node (interrupt only — the only code before interrupt() is building the same context dict, which is pure/side-effect-free and safe to replay):
  context = {query, optimize_for, recommendation}
  human_score = interrupt(context)     # human doesn't see judge_score (avoid anchoring)
  record_score(trace_id, context, human_score, state["judge_score"])
```

`judge_score` is carried from `judge_node` to `eval_node` via `GraphState`, the same mechanism already used for `last_search_input`/`last_search_results` between `agent_node` and `tools_node`. Graph edges: `agent` → (conditional) → `judge` → `eval` → `END`, replacing the old `agent` → (conditional) → `eval` → `END`.

The human interrupt prompt is updated to state the scale direction explicitly (1 = poor/unhelpful, 5 = excellent/highly useful), fixing the ambiguity noted in two of the three existing `scores.jsonl` rows.

## Components

### `agent.py` — new

```python
@dataclass
class JudgeScore:
    relevance: int   # 1-5, 1=poor match to query, 5=excellent match
    fit: int         # 1-5, 1=ignores optimize_for, 5=perfectly matches it
    quality: int     # 1-5, 1=unclear/incomplete presentation, 5=clear & complete
    overall: float   # average of the three, rounded to 1 decimal
    note: str        # short rationale naming what was weighed

def judge_recommendation(client, model_config, context: dict) -> JudgeScore | None:
    """Ask the LLM to score its own session's recommendation. Retries once on
    failure (API error or unparseable JSON). Returns None if both attempts fail —
    caller must not treat that as fatal."""
```

Prompt sent to the judge includes `query`, `optimize_for`, and `recommendation`, with the same fixed anchor language used in the human prompt, and asks for structured JSON output (`{"relevance": int, "fit": int, "quality": int, "note": str}`); `overall` is computed locally as the mean, not asked of the model.

**Revision (post-review):** the parser strips a leading/trailing markdown code fence (` ```json ` / ` ``` `) before `json.loads`, matching the normalization already done by the sibling judge `tools/cache_judge.py`. Without this, a model that wraps its JSON in a fence fails deterministically on both the first attempt and the identical retry, burning both attempts on a recoverable formatting quirk rather than a real error.

### `agent.py` — `record_score()` changes

Signature becomes `record_score(trace_id, context, human_score: EvalScore | None, judge_score: JudgeScore | None) -> None`.

- **Langfuse**: existing `usefulness` score (from `human_score`) unchanged; adds four more `create_score()` calls when `judge_score` is not `None`: `judge_relevance`, `judge_fit`, `judge_quality`, `judge_overall`. Each wrapped in the existing try/except-and-continue pattern.
- **Local JSONL** (`evals/scores.jsonl`): existing fields (`timestamp`, `query`, `optimize_for`, `recommendation`, `overall`, `note`) unchanged; adds `judge_relevance`, `judge_fit`, `judge_quality`, `judge_overall`, `judge_note` — all `None` when the judge failed twice.

### `graph.py` — `judge_node` (new) and `eval_node` changes

- `judge_node(state, runtime) -> dict`: builds the context dict, calls `agent.judge_recommendation(...)` wrapped so a judge exception never escapes the node, prints a console message on failure (`judge_score is None`, whether from an exception or from `judge_recommendation`'s own retry-exhaustion), returns `{"judge_score": judge_score}`. Runs exactly once — no `interrupt()` in this node, so LangGraph checkpoints it as a completed step and never replays it.
- `eval_node(state) -> dict`: rebuilds the same context dict, calls `interrupt(context)` for the human score, then `record_score(trace_id, context, human_score, state["judge_score"])`. No longer needs `runtime` — the judge already ran in `judge_node`.
- `GraphState` gains a `judge_score: "JudgeScore | None"` field, carried from `judge_node` to `eval_node` the same way `last_search_input`/`last_search_results` already carry data between `agent_node` and `tools_node`.
- `route_after_agent` returns `"judge"` (was `"eval"`) when a turn produced a recommendation. `build_graph()` edges become `agent` → (conditional) → `judge` → `eval` → `END` (previously `agent` → (conditional) → `eval` → `END`).

## Error Handling

| Scenario | Behavior |
|---|---|
| Judge LLM call raises or returns unparseable JSON | Retry once |
| Judge fails on retry too | Print console message, continue with `judge_score=None`, human interrupt still runs |
| Langfuse judge score calls fail | Catch, continue (matches existing pattern) |
| JSONL write fails | Print warning, continue (existing behavior, unchanged) |

## Testing

Extend `test_graph.py` / `test_main.py` (mocked LLM client, no real API calls):
- Judge succeeds → both human and judge fields present in the recorded row.
- Judge returns malformed JSON once, then valid JSON on retry → succeeds, one retry observed.
- Judge fails twice → `judge_*` fields are `None`, console message printed, human interrupt still proceeds normally, turn completes.
- `record_score()` writes all fields correctly, including when `judge_score is None`.

## Out of Scope (new backlog items)

- **Self-grading bias**: the judge currently reuses the active session's own model/provider, so a recommendation can be graded by the same model that produced it. Future work: dedicated fixed judge model, independent of session choice.
- **Async judge call**: judge call is synchronous and blocks before the human interrupt fires. Future work: run it concurrently with, or after, the human interrupt so it doesn't add latency to the human's wait.
- Human interrupt switching to per-criterion scoring (kept as single overall 1-5 for now, per this design).
- Confirmation-before-search interrupt (unrelated backlog item, see [[project-anz-agent-backlog]]).
- Search result caching, long-term cross-session memory (unrelated backlog items).
