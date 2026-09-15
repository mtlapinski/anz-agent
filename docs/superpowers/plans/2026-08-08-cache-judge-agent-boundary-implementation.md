# Cache-Judge Agent Boundary (Phase 1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bring `tools/cache_judge.py`'s fuzzy-match judge up to the same bar as `agent.py`'s LLM judge (structured result, 2-attempt retry, Langfuse tracing), then extract it into a standalone, network-invoked FastAPI service gated behind `CACHE_JUDGE_URL` — a rollback-able canary for the network-hop pattern that a later AWS Bedrock AgentCore migration will reuse.

**Architecture:** `tools/cache_judge.py` gets a structured `CacheMatch` result, 2-attempt retry, and a Langfuse `generation` observation, using a new shared `tracing.py` singleton (deduped out of `agent.py`). A new `tools/cache_judge_client.py` dispatcher — same name/signature as `find_match()` — reads `CACHE_JUDGE_URL` and either calls the judge in-process or POSTs to a new standalone `judge_service.py` (same Docker image, different `uvicorn` command, no host port). `tools/cache.py` swaps its import to the dispatcher; `trace_id` is threaded down through `agent.run_tool()` → `tools/amazon.py`'s `search_amazon()` → `tools/cache.py`'s `lookup()` so the judge call lands in the same Langfuse trace as the rest of the turn.

**Tech Stack:** Python 3.12, FastAPI + uvicorn (already a dependency), httpx (already a dependency), Langfuse SDK, pytest + unittest.mock, Docker Compose.

## Global Constraints

- **2-attempt retry** for the judge LLM call, matching `agent.judge_recommendation()`'s retry count — not more, not fewer.
- A retry is triggered by **either** the LLM call raising **or** the answer being neither `"NONE"` nor an exact candidate string (malformed/hallucinated).
- A literal `"NONE"` answer is a **successful** judge decision (`CacheMatch(None, "no_match")`) and is **never** retried.
- No candidates → `CacheMatch(None, "no_match")` immediately — no LLM call, no Langfuse span.
- Both attempts exhausted with no valid answer → `CacheMatch(None, "error")`.
- `JUDGE_MODEL_CONFIG` (`gemini-flash-lite-latest`) and `SYSTEM_PROMPT` are unchanged — this is a harness change only, never a judge-behavior change.
- Langfuse observation: `name="cache_judge"`, `as_type="generation"`, opened only when `trace_id` is given. Span `create`/`update`/`end` are each wrapped in their **own** try/except — a Langfuse outage must never break a cache lookup.
- `cache_judge_client.find_match()` has the **same name and signature** as `cache_judge.find_match()` so `tools/cache.py`'s import is a one-line swap.
- Network POST uses a module-level `httpx.Client` singleton, **15s timeout**, **no HTTP-level retry** (the service already retries the LLM call internally — stacking retries risks 4 LLM attempts for one lookup).
- Any client-side failure (timeout, connection error, non-2xx, malformed JSON) fails open to `CacheMatch(None, "error")` — it must **never** fall back to calling `cache_judge.find_match()` in-process, which would blur the canary signal.
- `judge_service.py` exposes exactly one endpoint, `POST /match`, body `{query, candidates, trace_id}`, response `{matched_query, outcome}`. No DB access, no other endpoints, no state, no auth (v1 — see spec's Out of Scope section).
- `judge` runs from the **same** `Dockerfile`, only the `command:` differs (`uvicorn judge_service:app --port 8001`). It gets **no** `ports:` entry in `docker-compose.yml` — reachable only from the Compose network, same trust level as `postgres`.
- `tools/cache.py`'s `store()` is **not** touched — it never calls the judge, so it needs no `trace_id` parameter.
- CI never sets `CACHE_JUDGE_URL` — the existing test suite must stay Docker-free and byte-for-byte unaffected when the var is unset.

---

### Task 1: `tracing.py` — shared Langfuse client singleton

**Files:**
- Create: `tracing.py`
- Modify: `agent.py:1-8` (imports), `agent.py:71-82` (`_langfuse` global + `_get_langfuse()`)
- Test: `tests/test_tracing.py` (new)

**Interfaces:**
- Produces: `tracing.get_langfuse() -> Langfuse` — module-level singleton, lazily constructed from `LANGFUSE_PUBLIC_KEY`/`LANGFUSE_SECRET_KEY`/`LANGFUSE_HOST` env vars.
- Consumes: nothing new (this is a lift of existing `agent._get_langfuse()` code, verbatim).

This is a pure dedup: `agent.py` keeps its `_get_langfuse` name (existing call sites in `graph.py`, `main.py`, and all of `test_agent.py`'s `@patch("agent._get_langfuse")` decorators keep working unchanged) but the implementation now lives in `tracing.py` so `tools/cache_judge.py` (Task 2) can reuse it without importing `agent.py`.

- [ ] **Step 1: Write the failing test for `tracing.get_langfuse()`**

Create `tests/test_tracing.py`:

```python
import os
from unittest.mock import patch
import pytest


@pytest.fixture(autouse=True)
def reset_singleton():
    import tracing
    tracing._langfuse = None
    yield
    tracing._langfuse = None


@patch.dict(os.environ, {
    "LANGFUSE_PUBLIC_KEY": "pk-test",
    "LANGFUSE_SECRET_KEY": "sk-test",
})
@patch("tracing.Langfuse")
def test_get_langfuse_constructs_client_from_env(mock_langfuse_class):
    import tracing
    mock_langfuse_class.return_value = "the-client"

    result = tracing.get_langfuse()

    mock_langfuse_class.assert_called_once_with(
        public_key="pk-test",
        secret_key="sk-test",
        host="https://cloud.langfuse.com",
    )
    assert result == "the-client"


@patch.dict(os.environ, {
    "LANGFUSE_PUBLIC_KEY": "pk-test",
    "LANGFUSE_SECRET_KEY": "sk-test",
    "LANGFUSE_HOST": "https://self-hosted.example.com",
})
@patch("tracing.Langfuse")
def test_get_langfuse_respects_custom_host(mock_langfuse_class):
    import tracing
    tracing.get_langfuse()

    mock_langfuse_class.assert_called_once_with(
        public_key="pk-test",
        secret_key="sk-test",
        host="https://self-hosted.example.com",
    )


@patch.dict(os.environ, {
    "LANGFUSE_PUBLIC_KEY": "pk-test",
    "LANGFUSE_SECRET_KEY": "sk-test",
})
@patch("tracing.Langfuse")
def test_get_langfuse_is_a_singleton(mock_langfuse_class):
    import tracing
    mock_langfuse_class.return_value = "the-client"

    first = tracing.get_langfuse()
    second = tracing.get_langfuse()

    mock_langfuse_class.assert_called_once()
    assert first is second
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_tracing.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'tracing'`

- [ ] **Step 3: Create `tracing.py`**

```python
import os
from langfuse import Langfuse

_langfuse: Langfuse | None = None


def get_langfuse() -> Langfuse:
    global _langfuse
    if _langfuse is None:
        _langfuse = Langfuse(
            public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
            secret_key=os.environ["LANGFUSE_SECRET_KEY"],
            host=os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com"),
        )
    return _langfuse
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_tracing.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Point `agent.py` at the shared singleton**

In `agent.py`, replace the `Langfuse` import and the `_langfuse`/`_get_langfuse()` block:

```python
# Before (agent.py:1-8):
import json
import os
import llm
from dataclasses import dataclass
from datetime import datetime, timezone
from langfuse import Langfuse
from llm import ModelConfig
from tools.amazon import search_amazon
```

becomes:

```python
import json
import os
import llm
from dataclasses import dataclass
from datetime import datetime, timezone
from llm import ModelConfig
from tools.amazon import search_amazon
from tracing import get_langfuse as _get_langfuse
```

And delete the old definition entirely:

```python
# Delete this whole block (agent.py:71-82):
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
```

Nothing else in `agent.py` changes — `run_tool()`, `record_score()`, etc. keep calling `_get_langfuse()` exactly as before, and `graph.py`/`main.py` keep calling `agent._get_langfuse()` exactly as before.

- [ ] **Step 6: Run the full existing test suite to verify zero behavior change**

Run: `pytest tests/test_agent.py tests/test_graph.py tests/test_main.py -v`
Expected: PASS — all existing Langfuse-related tests (`test_run_tool_creates_langfuse_span_with_trace_id`, `test_record_score_calls_langfuse_create_score`, etc.) pass unchanged, since `@patch("agent._get_langfuse")` still targets a real module-level name in `agent`.

- [ ] **Step 7: Commit**

```bash
git add tracing.py tests/test_tracing.py agent.py
git commit -m "refactor: extract shared Langfuse singleton into tracing.py"
```

---

### Task 2: `tools/cache_judge.py` — structured result, retry, tracing

**Files:**
- Modify: `tools/cache_judge.py` (entire `find_match()` function and its return type)
- Test: `tests/test_cache_judge.py` (rewrite)

**Interfaces:**
- Consumes: `tracing.get_langfuse() -> Langfuse` (Task 1).
- Produces: `CacheMatch` dataclass (`matched_query: str | None`, `outcome: str` — one of `"matched"`, `"no_match"`, `"error"`); `find_match(query: str, candidates: list[str], trace_id: str | None = None) -> CacheMatch` (module `tools.cache_judge`). Consumed by Task 3 and Task 7.

- [ ] **Step 1: Write the failing tests**

Replace `tests/test_cache_judge.py` entirely:

```python
import pytest
from unittest.mock import MagicMock, patch
from llm import LLMResponse
import tools.cache_judge as cache_judge
from tools.cache_judge import CacheMatch


@pytest.fixture(autouse=True)
def reset_client():
    cache_judge._client = None
    yield
    cache_judge._client = None


@patch("tools.cache_judge.llm")
def test_find_match_returns_matched_candidate(mock_llm):
    mock_llm.complete.return_value = LLMResponse(
        text="purple balance beam", tool_calls=None, input_tokens=10, output_tokens=2
    )

    result = cache_judge.find_match("balance beam", ["purple balance beam", "yoga mat"])

    assert result == CacheMatch("purple balance beam", "matched")


@patch("tools.cache_judge.llm")
def test_find_match_returns_no_match_when_judge_says_none(mock_llm):
    mock_llm.complete.return_value = LLMResponse(
        text="NONE", tool_calls=None, input_tokens=10, output_tokens=2
    )

    result = cache_judge.find_match("kettlebell", ["yoga mat"])

    assert result == CacheMatch(None, "no_match")
    mock_llm.complete.assert_called_once()  # a literal NONE is not retried


@patch("tools.cache_judge.llm")
def test_find_match_retries_then_succeeds_on_malformed_first_answer(mock_llm):
    mock_llm.complete.side_effect = [
        LLMResponse(text="something not in the candidate list", tool_calls=None, input_tokens=10, output_tokens=2),
        LLMResponse(text="yoga mat", tool_calls=None, input_tokens=10, output_tokens=2),
    ]

    result = cache_judge.find_match("kettlebell", ["yoga mat"])

    assert result == CacheMatch("yoga mat", "matched")
    assert mock_llm.complete.call_count == 2


@patch("tools.cache_judge.llm")
def test_find_match_retries_then_succeeds_on_transient_exception(mock_llm):
    mock_llm.complete.side_effect = [
        Exception("rate limited"),
        LLMResponse(text="yoga mat", tool_calls=None, input_tokens=10, output_tokens=2),
    ]

    result = cache_judge.find_match("kettlebell", ["yoga mat"])

    assert result == CacheMatch("yoga mat", "matched")
    assert mock_llm.complete.call_count == 2


@patch("tools.cache_judge.llm")
def test_find_match_returns_error_when_both_attempts_raise(mock_llm):
    mock_llm.complete.side_effect = Exception("rate limited")

    result = cache_judge.find_match("kettlebell", ["yoga mat"])

    assert result == CacheMatch(None, "error")
    assert mock_llm.complete.call_count == 2


@patch("tools.cache_judge.llm")
def test_find_match_returns_error_when_both_answers_malformed(mock_llm):
    mock_llm.complete.return_value = LLMResponse(
        text="something not in the candidate list", tool_calls=None, input_tokens=10, output_tokens=2
    )

    result = cache_judge.find_match("kettlebell", ["yoga mat"])

    assert result == CacheMatch(None, "error")
    assert mock_llm.complete.call_count == 2


@patch("tools.cache_judge.llm")
def test_find_match_returns_error_on_client_creation_error(mock_llm):
    mock_llm.create_client.side_effect = Exception("missing GOOGLE_API_KEY")

    result = cache_judge.find_match("kettlebell", ["yoga mat"])

    assert result == CacheMatch(None, "error")


def test_find_match_returns_no_match_when_no_candidates():
    result = cache_judge.find_match("kettlebell", [])
    assert result == CacheMatch(None, "no_match")


@patch("tools.cache_judge.get_langfuse")
def test_find_match_no_span_when_no_candidates_even_with_trace_id(mock_get_langfuse):
    result = cache_judge.find_match("kettlebell", [], trace_id="trace-123")

    assert result == CacheMatch(None, "no_match")
    mock_get_langfuse.assert_not_called()


@patch("tools.cache_judge.get_langfuse")
@patch("tools.cache_judge.llm")
def test_find_match_opens_and_ends_langfuse_generation_with_trace_id(mock_llm, mock_get_langfuse):
    mock_llm.complete.return_value = LLMResponse(
        text="yoga mat", tool_calls=None, input_tokens=10, output_tokens=2
    )
    mock_span = MagicMock()
    mock_get_langfuse.return_value.start_observation.return_value = mock_span

    cache_judge.find_match("kettlebell", ["yoga mat"], trace_id="trace-123")

    mock_get_langfuse.return_value.start_observation.assert_called_once_with(
        trace_context={"trace_id": "trace-123"},
        name="cache_judge",
        as_type="generation",
        input={"query": "kettlebell", "candidates": ["yoga mat"]},
        model="google/gemini-flash-lite-latest",
    )
    mock_span.update.assert_called_once_with(
        output={"matched_query": "yoga mat", "outcome": "matched", "attempts": 1}
    )
    mock_span.end.assert_called_once()


@patch("tools.cache_judge.get_langfuse")
@patch("tools.cache_judge.llm")
def test_find_match_no_span_when_trace_id_none(mock_llm, mock_get_langfuse):
    mock_llm.complete.return_value = LLMResponse(
        text="yoga mat", tool_calls=None, input_tokens=10, output_tokens=2
    )

    cache_judge.find_match("kettlebell", ["yoga mat"])

    mock_get_langfuse.assert_not_called()


@patch("tools.cache_judge.get_langfuse")
@patch("tools.cache_judge.llm")
def test_find_match_fails_open_when_span_creation_raises(mock_llm, mock_get_langfuse):
    mock_llm.complete.return_value = LLMResponse(
        text="yoga mat", tool_calls=None, input_tokens=10, output_tokens=2
    )
    mock_get_langfuse.return_value.start_observation.side_effect = Exception("langfuse down")

    result = cache_judge.find_match("kettlebell", ["yoga mat"], trace_id="trace-123")

    assert result == CacheMatch("yoga mat", "matched")


@patch("tools.cache_judge.get_langfuse")
@patch("tools.cache_judge.llm")
def test_find_match_fails_open_when_span_update_raises(mock_llm, mock_get_langfuse):
    mock_llm.complete.return_value = LLMResponse(
        text="yoga mat", tool_calls=None, input_tokens=10, output_tokens=2
    )
    mock_span = MagicMock()
    mock_span.update.side_effect = Exception("langfuse down")
    mock_get_langfuse.return_value.start_observation.return_value = mock_span

    result = cache_judge.find_match("kettlebell", ["yoga mat"], trace_id="trace-123")

    assert result == CacheMatch("yoga mat", "matched")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_cache_judge.py -v`
Expected: FAIL — `ImportError: cannot import name 'CacheMatch' from 'tools.cache_judge'`

- [ ] **Step 3: Rewrite `tools/cache_judge.py`**

```python
from __future__ import annotations
from dataclasses import dataclass
import llm
from tracing import get_langfuse

JUDGE_MODEL_CONFIG = llm.ModelConfig(provider="google", model="gemini-flash-lite-latest")

SYSTEM_PROMPT = (
    "You match a new Amazon product search query against a list of previously cached "
    "search queries, to decide whether a prior search's results can be reused instead "
    "of running a fresh search.\n\n"
    "Two queries match if they describe the same or closely related search intent, "
    "regardless of word order, extra descriptive words (e.g. color, brand, size), or "
    "minor rewording. For example, \"balance beam\" matches \"purple balance beam\", "
    "and \"purple balance beam\" matches \"balance beam purple\".\n\n"
    "Respond with ONLY the exact text of the single best matching cached query, copied "
    "verbatim from the list below. If none of the cached queries are a good match, "
    "respond with exactly: NONE\n"
    "Do not include any other text, explanation, or punctuation in your response."
)

_client = None


def _get_client():
    global _client
    if _client is None:
        _client = llm.create_client(JUDGE_MODEL_CONFIG)
    return _client


@dataclass
class CacheMatch:
    matched_query: str | None   # cached query text if judge found a match, else None
    outcome: str                 # "matched" | "no_match" | "error"


def find_match(query: str, candidates: list[str], trace_id: str | None = None) -> CacheMatch:
    if not candidates:
        return CacheMatch(None, "no_match")

    span = None
    if trace_id:
        try:
            span = get_langfuse().start_observation(
                trace_context={"trace_id": trace_id},
                name="cache_judge",
                as_type="generation",
                input={"query": query, "candidates": candidates},
                model=f"{JUDGE_MODEL_CONFIG.provider}/{JUDGE_MODEL_CONFIG.model}",
            )
        except Exception:
            span = None

    result, attempts_used = _run_attempts(query, candidates)

    if span:
        try:
            span.update(output={
                "matched_query": result.matched_query,
                "outcome": result.outcome,
                "attempts": attempts_used,
            })
            span.end()
        except Exception:
            pass

    return result


def _run_attempts(query: str, candidates: list[str]) -> tuple[CacheMatch, int]:
    candidate_list = "\n".join(f"- {c}" for c in candidates)
    user_message = f"New query: {query}\n\nCached queries:\n{candidate_list}"

    attempts_used = 0
    for attempt in range(2):
        attempts_used = attempt + 1
        try:
            response = llm.complete(
                _get_client(),
                JUDGE_MODEL_CONFIG,
                SYSTEM_PROMPT,
                [],
                [{"role": "user", "content": user_message}],
            )
            answer = (response.text or "").strip()
        except Exception:
            continue

        if answer == "NONE":
            return CacheMatch(None, "no_match"), attempts_used
        if answer in candidates:
            return CacheMatch(answer, "matched"), attempts_used
        # malformed/hallucinated answer — falls through, retried if attempts remain

    return CacheMatch(None, "error"), attempts_used
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_cache_judge.py -v`
Expected: PASS (14 tests)

- [ ] **Step 5: Commit**

```bash
git add tools/cache_judge.py tests/test_cache_judge.py
git commit -m "feat: give cache_judge.find_match() retry, structured result, and tracing"
```

---

### Task 3: `tools/cache_judge_client.py` — the CACHE_JUDGE_URL dispatcher

**Files:**
- Create: `tools/cache_judge_client.py`
- Test: `tests/test_cache_judge_client.py` (new)

**Interfaces:**
- Consumes: `tools.cache_judge.CacheMatch`, `tools.cache_judge.find_match(query, candidates, trace_id=None) -> CacheMatch` (Task 2).
- Produces: `find_match(query: str, candidates: list[str], trace_id: str | None = None) -> CacheMatch` (module `tools.cache_judge_client`). Consumed by Task 4.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_cache_judge_client.py`:

```python
import httpx
import pytest
from unittest.mock import MagicMock, patch
from tools.cache_judge import CacheMatch
import tools.cache_judge_client as cache_judge_client


@pytest.fixture(autouse=True)
def reset_client():
    cache_judge_client._client = None
    yield
    cache_judge_client._client = None


def test_find_match_dispatches_in_process_when_url_unset(monkeypatch):
    monkeypatch.delenv("CACHE_JUDGE_URL", raising=False)
    with patch(
        "tools.cache_judge_client._find_match_in_process",
        return_value=CacheMatch("yoga mat", "matched"),
    ) as mock_find:
        result = cache_judge_client.find_match("kettlebell", ["yoga mat"], trace_id="trace-1")

    mock_find.assert_called_once_with("kettlebell", ["yoga mat"], trace_id="trace-1")
    assert result == CacheMatch("yoga mat", "matched")


def test_find_match_posts_to_service_when_url_set(monkeypatch):
    monkeypatch.setenv("CACHE_JUDGE_URL", "http://judge:8001/match")
    mock_response = MagicMock()
    mock_response.raise_for_status.return_value = None
    mock_response.json.return_value = {"matched_query": "yoga mat", "outcome": "matched"}

    with patch.object(httpx.Client, "post", return_value=mock_response) as mock_post:
        result = cache_judge_client.find_match("kettlebell", ["yoga mat"], trace_id="trace-1")

    mock_post.assert_called_once_with(
        "http://judge:8001/match",
        json={"query": "kettlebell", "candidates": ["yoga mat"], "trace_id": "trace-1"},
    )
    assert result == CacheMatch("yoga mat", "matched")


def test_find_match_forwards_none_trace_id_when_url_set(monkeypatch):
    monkeypatch.setenv("CACHE_JUDGE_URL", "http://judge:8001/match")
    mock_response = MagicMock()
    mock_response.raise_for_status.return_value = None
    mock_response.json.return_value = {"matched_query": None, "outcome": "no_match"}

    with patch.object(httpx.Client, "post", return_value=mock_response) as mock_post:
        cache_judge_client.find_match("kettlebell", ["yoga mat"])

    mock_post.assert_called_once_with(
        "http://judge:8001/match",
        json={"query": "kettlebell", "candidates": ["yoga mat"], "trace_id": None},
    )


def test_find_match_fails_open_on_timeout(monkeypatch):
    monkeypatch.setenv("CACHE_JUDGE_URL", "http://judge:8001/match")
    with patch.object(httpx.Client, "post", side_effect=httpx.TimeoutException("timed out")):
        result = cache_judge_client.find_match("kettlebell", ["yoga mat"])

    assert result == CacheMatch(None, "error")


def test_find_match_fails_open_on_connection_error(monkeypatch):
    monkeypatch.setenv("CACHE_JUDGE_URL", "http://judge:8001/match")
    with patch.object(httpx.Client, "post", side_effect=httpx.ConnectError("refused")):
        result = cache_judge_client.find_match("kettlebell", ["yoga mat"])

    assert result == CacheMatch(None, "error")


def test_find_match_fails_open_on_non_2xx(monkeypatch):
    monkeypatch.setenv("CACHE_JUDGE_URL", "http://judge:8001/match")
    mock_response = MagicMock()
    mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
        "server error", request=MagicMock(), response=MagicMock(status_code=500)
    )

    with patch.object(httpx.Client, "post", return_value=mock_response):
        result = cache_judge_client.find_match("kettlebell", ["yoga mat"])

    assert result == CacheMatch(None, "error")


def test_find_match_fails_open_on_malformed_json(monkeypatch):
    monkeypatch.setenv("CACHE_JUDGE_URL", "http://judge:8001/match")
    mock_response = MagicMock()
    mock_response.raise_for_status.return_value = None
    mock_response.json.side_effect = ValueError("not json")

    with patch.object(httpx.Client, "post", return_value=mock_response):
        result = cache_judge_client.find_match("kettlebell", ["yoga mat"])

    assert result == CacheMatch(None, "error")


def test_find_match_never_falls_back_to_in_process_on_network_failure(monkeypatch):
    """A network failure must fail open to an error result, not silently retry
    in-process — that would blur the CACHE_JUDGE_URL canary signal."""
    monkeypatch.setenv("CACHE_JUDGE_URL", "http://judge:8001/match")
    with patch.object(httpx.Client, "post", side_effect=httpx.ConnectError("refused")), \
         patch("tools.cache_judge_client._find_match_in_process") as mock_in_process:
        cache_judge_client.find_match("kettlebell", ["yoga mat"])

    mock_in_process.assert_not_called()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_cache_judge_client.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'tools.cache_judge_client'`

- [ ] **Step 3: Create `tools/cache_judge_client.py`**

```python
from __future__ import annotations
import os
import httpx
from tools.cache_judge import CacheMatch, find_match as _find_match_in_process

_client: httpx.Client | None = None


def _get_client() -> httpx.Client:
    global _client
    if _client is None:
        _client = httpx.Client(timeout=15.0)
    return _client


def find_match(query: str, candidates: list[str], trace_id: str | None = None) -> CacheMatch:
    judge_url = os.environ.get("CACHE_JUDGE_URL")
    if not judge_url:
        return _find_match_in_process(query, candidates, trace_id=trace_id)

    try:
        response = _get_client().post(
            judge_url,
            json={"query": query, "candidates": candidates, "trace_id": trace_id},
        )
        response.raise_for_status()
        data = response.json()
        return CacheMatch(matched_query=data["matched_query"], outcome=data["outcome"])
    except Exception:
        return CacheMatch(None, "error")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_cache_judge_client.py -v`
Expected: PASS (8 tests)

- [ ] **Step 5: Commit**

```bash
git add tools/cache_judge_client.py tests/test_cache_judge_client.py
git commit -m "feat: add cache_judge_client dispatcher gated on CACHE_JUDGE_URL"
```

---

### Task 4: `tools/cache.py` — switch to the dispatcher, thread `trace_id`

**Files:**
- Modify: `tools/cache.py:1-10` (imports), `tools/cache.py:37-38` (`lookup`), `tools/cache.py:68-95` (`_lookup_sqlite`), `tools/cache.py:116-140` (`_lookup_postgres`)
- Test: `tests/test_cache.py` (update mock targets + add `trace_id`-forwarding tests)

**Interfaces:**
- Consumes: `tools.cache_judge_client.find_match(query, candidates, trace_id=None) -> CacheMatch` (Task 3).
- Produces: `lookup(query: str, trace_id: str | None = None) -> list[dict] | None` (module `tools.cache`). Consumed by Task 5.

- [ ] **Step 1: Update existing mock targets and add new tests in `tests/test_cache.py`**

In `tests/test_cache.py`, every `patch("tools.cache_judge.find_match", ...)` moves to `patch("tools.cache_judge_client.find_match", ...)`, and every string return value (`"purple balance beam"`, `None`) becomes a `CacheMatch`. Apply these edits:

```python
# Add this import near the top of tests/test_cache.py:
from tools.cache_judge import CacheMatch
```

```python
# test_lookup_uses_judge_for_fuzzy_match — before:
def test_lookup_uses_judge_for_fuzzy_match():
    from tools.cache import store, lookup
    store("purple balance beam", [{"title": "Purple Beam"}])

    with patch("tools.cache_judge.find_match", return_value="purple balance beam"):
        result = lookup("balance beam")

    assert result == [{"title": "Purple Beam"}]

# after:
def test_lookup_uses_judge_for_fuzzy_match():
    from tools.cache import store, lookup
    store("purple balance beam", [{"title": "Purple Beam"}])

    with patch("tools.cache_judge_client.find_match", return_value=CacheMatch("purple balance beam", "matched")):
        result = lookup("balance beam")

    assert result == [{"title": "Purple Beam"}]
```

```python
# test_lookup_returns_none_when_judge_finds_no_match — before:
def test_lookup_returns_none_when_judge_finds_no_match():
    from tools.cache import store, lookup
    store("yoga mat", [{"title": "Mat"}])

    with patch("tools.cache_judge.find_match", return_value=None):
        result = lookup("kettlebell")

    assert result is None

# after:
def test_lookup_returns_none_when_judge_finds_no_match():
    from tools.cache import store, lookup
    store("yoga mat", [{"title": "Mat"}])

    with patch("tools.cache_judge_client.find_match", return_value=CacheMatch(None, "no_match")):
        result = lookup("kettlebell")

    assert result is None
```

```python
# test_lookup_does_not_call_judge_when_cache_empty — before:
def test_lookup_does_not_call_judge_when_cache_empty():
    from tools.cache import lookup

    with patch("tools.cache_judge.find_match") as mock_find_match:
        result = lookup("anything")

    assert result is None
    mock_find_match.assert_not_called()

# after:
def test_lookup_does_not_call_judge_when_cache_empty():
    from tools.cache import lookup

    with patch("tools.cache_judge_client.find_match") as mock_find_match:
        result = lookup("anything")

    assert result is None
    mock_find_match.assert_not_called()
```

```python
# test_lookup_returns_none_when_judge_raises — before:
def test_lookup_returns_none_when_judge_raises():
    from tools.cache import store, lookup
    store("yoga mat", [{"title": "Mat"}])

    with patch("tools.cache_judge.find_match", side_effect=Exception("boom")):
        result = lookup("kettlebell")

    assert result is None

# after: identical body, just the patch target:
def test_lookup_returns_none_when_judge_raises():
    from tools.cache import store, lookup
    store("yoga mat", [{"title": "Mat"}])

    with patch("tools.cache_judge_client.find_match", side_effect=Exception("boom")):
        result = lookup("kettlebell")

    assert result is None
```

```python
# test_lookup_sends_only_shortlisted_candidates_to_judge — before:
def test_lookup_sends_only_shortlisted_candidates_to_judge():
    from tools.cache import store, lookup
    for i in range(30):
        store(f"unrelated product {i}", [{"title": f"p{i}"}])
    store("purple balance beam", [{"title": "Purple Beam"}])

    with patch("tools.cache_judge.find_match", return_value=None) as mock_find_match:
        lookup("balance beam")

    args, _ = mock_find_match.call_args
    assert len(args[1]) <= 20

# after: identical body, just the patch target and return value:
def test_lookup_sends_only_shortlisted_candidates_to_judge():
    from tools.cache import store, lookup
    for i in range(30):
        store(f"unrelated product {i}", [{"title": f"p{i}"}])
    store("purple balance beam", [{"title": "Purple Beam"}])

    with patch("tools.cache_judge_client.find_match", return_value=CacheMatch(None, "no_match")) as mock_find_match:
        lookup("balance beam")

    args, _ = mock_find_match.call_args
    assert len(args[1]) <= 20
```

Add a new test for `trace_id` forwarding, right after `test_lookup_sends_only_shortlisted_candidates_to_judge`:

```python
def test_lookup_forwards_trace_id_to_judge():
    from tools.cache import store, lookup
    store("yoga mat", [{"title": "Mat"}])

    with patch("tools.cache_judge_client.find_match", return_value=CacheMatch(None, "no_match")) as mock_find_match:
        lookup("kettlebell", trace_id="trace-123")

    _, kwargs = mock_find_match.call_args
    assert kwargs["trace_id"] == "trace-123"


def test_lookup_forwards_none_trace_id_by_default():
    from tools.cache import store, lookup
    store("yoga mat", [{"title": "Mat"}])

    with patch("tools.cache_judge_client.find_match", return_value=CacheMatch(None, "no_match")) as mock_find_match:
        lookup("kettlebell")

    _, kwargs = mock_find_match.call_args
    assert kwargs["trace_id"] is None
```

In the `TestPostgresBackend` class, update its `test_lookup_uses_judge_for_fuzzy_match` the same way:

```python
# before:
    def test_lookup_uses_judge_for_fuzzy_match(self):
        from tools.cache import store, lookup
        store("purple balance beam", [{"title": "Purple Beam"}])

        with patch("tools.cache_judge.find_match", return_value="purple balance beam"):
            result = lookup("balance beam")

        assert result == [{"title": "Purple Beam"}]

# after:
    def test_lookup_uses_judge_for_fuzzy_match(self):
        from tools.cache import store, lookup
        store("purple balance beam", [{"title": "Purple Beam"}])

        with patch("tools.cache_judge_client.find_match", return_value=CacheMatch("purple balance beam", "matched")):
            result = lookup("balance beam")

        assert result == [{"title": "Purple Beam"}]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_cache.py -v -k "not TestPostgresBackend"`
Expected: FAIL — `tools/cache.py` still imports `tools.cache_judge` directly and calls its old 2-arg `find_match(query, shortlist)`, so patching `tools.cache_judge_client.find_match` never intercepts the real call. The real (unpatched) `cache_judge.find_match()` runs instead, and since no `GOOGLE_API_KEY` is configured in the test environment it fails and `lookup()`'s outer `except Exception` swallows it — tests see `AssertionError`s (mocked return values like `CacheMatch("purple balance beam", "matched")` never take effect) and `mock_find_match.assert_not_called()`-style assertions fail because the mock was never the function actually invoked.

- [ ] **Step 3: Update `tools/cache.py`**

```python
# Change the import (tools/cache.py:1-9):
from __future__ import annotations
import json
import os
import re
import sqlite3
from datetime import datetime, timezone

import db as db_module
from tools import cache_judge_client
```

```python
# Change lookup() (tools/cache.py:37-38):
def lookup(query: str, trace_id: str | None = None) -> list[dict] | None:
    return _lookup_postgres(query, trace_id) if _using_postgres() else _lookup_sqlite(query, trace_id)
```

```python
# Change _lookup_sqlite() (tools/cache.py:68-95):
def _lookup_sqlite(query: str, trace_id: str | None = None) -> list[dict] | None:
    try:
        conn = _connect_sqlite()
        try:
            normalized = normalize(query)
            row = conn.execute(
                "SELECT raw_results FROM searches WHERE normalized_query = ?", (normalized,)
            ).fetchone()
            if row:
                return json.loads(row[0])

            candidates = [r[0] for r in conn.execute("SELECT DISTINCT query FROM searches").fetchall()]
            if not candidates:
                return None

            shortlist = _shortlist_candidates(query, candidates)
            match = cache_judge_client.find_match(query, shortlist, trace_id=trace_id)
            if match.matched_query is None:
                return None

            row = conn.execute(
                "SELECT raw_results FROM searches WHERE query = ?", (match.matched_query,)
            ).fetchone()
            return json.loads(row[0]) if row else None
        finally:
            conn.close()
    except Exception:
        return None
```

```python
# Change _lookup_postgres() (tools/cache.py:116-140):
def _lookup_postgres(query: str, trace_id: str | None = None) -> list[dict] | None:
    try:
        with db_module.get_connection() as conn:
            normalized = normalize(query)
            row = conn.execute(
                "SELECT raw_results FROM searches WHERE normalized_query = %s", (normalized,)
            ).fetchone()
            if row:
                return row[0]  # JSONB comes back already deserialized

            candidates = [r[0] for r in conn.execute("SELECT DISTINCT query FROM searches").fetchall()]
            if not candidates:
                return None

            shortlist = _shortlist_candidates(query, candidates)
            match = cache_judge_client.find_match(query, shortlist, trace_id=trace_id)
            if match.matched_query is None:
                return None

            row = conn.execute(
                "SELECT raw_results FROM searches WHERE query = %s", (match.matched_query,)
            ).fetchone()
            return row[0] if row else None
    except Exception:
        return None
```

`_store_sqlite`, `_store_postgres`, `store()`, `_shortlist_candidates()`, `normalize()`, `_using_postgres()`, `_connect_sqlite()` are all unchanged.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_cache.py -v -k "not TestPostgresBackend"`
Expected: PASS (SQLite-backed tests; the `TestPostgresBackend` class needs a running Postgres via `testcontainers` — run it too if Docker is available: `pytest tests/test_cache.py -v`)

- [ ] **Step 5: Commit**

```bash
git add tools/cache.py tests/test_cache.py
git commit -m "feat: route cache.py's judge lookups through cache_judge_client, thread trace_id"
```

---

### Task 5: `tools/amazon.py` — thread `trace_id` through `search_amazon()`

**Files:**
- Modify: `tools/amazon.py:7-19`
- Test: `tests/test_amazon.py` (update two assertions)

**Interfaces:**
- Consumes: `tools.cache.lookup(query, trace_id=None) -> list[dict] | None` (Task 4).
- Produces: `search_amazon(query, optimize_for, max_results=10, max_price=None, view=None, trace_id=None) -> dict` (module `tools.amazon`). Consumed by Task 6.

- [ ] **Step 1: Update the two assertions in `tests/test_amazon.py` that check `cache.lookup`/`cache.store` call args**

```python
# test_search_returns_cached_results_without_calling_serpapi — no change needed
# (it doesn't assert call args on cache.lookup), skip.

# test_search_stores_raw_results_after_live_call — before:
@patch.dict(os.environ, {"SERPAPI_KEY": "fake_key"})
@patch("tools.amazon.GoogleSearch")
def test_search_stores_raw_results_after_live_call(mock_search_class):
    mock_search_class.return_value.get_dict.return_value = {
        "organic_results": [make_mock_result(title="Fresh Laptop")]
    }

    from tools.amazon import search_amazon

    with patch("tools.cache.lookup", return_value=None) as mock_lookup, \
         patch("tools.cache.store") as mock_store:
        search_amazon(query="laptop", optimize_for="price", max_results=5)

    mock_lookup.assert_called_once_with("laptop")
    mock_store.assert_called_once_with("laptop", [make_mock_result(title="Fresh Laptop")])

# after:
@patch.dict(os.environ, {"SERPAPI_KEY": "fake_key"})
@patch("tools.amazon.GoogleSearch")
def test_search_stores_raw_results_after_live_call(mock_search_class):
    mock_search_class.return_value.get_dict.return_value = {
        "organic_results": [make_mock_result(title="Fresh Laptop")]
    }

    from tools.amazon import search_amazon

    with patch("tools.cache.lookup", return_value=None) as mock_lookup, \
         patch("tools.cache.store") as mock_store:
        search_amazon(query="laptop", optimize_for="price", max_results=5)

    mock_lookup.assert_called_once_with("laptop", trace_id=None)
    mock_store.assert_called_once_with("laptop", [make_mock_result(title="Fresh Laptop")])
```

Add a new test right after it:

```python
@patch.dict(os.environ, {"SERPAPI_KEY": "fake_key"})
@patch("tools.amazon.GoogleSearch")
def test_search_forwards_trace_id_to_cache_lookup(mock_search_class):
    mock_search_class.return_value.get_dict.return_value = {"organic_results": []}

    from tools.amazon import search_amazon

    with patch("tools.cache.lookup", return_value=None) as mock_lookup, \
         patch("tools.cache.store"):
        search_amazon(query="laptop", optimize_for="price", max_results=5, trace_id="trace-123")

    mock_lookup.assert_called_once_with("laptop", trace_id="trace-123")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_amazon.py -v -k "trace_id or stores_raw_results"`
Expected: FAIL — `mock_lookup.assert_called_once_with("laptop", trace_id=None)` fails because `search_amazon()` doesn't accept/forward `trace_id` yet.

- [ ] **Step 3: Update `tools/amazon.py`**

```python
import os
from typing import Optional
from serpapi import GoogleSearch
from tools import cache


def search_amazon(
    query: str,
    optimize_for: str,
    max_results: int = 10,
    max_price: Optional[float] = None,
    view: Optional[str] = None,
    trace_id: Optional[str] = None,
) -> dict:
    """
    Search Amazon via SerpAPI, via a local cache when available. optimize_for is
    passed through from the agent and used by the caller to rank results — not
    applied here. trace_id, when given, is forwarded to the cache's fuzzy-match
    judge so its Langfuse span lands in the same trace as the rest of the turn.
    """
    raw_items = cache.lookup(query, trace_id=trace_id)

    if raw_items is None:
        params = {
            "engine": "amazon",
            "k": query,
            "amazon_domain": "amazon.com",
            "api_key": os.environ["SERPAPI_KEY"],
        }

        try:
            results = GoogleSearch(params).get_dict()
        except Exception as e:
            return {"error": str(e), "products": []}

        if "error" in results:
            return {"error": results["error"], "products": []}

        raw_items = results.get("organic_results") or []
        cache.store(query, raw_items)

    return {"products": _build_products(raw_items, max_results, max_price)}
```

(`_build_products` and `_has_free_delivery` are unchanged.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_amazon.py -v`
Expected: PASS (all tests in the file)

- [ ] **Step 5: Commit**

```bash
git add tools/amazon.py tests/test_amazon.py
git commit -m "feat: thread trace_id through search_amazon() to the cache lookup"
```

---

### Task 6: `agent.py` — thread `trace_id` through `run_tool()`'s `search_amazon` call

**Files:**
- Modify: `agent.py:85-106` (`run_tool`)
- Test: `tests/test_agent.py:11-17` (update one assertion)

**Interfaces:**
- Consumes: `tools.amazon.search_amazon(..., trace_id=None) -> dict` (Task 5).
- Produces: `run_tool(tool_name: str, tool_input: dict, trace_id: str | None = None) -> str` (module `agent`) — signature unchanged, but now forwards `trace_id` one level deeper.

- [ ] **Step 1: Update the assertion in `tests/test_agent.py`**

```python
# before:
def test_run_tool_search_amazon():
    from agent import run_tool
    with patch("agent.search_amazon") as mock_search:
        mock_search.return_value = {"products": []}
        result = run_tool("search_amazon", {"query": "laptop", "optimize_for": "price", "max_results": 5})
    assert json.loads(result) == {"products": []}
    mock_search.assert_called_once_with(query="laptop", optimize_for="price", max_results=5)

# after:
def test_run_tool_search_amazon():
    from agent import run_tool
    with patch("agent.search_amazon") as mock_search:
        mock_search.return_value = {"products": []}
        result = run_tool("search_amazon", {"query": "laptop", "optimize_for": "price", "max_results": 5})
    assert json.loads(result) == {"products": []}
    mock_search.assert_called_once_with(query="laptop", optimize_for="price", max_results=5, trace_id=None)
```

Add a new test right after it, verifying `trace_id` is forwarded (not just defaulted):

```python
def test_run_tool_forwards_trace_id_to_search_amazon():
    from agent import run_tool
    with patch("agent.search_amazon") as mock_search, patch("agent._get_langfuse"):
        mock_search.return_value = {"products": []}
        run_tool("search_amazon", {"query": "laptop", "optimize_for": "price", "max_results": 5},
                  trace_id="trace-123")
    mock_search.assert_called_once_with(query="laptop", optimize_for="price", max_results=5, trace_id="trace-123")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_agent.py -v -k "search_amazon"`
Expected: FAIL — `mock_search.assert_called_once_with(..., trace_id=None)` doesn't match the actual call (no `trace_id` kwarg passed yet).

- [ ] **Step 3: Update `run_tool()` in `agent.py`**

```python
# before (agent.py:85-106):
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

# after: only the search_amazon(**tool_input) call changes:
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
        result = search_amazon(**tool_input, trace_id=trace_id)
        if span:
            try:
                span.update(output=result)
                span.end()
            except Exception:
                pass
        return json.dumps(result)
    raise ValueError(f"Unknown tool: {tool_name}")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_agent.py -v`
Expected: PASS (all tests in the file, including the pre-existing Langfuse span tests, unaffected by this change)

- [ ] **Step 5: Commit**

```bash
git add agent.py tests/test_agent.py
git commit -m "feat: forward trace_id from run_tool() to search_amazon()"
```

---

### Task 7: `judge_service.py` — standalone FastAPI judge service

**Files:**
- Create: `judge_service.py`
- Test: `tests/test_judge_service.py` (new)

**Interfaces:**
- Consumes: `tools.cache_judge.find_match(query, candidates, trace_id=None) -> CacheMatch` (Task 2).
- Produces: FastAPI app (module-level `app`) with `POST /match`, body `{query: str, candidates: list[str], trace_id: str | None}`, response `{matched_query: str | None, outcome: str}`. Consumed by Task 8 (`docker-compose.yml`'s `command:` for the `judge` service).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_judge_service.py`:

```python
from unittest.mock import patch
from fastapi.testclient import TestClient

import judge_service
from tools.cache_judge import CacheMatch


def client():
    return TestClient(judge_service.app)


def test_match_endpoint_returns_matched_result():
    with patch("judge_service.cache_judge.find_match", return_value=CacheMatch("yoga mat", "matched")) as mock_find:
        response = client().post(
            "/match", json={"query": "kettlebell", "candidates": ["yoga mat"], "trace_id": "trace-1"}
        )

    assert response.status_code == 200
    assert response.json() == {"matched_query": "yoga mat", "outcome": "matched"}
    mock_find.assert_called_once_with("kettlebell", ["yoga mat"], trace_id="trace-1")


def test_match_endpoint_returns_no_match_result():
    with patch("judge_service.cache_judge.find_match", return_value=CacheMatch(None, "no_match")):
        response = client().post("/match", json={"query": "kettlebell", "candidates": ["yoga mat"]})

    assert response.status_code == 200
    assert response.json() == {"matched_query": None, "outcome": "no_match"}


def test_match_endpoint_defaults_trace_id_to_none():
    with patch("judge_service.cache_judge.find_match", return_value=CacheMatch(None, "no_match")) as mock_find:
        client().post("/match", json={"query": "kettlebell", "candidates": []})

    mock_find.assert_called_once_with("kettlebell", [], trace_id=None)


def test_match_endpoint_returns_error_result():
    with patch("judge_service.cache_judge.find_match", return_value=CacheMatch(None, "error")):
        response = client().post("/match", json={"query": "kettlebell", "candidates": ["yoga mat"]})

    assert response.status_code == 200
    assert response.json() == {"matched_query": None, "outcome": "error"}


def test_match_endpoint_rejects_missing_query():
    response = client().post("/match", json={"candidates": ["yoga mat"]})
    assert response.status_code == 422
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_judge_service.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'judge_service'`

- [ ] **Step 3: Create `judge_service.py`**

```python
from fastapi import FastAPI
from pydantic import BaseModel

from tools import cache_judge

app = FastAPI()


class MatchRequest(BaseModel):
    query: str
    candidates: list[str]
    trace_id: str | None = None


class MatchResponse(BaseModel):
    matched_query: str | None
    outcome: str


@app.post("/match", response_model=MatchResponse)
def match(request: MatchRequest) -> MatchResponse:
    result = cache_judge.find_match(request.query, request.candidates, trace_id=request.trace_id)
    return MatchResponse(matched_query=result.matched_query, outcome=result.outcome)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_judge_service.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add judge_service.py tests/test_judge_service.py
git commit -m "feat: add standalone judge_service FastAPI app with POST /match"
```

---

### Task 8: `docker-compose.yml` — add the `judge` service, wire `CACHE_JUDGE_URL`

**Files:**
- Modify: `docker-compose.yml`

**Interfaces:**
- Consumes: `judge_service.py` (Task 7, via the `command:` override); `CACHE_JUDGE_URL` env var, consumed by `tools/cache_judge_client.py` (Task 3).
- Produces: a `judge` Compose service reachable at `http://judge:8001/match` from `app-1`/`app-2`.

- [ ] **Step 1: Update `docker-compose.yml`**

```yaml
# docker-compose.yml
services:
  postgres:
    image: postgres:16-alpine
    environment:
      POSTGRES_USER: anz
      POSTGRES_PASSWORD: anz
      POSTGRES_DB: anz_agent
    ports:
      - "5432:5432"  # exposed to localhost for scripts/migrate_cache_to_postgres.py
    volumes:
      - pgdata:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U anz"]
      interval: 5s
      timeout: 3s
      retries: 5

  judge:
    build: .
    env_file: .env
    command: ["uvicorn", "judge_service:app", "--host", "0.0.0.0", "--port", "8001"]
    restart: unless-stopped
    # No ports: entry — reachable only from inside the Compose network,
    # same trust level as postgres. See docs/superpowers/specs/
    # 2026-08-08-cache-judge-agent-boundary-design.md for the auth backlog item.

  app-1:
    build: .
    env_file: .env
    environment:
      DATABASE_URL: postgresql://anz:anz@postgres:5432/anz_agent
      REPLICA_ID: app-1
      CACHE_JUDGE_URL: http://judge:8001/match
    depends_on:
      postgres:
        condition: service_healthy
    restart: unless-stopped

  app-2:
    build: .
    env_file: .env
    environment:
      DATABASE_URL: postgresql://anz:anz@postgres:5432/anz_agent
      REPLICA_ID: app-2
      CACHE_JUDGE_URL: http://judge:8001/match
    depends_on:
      postgres:
        condition: service_healthy
    restart: unless-stopped

  nginx:
    image: nginx:1.27-alpine
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
    ports:
      - "8000:80"
    depends_on:
      - app-1
      - app-2
    restart: unless-stopped

volumes:
  pgdata:
```

- [ ] **Step 2: Verify the YAML is well-formed and has the expected shape**

If Docker is available:

```bash
docker compose config -q
```
Expected: no output, exit code 0 (valid Compose file).

Either way (works without Docker), statically confirm the key properties:

```bash
grep -A2 "^  judge:" docker-compose.yml | grep -q "build: \." && echo "judge builds from same Dockerfile: OK"
grep -A6 "^  judge:" docker-compose.yml | grep -q "ports:" && echo "FAIL: judge exposes a port" || echo "judge has no ports: entry: OK"
grep -c "CACHE_JUDGE_URL: http://judge:8001/match" docker-compose.yml | grep -q "^2$" && echo "both app-1 and app-2 have CACHE_JUDGE_URL: OK"
```
Expected: all three `OK` lines print, no `FAIL` line.

- [ ] **Step 3: Commit**

```bash
git add docker-compose.yml
git commit -m "feat: add judge Compose service, wire CACHE_JUDGE_URL into app-1/app-2"
```

---

## Final verification

- [ ] Run the full test suite (SQLite-backed; skips Postgres-only tests if Docker isn't available):

```bash
pytest -v
```
Expected: all tests pass, including the full `tests/test_cache.py`, `tests/test_cache_judge.py`, `tests/test_cache_judge_client.py`, `tests/test_judge_service.py`, `tests/test_tracing.py`, `tests/test_amazon.py`, `tests/test_agent.py`, `tests/test_graph.py`, `tests/test_server.py`, `tests/test_main.py`.

- [ ] Confirm `CACHE_JUDGE_URL` is never set anywhere the existing (non-Docker) test suite or `main.py`/`server.py` startup paths run — this is what guarantees the in-process fallback (today's behavior) is exercised by default:

```bash
grep -rn "CACHE_JUDGE_URL" --include="*.py" .
```
Expected: only `tools/cache_judge_client.py` (reads it) and `judge_service.py`/tests reference it — no test or app-startup code sets it as an env var outside `docker-compose.yml`.
