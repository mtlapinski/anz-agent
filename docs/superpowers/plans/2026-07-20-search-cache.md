# Local Search Cache Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a local, persistent SQLite cache in front of `search_amazon`'s SerpAPI call, with an LLM-judged fuzzy match so reworded/related repeat searches reuse cached results instead of burning SerpAPI quota.

**Architecture:** Caching lives entirely inside `tools/amazon.py`/`tools/cache.py` — `graph.py` is untouched. `search_amazon` checks `cache.lookup(query)` first (exact normalized match, then an LLM judge subagent over all distinct cached queries) before falling back to a live SerpAPI call; results are cached pre-filter (raw SerpAPI items), with `max_price`/`max_results` filtering applied identically after either path.

**Tech Stack:** Python stdlib `sqlite3` and `re` for the cache; the existing `llm.py` abstraction (`ModelConfig`, `create_client`, `complete`) for the judge subagent, pinned to `google`/`gemini-flash-lite-latest` regardless of the conversation's selected model.

## Global Constraints

- Cache file lives at `~/.anz-agent/cache.db` (module-level `DB_PATH` in `tools/cache.py`, patchable by tests).
- No TTL/expiration — cache entries persist until the file is manually deleted. (Per design: manual clear only.)
- Caching must never break a search: any judge error, SQLite error, or malformed judge response is treated as a cache miss, falling through to a live SerpAPI call.
- The judge subagent always uses `ModelConfig(provider="google", model="gemini-flash-lite-latest")`, independent of whatever provider/model the user selected via `/model`. This means `GOOGLE_API_KEY` must be set for the judge to ever produce a fuzzy match — if it's missing, judge calls fail (caught) and every non-exact query is a cache miss (safe degradation, not a crash).
- Candidate selection for the judge is unfiltered in this plan (every distinct cached query is passed in) — this is a known scaling gap, tracked in `docs/BACKLOG.md`, not to be fixed here.
- Tests must never touch the real `~/.anz-agent/cache.db` or leak cache state between tests — every test gets an isolated temp-file cache via an autouse fixture.

---

### Task 1: Cache storage module (exact match only)

**Files:**
- Create: `tools/cache.py`
- Create: `tests/conftest.py`
- Create: `tests/test_cache.py`

**Interfaces:**
- Produces: `normalize(query: str) -> str`, `lookup(query: str) -> list[dict] | None`, `store(query: str, raw_results: list[dict]) -> None`, module-level `DB_PATH: str` in `tools/cache.py`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/conftest.py
import pytest


@pytest.fixture(autouse=True)
def isolated_cache_db(tmp_path, monkeypatch):
    """Every test gets its own empty cache file — never touches the real
    ~/.anz-agent/cache.db and never leaks state between tests."""
    monkeypatch.setattr("tools.cache.DB_PATH", str(tmp_path / "cache.db"))
```

```python
# tests/test_cache.py
def test_normalize_ignores_word_order_and_case():
    from tools.cache import normalize
    assert normalize("Purple Balance Beam") == normalize("balance beam purple")


def test_normalize_strips_punctuation():
    from tools.cache import normalize
    assert normalize("balance-beam!") == normalize("balance beam")


def test_store_then_lookup_exact_match_returns_results():
    from tools.cache import store, lookup
    store("balance beam", [{"title": "Beam"}])

    result = lookup("balance beam")

    assert result == [{"title": "Beam"}]


def test_lookup_exact_match_ignores_word_order():
    from tools.cache import store, lookup
    store("purple balance beam", [{"title": "Purple Beam"}])

    result = lookup("balance beam purple")

    assert result == [{"title": "Purple Beam"}]


def test_lookup_returns_none_on_empty_cache():
    from tools.cache import lookup
    assert lookup("anything") is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_cache.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'tools.cache'`

- [ ] **Step 3: Write the implementation**

```python
# tools/cache.py
from __future__ import annotations
import json
import os
import re
import sqlite3
from datetime import datetime, timezone

DB_PATH = os.path.expanduser("~/.anz-agent/cache.db")


def normalize(query: str) -> str:
    words = re.findall(r"[a-z0-9]+", query.lower())
    return " ".join(sorted(words))


def _connect() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS searches (
            id INTEGER PRIMARY KEY,
            query TEXT NOT NULL,
            normalized_query TEXT NOT NULL UNIQUE,
            raw_results TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_normalized_query ON searches(normalized_query)")
    return conn


def lookup(query: str) -> list[dict] | None:
    try:
        conn = _connect()
        try:
            normalized = normalize(query)
            row = conn.execute(
                "SELECT raw_results FROM searches WHERE normalized_query = ?", (normalized,)
            ).fetchone()
            if row:
                return json.loads(row[0])
            return None
        finally:
            conn.close()
    except Exception:
        return None


def store(query: str, raw_results: list[dict]) -> None:
    try:
        conn = _connect()
        try:
            conn.execute(
                "INSERT OR REPLACE INTO searches (query, normalized_query, raw_results, created_at) "
                "VALUES (?, ?, ?, ?)",
                (query, normalize(query), json.dumps(raw_results), datetime.now(timezone.utc).isoformat()),
            )
            conn.commit()
        finally:
            conn.close()
    except Exception:
        pass
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_cache.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add tools/cache.py tests/conftest.py tests/test_cache.py
git commit -m "feat: add exact-match local search cache"
```

---

### Task 2: Judge subagent module

**Files:**
- Create: `tools/cache_judge.py`
- Create: `tests/test_cache_judge.py`

**Interfaces:**
- Consumes: `llm.ModelConfig`, `llm.create_client`, `llm.complete`, `llm.LLMResponse` (all from `llm.py`, unchanged).
- Produces: `find_match(query: str, candidates: list[str]) -> str | None` in `tools/cache_judge.py`. Never raises — any internal error returns `None`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_cache_judge.py
import pytest
from unittest.mock import patch
from llm import LLMResponse
import tools.cache_judge as cache_judge


@pytest.fixture(autouse=True)
def reset_client():
    cache_judge._client = None
    yield
    cache_judge._client = None


@patch("tools.cache_judge.llm")
def test_find_match_returns_exact_candidate_text(mock_llm):
    mock_llm.complete.return_value = LLMResponse(
        text="purple balance beam", tool_calls=None, input_tokens=10, output_tokens=2
    )

    result = cache_judge.find_match("balance beam", ["purple balance beam", "yoga mat"])

    assert result == "purple balance beam"


@patch("tools.cache_judge.llm")
def test_find_match_returns_none_when_judge_says_none(mock_llm):
    mock_llm.complete.return_value = LLMResponse(
        text="NONE", tool_calls=None, input_tokens=10, output_tokens=2
    )

    result = cache_judge.find_match("kettlebell", ["yoga mat"])

    assert result is None


@patch("tools.cache_judge.llm")
def test_find_match_returns_none_when_judge_hallucinates_a_query(mock_llm):
    mock_llm.complete.return_value = LLMResponse(
        text="something not in the candidate list", tool_calls=None, input_tokens=10, output_tokens=2
    )

    result = cache_judge.find_match("kettlebell", ["yoga mat"])

    assert result is None


@patch("tools.cache_judge.llm")
def test_find_match_returns_none_on_completion_error(mock_llm):
    mock_llm.complete.side_effect = Exception("rate limited")

    result = cache_judge.find_match("kettlebell", ["yoga mat"])

    assert result is None


@patch("tools.cache_judge.llm")
def test_find_match_returns_none_on_client_creation_error(mock_llm):
    mock_llm.create_client.side_effect = Exception("missing GOOGLE_API_KEY")

    result = cache_judge.find_match("kettlebell", ["yoga mat"])

    assert result is None


def test_find_match_returns_none_when_no_candidates():
    result = cache_judge.find_match("kettlebell", [])
    assert result is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_cache_judge.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'tools.cache_judge'`

- [ ] **Step 3: Write the implementation**

```python
# tools/cache_judge.py
from __future__ import annotations
import llm

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


def find_match(query: str, candidates: list[str]) -> str | None:
    if not candidates:
        return None

    try:
        candidate_list = "\n".join(f"- {c}" for c in candidates)
        user_message = f"New query: {query}\n\nCached queries:\n{candidate_list}"
        response = llm.complete(
            _get_client(),
            JUDGE_MODEL_CONFIG,
            SYSTEM_PROMPT,
            [],
            [{"role": "user", "content": user_message}],
        )
        answer = (response.text or "").strip()
    except Exception:
        return None

    if answer not in candidates:
        return None
    return answer
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_cache_judge.py -v`
Expected: PASS (6 passed)

- [ ] **Step 5: Commit**

```bash
git add tools/cache_judge.py tests/test_cache_judge.py
git commit -m "feat: add LLM-judged fuzzy query matching for the search cache"
```

---

### Task 3: Wire the judge into cache lookups

**Files:**
- Modify: `tools/cache.py`
- Modify: `tests/test_cache.py`

**Interfaces:**
- Consumes: `tools.cache_judge.find_match(query: str, candidates: list[str]) -> str | None` (Task 2).
- Produces: `lookup()` now falls back to the judge on a normalized-match miss; public signature unchanged.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_cache.py`:

```python
from unittest.mock import patch


def test_lookup_uses_judge_for_fuzzy_match():
    from tools.cache import store, lookup
    store("purple balance beam", [{"title": "Purple Beam"}])

    with patch("tools.cache_judge.find_match", return_value="purple balance beam"):
        result = lookup("balance beam")

    assert result == [{"title": "Purple Beam"}]


def test_lookup_returns_none_when_judge_finds_no_match():
    from tools.cache import store, lookup
    store("yoga mat", [{"title": "Mat"}])

    with patch("tools.cache_judge.find_match", return_value=None):
        result = lookup("kettlebell")

    assert result is None


def test_lookup_does_not_call_judge_when_cache_empty():
    from tools.cache import lookup

    with patch("tools.cache_judge.find_match") as mock_find_match:
        result = lookup("anything")

    assert result is None
    mock_find_match.assert_not_called()


def test_lookup_returns_none_when_judge_raises():
    from tools.cache import store, lookup
    store("yoga mat", [{"title": "Mat"}])

    with patch("tools.cache_judge.find_match", side_effect=Exception("boom")):
        result = lookup("kettlebell")

    assert result is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_cache.py -v`
Expected: FAIL — `test_lookup_uses_judge_for_fuzzy_match` and `test_lookup_returns_none_when_judge_finds_no_match` fail because `lookup()` currently returns `None` on any normalized-match miss without ever calling the judge (so the "reuse cached results" case returns `None` instead of the stored products, and the `mock_find_match.assert_not_called()`/error tests pass vacuously — the real signal is the two reuse-case failures).

- [ ] **Step 3: Update the implementation**

Replace the body of `lookup()` in `tools/cache.py`:

```python
from tools import cache_judge


def lookup(query: str) -> list[dict] | None:
    try:
        conn = _connect()
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

            matched_query = cache_judge.find_match(query, candidates)
            if matched_query is None:
                return None

            row = conn.execute(
                "SELECT raw_results FROM searches WHERE query = ?", (matched_query,)
            ).fetchone()
            return json.loads(row[0]) if row else None
        finally:
            conn.close()
    except Exception:
        return None
```

Add `from tools import cache_judge` near the top of `tools/cache.py`, alongside the existing `json`/`os`/`re`/`sqlite3`/`datetime` imports.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_cache.py -v`
Expected: PASS (9 passed)

- [ ] **Step 5: Commit**

```bash
git add tools/cache.py tests/test_cache.py
git commit -m "feat: fall back to LLM judge for fuzzy cache matches"
```

---

### Task 4: Wire the cache into `search_amazon`

**Files:**
- Modify: `tools/amazon.py`
- Modify: `tests/test_amazon.py`

**Interfaces:**
- Consumes: `tools.cache.lookup(query: str) -> list[dict] | None`, `tools.cache.store(query: str, raw_results: list[dict]) -> None` (Task 1/3).
- Produces: `search_amazon(...)` unchanged public signature/return shape; new private helper `_build_products(raw_items: list[dict], max_results: int, max_price: float | None) -> list[dict]`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_amazon.py`:

```python
from unittest.mock import patch


@patch.dict(os.environ, {"SERPAPI_KEY": "fake_key"})
@patch("tools.amazon.GoogleSearch")
def test_search_returns_cached_results_without_calling_serpapi(mock_search_class):
    from tools.amazon import search_amazon

    with patch("tools.cache.lookup", return_value=[make_mock_result(title="Cached Laptop")]):
        result = search_amazon(query="laptop", optimize_for="price", max_results=5)

    mock_search_class.assert_not_called()
    assert result["products"][0]["title"] == "Cached Laptop"


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


@patch.dict(os.environ, {"SERPAPI_KEY": "fake_key"})
@patch("tools.amazon.GoogleSearch")
def test_search_does_not_store_on_api_error(mock_search_class):
    mock_search_class.return_value.get_dict.side_effect = Exception("Rate limit exceeded")

    from tools.amazon import search_amazon

    with patch("tools.cache.lookup", return_value=None), \
         patch("tools.cache.store") as mock_store:
        search_amazon(query="laptop", optimize_for="price", max_results=5)

    mock_store.assert_not_called()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_amazon.py -v`
Expected: FAIL — the three new tests fail (`tools.amazon` doesn't reference `tools.cache` yet, so patching `tools.cache.lookup`/`store` has no effect and `search_amazon` always calls `GoogleSearch`).

- [ ] **Step 3: Update the implementation**

```python
# tools/amazon.py
import os
from typing import Optional
from serpapi import GoogleSearch
from tools import cache


def search_amazon(
    query: str,
    optimize_for: str,
    max_results: int = 5,
    max_price: Optional[float] = None,
) -> dict:
    """
    Search Amazon via SerpAPI, via a local cache when available. optimize_for is
    passed through from the agent and used by the caller to rank results — not
    applied here.
    """
    raw_items = cache.lookup(query)

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


def _build_products(raw_items: list[dict], max_results: int, max_price: Optional[float]) -> list[dict]:
    products = []
    for item in raw_items:
        title = item.get("title")
        if not title:
            continue

        price = item.get("extracted_price")

        if max_price is not None and (price is None or price > max_price):
            continue

        products.append({
            "title": title,
            "price": price,
            "currency": "USD",
            "rating": item.get("rating"),
            "review_count": item.get("reviews"),
            "prime": _has_free_delivery(item.get("delivery")),
            "url": item.get("link"),
        })

        if len(products) >= max_results:
            break

    return products


def _has_free_delivery(delivery) -> bool:
    if not delivery:
        return False
    if isinstance(delivery, list):
        return any("free" in str(d).lower() for d in delivery)
    return "free" in str(delivery).lower()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_amazon.py -v`
Expected: PASS (10 passed) — the 7 pre-existing tests keep passing unchanged (they get an isolated empty cache via the `isolated_cache_db` fixture from Task 1, so `cache.lookup` naturally misses and behavior matches today's).

Then run the full suite to confirm no regressions elsewhere:

Run: `python -m pytest -v`
Expected: PASS, all tests green.

- [ ] **Step 5: Commit**

```bash
git add tools/amazon.py tests/test_amazon.py
git commit -m "feat: check local cache before calling SerpAPI in search_amazon"
```

---

### Task 5: Documentation

**Files:**
- Modify: `README.md`

**Interfaces:**
- None — documentation only.

- [ ] **Step 1: Update the Project Structure section**

In `README.md`, replace:

```
├── tools/
│   └── amazon.py    # SerpAPI search tool
```

with:

```
├── tools/
│   ├── amazon.py       # SerpAPI search tool, checks the local cache first
│   ├── cache.py         # SQLite-backed search result cache (~/.anz-agent/cache.db)
│   └── cache_judge.py   # LLM subagent that fuzzy-matches queries against the cache
```

- [ ] **Step 2: Update the Configuration table's `GOOGLE_API_KEY` row**

Replace:

```
| `GOOGLE_API_KEY` | Google provider — [aistudio.google.com](https://aistudio.google.com/app/apikey) |
```

with:

```
| `GOOGLE_API_KEY` | Google provider, and the search cache's fuzzy-match judge (fixed to `gemini-flash-lite-latest` regardless of your selected provider) — [aistudio.google.com](https://aistudio.google.com/app/apikey) |
```

- [ ] **Step 3: Add a short Cache section**

Insert after the `## Configuration` section (before `## Models`):

```markdown
## Search cache

`search_amazon` caches results locally in `~/.anz-agent/cache.db` (SQLite) to
conserve the SerpAPI free-tier quota. An exact reworded/reordered query
(e.g. "balance beam purple" vs. "purple balance beam") reuses the cache
directly; other queries are checked against past searches by a small LLM
judge (`tools/cache_judge.py`) that decides if a prior search is a close
enough match to reuse (e.g. "balance beam" reusing "purple balance beam"
results). Entries never expire — delete `~/.anz-agent/cache.db` to clear
the cache manually.
```

- [ ] **Step 4: Verify the mermaid diagram and rest of the README still render sensibly**

Run: `git diff README.md`
Expected: only the three additions/edits above; no unrelated changes.

- [ ] **Step 5: Commit**

```bash
git add README.md
git commit -m "docs: document the local search cache"
```
