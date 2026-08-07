# Strip Image Field From LLM Tool Result Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop resending each product's `image` (thumbnail URL) field to the LLM on every conversation turn, while the web UI keeps receiving it unchanged.

**Architecture:** `tools_node` in `graph.py` is the single point where a `search_amazon` result currently becomes both the `tool_result` sent to the LLM and `last_search_results` served to the web UI. Add a pure helper `agent.strip_image_for_llm(products)` that returns a copy of a product list with `image` removed from each item, and call it only when building the `tool_result` content — `last_search_results` keeps the untouched original.

**Tech Stack:** Python 3.14, pytest, unittest.mock (existing project conventions — no new dependencies).

## Global Constraints

- No new dependencies.
- Only the `image` field is removed from the LLM-facing copy — every other product field (title, price, currency, rating, review_count, prime, url) stays, per the spec's system-prompt requirement that the LLM surface them in its response text.
- The trim applies only to `search_amazon` tool results — no other tool exists today, and the split must not touch non-`search_amazon` tool_result content.
- `last_search_results` (used by `server.py` for the web UI) must be byte-for-byte unchanged by this work.
- Follow existing test conventions in the repo: `unittest.mock.patch`/`MagicMock`, one assertion focus per test, `from <module> import <name>` inside each test function (matches existing style in `tests/test_agent.py` and `tests/test_graph.py`).

---

### Task 1: Add `strip_image_for_llm` helper to `agent.py`

**Files:**
- Modify: `agent.py` (add new function directly after `run_tool`, i.e. after the line `raise ValueError(f"Unknown tool: {tool_name}")` and before `@dataclass class EvalScore`)
- Test: `tests/test_agent.py`

**Interfaces:**
- Produces: `strip_image_for_llm(products: list[dict]) -> list[dict]` — returns a **new** list of **new** dicts, each equal to the corresponding input dict but with the `"image"` key removed if present. Does not mutate its input. Works on an empty list. Works on dicts that never had an `"image"` key (no-op passthrough for that key).

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_agent.py` (place near the other standalone-function tests, e.g. right after `test_run_tool_unknown_raises`):

```python
def test_strip_image_for_llm_removes_image_field():
    from agent import strip_image_for_llm
    products = [
        {"title": "Laptop", "price": 999.0, "image": "https://example.com/thumb1.jpg"},
        {"title": "Mouse", "price": 19.99, "image": "https://example.com/thumb2.jpg"},
    ]
    result = strip_image_for_llm(products)
    assert result == [
        {"title": "Laptop", "price": 999.0},
        {"title": "Mouse", "price": 19.99},
    ]


def test_strip_image_for_llm_handles_missing_image_field():
    from agent import strip_image_for_llm
    products = [{"title": "Laptop", "price": 999.0}]
    result = strip_image_for_llm(products)
    assert result == [{"title": "Laptop", "price": 999.0}]


def test_strip_image_for_llm_handles_empty_list():
    from agent import strip_image_for_llm
    assert strip_image_for_llm([]) == []


def test_strip_image_for_llm_does_not_mutate_input():
    from agent import strip_image_for_llm
    original = {"title": "Laptop", "image": "https://example.com/thumb1.jpg"}
    products = [original]
    strip_image_for_llm(products)
    assert original == {"title": "Laptop", "image": "https://example.com/thumb1.jpg"}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_agent.py -k strip_image_for_llm -v`
Expected: FAIL with `ImportError: cannot import name 'strip_image_for_llm' from 'agent'` (function doesn't exist yet).

- [ ] **Step 3: Write minimal implementation**

In `agent.py`, insert directly after `run_tool`'s closing `raise ValueError(f"Unknown tool: {tool_name}")` line and before the `@dataclass` / `class EvalScore` block:

```python
def strip_image_for_llm(products: list[dict]) -> list[dict]:
    return [{k: v for k, v in p.items() if k != "image"} for p in products]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_agent.py -k strip_image_for_llm -v`
Expected: PASS — all 4 new tests.

- [ ] **Step 5: Run the full test suite**

Run: `python -m pytest -q`
Expected: PASS — no regressions (this is a new, unused-so-far function; nothing else calls it yet).

- [ ] **Step 6: Commit**

```bash
git add agent.py tests/test_agent.py
git commit -m "feat: add strip_image_for_llm helper for trimming LLM-facing product data"
```

---

### Task 2: Wire `strip_image_for_llm` into `tools_node`

**Files:**
- Modify: `graph.py:117-129` (`tools_node`)
- Test: `tests/test_graph.py`

**Interfaces:**
- Consumes: `agent.strip_image_for_llm(products: list[dict]) -> list[dict]` from Task 1.
- Produces: no new public interface — `tools_node(state: GraphState) -> dict` keeps its existing signature and return shape (`{"history": [...], "pending_tool_calls": None, "last_search_results": ...}`). Only the *content* of the `tool_result` history entry changes for `search_amazon` calls.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_graph.py`, right after the existing `test_tools_node_captures_last_search_results` test:

```python
@patch("graph.agent.run_tool")
def test_tools_node_strips_image_from_llm_tool_result_but_keeps_it_in_last_search_results(mock_run_tool):
    from graph import tools_node
    mock_run_tool.return_value = json.dumps({
        "products": [
            {"title": "Laptop", "price": 999.0, "image": "https://example.com/thumb.jpg"},
        ]
    })
    state = empty_state(
        pending_tool_calls=[{"name": "search_amazon", "id": "tu_1",
                               "input": {"query": "laptop", "optimize_for": "price", "max_results": 5}}],
        trace_id="trace-1",
    )

    result = tools_node(state)

    tool_result_msg = result["history"][0]
    llm_content = json.loads(tool_result_msg["content"][0]["content"])
    assert llm_content == {"products": [{"title": "Laptop", "price": 999.0}]}

    assert result["last_search_results"] == {
        "products": [{"title": "Laptop", "price": 999.0, "image": "https://example.com/thumb.jpg"}]
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_graph.py -k strips_image -v`
Expected: FAIL — `llm_content` still contains `"image"` (assertion mismatch), since `tools_node` currently forwards `run_tool`'s raw JSON string unchanged as the tool_result content.

- [ ] **Step 3: Write minimal implementation**

Replace `tools_node` in `graph.py`:

```python
def tools_node(state: GraphState) -> dict:
    trace_id = state.get("trace_id")
    tool_results = []
    last_search_results = state.get("last_search_results")
    for tc in state["pending_tool_calls"]:
        result = agent.run_tool(tc["name"], tc["input"], trace_id=trace_id)
        content = result
        if tc["name"] == "search_amazon":
            last_search_results = json.loads(result)
            llm_view = {
                **last_search_results,
                "products": agent.strip_image_for_llm(last_search_results.get("products", [])),
            }
            content = json.dumps(llm_view)
        tool_results.append({"type": "tool_result", "tool_use_id": tc["id"], "content": content})
    return {
        "history": [{"role": "user", "content": tool_results}],
        "pending_tool_calls": None,
        "last_search_results": last_search_results,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_graph.py -k strips_image -v`
Expected: PASS.

- [ ] **Step 5: Run the full test suite**

Run: `python -m pytest -q`
Expected: PASS — including the pre-existing `test_tools_node_executes_and_appends_results` (its mock result has an empty `products` list, so stripping is a no-op and the JSON string is unchanged) and `test_tools_node_captures_last_search_results` (its mock product has no `image` key, so stripping is a no-op there too). Also confirm `test_tools_node_missing_data_when_no_search_call` still passes unmodified — it uses a non-`search_amazon` tool name, which now takes the `content = result` passthrough branch, identical to before.

- [ ] **Step 6: Commit**

```bash
git add graph.py tests/test_graph.py
git commit -m "feat: strip image field from search_amazon tool_result sent to the LLM"
```

---

## Manual Verification

After both tasks are committed, no manual/live-API verification is needed beyond the automated tests — this change has no external side effects (no new API calls, no schema change to `search_amazon`'s public signature) and is fully exercised by mocked unit tests, consistent with the rest of the test suite (see the "Why no integration tests" note already established for this project).
