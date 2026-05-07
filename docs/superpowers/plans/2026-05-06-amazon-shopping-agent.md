# Amazon Shopping Agent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a CLI chat agent that uses Claude + Amazon PA API to find the top 3–5 products matching the user's needs and optimization goal.

**Architecture:** A Python CLI runs a conversation loop where Claude drives the dialogue, asks clarifying questions, then calls a `search_amazon` tool when ready. All LLM and tool calls are traced through Langfuse. The Anthropic client is instantiated in `main.py` and passed into `agent.py` for testability.

**Tech Stack:** Python 3.11+, `anthropic` SDK, `python-amazon-paapi`, `langfuse`, `python-dotenv`, `pytest`

---

## File Map

| File | Responsibility |
|---|---|
| `main.py` | CLI entry point: load env, validate credentials, run chat loop |
| `agent.py` | Conversation loop, system prompt, tool routing, Langfuse tracing |
| `tools/amazon.py` | `search_amazon` function: calls Amazon PA API, returns structured product list |
| `tools/__init__.py` | Empty package marker |
| `tests/test_amazon.py` | Unit tests for `search_amazon` (Amazon API mocked) |
| `tests/test_agent.py` | Unit tests for agent loop and tool routing (Anthropic API mocked) |
| `tests/__init__.py` | Empty package marker |
| `.env.example` | Template listing all required environment variable names |
| `.gitignore` | Ignore `.env`, `__pycache__`, `.pytest_cache`, `*.pyc` |
| `requirements.txt` | Pinned dependencies |

---

## Task 1: Project Scaffolding

**Files:**
- Create: `requirements.txt`
- Create: `.env.example`
- Create: `.gitignore`
- Create: `tools/__init__.py`
- Create: `tests/__init__.py`

- [ ] **Step 1: Create requirements.txt**

```
anthropic>=0.40.0
python-amazon-paapi>=1.1.0
langfuse>=2.0.0
python-dotenv>=1.0.0
pytest>=8.0.0
pytest-mock>=3.12.0
```

- [ ] **Step 2: Create .env.example**

```
ANTHROPIC_API_KEY=
AMAZON_ACCESS_KEY=
AMAZON_SECRET_KEY=
AMAZON_PARTNER_TAG=
LANGFUSE_PUBLIC_KEY=
LANGFUSE_SECRET_KEY=
LANGFUSE_HOST=https://cloud.langfuse.com
```

- [ ] **Step 3: Create .gitignore**

```
.env
__pycache__/
*.pyc
.pytest_cache/
*.egg-info/
dist/
.venv/
```

- [ ] **Step 4: Create empty package markers**

`tools/__init__.py` — empty file
`tests/__init__.py` — empty file

- [ ] **Step 5: Install dependencies**

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Expected: all packages install without errors.

- [ ] **Step 6: Commit**

```bash
git add requirements.txt .env.example .gitignore tools/__init__.py tests/__init__.py
git commit -m "chore: project scaffolding"
```

---

## Task 2: Amazon Search Tool (TDD)

**Files:**
- Create: `tools/amazon.py`
- Create: `tests/test_amazon.py`

The `python-amazon-paapi` library provides `AmazonApi`. You instantiate it with your credentials and call `search_items(keywords=..., item_count=...)`. It returns a list of item objects with nested attributes for title, price, ratings, etc.

- [ ] **Step 1: Write failing tests**

Create `tests/test_amazon.py`:

```python
import json
import os
import pytest
from unittest.mock import patch, MagicMock


def make_mock_item(
    title="Test Laptop",
    price=499.99,
    currency="USD",
    rating=4.5,
    review_count=1234,
    prime=True,
    url="https://www.amazon.com/dp/B001TEST",
):
    item = MagicMock()
    item.item_info.title.display_value = title
    item.offers.listings[0].price.amount = price
    item.offers.listings[0].price.currency = currency
    item.customer_reviews.star_rating.value = rating
    item.customer_reviews.count.display_value = review_count
    item.offers.listings[0].delivery_info.is_prime_eligible = prime
    item.detail_page_url = url
    return item


@patch.dict(os.environ, {
    "AMAZON_ACCESS_KEY": "fake_key",
    "AMAZON_SECRET_KEY": "fake_secret",
    "AMAZON_PARTNER_TAG": "fake_tag",
})
@patch("tools.amazon.AmazonApi")
def test_search_returns_products(mock_api_class):
    mock_api = mock_api_class.return_value
    mock_api.search_items.return_value = [make_mock_item()]

    from tools.amazon import search_amazon
    result = search_amazon(query="laptop", optimize_for="price", max_results=5)

    assert "products" in result
    assert len(result["products"]) == 1
    p = result["products"][0]
    assert p["title"] == "Test Laptop"
    assert p["price"] == 499.99
    assert p["currency"] == "USD"
    assert p["rating"] == 4.5
    assert p["review_count"] == 1234
    assert p["prime"] is True
    assert p["url"] == "https://www.amazon.com/dp/B001TEST"


@patch.dict(os.environ, {
    "AMAZON_ACCESS_KEY": "fake_key",
    "AMAZON_SECRET_KEY": "fake_secret",
    "AMAZON_PARTNER_TAG": "fake_tag",
})
@patch("tools.amazon.AmazonApi")
def test_search_filters_by_max_price(mock_api_class):
    mock_api = mock_api_class.return_value
    mock_api.search_items.return_value = [
        make_mock_item(title="Cheap Laptop", price=199.99),
        make_mock_item(title="Expensive Laptop", price=999.99),
    ]

    from tools.amazon import search_amazon
    result = search_amazon(query="laptop", optimize_for="price", max_results=5, max_price=300.0)

    assert len(result["products"]) == 1
    assert result["products"][0]["title"] == "Cheap Laptop"


@patch.dict(os.environ, {
    "AMAZON_ACCESS_KEY": "fake_key",
    "AMAZON_SECRET_KEY": "fake_secret",
    "AMAZON_PARTNER_TAG": "fake_tag",
})
@patch("tools.amazon.AmazonApi")
def test_search_handles_no_results(mock_api_class):
    mock_api = mock_api_class.return_value
    mock_api.search_items.return_value = []

    from tools.amazon import search_amazon
    result = search_amazon(query="xyznonexistent", optimize_for="price", max_results=5)

    assert result["products"] == []


@patch.dict(os.environ, {
    "AMAZON_ACCESS_KEY": "fake_key",
    "AMAZON_SECRET_KEY": "fake_secret",
    "AMAZON_PARTNER_TAG": "fake_tag",
})
@patch("tools.amazon.AmazonApi")
def test_search_handles_api_error(mock_api_class):
    mock_api = mock_api_class.return_value
    mock_api.search_items.side_effect = Exception("Rate limit exceeded")

    from tools.amazon import search_amazon
    result = search_amazon(query="laptop", optimize_for="price", max_results=5)

    assert "error" in result
    assert "Rate limit exceeded" in result["error"]
    assert result["products"] == []


@patch.dict(os.environ, {
    "AMAZON_ACCESS_KEY": "fake_key",
    "AMAZON_SECRET_KEY": "fake_secret",
    "AMAZON_PARTNER_TAG": "fake_tag",
})
@patch("tools.amazon.AmazonApi")
def test_search_respects_max_results(mock_api_class):
    mock_api = mock_api_class.return_value
    mock_api.search_items.return_value = [make_mock_item(title=f"Item {i}") for i in range(10)]

    from tools.amazon import search_amazon
    result = search_amazon(query="laptop", optimize_for="price", max_results=3)

    assert len(result["products"]) == 3
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_amazon.py -v
```

Expected: `ImportError` or `ModuleNotFoundError` — `tools.amazon` does not exist yet.

- [ ] **Step 3: Implement tools/amazon.py**

```python
import os
from typing import Optional
from amazon_paapi import AmazonApi


def search_amazon(
    query: str,
    optimize_for: str,
    max_results: int = 5,
    max_price: Optional[float] = None,
) -> dict:
    amazon = AmazonApi(
        key=os.environ["AMAZON_ACCESS_KEY"],
        secret=os.environ["AMAZON_SECRET_KEY"],
        tag=os.environ["AMAZON_PARTNER_TAG"],
        country="US",
    )

    try:
        items = amazon.search_items(keywords=query, item_count=max_results * 2)
    except Exception as e:
        return {"error": str(e), "products": []}

    products = []
    for item in items:
        try:
            price = item.offers.listings[0].price.amount
        except (AttributeError, IndexError):
            price = None

        if max_price is not None and price is not None and price > max_price:
            continue

        try:
            currency = item.offers.listings[0].price.currency
        except (AttributeError, IndexError):
            currency = "USD"

        try:
            rating = item.customer_reviews.star_rating.value
        except AttributeError:
            rating = None

        try:
            review_count = item.customer_reviews.count.display_value
        except AttributeError:
            review_count = 0

        try:
            prime = item.offers.listings[0].delivery_info.is_prime_eligible
        except (AttributeError, IndexError):
            prime = False

        try:
            title = item.item_info.title.display_value
        except AttributeError:
            continue  # skip items with no title

        products.append({
            "title": title,
            "price": price,
            "currency": currency,
            "rating": rating,
            "review_count": review_count,
            "prime": prime,
            "url": item.detail_page_url,
        })

        if len(products) >= max_results:
            break

    return {"products": products}
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_amazon.py -v
```

Expected: all 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add tools/amazon.py tests/test_amazon.py
git commit -m "feat: add search_amazon tool with Amazon PA API"
```

---

## Task 3: Agent Conversation Loop (TDD)

**Files:**
- Create: `agent.py`
- Create: `tests/test_agent.py`

The agent loop calls `client.messages.create()` with the Anthropic SDK. If the response `stop_reason` is `"tool_use"`, it extracts tool call blocks, runs the tool, appends a `tool_result` message, and calls Claude again. If `stop_reason` is `"end_turn"`, it extracts the text block and returns it.

The Anthropic multi-turn format requires that when you append a tool result, the role is `"user"` and the content is a list of `tool_result` dicts.

- [ ] **Step 1: Write failing tests**

Create `tests/test_agent.py`:

```python
import json
import pytest
from unittest.mock import MagicMock, patch, call


def make_text_response(text="What are you looking for?"):
    block = MagicMock()
    block.type = "text"
    block.text = text
    response = MagicMock()
    response.stop_reason = "end_turn"
    response.content = [block]
    return response


def make_tool_response(tool_name="search_amazon", tool_input=None, tool_id="tu_123"):
    if tool_input is None:
        tool_input = {"query": "laptop", "optimize_for": "price", "max_results": 5}
    block = MagicMock()
    block.type = "tool_use"
    block.name = tool_name
    block.id = tool_id
    block.input = tool_input
    response = MagicMock()
    response.stop_reason = "tool_use"
    response.content = [block]
    return response


def test_run_tool_search_amazon():
    from agent import run_tool
    with patch("agent.search_amazon") as mock_search:
        mock_search.return_value = {"products": []}
        result = run_tool("search_amazon", {"query": "laptop", "optimize_for": "price", "max_results": 5})
    assert json.loads(result) == {"products": []}
    mock_search.assert_called_once_with(query="laptop", optimize_for="price", max_results=5)


def test_run_tool_unknown_raises():
    from agent import run_tool
    with pytest.raises(ValueError, match="Unknown tool"):
        run_tool("nonexistent_tool", {})


def test_chat_text_response_updates_history():
    from agent import chat
    mock_client = MagicMock()
    mock_client.messages.create.return_value = make_text_response("What are you looking for?")

    history = []
    result = chat(mock_client, history, "Hello")

    assert result == "What are you looking for?"
    assert history[0] == {"role": "user", "content": "Hello"}
    assert history[1]["role"] == "assistant"


def test_chat_tool_call_then_text():
    from agent import chat
    mock_client = MagicMock()
    mock_client.messages.create.side_effect = [
        make_tool_response(),
        make_text_response("Here are the top laptops..."),
    ]

    with patch("agent.search_amazon") as mock_search:
        mock_search.return_value = {"products": []}
        history = []
        result = chat(mock_client, history, "Find me a laptop")

    assert result == "Here are the top laptops..."
    assert mock_client.messages.create.call_count == 2
    # history: user, assistant (tool call), user (tool result), assistant (text)
    assert len(history) == 4


def test_chat_passes_system_prompt():
    from agent import chat, SYSTEM_PROMPT
    mock_client = MagicMock()
    mock_client.messages.create.return_value = make_text_response()

    chat(mock_client, [], "hi")

    call_kwargs = mock_client.messages.create.call_args.kwargs
    assert call_kwargs["system"] == SYSTEM_PROMPT
    assert call_kwargs["model"] == "claude-sonnet-4-6"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_agent.py -v
```

Expected: `ImportError` — `agent` module does not exist yet.

- [ ] **Step 3: Implement agent.py**

```python
import json
import anthropic
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


def run_tool(tool_name: str, tool_input: dict) -> str:
    if tool_name == "search_amazon":
        result = search_amazon(**tool_input)
        return json.dumps(result)
    raise ValueError(f"Unknown tool: {tool_name}")


def chat(client: anthropic.Anthropic, history: list, user_message: str) -> str:
    history.append({"role": "user", "content": user_message})

    while True:
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=2048,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=history,
        )

        if response.stop_reason == "tool_use":
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    result = run_tool(block.name, block.input)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                    })
            history.append({"role": "assistant", "content": response.content})
            history.append({"role": "user", "content": tool_results})
        else:
            text = next(b.text for b in response.content if hasattr(b, "text"))
            history.append({"role": "assistant", "content": response.content})
            return text
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_agent.py -v
```

Expected: all 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add agent.py tests/test_agent.py
git commit -m "feat: add agent conversation loop with tool routing"
```

---

## Task 4: CLI Entry Point

**Files:**
- Create: `main.py`

No tests for `main.py` — it's a thin I/O wrapper. Validate it manually by running it.

- [ ] **Step 1: Implement main.py**

```python
import os
import sys
import anthropic
from dotenv import load_dotenv
from agent import chat

REQUIRED_KEYS = [
    "ANTHROPIC_API_KEY",
    "AMAZON_ACCESS_KEY",
    "AMAZON_SECRET_KEY",
    "AMAZON_PARTNER_TAG",
    "LANGFUSE_PUBLIC_KEY",
    "LANGFUSE_SECRET_KEY",
]


def check_credentials() -> None:
    missing = [k for k in REQUIRED_KEYS if not os.environ.get(k)]
    if missing:
        print(f"Error: missing environment variables: {', '.join(missing)}")
        print("Copy .env.example to .env and fill in your keys.")
        sys.exit(1)


def main() -> None:
    load_dotenv()
    check_credentials()

    client = anthropic.Anthropic()
    history: list = []

    print("Amazon Shopping Assistant")
    print("Type 'quit' or 'exit' to stop.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        if not user_input:
            continue

        try:
            response = chat(client, history, user_input)
            print(f"\nAssistant: {response}\n")
        except Exception as e:
            print(f"\nError: {e}. Please try again.\n")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Create a real .env file and smoke-test**

Copy `.env.example` to `.env` and fill in your Amazon PA API credentials and Anthropic API key. For Langfuse, sign up at cloud.langfuse.com (free) and get your public/secret keys. Then:

```bash
python main.py
```

Expected: the assistant greets you and asks what you're looking for. Type a product (e.g., "I'm looking for a coffee maker") and verify it asks about your optimization goal before searching.

- [ ] **Step 3: Commit**

```bash
git add main.py
git commit -m "feat: add CLI entry point"
```

---

## Task 5: Langfuse Observability

**Files:**
- Modify: `agent.py`

Add Langfuse tracing so each conversation turn becomes a trace, the Claude call is a generation span, and the Amazon tool call is a child span.

- [ ] **Step 1: Update agent.py to add Langfuse tracing**

Replace the contents of `agent.py` with:

```python
import json
import os
import anthropic
from langfuse import Langfuse
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
        span = trace.span(name="search_amazon", input=tool_input) if trace else None
        result = search_amazon(**tool_input)
        if span:
            span.end(output=result)
        return json.dumps(result)
    raise ValueError(f"Unknown tool: {tool_name}")


def chat(client: anthropic.Anthropic, history: list, user_message: str) -> str:
    lf = _get_langfuse()
    trace = lf.trace(name="shopping-turn", input={"message": user_message})

    history.append({"role": "user", "content": user_message})

    while True:
        generation = trace.generation(
            name="claude",
            model="claude-sonnet-4-6",
            input={"system": SYSTEM_PROMPT, "messages": history},
        )

        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=2048,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=history,
        )

        generation.end(
            output=str(response.content),
            usage={
                "input": response.usage.input_tokens,
                "output": response.usage.output_tokens,
            },
        )

        if response.stop_reason == "tool_use":
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    result = run_tool(block.name, block.input, trace=trace)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                    })
            history.append({"role": "assistant", "content": response.content})
            history.append({"role": "user", "content": tool_results})
        else:
            text = next(b.text for b in response.content if hasattr(b, "text"))
            history.append({"role": "assistant", "content": response.content})
            trace.update(output={"response": text})
            return text
```

- [ ] **Step 2: Run existing tests to verify nothing broke**

```bash
pytest tests/ -v
```

Expected: all tests PASS. (Tests mock `search_amazon` directly and don't hit Langfuse since `_get_langfuse()` is only called in `chat()`, which the agent tests already mock via `mock_client`.)

Note: If the tests fail because `_get_langfuse()` is called during `chat()` and env vars are missing, add `@patch("agent._get_langfuse")` to the affected tests to mock the Langfuse instance.

- [ ] **Step 3: Smoke-test observability**

Run `python main.py`, have a conversation, complete a search. Then open your Langfuse dashboard (cloud.langfuse.com or localhost) and verify:
- A trace named `shopping-turn` appears for each user message
- The trace contains a `claude` generation with token counts
- When a search runs, a `search_amazon` span appears inside the trace

- [ ] **Step 4: Commit**

```bash
git add agent.py
git commit -m "feat: add Langfuse tracing for all LLM and tool calls"
```

---

## Self-Review Notes

- **Spec coverage:** All spec sections covered — architecture, conversation flow, tool schema, error handling, observability, credential validation, project structure.
- **Error handling:** Amazon no-results returns `{"products": []}` which Claude handles gracefully via its system prompt instruction. API errors surface the error string in the JSON so Claude can relay it.
- **Credential validation:** `check_credentials()` in `main.py` exits immediately with a clear message listing which keys are missing.
- **Types:** `run_tool` signature in Task 3 and Task 5 match — both accept `(tool_name: str, tool_input: dict)` with optional `trace` kwarg added in Task 5.
- **Test isolation:** Amazon API and Anthropic SDK are mocked in all unit tests; no real API calls needed to run the test suite.
