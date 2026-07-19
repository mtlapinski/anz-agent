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
        try:
            span = trace.span(name="search_amazon", input=tool_input) if trace else None
        except Exception:
            span = None
        result = search_amazon(**tool_input)
        try:
            if span:
                span.end(output=result)
        except Exception:
            pass
        return json.dumps(result)
    raise ValueError(f"Unknown tool: {tool_name}")


def chat(client: anthropic.Anthropic, history: list, user_message: str) -> str:
    try:
        lf = _get_langfuse()
        trace = lf.trace(name="shopping-turn", input={"message": user_message})
    except Exception:
        trace = None

    history.append({"role": "user", "content": user_message})

    while True:
        try:
            generation = trace.generation(
                name="claude",
                model="claude-sonnet-4-6",
                input={"system": SYSTEM_PROMPT, "messages": history},
            ) if trace else None
        except Exception:
            generation = None

        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=2048,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=history,
        )

        try:
            if generation:
                generation.end(
                    output=str(response.content),
                    usage={
                        "input": response.usage.input_tokens,
                        "output": response.usage.output_tokens,
                    },
                )
        except Exception:
            pass

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
            history.append({"role": "assistant", "content": [
                {"type": "tool_use", "id": b.id, "name": b.name, "input": b.input}
                for b in response.content if b.type == "tool_use"
            ]})
            history.append({"role": "user", "content": tool_results})
        else:
            if response.stop_reason not in ("end_turn", "stop_sequence"):
                raise RuntimeError(f"Unexpected stop_reason: {response.stop_reason!r}")
            text = next((b.text for b in response.content if hasattr(b, "text")), "")
            history.append({"role": "assistant", "content": [{"type": "text", "text": text}]})
            try:
                if trace:
                    trace.update(output={"response": text})
            except Exception:
                pass
            return text
