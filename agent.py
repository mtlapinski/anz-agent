import json
import os
import llm
from dataclasses import dataclass
from datetime import datetime, timezone
from langfuse import Langfuse
from llm import ModelConfig
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
Keep the presentation clear and scannable.

When calling search_amazon, also set the 'view' parameter based on how the user is thinking about
the choice: 'cards' for general browsing, 'table' when they want to compare specific options
side by side, or 'chart' when they're weighing price against rating/quality. Default to 'cards'
if you're unsure."""

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
                "view": {
                    "type": "string",
                    "enum": ["cards", "table", "chart"],
                    "description": (
                        "How the web UI should visualize these results: 'cards' for general "
                        "browsing, 'table' when the user wants to compare options side by side, "
                        "'chart' when the user is weighing price against rating/quality trade-offs. "
                        "Defaults to 'cards' if omitted."
                    ),
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


@dataclass
class EvalScore:
    overall: int | None
    note: str | None
    criteria: dict[str, int] | None = None


def record_score(trace_id: str | None, context: dict, score: "EvalScore | None") -> None:
    if score is None:
        return

    if trace_id:
        try:
            _get_langfuse().create_score(
                trace_id=trace_id,
                name="usefulness",
                value=score.overall,
                comment=score.note,
            )
        except Exception:
            pass

    try:
        os.makedirs("evals", exist_ok=True)
        row = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "query": context.get("query", ""),
            "optimize_for": context.get("optimize_for", ""),
            "recommendation": context.get("recommendation", ""),
            "overall": score.overall,
            "note": score.note,
        }
        with open("evals/scores.jsonl", "a") as f:
            f.write(json.dumps(row) + "\n")
    except Exception as e:
        print(f"Warning: failed to write eval score: {e}")
