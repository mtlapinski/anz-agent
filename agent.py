import json
import os
import llm
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


def chat(client, history: list, user_message: str, model_config: ModelConfig) -> str:
    try:
        trace_id = _get_langfuse().create_trace_id()
    except Exception:
        trace_id = None

    history.append({"role": "user", "content": user_message})

    while True:
        generation = None
        if trace_id:
            try:
                generation = _get_langfuse().start_observation(
                    trace_context={"trace_id": trace_id},
                    name="llm",
                    as_type="generation",
                    input={"system": SYSTEM_PROMPT, "messages": list(history)},
                    model=f"{model_config.provider}/{model_config.model}",
                )
            except Exception:
                generation = None

        llm_response = llm.complete(client, model_config, SYSTEM_PROMPT, TOOLS, history)

        print(f"[tokens: {llm_response.input_tokens} in / {llm_response.output_tokens} out]")

        if generation:
            try:
                generation.update(
                    output=str(llm_response.text or llm_response.tool_calls),
                    usage_details={
                        "input": llm_response.input_tokens,
                        "output": llm_response.output_tokens,
                    },
                )
                generation.end()
            except Exception:
                pass

        if llm_response.tool_calls:
            tool_results = []
            for tc in llm_response.tool_calls:
                result = run_tool(tc["name"], tc["input"], trace_id=trace_id)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tc["id"],
                    "content": result,
                })
            history.append({"role": "assistant", "content": [
                {"type": "tool_use", "id": tc["id"], "name": tc["name"], "input": tc["input"]}
                for tc in llm_response.tool_calls
            ]})
            history.append({"role": "user", "content": tool_results})
        else:
            text = llm_response.text or ""
            history.append({"role": "assistant", "content": [{"type": "text", "text": text}]})
            return text
