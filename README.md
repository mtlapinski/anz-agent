# Amazon Shopping Agent

A CLI chat agent that helps you find products on Amazon at the right price.

Describe what you want in plain English. The agent asks clarifying questions, searches Amazon, and returns the top results ranked by your goal (price, quality, or balance).

## Stack

- **Claude** (Anthropic SDK) — drives the conversation and decides when to search
- **SerpAPI** — Amazon product search (100 free searches/month)
- **Langfuse** — optional LLM observability (traces, token counts)

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# Fill in ANTHROPIC_API_KEY and SERPAPI_KEY
```

## Run

```bash
python main.py
```

Token usage is printed after each LLM call. Type `quit` or `exit` to stop.

## Configuration

| Variable | Required | Notes |
|---|---|---|
| `ANTHROPIC_API_KEY` | Yes | [console.anthropic.com](https://console.anthropic.com) |
| `SERPAPI_KEY` | Yes | [serpapi.com](https://serpapi.com) — 100 free searches/month |
| `LANGFUSE_PUBLIC_KEY` | No | Optional observability |
| `LANGFUSE_SECRET_KEY` | No | Optional observability |
| `LANGFUSE_HOST` | No | Defaults to Langfuse cloud |

## Project Structure

```
anz-agent/
├── main.py          # CLI entry point
├── agent.py         # Conversation loop + Langfuse tracing
├── tools/
│   └── amazon.py    # SerpAPI search tool
├── tests/
├── .env.example
└── requirements.txt
```
