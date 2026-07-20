# Amazon Shopping Agent

A CLI chat agent that helps you find products on Amazon at the right price.

Describe what you want in plain English. The agent asks clarifying questions, searches Amazon, and returns the top results ranked by your goal (price, quality, or balance).

## Stack

- **Claude** (Anthropic SDK) — drives the conversation and decides when to search
- **SerpAPI** — Amazon product search (100 free searches/month)
- **Langfuse** — optional LLM observability (traces, token counts)

## Architecture

After a search, the agent pauses to ask you to rate the recommendation before continuing:

```mermaid
flowchart LR
    subgraph CLI
        M[main.py]
    end
    subgraph SG["LangGraph StateGraph (graph.py)"]
        A[agent<br/>calls the LLM]
        T[tools<br/>runs search_amazon]
        E[eval<br/>interrupt for rating]
        END([END])
        A -->|tool call| T
        T -->|loop back| A
        A -->|recommendation ready| E
        A -->|no tool call| END
    end
    subgraph Human
        P[CLI prompt<br/>rate usefulness 1-5]
    end
    subgraph Storage["agent.py"]
        L[(Langfuse v4<br/>create_score)]
        J[(evals/scores.jsonl)]
    end

    M --> A
    E -->|interrupt value| P
    P -->|"Command(resume=score)"| E
    E --> L
    E --> J
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# Fill in SERPAPI_KEY and either GOOGLE_API_KEY or ANTHROPIC_API_KEY
```

## Run

```bash
python main.py
```

At startup you'll be prompted to choose a provider and model:
```
Provider? [1] Google (default)  [2] Anthropic :
Model? [gemini-2.0-flash-lite] :
```

Press Enter to accept the defaults. Type `/model` at any time to switch mid-session.

Token usage is printed after each LLM call. Type `quit` or `exit` to stop.

## Configuration

| Variable | Required for |
|---|---|
| `GOOGLE_API_KEY` | Google provider — [aistudio.google.com](https://aistudio.google.com/app/apikey) |
| `ANTHROPIC_API_KEY` | Anthropic provider — [console.anthropic.com](https://console.anthropic.com) |
| `SERPAPI_KEY` | Always — [serpapi.com](https://serpapi.com) (100 free searches/month) |
| `LANGFUSE_PUBLIC_KEY` | No — optional observability |
| `LANGFUSE_SECRET_KEY` | No — optional observability |
| `LANGFUSE_HOST` | No — defaults to Langfuse cloud |

## Models

| Provider | Recommended model | Notes |
|---|---|---|
| Google | `gemini-2.0-flash-lite` | Default — cheapest |
| Google | `gemini-2.0-flash` | Better quality |
| Anthropic | `claude-haiku-4-5-20251001` | Fast and cheap |
| Anthropic | `claude-sonnet-4-6` | Higher quality |

## Project Structure

```
anz-agent/
├── main.py          # CLI entry point — drives the graph, handles eval prompts
├── graph.py         # LangGraph StateGraph — agent/tools/eval nodes
├── agent.py         # LLM prompt/tools, Langfuse tracing, eval scoring
├── tools/
│   └── amazon.py    # SerpAPI search tool
├── tests/
├── evals/           # scores.jsonl — eval ratings (gitignored)
├── .env.example
└── requirements.txt
```
