# Amazon Shopping Agent

A CLI chat agent that helps you find products on Amazon at the right price.

Describe what you want in plain English. The agent asks clarifying questions, searches Amazon, and returns the top results ranked by your goal (price, quality, or balance).

## Stack

- **Claude** (Anthropic SDK) — drives the conversation and decides when to search
- **SerpAPI** — Amazon product search (100 free searches/month)
- **Langfuse** — optional LLM observability (traces, token counts)

## Architecture

### Agent conversation flow

After a search, the agent's recommendation is scored twice — once by an LLM
judge, automatically, then by you — before the turn ends:

```mermaid
flowchart LR
    subgraph CLI
        M[main.py]
    end
    subgraph SG["LangGraph StateGraph (graph.py)"]
        A[agent<br/>calls the LLM]
        T["tools<br/>runs search_amazon<br/>(cache + cache-judge subagent —<br/>see Search cache below)"]
        J["judge<br/>LLM subagent, same model config<br/>scores relevance / fit / quality"]
        E[eval<br/>interrupt for human rating]
        END([END])
        A -->|tool call| T
        T -->|loop back| A
        A -->|recommendation ready| J
        J -->|judge_score| E
        A -->|no tool call| END
    end
    subgraph Human
        P[CLI prompt<br/>rate usefulness 1-5]
    end
    subgraph Storage["agent.py"]
        L[(Langfuse v4<br/>create_score)]
        SC[(evals/scores.jsonl)]
    end

    M --> A
    E -->|interrupt value| P
    P -->|"Command(resume=score)"| E
    E --> L
    E --> SC
```

Two distinct subagents show up in this codebase, easy to conflate since both
are called "judge": the **eval judge** above (`agent.judge_recommendation`,
`graph.py`'s `judge` node) scores every recommendation automatically on the
main conversation's own model, alongside your 1-5 rating; the **cache
judge** (`tools/cache_judge.py`) is unrelated — a separate, fixed cheap-model
subagent that only fires on a cache miss, decides whether a new search query
is close enough to a previously cached one to reuse its results. See
"Search cache" below for that one's flow.

This part of the app — the graph, its nodes, and the eval flow — is identical
in both deployment modes below; only where state lives changes.

### Deployment: single process vs. scaled (Docker Compose)

Everything that needs shared state — the LangGraph checkpointer, the session
registry, and the search cache — is gated on one environment variable,
`DATABASE_URL`. Unset, the app runs exactly as it always has: one process,
state held in memory or a local SQLite file. Set (as in the Docker Compose
stack), the same code runs against Postgres instead, which is what makes it
safe to run more than one replica behind a load balancer.

**Default — `python main.py` / `python server.py`, no `DATABASE_URL`:**

```mermaid
flowchart LR
    Client[CLI / Web UI] --> S[server.py<br/>single process]
    S --> MS[("MemorySaver<br/>in-process, per-thread_id")]
    S --> SQLite[("SQLite cache<br/>~/.anz-agent/cache.db")]
    S -.-> LLM[(Anthropic / Google)]
    S -.->|on cache miss| SerpAPI[(SerpAPI)]
```

State lives in that one process. Restart it, or run a second instance, and
sessions/cache don't carry over — fine for local single-user use, the
failure mode this branch's Docker Compose stack exists to fix.

**Scaled — `docker compose up`, `DATABASE_URL` set:**

```mermaid
flowchart TD
    subgraph Compose["Docker Compose"]
        LB["nginx<br/>load balancer<br/>:8000"]
        A1[app-1<br/>server.py]
        A2[app-2<br/>server.py]
        PG[("Postgres<br/>checkpoints · sessions · cache")]
        J[judge<br/>judge_service.py]
    end
    Client[CLI / Web UI] -->|localhost:8000| LB
    LB -->|round robin| A1
    LB -->|round robin| A2
    A1 --> PG
    A2 --> PG
    A1 -.-> LLM[(Anthropic / Google)]
    A2 -.-> LLM
    A1 -.->|on cache miss| SerpAPI[(SerpAPI)]
    A2 -.->|on cache miss| SerpAPI
    A1 -.->|CACHE_JUDGE_URL| J
    A2 -.->|CACHE_JUDGE_URL| J
```

nginx round-robins between two identical `server.py` replicas; neither holds
state a request depends on — checkpoints, sessions, and cache all live in
Postgres, so either replica can serve any request for any `thread_id`, and a
crashed replica (nginx routes around it) or a restart doesn't lose in-flight
conversations. See "Running the scaled stack locally" below to try it, and
[docs/superpowers/specs/2026-08-06-local-scaling-environment-design.md](docs/superpowers/specs/2026-08-06-local-scaling-environment-design.md)
for the full design rationale.

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
Model? [gemini-flash-lite-latest] :
```

Press Enter to accept the defaults. Type `/model` at any time to switch mid-session.

Token usage is printed after each LLM call. Type `quit` or `exit` to stop.

## Web UI

An alternative to the CLI: a local web UI with a chat pane and a results panel that
visualizes search results as cards, a sortable table, or a price/rating chart —
whichever the agent judges best for the query.

Run the backend and frontend in two terminals:

```bash
python server.py
```

```bash
cd web
npm install   # first time only
npm run dev
```

Open the URL Vite prints (typically `http://localhost:5173`). Choose a provider/model
and click **Start** — this mirrors the CLI's startup prompt and requires the same
environment variables (see Configuration below).

Notes:
- The web UI does not support switching models mid-session (use the CLI's `/model`
  command for that).
- Responses are not streamed — the chat pane shows the full reply once the agent
  finishes, same latency profile as the CLI.
- Without `DATABASE_URL` set, sessions live only in the running `server.py`
  process's memory; restarting it invalidates all open sessions (same
  `MemorySaver` limitation the CLI has). With `DATABASE_URL` set (as in the
  Docker Compose stack — see "Running the scaled stack locally" below),
  sessions persist in Postgres across restarts and are shared across
  replicas.

## Running the scaled stack locally

A Docker Compose stack runs two `server.py` replicas behind nginx, backed by
Postgres for both session storage and the search cache — a local stand-in for
horizontal scaling.

A `.env` file must exist before starting (`env_file: .env` is required by
`docker-compose.yml`); create one from `.env.example` as in Setup above if
you haven't already.

```bash
docker compose up --build
```

The stack is then reachable at `http://localhost:8000` — through nginx, not
directly against either replica. `GET /healthz` reports which replica served
a given request.

To bring an existing local SQLite cache (`~/.anz-agent/cache.db`) along, run
`DATABASE_URL=postgresql://anz:anz@localhost:5432/anz_agent python
scripts/migrate_cache_to_postgres.py` once, against the Compose stack's
exposed Postgres port.

`docker-compose.yml` also starts a `judge` service and sets
`CACHE_JUDGE_URL=http://judge:8001/match` on `app-1`/`app-2`, so the
Compose stack's cache-matching judge calls happen over the network to that
standalone service instead of in-process. Unset `CACHE_JUDGE_URL` (or run
`python main.py`/`python server.py` directly, outside Compose) and the
judge call falls back to running in-process — today's behavior, unchanged.
This is a rollback-able canary: if the network judge misbehaves, removing
`CACHE_JUDGE_URL` from `app-1`/`app-2`'s environment in `docker-compose.yml`
reverts to the in-process path with no other code changes.

## Configuration

| Variable | Required for |
|---|---|
| `GOOGLE_API_KEY` | Google provider, and the search cache's fuzzy-match judge (fixed to `gemini-flash-lite-latest` regardless of your selected provider) — [aistudio.google.com](https://aistudio.google.com/app/apikey) |
| `ANTHROPIC_API_KEY` | Anthropic provider — [console.anthropic.com](https://console.anthropic.com) |
| `SERPAPI_KEY` | Always — [serpapi.com](https://serpapi.com) (100 free searches/month) |
| `LANGFUSE_PUBLIC_KEY` | No — optional observability |
| `LANGFUSE_SECRET_KEY` | No — optional observability |
| `LANGFUSE_HOST` | No — defaults to Langfuse cloud |
| `CACHE_JUDGE_URL` | No — defaults to running the search cache's judge in-process; when set, routes judge calls over HTTP to the URL instead |

## Testing

Run tests with:
```bash
pytest
```

Tests that use testcontainers (e.g., Postgres tests) require Docker to be running.

### Testcontainers on macOS with Colima

If you develop on macOS using Colima, testcontainers may need the following environment variable set to work correctly:

```bash
export TESTCONTAINERS_DOCKER_SOCKET_OVERRIDE=/var/run/docker.sock
```

Add this to your shell profile or set it before running pytest if you encounter Docker socket errors during test runs.

## Search cache

`search_amazon` caches results to conserve the SerpAPI free-tier quota, via
`tools/cache.py`, called on every search before SerpAPI:

```mermaid
flowchart TD
    S["search_amazon()<br/>tools/amazon.py"] --> L["cache.lookup(query)"]
    L --> N{"normalized query<br/>exact match?"}
    N -->|yes| Hit[reuse cached raw results]
    N -->|no| D{any queries<br/>cached at all?}
    D -->|no| API[SerpAPI call]
    D -->|yes| SL["_shortlist_candidates()<br/>top 20 by token overlap"]
    SL --> CL["cache_judge_client.find_match()<br/>tools/cache_judge_client.py"]
    CL -->|"CACHE_JUDGE_URL unset (default)"| CJ[["cache-judge subagent<br/>tools/cache_judge.py, in-process<br/>fixed model: gemini-flash-lite-latest"]]
    CL -->|"CACHE_JUDGE_URL set<br/>(Docker Compose)"| SVC["judge_service.py<br/>standalone FastAPI, POST /match"]
    SVC --> CJ
    CJ -->|match found| Hit
    CJ -->|"NONE / network error"| API
    API --> Store["cache.store(query, raw_results)"]
    Hit --> Filt["_build_products()<br/>filter by max_price / max_results"]
    Store --> Filt
    Filt --> Products[products returned to agent]

    style CJ fill:#f9e79f,stroke:#b7950b
    style SVC fill:#dcecec,stroke:#2f6e72
```

By default (no `DATABASE_URL` set) `cache.lookup`/`cache.store` read and
write `~/.anz-agent/cache.db` (SQLite); when `DATABASE_URL` is set (as in
the Docker Compose stack) they hit Postgres instead, shared across
replicas — `search_amazon` and the diagram above are unaware which backend
is active. An exact reworded/reordered query (e.g. "balance beam purple" vs.
"purple balance beam") reuses the cache directly with no LLM call; other
queries are shortlisted by token overlap (top 20, to bound the judge's
prompt size as the cache grows) and handed to `cache_judge_client`, which
picks the same judge's call boundary the same way `DATABASE_URL` picks the
cache backend: in-process by default, or — when `CACHE_JUDGE_URL` is set,
as in the Docker Compose stack — over the network to the standalone `judge`
service, treating a network error the same as no match (falls back to a
fresh SerpAPI search rather than failing the request). Entries never
expire — delete `~/.anz-agent/cache.db` (SQLite) or the `searches` table
(Postgres) to clear the cache manually.

## Eval report

`evals/scores.jsonl` accumulates one row per rated recommendation (human
`overall` score plus the LLM judge's `relevance`/`fit`/`quality`/`overall`
scores — see [docs/superpowers/specs/2026-07-26-llm-judge-design.md](docs/superpowers/specs/2026-07-26-llm-judge-design.md)).
To see it as a dashboard instead of raw JSON lines:

```bash
python scripts/eval_report.py
```

This writes `evals/report.html` and opens it in your browser: summary stats,
a sortable table of every rated turn, the 10 lowest-scoring recommendations,
and a human-vs-judge score chart over time. Re-run it any time to pick up new
rows — it's a static snapshot, not a live-updating page.

## Models

| Provider | Recommended model | Notes |
|---|---|---|
| Google | `gemini-flash-lite-latest` | Default — cheapest, tracks Google's current lite model |
| Google | `gemini-flash-latest` | Better quality |
| Anthropic | `claude-haiku-4-5-20251001` | Fast and cheap |
| Anthropic | `claude-sonnet-4-6` | Higher quality |

## Project Structure

```
anz-agent/
├── main.py          # CLI entry point — drives the graph, handles eval prompts
├── graph.py         # LangGraph StateGraph — agent/tools/eval nodes
├── agent.py         # LLM prompt/tools, Langfuse tracing, eval scoring
├── server.py        # FastAPI backend for the web UI — /session, /chat, /resume
├── db.py             # Postgres connection helper + schema migrations, used when DATABASE_URL is set
├── tracing.py         # Shared Langfuse client singleton, used by agent.py and tools/cache_judge.py
├── judge_service.py   # Standalone FastAPI wrapper around tools/cache_judge.py, run by the judge Compose service
├── tools/
│   ├── amazon.py       # SerpAPI search tool, checks the local cache first
│   ├── cache.py         # Search result cache — SQLite by default, Postgres when DATABASE_URL is set
│   ├── cache_judge.py   # LLM subagent that fuzzy-matches queries against the cache — the judge itself
│   ├── cache_judge_client.py # Dispatcher — calls the judge in-process or over the network depending on CACHE_JUDGE_URL
│   └── session_store.py # Session config storage — Postgres-backed when DATABASE_URL is set
├── scripts/
│   └── migrate_cache_to_postgres.py  # One-time migration of the local SQLite cache into Postgres
├── web/              # Vite/React/TypeScript frontend for the web UI
├── tests/
├── evals/           # scores.jsonl — eval ratings (gitignored)
├── Dockerfile         # Container image for server.py, used by docker-compose.yml
├── docker-compose.yml # Local 2-replica + nginx + Postgres + judge stack (see below)
├── nginx.conf          # Load-balancer config for the docker-compose stack
├── .env.example
└── requirements.txt
```

## Backlog

Deferred work items live in [docs/BACKLOG.md](docs/BACKLOG.md).
