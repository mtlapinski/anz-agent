# Local Scaling Environment (Phase 1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stand up a full-fidelity local environment — Postgres-backed checkpointer, shared session registry, shared search cache with a pre-filtered cache-judge, and 2 FastAPI replicas behind an nginx load balancer with non-blocking request handling — all runnable via `docker compose up`, per [docs/superpowers/specs/2026-08-06-local-scaling-environment-design.md](../specs/2026-08-06-local-scaling-environment-design.md).

**Architecture:** Every backend swap (checkpointer, cache, session store) is env-var gated on `DATABASE_URL`. Unset (the default for `pytest`), the app behaves exactly as it does today — `MemorySaver`, SQLite cache, in-memory `_sessions`. Set (inside Docker Compose), it runs fully on Postgres. This keeps the existing test suite untouched and fast, while a new Postgres-backed integration test suite (via `testcontainers`) covers the new paths.

**Tech Stack:** FastAPI, LangGraph (`langgraph-checkpoint-postgres`), `psycopg[binary]` (psycopg3), Postgres 16, nginx, Docker Compose, `testcontainers-python` for integration tests.

## Global Constraints

- `DATABASE_URL` unset → existing `MemorySaver`/SQLite behavior, unchanged. `DATABASE_URL` set → Postgres for checkpointer, sessions, and cache. This branch must hold in every task below.
- `tools/cache.py`'s public API (`normalize()`, `lookup()`, `store()`) keeps its exact existing signatures — `tools/amazon.py` must not need to change.
- No changes to `graph.py` node functions or the `interrupt()`/`Command(resume=...)` flow — only the checkpointer construction in `build_graph()` changes.
- Cache entries still never expire (no TTL) — unchanged from today.
- Local Compose Postgres credentials (`anz`/`anz`) are local-dev-only, not a production secret.
- New Python dependencies go in `requirements.txt` (this repo has no separate dev-requirements file — `pytest`/`pytest-mock` already live there).

---

## File Structure

New files:
- `db.py` — Postgres connection helper + idempotent schema setup for the `sessions` and `searches` tables (checkpoint tables are created separately by `PostgresSaver.setup()`)
- `tools/session_store.py` — Postgres-backed session registry (`create_session`, `get_session`, `SessionNotFound`)
- `scripts/migrate_cache_to_postgres.py` — one-time SQLite → Postgres cache migration
- `Dockerfile` — app image
- `docker-compose.yml` — `postgres`, `app-1`, `app-2`, `nginx`
- `nginx.conf` — round-robin load balancer config
- `tests/test_db.py`, `tests/test_session_store.py`, `tests/test_migrate_cache.py` — new test files
- `tests/postgres_fixtures.py` — shared `testcontainers` Postgres fixture, imported by the test files above and by `tests/test_cache.py`/`tests/test_graph.py`'s new Postgres cases

Modified files:
- `graph.py` — `build_graph()` selects `PostgresSaver` vs `MemorySaver` based on `DATABASE_URL`
- `server.py` — replaces `_sessions` dict with `tools/session_store.py`; `/chat` and `/resume` become async with `run_in_threadpool`; adds `GET /healthz`
- `tools/cache.py` — adds a Postgres-backed path alongside the existing SQLite path; adds the candidate pre-filter
- `requirements.txt` — adds `psycopg[binary]`, `langgraph-checkpoint-postgres`, `testcontainers[postgres]`
- `.env.example` — documents `DATABASE_URL`
- `docs/BACKLOG.md` — remove the "Judge candidate pre-filtering" entry once Task 7 ships
- `tests/test_server.py` — updated for the `session_store`-based session lookup instead of the `_sessions` dict

---

### Task 1: Postgres foundation — connection helper, schema, test fixture

**Files:**
- Create: `db.py`
- Create: `tests/postgres_fixtures.py`
- Create: `tests/test_db.py`
- Modify: `requirements.txt`

**Interfaces:**
- Produces: `db.get_connection() -> psycopg.Connection`, `db.run_migrations(conn_str: str) -> None`, `tests.postgres_fixtures.postgres_url` (pytest fixture yielding a live `DATABASE_URL` string for a throwaway container)

- [ ] **Step 1: Add new dependencies**

Add to `requirements.txt`:
```
psycopg[binary]>=3.1.0
langgraph-checkpoint-postgres>=2.0.0
testcontainers[postgres]>=4.8.0
```

Run: `pip install -r requirements.txt`

- [ ] **Step 2: Write the shared Postgres test fixture**

```python
# tests/postgres_fixtures.py
import pytest
from testcontainers.postgres import PostgresContainer


@pytest.fixture(scope="session")
def postgres_url():
    """Session-scoped: one throwaway Postgres container for the whole test run."""
    with PostgresContainer("postgres:16-alpine") as pg:
        yield pg.get_connection_url().replace("postgresql+psycopg2://", "postgresql://")
```

- [ ] **Step 3: Write the failing test for schema setup**

```python
# tests/test_db.py
import psycopg
from tests.postgres_fixtures import postgres_url  # noqa: F401


def test_run_migrations_creates_sessions_and_searches_tables(postgres_url):
    import db
    db.run_migrations(postgres_url)

    with psycopg.connect(postgres_url) as conn:
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'"
            ).fetchall()
        }
    assert "sessions" in tables
    assert "searches" in tables


def test_run_migrations_is_idempotent(postgres_url):
    import db
    db.run_migrations(postgres_url)
    db.run_migrations(postgres_url)  # must not raise
```

- [ ] **Step 4: Run the test to verify it fails**

Run: `pytest tests/test_db.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'db'`

- [ ] **Step 5: Implement `db.py`**

```python
# db.py
from __future__ import annotations
import os
import psycopg


def get_connection() -> psycopg.Connection:
    # Reads the env var fresh on every call, not once at import time — tests
    # rely on monkeypatch.setenv("DATABASE_URL", ...) changing behavior
    # mid-test, which a module-level cached constant would miss.
    return psycopg.connect(os.environ["DATABASE_URL"])


def run_migrations(conn_str: str | None = None) -> None:
    """Creates the sessions and searches tables if they don't exist.
    Idempotent — safe to call on every app startup."""
    with psycopg.connect(conn_str or os.environ["DATABASE_URL"]) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                thread_id TEXT PRIMARY KEY,
                provider TEXT NOT NULL,
                model TEXT NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT now()
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS searches (
                id SERIAL PRIMARY KEY,
                query TEXT NOT NULL,
                normalized_query TEXT NOT NULL UNIQUE,
                raw_results JSONB NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT now()
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_normalized_query ON searches(normalized_query)")
        conn.commit()
```

- [ ] **Step 6: Run the test to verify it passes**

Run: `pytest tests/test_db.py -v`
Expected: PASS (first run pulls the `postgres:16-alpine` image via `testcontainers` — needs Docker running locally)

- [ ] **Step 7: Commit**

```bash
git add db.py tests/postgres_fixtures.py tests/test_db.py requirements.txt
git commit -m "feat: add Postgres connection helper and schema migrations"
```

---

### Task 2: Postgres-backed LangGraph checkpointer

**Files:**
- Modify: `graph.py:9`, `graph.py:170-187` (`build_graph()`)
- Test: `tests/test_graph.py`

**Interfaces:**
- Consumes: `db.run_migrations(conn_str)` (Task 1) is NOT called from here — `PostgresSaver.setup()` creates its own checkpoint tables independently
- Produces: `build_graph()` signature unchanged — still returns a compiled graph; callers (`main.py`, `server.py`) don't change

- [ ] **Step 1: Write the failing test**

```python
# tests/test_graph.py — add this test
from tests.postgres_fixtures import postgres_url  # noqa: F401


def test_build_graph_uses_postgres_checkpointer_and_shares_state_across_instances(postgres_url, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", postgres_url)
    import importlib
    import graph
    importlib.reload(graph)  # picks up the env var at build_graph() call time

    config = {"configurable": {"thread_id": "shared-thread"}}

    # Simulate replica A handling turn 1
    graph_a = graph.build_graph()
    graph_a.get_state(config)  # no turns yet, just confirms it doesn't error

    # Simulate replica B (a fresh graph instance, same DB) reading turn 1's checkpoint
    graph_b = graph.build_graph()
    state_b = graph_b.get_state(config)

    assert state_b is not None  # both instances see the same Postgres-backed thread
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_graph.py::test_build_graph_uses_postgres_checkpointer_and_shares_state_across_instances -v`
Expected: FAIL — `build_graph()` still always uses `MemorySaver`, so each instance has its own state (no shared-instance failure yet, but the assertion below will be revisited once the real behavior is wired — this test's purpose is to fail for the *right* reason: no Postgres wiring exists yet). Confirm the failure is an import/attribute error, not a false pass.

- [ ] **Step 3: Implement the checkpointer swap**

```python
# graph.py — replace the MemorySaver import and build_graph()'s return
```
At the top of `graph.py`, replace:
```python
from langgraph.checkpoint.memory import MemorySaver
```
with:
```python
import os
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.postgres import PostgresSaver
```

Replace the last line of `build_graph()`:
```python
    return builder.compile(checkpointer=MemorySaver())
```
with:
```python
    database_url = os.environ.get("DATABASE_URL")
    if database_url:
        # Kept open for the process lifetime — this app builds one graph at
        # import time (see server.py/main.py) and reuses it for every request.
        checkpointer = PostgresSaver.from_conn_string(database_url).__enter__()
        checkpointer.setup()
    else:
        checkpointer = MemorySaver()
    return builder.compile(checkpointer=checkpointer)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_graph.py::test_build_graph_uses_postgres_checkpointer_and_shares_state_across_instances -v`
Expected: PASS

- [ ] **Step 5: Run the full existing graph test suite to confirm no regression**

Run: `pytest tests/test_graph.py -v`
Expected: All PASS — tests that don't set `DATABASE_URL` still exercise `MemorySaver` exactly as before.

- [ ] **Step 6: Commit**

```bash
git add graph.py tests/test_graph.py
git commit -m "feat: swap MemorySaver for a Postgres checkpointer when DATABASE_URL is set"
```

*Note: a test reconciling the design spec's "checkpointer fails loud (500)" wording against `server.py`'s actual error handling is deferred to Task 4, Step 6 — it needs the `isolated_session_store` fixture introduced there.*

---

### Task 3: Session registry module

**Files:**
- Create: `tools/session_store.py`
- Create: `tests/test_session_store.py`

**Interfaces:**
- Consumes: `db.get_connection()` (Task 1)
- Produces: `session_store.create_session(thread_id: str, config: ModelConfig) -> None`, `session_store.get_session(thread_id: str) -> ModelConfig`, `session_store.SessionNotFound` (exception class) — used by Task 4

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_session_store.py
import pytest
from tests.postgres_fixtures import postgres_url  # noqa: F401
from llm import ModelConfig


@pytest.fixture(autouse=True)
def _setup_schema(postgres_url, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", postgres_url)
    import db
    db.run_migrations(postgres_url)


def test_create_then_get_session_round_trips():
    from tools import session_store
    session_store.create_session("t-1", ModelConfig(provider="anthropic", model="claude-sonnet-4-6"))

    config = session_store.get_session("t-1")

    assert config == ModelConfig(provider="anthropic", model="claude-sonnet-4-6")


def test_get_session_raises_session_not_found_for_unknown_thread_id():
    from tools import session_store

    with pytest.raises(session_store.SessionNotFound):
        session_store.get_session("does-not-exist")


def test_get_session_propagates_infra_errors_distinctly_from_not_found(monkeypatch):
    from tools import session_store

    def _broken_connection():
        raise psycopg.OperationalError("connection refused")

    import psycopg
    monkeypatch.setattr("db.get_connection", _broken_connection)

    with pytest.raises(psycopg.OperationalError):
        session_store.get_session("t-1")
    # Specifically NOT a SessionNotFound — infra failures must not look like a 404
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_session_store.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'tools.session_store'`

- [ ] **Step 3: Implement `tools/session_store.py`**

```python
# tools/session_store.py
from __future__ import annotations
import db
from llm import ModelConfig


class SessionNotFound(Exception):
    pass


def create_session(thread_id: str, config: ModelConfig) -> None:
    with db.get_connection() as conn:
        conn.execute(
            "INSERT INTO sessions (thread_id, provider, model) VALUES (%s, %s, %s)",
            (thread_id, config.provider, config.model),
        )
        conn.commit()


def get_session(thread_id: str) -> ModelConfig:
    """Raises SessionNotFound if no row exists for thread_id. Any other
    exception (e.g. a connection error) propagates unchanged — a session
    lookup FAILURE must not be indistinguishable from a legitimate MISS."""
    with db.get_connection() as conn:
        row = conn.execute(
            "SELECT provider, model FROM sessions WHERE thread_id = %s", (thread_id,)
        ).fetchone()
    if row is None:
        raise SessionNotFound(thread_id)
    return ModelConfig(provider=row[0], model=row[1])
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_session_store.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tools/session_store.py tests/test_session_store.py
git commit -m "feat: add Postgres-backed session registry"
```

---

### Task 4: Wire session registry into server.py, replacing `_sessions`

**Files:**
- Modify: `server.py:20` (remove `_sessions`), `server.py:43-63` (`create_session`, `_get_context`)
- Modify: `tests/test_server.py` (remove `_sessions`-based fixtures/assertions, mock `session_store` instead)

**Interfaces:**
- Consumes: `session_store.create_session`, `session_store.get_session`, `session_store.SessionNotFound` (Task 3)

- [ ] **Step 1: Update `tests/test_server.py`'s fixture and session-dependent tests**

Replace:
```python
@pytest.fixture(autouse=True)
def reset_sessions():
    server._sessions.clear()
    yield
    server._sessions.clear()
```
with:
```python
@pytest.fixture(autouse=True)
def isolated_session_store(monkeypatch):
    """In-memory fake so these unit tests don't need a real Postgres."""
    fake_sessions: dict[str, ModelConfig] = {}
    monkeypatch.setattr("server.session_store.create_session", lambda tid, cfg: fake_sessions.__setitem__(tid, cfg))

    def _get(tid):
        if tid not in fake_sessions:
            from tools.session_store import SessionNotFound
            raise SessionNotFound(tid)
        return fake_sessions[tid]

    monkeypatch.setattr("server.session_store.get_session", _get)
    return fake_sessions
```

Replace every `server._sessions["t-N"] = GraphContext(client=MagicMock(), model_config=ModelConfig(provider="google", model="m"))` line (there are six: `t-1`, `t-1b`, `t-2`, `t-3`, `t-4`, `t-5`) with:
```python
    isolated_session_store["t-1"] = ModelConfig(provider="google", model="m")
```
(substituting the matching thread id), and add `@patch("server.create_client")` with `mock_create_client.return_value = MagicMock()` to each of those tests (they need it now that `create_client` is called per-request rather than once at session creation).

Update `test_create_session_success`:
```python
@patch.dict("os.environ", {"SERPAPI_KEY": "fake", "GOOGLE_API_KEY": "fake"})
@patch("server.create_client")
def test_create_session_success(mock_create_client, client, isolated_session_store):
    mock_create_client.return_value = MagicMock()

    response = client.post("/session", json={"provider": "google", "model": "gemini-flash-lite-latest"})

    assert response.status_code == 200
    thread_id = response.json()["thread_id"]
    assert thread_id in isolated_session_store
```

- [ ] **Step 2: Run the test suite to confirm it fails against current server.py**

Run: `pytest tests/test_server.py -v`
Expected: FAIL — `server.session_store` doesn't exist yet.

- [ ] **Step 3: Implement the `server.py` changes**

Add the import:
```python
from tools import session_store
from tools.session_store import SessionNotFound
```

Delete:
```python
_sessions: dict[str, GraphContext] = {}
```

Replace `create_session`:
```python
@app.post("/session", response_model=SessionResponse)
def create_session(req: SessionRequest) -> SessionResponse:
    missing = [k for k in ["SERPAPI_KEY", PROVIDER_KEYS[req.provider]] if not os.environ.get(k)]
    if missing:
        raise HTTPException(status_code=400, detail=f"missing environment variables: {', '.join(missing)}")

    config = ModelConfig(provider=req.provider, model=req.model)
    thread_id = str(uuid.uuid4())
    session_store.create_session(thread_id, config)
    return SessionResponse(thread_id=thread_id)
```

Replace `_get_context`:
```python
def _get_context(thread_id: str) -> GraphContext:
    try:
        config = session_store.get_session(thread_id)
    except SessionNotFound:
        raise HTTPException(status_code=404, detail="unknown thread_id")
    return GraphContext(client=create_client(config), model_config=config)
```
(Any other exception — e.g. a Postgres connectivity error — is left unhandled here and propagates to FastAPI's default 500 handler, satisfying the spec's "fails loud" requirement with no extra code.)

- [ ] **Step 4: Run the test suite to verify it passes**

Run: `pytest tests/test_server.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add server.py tests/test_server.py
git commit -m "feat: replace in-memory _sessions dict with Postgres session registry"
```

- [ ] **Step 6: Reconcile "checkpointer fails loud" with the existing error envelope**

The design spec says a checkpointer failure should "surface as a request error (500)". `/chat` and `/resume` wrap `_graph.invoke(...)` in a broad `except Exception: return {"type": "error", ...}` — a 200 response with an explicit error payload, the same path any other graph-level failure (a down LLM API, a down SerpAPI) already takes, per `test_chat_graph_exception_returns_error_payload`. A mid-request checkpointer error takes this same path, not a distinct HTTP 500.

This still satisfies the spec's actual intent — the user sees an explicit error, never a silently-reset empty conversation — just not via the literal status code the spec named. Document this instead of leaving it ambiguous, and lock in the behavior with a test:

```python
# tests/test_server.py — add this test
@patch("server._graph")
def test_chat_checkpointer_error_surfaces_as_explicit_error_not_silent_reset(mock_graph, client, isolated_session_store):
    isolated_session_store["t-checkpointer-down"] = ModelConfig(provider="google", model="m")
    mock_graph.invoke.side_effect = OSError("could not connect to Postgres")

    with patch("server.create_client", return_value=MagicMock()):
        response = client.post("/chat", json={"thread_id": "t-checkpointer-down", "message": "hi"})

    # Not a fabricated 200 success, and not a fake empty-history response —
    # an explicit, visible error the client can show to the user.
    assert response.status_code == 200
    assert response.json() == {"type": "error", "message": "could not connect to Postgres"}
```

Run: `pytest tests/test_server.py::test_chat_checkpointer_error_surfaces_as_explicit_error_not_silent_reset -v`
Expected: PASS with no further code changes — this test documents existing, already-correct-in-spirit behavior.

- [ ] **Step 7: Commit**

```bash
git add tests/test_server.py
git commit -m "test: document checkpointer-failure behavior against the design spec's error philosophy"
```

---

### Task 5: Non-blocking `/chat` and `/resume`

**Files:**
- Modify: `server.py:91-109` (`chat`, `resume` route handlers)
- Test: `tests/test_server.py`

**Interfaces:**
- No new interfaces — existing `_graph.invoke(...)` call signature is unchanged, just invoked differently

- [ ] **Step 1: Write the failing test — handlers must not block each other**

```python
# tests/test_server.py — add this test
import time
import threading


@patch("server._graph")
def test_chat_requests_do_not_serialize_on_slow_graph_invoke(mock_graph, client, isolated_session_store):
    isolated_session_store["t-slow-1"] = ModelConfig(provider="google", model="m")
    isolated_session_store["t-slow-2"] = ModelConfig(provider="google", model="m")

    def slow_invoke(*args, **kwargs):
        time.sleep(0.3)
        return {"response": "done", "made_tool_call_this_turn": False, "last_search_results": None, "last_search_input": None}

    mock_graph.invoke.side_effect = slow_invoke

    results = []

    def _call(thread_id):
        with patch("server.create_client", return_value=MagicMock()):
            r = client.post("/chat", json={"thread_id": thread_id, "message": "hi"})
            results.append(r.status_code)

    start = time.monotonic()
    t1 = threading.Thread(target=_call, args=("t-slow-1",))
    t2 = threading.Thread(target=_call, args=("t-slow-2",))
    t1.start(); t2.start()
    t1.join(); t2.join()
    elapsed = time.monotonic() - start

    assert results == [200, 200]
    # Serialized would take ~0.6s; concurrent should take ~0.3s. Generous bound for CI jitter.
    assert elapsed < 0.5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_server.py::test_chat_requests_do_not_serialize_on_slow_graph_invoke -v`
Expected: FAIL (or flaky-pass near the 0.5s boundary) — today's handlers are synchronous `def`, and while `TestClient`/Starlette already runs sync `def` routes in a thread pool by default, this test locks in the requirement explicitly so the behavior can't regress silently if someone "simplifies" the handler back to fully synchronous blocking code later. Confirm today's elapsed time before changing anything, then proceed.

- [ ] **Step 3: Implement async handlers**

Add the import:
```python
from starlette.concurrency import run_in_threadpool
```

Replace `chat`:
```python
@app.post("/chat")
async def chat(req: ChatRequest) -> dict:
    context = _get_context(req.thread_id)
    try:
        result = await run_in_threadpool(_graph.invoke, {"new_message": req.message}, config=_graph_config(req.thread_id), context=context)
    except Exception as e:
        return {"type": "error", "message": str(e)}
    return _format_chat_result(result)
```

Replace `resume`:
```python
@app.post("/resume")
async def resume(req: ResumeRequest) -> dict:
    context = _get_context(req.thread_id)
    score = EvalScore(overall=req.score, note=req.note)
    try:
        await run_in_threadpool(_graph.invoke, Command(resume=score), config=_graph_config(req.thread_id), context=context)
    except Exception as e:
        return {"type": "error", "message": str(e)}
    return {"type": "message", "text": "Thanks for the rating!", "products": None, "view": None}
```

- [ ] **Step 4: Run the full server test suite**

Run: `pytest tests/test_server.py -v`
Expected: All PASS, including the new concurrency test.

- [ ] **Step 5: Commit**

```bash
git add server.py tests/test_server.py
git commit -m "feat: run graph.invoke in a thread pool so /chat and /resume don't block the event loop"
```

---

### Task 6: Postgres-backed search cache

**Files:**
- Modify: `tools/cache.py` (add a Postgres path alongside the existing SQLite path)
- Test: `tests/test_cache.py` (new Postgres-backed test class)

**Interfaces:**
- Consumes: `db.get_connection()` (Task 1)
- Produces: `lookup()`/`store()` signatures unchanged — `tools/amazon.py` requires no changes

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_cache.py — add this section
import pytest
from tests.postgres_fixtures import postgres_url  # noqa: F401


class TestPostgresBackend:
    @pytest.fixture(autouse=True)
    def _setup(self, postgres_url, monkeypatch):
        monkeypatch.setenv("DATABASE_URL", postgres_url)
        import db
        db.run_migrations(postgres_url)
        # Each test gets a clean searches table
        with db.get_connection() as conn:
            conn.execute("TRUNCATE searches")
            conn.commit()

    def test_store_then_lookup_exact_match_returns_results(self):
        from tools.cache import store, lookup
        store("balance beam", [{"title": "Beam"}])

        assert lookup("balance beam") == [{"title": "Beam"}]

    def test_lookup_exact_match_ignores_word_order(self):
        from tools.cache import store, lookup
        store("purple balance beam", [{"title": "Purple Beam"}])

        assert lookup("balance beam purple") == [{"title": "Purple Beam"}]

    def test_lookup_returns_none_on_empty_cache(self):
        from tools.cache import lookup
        assert lookup("anything") is None

    def test_lookup_gracefully_handles_unreachable_db(self, monkeypatch):
        from tools.cache import lookup
        monkeypatch.setenv("DATABASE_URL", "postgresql://bad:bad@localhost:1/nope")
        assert lookup("anything") is None

    def test_store_gracefully_handles_unreachable_db(self, monkeypatch):
        from tools.cache import store
        monkeypatch.setenv("DATABASE_URL", "postgresql://bad:bad@localhost:1/nope")
        store("anything", [{"title": "x"}])  # must not raise
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_cache.py::TestPostgresBackend -v`
Expected: FAIL — `lookup()`/`store()` only know about SQLite today, so these hit the SQLite path (env var is ignored) and the Postgres `searches` table stays empty.

- [ ] **Step 3: Implement the Postgres path in `tools/cache.py`**

```python
# tools/cache.py — replace the whole file
from __future__ import annotations
import json
import os
import re
import sqlite3
from datetime import datetime, timezone

import db as db_module
from tools import cache_judge

DB_PATH = os.path.expanduser("~/.anz-agent/cache.db")


def normalize(query: str) -> str:
    words = re.findall(r"[a-z0-9]+", query.lower())
    return " ".join(sorted(words))


def _shortlist_candidates(query: str, candidates: list[str], k: int = 20) -> list[str]:
    """Ranks candidates by token overlap with query, returns the top k.
    Pure function — no DB, backend-agnostic, used by both SQLite and
    Postgres paths ahead of the judge call."""
    query_tokens = set(normalize(query).split())
    scored = []
    for candidate in candidates:
        candidate_tokens = set(normalize(candidate).split())
        overlap = len(query_tokens & candidate_tokens)
        scored.append((overlap, candidate))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [candidate for _, candidate in scored[:k]]


def _using_postgres() -> bool:
    return bool(os.environ.get("DATABASE_URL"))


def lookup(query: str) -> list[dict] | None:
    return _lookup_postgres(query) if _using_postgres() else _lookup_sqlite(query)


def store(query: str, raw_results: list[dict]) -> None:
    if _using_postgres():
        _store_postgres(query, raw_results)
    else:
        _store_sqlite(query, raw_results)


# --- SQLite path (default; used when DATABASE_URL is unset) ---

def _connect_sqlite() -> sqlite3.Connection:
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


def _lookup_sqlite(query: str) -> list[dict] | None:
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
            matched_query = cache_judge.find_match(query, shortlist)
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


def _store_sqlite(query: str, raw_results: list[dict]) -> None:
    try:
        conn = _connect_sqlite()
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


# --- Postgres path (used when DATABASE_URL is set) ---

def _lookup_postgres(query: str) -> list[dict] | None:
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
            matched_query = cache_judge.find_match(query, shortlist)
            if matched_query is None:
                return None

            row = conn.execute(
                "SELECT raw_results FROM searches WHERE query = %s", (matched_query,)
            ).fetchone()
            return row[0] if row else None
    except Exception:
        return None


def _store_postgres(query: str, raw_results: list[dict]) -> None:
    try:
        with db_module.get_connection() as conn:
            conn.execute(
                """
                INSERT INTO searches (query, normalized_query, raw_results)
                VALUES (%s, %s, %s)
                ON CONFLICT (normalized_query) DO UPDATE SET raw_results = EXCLUDED.raw_results
                """,
                (query, normalize(query), json.dumps(raw_results)),
            )
            conn.commit()
    except Exception:
        pass
```

- [ ] **Step 4: Run the full cache test suite**

Run: `pytest tests/test_cache.py -v`
Expected: All PASS — both the pre-existing SQLite-backed tests (untouched, `DATABASE_URL` unset) and the new `TestPostgresBackend` class.

- [ ] **Step 5: Commit**

```bash
git add tools/cache.py tests/test_cache.py
git commit -m "feat: add Postgres-backed search cache alongside the SQLite default"
```

---

### Task 7: Cache-judge candidate pre-filter

**Files:**
- Modify: `tests/test_cache.py` (verify the shortlist is what actually reaches the judge)
- Modify: `docs/BACKLOG.md`

**Interfaces:**
- `_shortlist_candidates()` already exists from Task 6 — this task only adds test coverage proving it's wired into the judge call path, plus the backlog cleanup

- [ ] **Step 1: Write the failing test — pure ranking function**

```python
# tests/test_cache.py — add this test
def test_shortlist_candidates_ranks_by_token_overlap():
    from tools.cache import _shortlist_candidates
    candidates = ["yoga mat", "purple balance beam", "kettlebell 20lb", "balance beam blue"]

    result = _shortlist_candidates("balance beam", candidates, k=2)

    assert set(result) == {"purple balance beam", "balance beam blue"}


def test_shortlist_candidates_respects_k():
    from tools.cache import _shortlist_candidates
    candidates = [f"balance beam variant {i}" for i in range(50)]

    result = _shortlist_candidates("balance beam", candidates, k=20)

    assert len(result) == 20
```

- [ ] **Step 2: Write the failing test — judge only sees the shortlist, not every candidate**

```python
# tests/test_cache.py — add this test (SQLite path; DATABASE_URL unset)
def test_lookup_sends_only_shortlisted_candidates_to_judge():
    from tools.cache import store, lookup
    for i in range(30):
        store(f"unrelated product {i}", [{"title": f"p{i}"}])
    store("purple balance beam", [{"title": "Purple Beam"}])

    with patch("tools.cache_judge.find_match", return_value=None) as mock_find_match:
        lookup("balance beam")

    args, _ = mock_find_match.call_args
    assert len(args[1]) <= 20
```

- [ ] **Step 3: Run tests to verify current state**

Run: `pytest tests/test_cache.py::test_shortlist_candidates_ranks_by_token_overlap tests/test_cache.py::test_shortlist_candidates_respects_k tests/test_cache.py::test_lookup_sends_only_shortlisted_candidates_to_judge -v`
Expected: All PASS already — `_shortlist_candidates` and its wiring into `_lookup_sqlite`/`_lookup_postgres` were implemented in Task 6 (the ranking function was written there since both backends need it). This task exists to lock the behavior in with dedicated tests and close the loop on the backlog item. If any of these fail, fix `tools/cache.py` before proceeding — do not silently skip.

- [ ] **Step 4: Remove the resolved backlog entry**

In `docs/BACKLOG.md`, delete the "Judge candidate pre-filtering" bullet entirely (it now says "Addressed by the design in ... — remove this entry once that's implemented").

- [ ] **Step 5: Commit**

```bash
git add tests/test_cache.py docs/BACKLOG.md
git commit -m "test: lock in cache-judge candidate pre-filtering; close out BACKLOG.md item"
```

---

### Task 8: Cache data migration script

**Files:**
- Create: `scripts/migrate_cache_to_postgres.py`
- Create: `tests/test_migrate_cache.py`

**Interfaces:**
- Consumes: `db.get_connection()` (Task 1)
- Produces: `migrate_cache_to_postgres.migrate(sqlite_path: str, database_url: str) -> int` (returns rows migrated) — a thin `if __name__ == "__main__":` wrapper calls it with real paths

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_migrate_cache.py
import json
import sqlite3
from datetime import datetime, timezone
import pytest
from tests.postgres_fixtures import postgres_url  # noqa: F401


def _seed_sqlite(path: str):
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE searches (id INTEGER PRIMARY KEY, query TEXT, normalized_query TEXT UNIQUE, "
        "raw_results TEXT, created_at TEXT)"
    )
    conn.execute(
        "INSERT INTO searches (query, normalized_query, raw_results, created_at) VALUES (?, ?, ?, ?)",
        ("balance beam", "balance beam", json.dumps([{"title": "Beam"}]), datetime.now(timezone.utc).isoformat()),
    )
    conn.commit()
    conn.close()


@pytest.fixture(autouse=True)
def _schema(postgres_url):
    import db
    db.run_migrations(postgres_url)


def test_migrate_copies_all_rows(tmp_path, postgres_url):
    from scripts.migrate_cache_to_postgres import migrate
    sqlite_path = str(tmp_path / "cache.db")
    _seed_sqlite(sqlite_path)

    count = migrate(sqlite_path, postgres_url)

    assert count == 1
    from tools.cache import lookup
    import os
    os.environ["DATABASE_URL"] = postgres_url
    assert lookup("balance beam") == [{"title": "Beam"}]


def test_migrate_is_idempotent(tmp_path, postgres_url):
    from scripts.migrate_cache_to_postgres import migrate
    sqlite_path = str(tmp_path / "cache.db")
    _seed_sqlite(sqlite_path)

    migrate(sqlite_path, postgres_url)
    second_count = migrate(sqlite_path, postgres_url)  # re-run must not error or duplicate

    assert second_count == 1


def test_migrate_handles_missing_sqlite_file(tmp_path, postgres_url):
    from scripts.migrate_cache_to_postgres import migrate
    count = migrate(str(tmp_path / "does-not-exist.db"), postgres_url)
    assert count == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_migrate_cache.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts'`

- [ ] **Step 3: Implement the migration script**

```python
# scripts/migrate_cache_to_postgres.py
"""One-time migration of the local SQLite search cache into Postgres.

Run manually from the host after Compose's postgres service is up and its
port is exposed to localhost:

    DATABASE_URL=postgresql://anz:anz@localhost:5432/anz_agent \
        python scripts/migrate_cache_to_postgres.py
"""
from __future__ import annotations
import os
import sqlite3
from datetime import datetime

import psycopg


def migrate(sqlite_path: str, database_url: str) -> int:
    if not os.path.exists(sqlite_path):
        return 0

    sqlite_conn = sqlite3.connect(sqlite_path)
    rows = sqlite_conn.execute("SELECT query, normalized_query, raw_results, created_at FROM searches").fetchall()
    sqlite_conn.close()

    migrated = 0
    with psycopg.connect(database_url) as pg_conn:
        for query, normalized_query, raw_results, created_at in rows:
            pg_conn.execute(
                """
                INSERT INTO searches (query, normalized_query, raw_results, created_at)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (normalized_query) DO NOTHING
                """,
                (query, normalized_query, raw_results, datetime.fromisoformat(created_at)),
            )
            migrated += 1
        pg_conn.commit()
    return migrated


if __name__ == "__main__":
    sqlite_path = os.path.expanduser("~/.anz-agent/cache.db")
    database_url = os.environ["DATABASE_URL"]
    count = migrate(sqlite_path, database_url)
    print(f"Migrated {count} cache entries from {sqlite_path} to Postgres.")
```

Add `scripts/__init__.py` (empty) so `scripts.migrate_cache_to_postgres` is importable in tests:
```python
# scripts/__init__.py
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_migrate_cache.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/__init__.py scripts/migrate_cache_to_postgres.py tests/test_migrate_cache.py
git commit -m "feat: add one-time SQLite-to-Postgres cache migration script"
```

---

### Task 9: Dockerfile, Compose topology, nginx load balancer

**Files:**
- Create: `Dockerfile`
- Create: `docker-compose.yml`
- Create: `nginx.conf`
- Modify: `server.py` (add `GET /healthz`)
- Modify: `.env.example`
- Test: `tests/test_server.py` (unit test for `/healthz`); rest of this task is verified manually (infra config, not unit-testable)

**Interfaces:**
- No new Python interfaces beyond the `/healthz` route

- [ ] **Step 1: Write the failing test for `/healthz`**

```python
# tests/test_server.py — add this test
def test_healthz_returns_replica_id(client, monkeypatch):
    monkeypatch.setenv("REPLICA_ID", "app-1")
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"replica": "app-1"}


def test_healthz_defaults_to_unknown_without_replica_id(client, monkeypatch):
    monkeypatch.delenv("REPLICA_ID", raising=False)
    response = client.get("/healthz")
    assert response.json() == {"replica": "unknown"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_server.py::test_healthz_returns_replica_id -v`
Expected: FAIL — 404, no such route yet.

- [ ] **Step 3: Add the `/healthz` route**

In `server.py`, after the existing route definitions:
```python
@app.get("/healthz")
def healthz() -> dict:
    return {"replica": os.environ.get("REPLICA_ID", "unknown")}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_server.py -v`
Expected: All PASS

- [ ] **Step 5: Write the Dockerfile**

```dockerfile
# Dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
```

- [ ] **Step 6: Write the nginx config**

```nginx
# nginx.conf
events {}
http {
    upstream anz_agent {
        server app-1:8000 max_fails=2 fail_timeout=10s;
        server app-2:8000 max_fails=2 fail_timeout=10s;
    }
    server {
        listen 80;
        location / {
            proxy_pass http://anz_agent;
            proxy_set_header Host $host;
        }
    }
}
```

- [ ] **Step 7: Write docker-compose.yml**

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

  app-1:
    build: .
    env_file: .env
    environment:
      DATABASE_URL: postgresql://anz:anz@postgres:5432/anz_agent
      REPLICA_ID: app-1
    depends_on:
      postgres:
        condition: service_healthy

  app-2:
    build: .
    env_file: .env
    environment:
      DATABASE_URL: postgresql://anz:anz@postgres:5432/anz_agent
      REPLICA_ID: app-2
    depends_on:
      postgres:
        condition: service_healthy

  nginx:
    image: nginx:1.27-alpine
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
    ports:
      - "8000:80"
    depends_on:
      - app-1
      - app-2

volumes:
  pgdata:
```

- [ ] **Step 8: Document `DATABASE_URL` in `.env.example`**

Add:
```
# Set automatically by docker-compose.yml for app-1/app-2. Set manually
# (pointed at localhost:5432) when running scripts/migrate_cache_to_postgres.py
# from the host, or when running the app outside Compose against Postgres.
DATABASE_URL=
```

- [ ] **Step 9: Commit**

```bash
git add Dockerfile docker-compose.yml nginx.conf server.py tests/test_server.py .env.example
git commit -m "feat: add Docker Compose topology (2 replicas + nginx + Postgres)"
```

- [ ] **Step 10: Manual verification — build and run the full stack**

```bash
docker compose up --build -d
docker compose ps  # all four services should show healthy/running
```
Expected: `postgres`, `app-1`, `app-2`, `nginx` all up.

- [ ] **Step 11: Manual verification — round-robin load balancing**

```bash
curl -s http://localhost:8000/healthz
curl -s http://localhost:8000/healthz
curl -s http://localhost:8000/healthz
curl -s http://localhost:8000/healthz
```
Expected: responses alternate between `{"replica":"app-1"}` and `{"replica":"app-2"}`.

- [ ] **Step 12: Manual verification — passive health check routes around a dead replica**

```bash
docker compose stop app-2
curl -s http://localhost:8000/healthz
curl -s http://localhost:8000/healthz
curl -s http://localhost:8000/healthz
```
Expected: after at most `max_fails` (2) failed attempts, all subsequent responses are `{"replica":"app-1"}` — nginx stops routing to the stopped container.

```bash
docker compose start app-2  # restore for the next check
```

- [ ] **Step 13: Manual verification — a full chat round-trip through the stack**

```bash
curl -s -X POST http://localhost:8000/session -H 'Content-Type: application/json' \
  -d '{"provider": "google", "model": "gemini-flash-lite-latest"}'
# copy the returned thread_id into the next call
curl -s -X POST http://localhost:8000/chat -H 'Content-Type: application/json' \
  -d '{"thread_id": "<thread_id>", "message": "find me a yoga mat under $20"}'
```
Expected: a normal chat response — confirms the session registry, Postgres checkpointer, and cache all work end to end through nginx and either replica.

- [ ] **Step 14: Manual verification — cache migration**

```bash
DATABASE_URL=postgresql://anz:anz@localhost:5432/anz_agent python scripts/migrate_cache_to_postgres.py
```
Expected: prints `Migrated N cache entries...` where N matches however many rows exist in `~/.anz-agent/cache.db` (0 is fine if that file doesn't exist locally yet).

---

## Post-plan verification

- [ ] Run the full unit test suite with `DATABASE_URL` unset: `pytest` — expect all PASS, no Docker required, same speed as before this plan.
- [ ] Run the full suite including Postgres-backed tests: `pytest` with Docker running (the `testcontainers` fixtures start automatically) — expect all PASS.
- [ ] Confirm `docs/BACKLOG.md` no longer lists "Judge candidate pre-filtering" (removed in Task 7) and now lists "Vector similarity search for the cache judge" as the sole remaining cache-judge backlog item.
