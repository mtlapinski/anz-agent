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
