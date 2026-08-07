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
                ON CONFLICT (normalized_query) DO UPDATE SET
                    query = EXCLUDED.query,
                    raw_results = EXCLUDED.raw_results,
                    created_at = EXCLUDED.created_at
                """,
                (query, normalize(query), json.dumps(raw_results)),
            )
            conn.commit()
    except Exception:
        pass
