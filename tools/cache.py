from __future__ import annotations
import json
import os
import re
import sqlite3
from datetime import datetime, timezone

DB_PATH = os.path.expanduser("~/.anz-agent/cache.db")


def normalize(query: str) -> str:
    words = re.findall(r"[a-z0-9]+", query.lower())
    return " ".join(sorted(words))


def _connect() -> sqlite3.Connection:
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


def lookup(query: str) -> list[dict] | None:
    try:
        conn = _connect()
        try:
            normalized = normalize(query)
            row = conn.execute(
                "SELECT raw_results FROM searches WHERE normalized_query = ?", (normalized,)
            ).fetchone()
            if row:
                return json.loads(row[0])
            return None
        finally:
            conn.close()
    except Exception:
        return None


def store(query: str, raw_results: list[dict]) -> None:
    try:
        conn = _connect()
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
