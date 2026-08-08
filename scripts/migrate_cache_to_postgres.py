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
