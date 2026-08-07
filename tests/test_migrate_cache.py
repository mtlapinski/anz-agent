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
