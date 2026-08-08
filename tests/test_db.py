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
