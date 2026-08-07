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
