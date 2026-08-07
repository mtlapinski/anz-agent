import pytest
from tests.postgres_fixtures import postgres_url  # noqa: F401


@pytest.fixture(autouse=True)
def isolated_cache_db(tmp_path, monkeypatch):
    """Every test gets its own empty cache file — never touches the real
    ~/.anz-agent/cache.db and never leaks state between tests."""
    monkeypatch.setattr("tools.cache.DB_PATH", str(tmp_path / "cache.db"))
