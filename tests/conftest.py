import os
import pytest


@pytest.fixture(scope="session", autouse=True)
def setup_testcontainers_env():
    """Configure testcontainers for use with Colima on macOS."""
    os.environ.setdefault(
        "TESTCONTAINERS_DOCKER_SOCKET_OVERRIDE", "/var/run/docker.sock"
    )


@pytest.fixture(autouse=True)
def isolated_cache_db(tmp_path, monkeypatch):
    """Every test gets its own empty cache file — never touches the real
    ~/.anz-agent/cache.db and never leaks state between tests."""
    monkeypatch.setattr("tools.cache.DB_PATH", str(tmp_path / "cache.db"))
