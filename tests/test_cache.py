import pytest
from unittest.mock import patch


def test_normalize_ignores_word_order_and_case():
    from tools.cache import normalize
    assert normalize("Purple Balance Beam") == normalize("balance beam purple")


def test_normalize_strips_punctuation():
    from tools.cache import normalize
    assert normalize("balance-beam!") == normalize("balance beam")


def test_store_then_lookup_exact_match_returns_results():
    from tools.cache import store, lookup
    store("balance beam", [{"title": "Beam"}])

    result = lookup("balance beam")

    assert result == [{"title": "Beam"}]


def test_lookup_exact_match_ignores_word_order():
    from tools.cache import store, lookup
    store("purple balance beam", [{"title": "Purple Beam"}])

    result = lookup("balance beam purple")

    assert result == [{"title": "Purple Beam"}]


def test_lookup_returns_none_on_empty_cache():
    from tools.cache import lookup
    assert lookup("anything") is None


def test_lookup_gracefully_handles_unreachable_db(monkeypatch, tmp_path):
    """Verify lookup returns None (cache miss) when DB is unreachable, not an exception."""
    from tools.cache import lookup
    # Point DB_PATH to a directory, not a file — sqlite3.connect() will fail
    monkeypatch.setattr("tools.cache.DB_PATH", str(tmp_path))

    result = lookup("test query")

    # Should return None (cache miss) not raise an exception
    assert result is None


def test_store_gracefully_handles_unreachable_db(monkeypatch, tmp_path):
    """Verify store does not raise when DB is unreachable."""
    from tools.cache import store
    # Point DB_PATH to a directory, not a file — sqlite3.connect() will fail
    monkeypatch.setattr("tools.cache.DB_PATH", str(tmp_path))

    # Should not raise an exception, even though DB is unreachable
    store("test query", [{"title": "test"}])


def test_lookup_uses_judge_for_fuzzy_match():
    from tools.cache import store, lookup
    store("purple balance beam", [{"title": "Purple Beam"}])

    with patch("tools.cache_judge.find_match", return_value="purple balance beam"):
        result = lookup("balance beam")

    assert result == [{"title": "Purple Beam"}]


def test_lookup_returns_none_when_judge_finds_no_match():
    from tools.cache import store, lookup
    store("yoga mat", [{"title": "Mat"}])

    with patch("tools.cache_judge.find_match", return_value=None):
        result = lookup("kettlebell")

    assert result is None


def test_lookup_does_not_call_judge_when_cache_empty():
    from tools.cache import lookup

    with patch("tools.cache_judge.find_match") as mock_find_match:
        result = lookup("anything")

    assert result is None
    mock_find_match.assert_not_called()


def test_lookup_returns_none_when_judge_raises():
    from tools.cache import store, lookup
    store("yoga mat", [{"title": "Mat"}])

    with patch("tools.cache_judge.find_match", side_effect=Exception("boom")):
        result = lookup("kettlebell")

    assert result is None


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


def test_shortlist_candidates_breaks_ties_alphabetically():
    from tools.cache import _shortlist_candidates
    # All three share the same token overlap with "balance beam" (score 2),
    # so the tie must be broken deterministically by candidate string.
    candidates = ["beam balance zebra", "beam balance apple", "beam balance mango"]

    result = _shortlist_candidates("balance beam", candidates, k=2)

    assert result == ["beam balance apple", "beam balance mango"]


def test_lookup_sends_only_shortlisted_candidates_to_judge():
    from tools.cache import store, lookup
    for i in range(30):
        store(f"unrelated product {i}", [{"title": f"p{i}"}])
    store("purple balance beam", [{"title": "Purple Beam"}])

    with patch("tools.cache_judge.find_match", return_value=None) as mock_find_match:
        lookup("balance beam")

    args, _ = mock_find_match.call_args
    assert len(args[1]) <= 20


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

    def test_lookup_uses_judge_for_fuzzy_match(self):
        from tools.cache import store, lookup
        store("purple balance beam", [{"title": "Purple Beam"}])

        with patch("tools.cache_judge.find_match", return_value="purple balance beam"):
            result = lookup("balance beam")

        assert result == [{"title": "Purple Beam"}]

    def test_store_replaces_entire_row_on_conflict(self):
        """Verify that storing with the same normalized query but different
        case/punctuation replaces the entire row (query, raw_results, created_at)."""
        from tools.cache import store, lookup
        # First store with lowercase
        store("balance beam", [{"title": "Beam"}])
        # Verify it's there
        assert lookup("balance beam") == [{"title": "Beam"}]

        # Store again with same normalized form but different casing
        store("Balance Beam", [{"title": "Better Beam"}])
        # Second store should win — both result and query string should be updated
        assert lookup("balance beam") == [{"title": "Better Beam"}]
