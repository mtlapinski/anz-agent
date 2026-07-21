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


from unittest.mock import patch


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
