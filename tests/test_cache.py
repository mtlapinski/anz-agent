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
