import httpx
import pytest
from unittest.mock import MagicMock, patch
from tools.cache_judge import CacheMatch
import tools.cache_judge_client as cache_judge_client


@pytest.fixture(autouse=True)
def reset_client():
    cache_judge_client._client = None
    yield
    cache_judge_client._client = None


def test_find_match_dispatches_in_process_when_url_unset(monkeypatch):
    monkeypatch.delenv("CACHE_JUDGE_URL", raising=False)
    with patch(
        "tools.cache_judge_client._find_match_in_process",
        return_value=CacheMatch("yoga mat", "matched"),
    ) as mock_find:
        result = cache_judge_client.find_match("kettlebell", ["yoga mat"], trace_id="trace-1")

    mock_find.assert_called_once_with("kettlebell", ["yoga mat"], trace_id="trace-1")
    assert result == CacheMatch("yoga mat", "matched")


def test_find_match_posts_to_service_when_url_set(monkeypatch):
    monkeypatch.setenv("CACHE_JUDGE_URL", "http://judge:8001/match")
    mock_response = MagicMock()
    mock_response.raise_for_status.return_value = None
    mock_response.json.return_value = {"matched_query": "yoga mat", "outcome": "matched"}

    with patch.object(httpx.Client, "post", return_value=mock_response) as mock_post:
        result = cache_judge_client.find_match("kettlebell", ["yoga mat"], trace_id="trace-1")

    mock_post.assert_called_once_with(
        "http://judge:8001/match",
        json={"query": "kettlebell", "candidates": ["yoga mat"], "trace_id": "trace-1"},
    )
    assert result == CacheMatch("yoga mat", "matched")


def test_find_match_forwards_none_trace_id_when_url_set(monkeypatch):
    monkeypatch.setenv("CACHE_JUDGE_URL", "http://judge:8001/match")
    mock_response = MagicMock()
    mock_response.raise_for_status.return_value = None
    mock_response.json.return_value = {"matched_query": None, "outcome": "no_match"}

    with patch.object(httpx.Client, "post", return_value=mock_response) as mock_post:
        cache_judge_client.find_match("kettlebell", ["yoga mat"])

    mock_post.assert_called_once_with(
        "http://judge:8001/match",
        json={"query": "kettlebell", "candidates": ["yoga mat"], "trace_id": None},
    )


def test_find_match_fails_open_on_timeout(monkeypatch):
    monkeypatch.setenv("CACHE_JUDGE_URL", "http://judge:8001/match")
    with patch.object(httpx.Client, "post", side_effect=httpx.TimeoutException("timed out")):
        result = cache_judge_client.find_match("kettlebell", ["yoga mat"])

    assert result == CacheMatch(None, "error")


def test_find_match_fails_open_on_connection_error(monkeypatch):
    monkeypatch.setenv("CACHE_JUDGE_URL", "http://judge:8001/match")
    with patch.object(httpx.Client, "post", side_effect=httpx.ConnectError("refused")):
        result = cache_judge_client.find_match("kettlebell", ["yoga mat"])

    assert result == CacheMatch(None, "error")


def test_find_match_fails_open_on_non_2xx(monkeypatch):
    monkeypatch.setenv("CACHE_JUDGE_URL", "http://judge:8001/match")
    mock_response = MagicMock()
    mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
        "server error", request=MagicMock(), response=MagicMock(status_code=500)
    )

    with patch.object(httpx.Client, "post", return_value=mock_response):
        result = cache_judge_client.find_match("kettlebell", ["yoga mat"])

    assert result == CacheMatch(None, "error")


def test_find_match_fails_open_on_malformed_json(monkeypatch):
    monkeypatch.setenv("CACHE_JUDGE_URL", "http://judge:8001/match")
    mock_response = MagicMock()
    mock_response.raise_for_status.return_value = None
    mock_response.json.side_effect = ValueError("not json")

    with patch.object(httpx.Client, "post", return_value=mock_response):
        result = cache_judge_client.find_match("kettlebell", ["yoga mat"])

    assert result == CacheMatch(None, "error")


def test_find_match_never_falls_back_to_in_process_on_network_failure(monkeypatch):
    """A network failure must fail open to an error result, not silently retry
    in-process — that would blur the CACHE_JUDGE_URL canary signal."""
    monkeypatch.setenv("CACHE_JUDGE_URL", "http://judge:8001/match")
    with patch.object(httpx.Client, "post", side_effect=httpx.ConnectError("refused")), \
         patch("tools.cache_judge_client._find_match_in_process") as mock_in_process:
        cache_judge_client.find_match("kettlebell", ["yoga mat"])

    mock_in_process.assert_not_called()


@patch("tools.cache_judge_client.get_langfuse")
def test_find_match_opens_and_ends_langfuse_span_on_network_failure_with_trace_id(mock_get_langfuse, monkeypatch):
    monkeypatch.setenv("CACHE_JUDGE_URL", "http://judge:8001/match")
    mock_span = MagicMock()
    mock_get_langfuse.return_value.start_observation.return_value = mock_span

    with patch.object(httpx.Client, "post", side_effect=httpx.ConnectError("refused")):
        result = cache_judge_client.find_match("kettlebell", ["yoga mat"], trace_id="trace-1")

    mock_get_langfuse.return_value.start_observation.assert_called_once_with(
        trace_context={"trace_id": "trace-1"},
        name="cache_judge",
        as_type="generation",
        input={"query": "kettlebell", "candidates": ["yoga mat"]},
    )
    mock_span.update.assert_called_once_with(
        output={"matched_query": None, "outcome": "error", "transport": "network"}
    )
    mock_span.end.assert_called_once()
    assert result == CacheMatch(None, "error")


@patch("tools.cache_judge_client.get_langfuse")
def test_find_match_network_failure_with_no_trace_id_skips_langfuse(mock_get_langfuse, monkeypatch):
    monkeypatch.setenv("CACHE_JUDGE_URL", "http://judge:8001/match")

    with patch.object(httpx.Client, "post", side_effect=httpx.ConnectError("refused")):
        result = cache_judge_client.find_match("kettlebell", ["yoga mat"])

    mock_get_langfuse.assert_not_called()
    assert result == CacheMatch(None, "error")


@patch("tools.cache_judge_client.get_langfuse")
def test_find_match_fails_open_when_langfuse_itself_fails_during_network_error_handling(mock_get_langfuse, monkeypatch):
    monkeypatch.setenv("CACHE_JUDGE_URL", "http://judge:8001/match")
    mock_get_langfuse.return_value.start_observation.side_effect = Exception("langfuse down")

    with patch.object(httpx.Client, "post", side_effect=httpx.ConnectError("refused")):
        result = cache_judge_client.find_match("kettlebell", ["yoga mat"], trace_id="trace-1")

    assert result == CacheMatch(None, "error")
