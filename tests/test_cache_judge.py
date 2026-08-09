import pytest
from unittest.mock import MagicMock, patch
from llm import LLMResponse
import tools.cache_judge as cache_judge
from tools.cache_judge import CacheMatch


@pytest.fixture(autouse=True)
def reset_client():
    cache_judge._client = None
    yield
    cache_judge._client = None


@patch("tools.cache_judge.llm")
def test_find_match_returns_matched_candidate(mock_llm):
    mock_llm.complete.return_value = LLMResponse(
        text="purple balance beam", tool_calls=None, input_tokens=10, output_tokens=2
    )

    result = cache_judge.find_match("balance beam", ["purple balance beam", "yoga mat"])

    assert result == CacheMatch("purple balance beam", "matched")


@patch("tools.cache_judge.llm")
def test_find_match_returns_no_match_when_judge_says_none(mock_llm):
    mock_llm.complete.return_value = LLMResponse(
        text="NONE", tool_calls=None, input_tokens=10, output_tokens=2
    )

    result = cache_judge.find_match("kettlebell", ["yoga mat"])

    assert result == CacheMatch(None, "no_match")
    mock_llm.complete.assert_called_once()  # a literal NONE is not retried


@patch("tools.cache_judge.llm")
def test_find_match_retries_then_succeeds_on_malformed_first_answer(mock_llm):
    mock_llm.complete.side_effect = [
        LLMResponse(text="something not in the candidate list", tool_calls=None, input_tokens=10, output_tokens=2),
        LLMResponse(text="yoga mat", tool_calls=None, input_tokens=10, output_tokens=2),
    ]

    result = cache_judge.find_match("kettlebell", ["yoga mat"])

    assert result == CacheMatch("yoga mat", "matched")
    assert mock_llm.complete.call_count == 2


@patch("tools.cache_judge.llm")
def test_find_match_retries_then_succeeds_on_transient_exception(mock_llm):
    mock_llm.complete.side_effect = [
        Exception("rate limited"),
        LLMResponse(text="yoga mat", tool_calls=None, input_tokens=10, output_tokens=2),
    ]

    result = cache_judge.find_match("kettlebell", ["yoga mat"])

    assert result == CacheMatch("yoga mat", "matched")
    assert mock_llm.complete.call_count == 2


@patch("tools.cache_judge.llm")
def test_find_match_returns_error_when_both_attempts_raise(mock_llm):
    mock_llm.complete.side_effect = Exception("rate limited")

    result = cache_judge.find_match("kettlebell", ["yoga mat"])

    assert result == CacheMatch(None, "error")
    assert mock_llm.complete.call_count == 2


@patch("tools.cache_judge.llm")
def test_find_match_returns_error_when_both_answers_malformed(mock_llm):
    mock_llm.complete.return_value = LLMResponse(
        text="something not in the candidate list", tool_calls=None, input_tokens=10, output_tokens=2
    )

    result = cache_judge.find_match("kettlebell", ["yoga mat"])

    assert result == CacheMatch(None, "error")
    assert mock_llm.complete.call_count == 2


@patch("tools.cache_judge.llm")
def test_find_match_returns_error_on_client_creation_error(mock_llm):
    mock_llm.create_client.side_effect = Exception("missing GOOGLE_API_KEY")

    result = cache_judge.find_match("kettlebell", ["yoga mat"])

    assert result == CacheMatch(None, "error")


def test_find_match_returns_no_match_when_no_candidates():
    result = cache_judge.find_match("kettlebell", [])
    assert result == CacheMatch(None, "no_match")


@patch("tools.cache_judge.get_langfuse")
def test_find_match_no_span_when_no_candidates_even_with_trace_id(mock_get_langfuse):
    result = cache_judge.find_match("kettlebell", [], trace_id="trace-123")

    assert result == CacheMatch(None, "no_match")
    mock_get_langfuse.assert_not_called()


@patch("tools.cache_judge.get_langfuse")
@patch("tools.cache_judge.llm")
def test_find_match_opens_and_ends_langfuse_generation_with_trace_id(mock_llm, mock_get_langfuse):
    mock_llm.complete.return_value = LLMResponse(
        text="yoga mat", tool_calls=None, input_tokens=10, output_tokens=2
    )
    mock_span = MagicMock()
    mock_get_langfuse.return_value.start_observation.return_value = mock_span

    cache_judge.find_match("kettlebell", ["yoga mat"], trace_id="trace-123")

    mock_get_langfuse.return_value.start_observation.assert_called_once_with(
        trace_context={"trace_id": "trace-123"},
        name="cache_judge",
        as_type="generation",
        input={"query": "kettlebell", "candidates": ["yoga mat"]},
        model="google/gemini-flash-lite-latest",
    )
    mock_span.update.assert_called_once_with(
        output={"matched_query": "yoga mat", "outcome": "matched", "attempts": 1}
    )
    mock_span.end.assert_called_once()


@patch("tools.cache_judge.get_langfuse")
@patch("tools.cache_judge.llm")
def test_find_match_no_span_when_trace_id_none(mock_llm, mock_get_langfuse):
    mock_llm.complete.return_value = LLMResponse(
        text="yoga mat", tool_calls=None, input_tokens=10, output_tokens=2
    )

    cache_judge.find_match("kettlebell", ["yoga mat"])

    mock_get_langfuse.assert_not_called()


@patch("tools.cache_judge.get_langfuse")
@patch("tools.cache_judge.llm")
def test_find_match_fails_open_when_span_creation_raises(mock_llm, mock_get_langfuse):
    mock_llm.complete.return_value = LLMResponse(
        text="yoga mat", tool_calls=None, input_tokens=10, output_tokens=2
    )
    mock_get_langfuse.return_value.start_observation.side_effect = Exception("langfuse down")

    result = cache_judge.find_match("kettlebell", ["yoga mat"], trace_id="trace-123")

    assert result == CacheMatch("yoga mat", "matched")


@patch("tools.cache_judge.get_langfuse")
@patch("tools.cache_judge.llm")
def test_find_match_fails_open_when_span_update_raises(mock_llm, mock_get_langfuse):
    mock_llm.complete.return_value = LLMResponse(
        text="yoga mat", tool_calls=None, input_tokens=10, output_tokens=2
    )
    mock_span = MagicMock()
    mock_span.update.side_effect = Exception("langfuse down")
    mock_get_langfuse.return_value.start_observation.return_value = mock_span

    result = cache_judge.find_match("kettlebell", ["yoga mat"], trace_id="trace-123")

    assert result == CacheMatch("yoga mat", "matched")
