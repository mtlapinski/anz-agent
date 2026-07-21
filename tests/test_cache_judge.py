import pytest
from unittest.mock import patch
from llm import LLMResponse
import tools.cache_judge as cache_judge


@pytest.fixture(autouse=True)
def reset_client():
    cache_judge._client = None
    yield
    cache_judge._client = None


@patch("tools.cache_judge.llm")
def test_find_match_returns_exact_candidate_text(mock_llm):
    mock_llm.complete.return_value = LLMResponse(
        text="purple balance beam", tool_calls=None, input_tokens=10, output_tokens=2
    )

    result = cache_judge.find_match("balance beam", ["purple balance beam", "yoga mat"])

    assert result == "purple balance beam"


@patch("tools.cache_judge.llm")
def test_find_match_returns_none_when_judge_says_none(mock_llm):
    mock_llm.complete.return_value = LLMResponse(
        text="NONE", tool_calls=None, input_tokens=10, output_tokens=2
    )

    result = cache_judge.find_match("kettlebell", ["yoga mat"])

    assert result is None


@patch("tools.cache_judge.llm")
def test_find_match_returns_none_when_judge_hallucinates_a_query(mock_llm):
    mock_llm.complete.return_value = LLMResponse(
        text="something not in the candidate list", tool_calls=None, input_tokens=10, output_tokens=2
    )

    result = cache_judge.find_match("kettlebell", ["yoga mat"])

    assert result is None


@patch("tools.cache_judge.llm")
def test_find_match_returns_none_on_completion_error(mock_llm):
    mock_llm.complete.side_effect = Exception("rate limited")

    result = cache_judge.find_match("kettlebell", ["yoga mat"])

    assert result is None


@patch("tools.cache_judge.llm")
def test_find_match_returns_none_on_client_creation_error(mock_llm):
    mock_llm.create_client.side_effect = Exception("missing GOOGLE_API_KEY")

    result = cache_judge.find_match("kettlebell", ["yoga mat"])

    assert result is None


def test_find_match_returns_none_when_no_candidates():
    result = cache_judge.find_match("kettlebell", [])
    assert result is None
