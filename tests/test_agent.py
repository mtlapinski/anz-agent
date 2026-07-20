import json
import pytest
from unittest.mock import MagicMock, patch, call
from llm import ModelConfig, LLMResponse


def default_config():
    return ModelConfig(provider="anthropic", model="claude-haiku-4-5-20251001")


def test_run_tool_search_amazon():
    from agent import run_tool
    with patch("agent.search_amazon") as mock_search:
        mock_search.return_value = {"products": []}
        result = run_tool("search_amazon", {"query": "laptop", "optimize_for": "price", "max_results": 5})
    assert json.loads(result) == {"products": []}
    mock_search.assert_called_once_with(query="laptop", optimize_for="price", max_results=5)


def test_run_tool_unknown_raises():
    from agent import run_tool
    with pytest.raises(ValueError, match="Unknown tool"):
        run_tool("nonexistent_tool", {})


@patch("agent._get_langfuse")
@patch("agent.llm")
def test_chat_text_response_updates_history(mock_llm, mock_lf):
    from agent import chat
    mock_llm.complete.return_value = LLMResponse(
        text="What are you looking for?", tool_calls=None, input_tokens=10, output_tokens=5
    )
    history = []
    result = chat(MagicMock(), history, "Hello", default_config())
    assert result == "What are you looking for?"
    assert history[0] == {"role": "user", "content": "Hello"}
    assert history[1]["role"] == "assistant"


@patch("agent._get_langfuse")
@patch("agent.llm")
def test_chat_tool_call_then_text(mock_llm, mock_lf):
    from agent import chat
    mock_llm.complete.side_effect = [
        LLMResponse(text=None, tool_calls=[{"name": "search_amazon", "id": "tu_123",
            "input": {"query": "laptop", "optimize_for": "price", "max_results": 5}}],
            input_tokens=80, output_tokens=20),
        LLMResponse(text="Here are the top laptops...", tool_calls=None, input_tokens=200, output_tokens=50),
    ]
    with patch("agent.search_amazon") as mock_search:
        mock_search.return_value = {"products": []}
        history = []
        result = chat(MagicMock(), history, "Find me a laptop", default_config())
    assert result == "Here are the top laptops..."
    assert mock_llm.complete.call_count == 2
    assert len(history) == 4
    tool_result_msg = history[2]
    assert tool_result_msg["role"] == "user"
    assert tool_result_msg["content"][0]["type"] == "tool_result"
    assert tool_result_msg["content"][0]["tool_use_id"] == "tu_123"


@patch("agent._get_langfuse")
@patch("agent.llm")
def test_chat_passes_system_prompt(mock_llm, mock_lf):
    from agent import chat, SYSTEM_PROMPT
    mock_llm.complete.return_value = LLMResponse(
        text="hi", tool_calls=None, input_tokens=10, output_tokens=5
    )
    config = default_config()
    chat(MagicMock(), [], "hi", config)
    call_kwargs = mock_llm.complete.call_args
    assert call_kwargs.args[2] == SYSTEM_PROMPT   # system is 3rd positional arg


@patch("agent._get_langfuse")
@patch("agent.llm")
def test_chat_logs_token_usage(mock_llm, mock_lf):
    from agent import chat
    mock_llm.complete.return_value = LLMResponse(
        text="Here are results...", tool_calls=None, input_tokens=100, output_tokens=50
    )
    mock_trace = MagicMock()
    mock_lf.return_value.trace.return_value = mock_trace
    mock_generation = MagicMock()
    mock_trace.generation.return_value = mock_generation

    chat(MagicMock(), [], "find me a blender", default_config())

    mock_generation.end.assert_called_once()
    call_kwargs = mock_generation.end.call_args.kwargs
    assert call_kwargs["usage"]["input"] == 100
    assert call_kwargs["usage"]["output"] == 50
