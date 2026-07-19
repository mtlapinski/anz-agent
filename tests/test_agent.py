import json
import pytest
from unittest.mock import MagicMock, patch, call


def make_text_response(text="What are you looking for?"):
    block = MagicMock()
    block.type = "text"
    block.text = text
    response = MagicMock()
    response.stop_reason = "end_turn"
    response.content = [block]
    return response


def make_tool_response(tool_name="search_amazon", tool_input=None, tool_id="tu_123"):
    if tool_input is None:
        tool_input = {"query": "laptop", "optimize_for": "price", "max_results": 5}
    block = MagicMock()
    block.type = "tool_use"
    block.name = tool_name
    block.id = tool_id
    block.input = tool_input
    response = MagicMock()
    response.stop_reason = "tool_use"
    response.content = [block]
    return response


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


def test_chat_text_response_updates_history():
    from agent import chat
    mock_client = MagicMock()
    mock_client.messages.create.return_value = make_text_response("What are you looking for?")

    history = []
    result = chat(mock_client, history, "Hello")

    assert result == "What are you looking for?"
    assert history[0] == {"role": "user", "content": "Hello"}
    assert history[1]["role"] == "assistant"


def test_chat_tool_call_then_text():
    from agent import chat
    mock_client = MagicMock()
    mock_client.messages.create.side_effect = [
        make_tool_response(),
        make_text_response("Here are the top laptops..."),
    ]

    with patch("agent.search_amazon") as mock_search:
        mock_search.return_value = {"products": []}
        history = []
        result = chat(mock_client, history, "Find me a laptop")

    assert result == "Here are the top laptops..."
    assert mock_client.messages.create.call_count == 2
    # history: user, assistant (tool call), user (tool result), assistant (text)
    assert len(history) == 4
    tool_result_msg = history[2]
    assert tool_result_msg["role"] == "user"
    assert tool_result_msg["content"][0]["type"] == "tool_result"
    assert tool_result_msg["content"][0]["tool_use_id"] == "tu_123"
    assert isinstance(tool_result_msg["content"][0]["content"], str)


def test_chat_passes_system_prompt():
    from agent import chat, SYSTEM_PROMPT
    mock_client = MagicMock()
    mock_client.messages.create.return_value = make_text_response()

    chat(mock_client, [], "hi")

    call_kwargs = mock_client.messages.create.call_args.kwargs
    assert call_kwargs["system"] == SYSTEM_PROMPT
    assert call_kwargs["model"] == "claude-sonnet-4-6"
