import pytest
from unittest.mock import MagicMock, patch
from llm import ModelConfig, LLMResponse, create_client, complete


def make_anthropic_text_response(text="Here are results"):
    block = MagicMock()
    block.type = "text"
    block.text = text
    response = MagicMock()
    response.stop_reason = "end_turn"
    response.content = [block]
    response.usage.input_tokens = 100
    response.usage.output_tokens = 50
    return response


def make_anthropic_tool_response(name="search_amazon", tool_id="tu_1", input=None):
    if input is None:
        input = {"query": "blender", "optimize_for": "price", "max_results": 5}
    block = MagicMock()
    block.type = "tool_use"
    block.name = name
    block.id = tool_id
    block.input = input
    response = MagicMock()
    response.stop_reason = "tool_use"
    response.content = [block]
    response.usage.input_tokens = 80
    response.usage.output_tokens = 20
    return response


def test_create_client_anthropic():
    config = ModelConfig(provider="anthropic", model="claude-haiku-4-5-20251001")
    with patch("llm.anthropic.Anthropic") as mock_cls, \
         patch("llm.os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
        mock_cls.return_value = MagicMock()
        client = create_client(config)
    mock_cls.assert_called_once()
    assert client is not None


def test_create_client_unknown_provider_raises():
    config = ModelConfig(provider="openai", model="gpt-4")
    with pytest.raises(ValueError, match="Unknown provider"):
        create_client(config)


def test_complete_anthropic_text_response():
    config = ModelConfig(provider="anthropic", model="claude-haiku-4-5-20251001")
    mock_client = MagicMock()
    mock_client.messages.create.return_value = make_anthropic_text_response("Found it!")

    result = complete(mock_client, config, "system", [], [{"role": "user", "content": "hi"}])

    assert isinstance(result, LLMResponse)
    assert result.text == "Found it!"
    assert result.tool_calls is None
    assert result.input_tokens == 100
    assert result.output_tokens == 50


def test_complete_anthropic_tool_response():
    config = ModelConfig(provider="anthropic", model="claude-haiku-4-5-20251001")
    mock_client = MagicMock()
    mock_client.messages.create.return_value = make_anthropic_tool_response()

    result = complete(mock_client, config, "system", [], [{"role": "user", "content": "find blender"}])

    assert result.text is None
    assert result.tool_calls is not None
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0]["name"] == "search_amazon"
    assert result.tool_calls[0]["id"] == "tu_1"
    assert result.tool_calls[0]["input"] == {"query": "blender", "optimize_for": "price", "max_results": 5}
    assert result.input_tokens == 80
    assert result.output_tokens == 20


def test_complete_anthropic_unexpected_stop_reason_raises():
    config = ModelConfig(provider="anthropic", model="claude-haiku-4-5-20251001")
    mock_client = MagicMock()
    response = MagicMock()
    response.stop_reason = "max_tokens"
    response.content = []
    response.usage.input_tokens = 10
    response.usage.output_tokens = 5
    mock_client.messages.create.return_value = response

    with pytest.raises(RuntimeError, match="Unexpected stop_reason"):
        complete(mock_client, config, "system", [], [{"role": "user", "content": "hi"}])
