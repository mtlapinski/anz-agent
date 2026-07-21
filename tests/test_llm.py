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


def make_google_text_response(text="Here are results"):
    part = MagicMock()
    part.text = text
    part.function_call = None  # falsy, so the tool-call branch is skipped
    candidate = MagicMock()
    candidate.content.parts = [part]
    response = MagicMock()
    response.candidates = [candidate]
    response.usage_metadata.prompt_token_count = 200
    response.usage_metadata.candidates_token_count = 60
    return response


def make_google_tool_response(name="search_amazon", args=None, thought_signature=b"sig-123"):
    if args is None:
        args = {"query": "blender", "optimize_for": "price", "max_results": 5}
    fc = MagicMock()
    fc.name = name
    fc.args = args
    part = MagicMock()
    part.function_call = fc
    part.text = None
    part.thought_signature = thought_signature
    candidate = MagicMock()
    candidate.content.parts = [part]
    response = MagicMock()
    response.candidates = [candidate]
    response.usage_metadata.prompt_token_count = 150
    response.usage_metadata.candidates_token_count = 30
    return response


def test_create_client_google():
    config = ModelConfig(provider="google", model="gemini-flash-lite-latest")
    with patch("google.genai.Client") as mock_cls, \
         patch("llm.os.environ", {"GOOGLE_API_KEY": "test-key"}):
        mock_cls.return_value = MagicMock()
        client = create_client(config)
    mock_cls.assert_called_once_with(api_key="test-key")
    assert client is not None


def test_complete_google_text_response():
    config = ModelConfig(provider="google", model="gemini-flash-lite-latest")

    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = make_google_text_response("Great choice!")

    result = complete(mock_client, config, "system", [], [{"role": "user", "content": "hi"}])

    assert result.text == "Great choice!"
    assert result.tool_calls is None
    assert result.input_tokens == 200
    assert result.output_tokens == 60


def test_complete_google_tool_response():
    config = ModelConfig(provider="google", model="gemini-flash-lite-latest")

    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = make_google_tool_response()

    result = complete(mock_client, config, "system", [], [{"role": "user", "content": "find blender"}])

    assert result.text is None
    assert result.tool_calls is not None
    assert result.tool_calls[0]["name"] == "search_amazon"
    assert result.tool_calls[0]["input"] == {"query": "blender", "optimize_for": "price", "max_results": 5}
    assert result.tool_calls[0]["thought_signature"] == b"sig-123"
    assert result.input_tokens == 150
    assert result.output_tokens == 30


def test_anthropic_messages_to_google_roundtrips_thought_signature():
    from llm import _anthropic_messages_to_google
    messages = [
        {"role": "user", "content": "find a purple balance beam"},
        {"role": "assistant", "content": [
            {
                "type": "tool_use",
                "id": "google_search_amazon",
                "name": "search_amazon",
                "input": {"query": "balance beam", "optimize_for": "price", "max_results": 5},
                "thought_signature": b"sig-123",
            }
        ]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "google_search_amazon", "content": '{"results": []}'}
        ]},
    ]
    result = _anthropic_messages_to_google(messages)
    fc_part = result[1]["parts"][0]
    assert fc_part["function_call"]["name"] == "search_amazon"
    assert fc_part["thought_signature"] == b"sig-123"


def test_anthropic_tools_to_google():
    from llm import _anthropic_tools_to_google
    anthropic_tools = [{
        "name": "search_amazon",
        "description": "Search Amazon",
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    }]
    result = _anthropic_tools_to_google(anthropic_tools)
    assert len(result) == 1
    decl = result[0]["function_declarations"][0]
    assert decl["name"] == "search_amazon"
    assert decl["description"] == "Search Amazon"
    assert decl["parameters_json_schema"]["properties"]["query"]["type"] == "string"


def test_anthropic_messages_to_google_plain_text():
    from llm import _anthropic_messages_to_google
    messages = [
        {"role": "user", "content": "Find me a blender"},
        {"role": "assistant", "content": [{"type": "text", "text": "What's your budget?"}]},
        {"role": "user", "content": "Under $50"},
    ]
    result = _anthropic_messages_to_google(messages)
    assert result[0] == {"role": "user", "parts": [{"text": "Find me a blender"}]}
    assert result[1] == {"role": "model", "parts": [{"text": "What's your budget?"}]}
    assert result[2] == {"role": "user", "parts": [{"text": "Under $50"}]}
