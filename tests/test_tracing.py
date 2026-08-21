import os
from unittest.mock import patch
import pytest


@pytest.fixture(autouse=True)
def reset_singleton():
    import tracing
    tracing._langfuse = None
    yield
    tracing._langfuse = None


@patch.dict(os.environ, {
    "LANGFUSE_PUBLIC_KEY": "pk-test",
    "LANGFUSE_SECRET_KEY": "sk-test",
}, clear=True)
@patch("tracing.Langfuse")
def test_get_langfuse_constructs_client_from_env(mock_langfuse_class):
    import tracing
    mock_langfuse_class.return_value = "the-client"

    result = tracing.get_langfuse()

    mock_langfuse_class.assert_called_once_with(
        public_key="pk-test",
        secret_key="sk-test",
        host="https://cloud.langfuse.com",
    )
    assert result == "the-client"


@patch.dict(os.environ, {
    "LANGFUSE_PUBLIC_KEY": "pk-test",
    "LANGFUSE_SECRET_KEY": "sk-test",
    "LANGFUSE_HOST": "https://self-hosted.example.com",
}, clear=True)
@patch("tracing.Langfuse")
def test_get_langfuse_respects_custom_host(mock_langfuse_class):
    import tracing
    tracing.get_langfuse()

    mock_langfuse_class.assert_called_once_with(
        public_key="pk-test",
        secret_key="sk-test",
        host="https://self-hosted.example.com",
    )


@patch.dict(os.environ, {
    "LANGFUSE_PUBLIC_KEY": "pk-test",
    "LANGFUSE_SECRET_KEY": "sk-test",
}, clear=True)
@patch("tracing.Langfuse")
def test_get_langfuse_is_a_singleton(mock_langfuse_class):
    import tracing
    mock_langfuse_class.return_value = "the-client"

    first = tracing.get_langfuse()
    second = tracing.get_langfuse()

    mock_langfuse_class.assert_called_once()
    assert first is second
