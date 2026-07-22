import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

import server
from llm import ModelConfig
from graph import GraphContext


@pytest.fixture(autouse=True)
def reset_sessions():
    server._sessions.clear()
    yield
    server._sessions.clear()


@pytest.fixture
def client():
    return TestClient(server.app)


@patch.dict("os.environ", {"SERPAPI_KEY": "fake", "GOOGLE_API_KEY": "fake"})
@patch("server.create_client")
def test_create_session_success(mock_create_client, client):
    mock_create_client.return_value = MagicMock()

    response = client.post("/session", json={"provider": "google", "model": "gemini-flash-lite-latest"})

    assert response.status_code == 200
    thread_id = response.json()["thread_id"]
    assert thread_id in server._sessions


@patch.dict("os.environ", {}, clear=True)
def test_create_session_missing_credentials(client):
    response = client.post("/session", json={"provider": "google", "model": "gemini-flash-lite-latest"})

    assert response.status_code == 400
    assert "SERPAPI_KEY" in response.json()["detail"]


def test_chat_unknown_thread_returns_404(client):
    response = client.post("/chat", json={"thread_id": "nope", "message": "hi"})

    assert response.status_code == 404


@patch("server._graph")
def test_chat_message_response(mock_graph, client):
    server._sessions["t-1"] = GraphContext(client=MagicMock(), model_config=ModelConfig(provider="google", model="m"))
    mock_graph.invoke.return_value = {
        "response": "Here are some laptops",
        "made_tool_call_this_turn": True,
        "last_search_results": {"products": [{"title": "Laptop"}]},
        "last_search_input": {"query": "laptop", "optimize_for": "price", "max_results": 5, "view": "cards"},
    }

    response = client.post("/chat", json={"thread_id": "t-1", "message": "find me a laptop"})

    assert response.status_code == 200
    assert response.json() == {
        "type": "message",
        "text": "Here are some laptops",
        "products": [{"title": "Laptop"}],
        "view": "cards",
    }


@patch("server._graph")
def test_chat_message_omits_products_when_no_search_this_turn(mock_graph, client):
    server._sessions["t-1b"] = GraphContext(client=MagicMock(), model_config=ModelConfig(provider="google", model="m"))
    mock_graph.invoke.return_value = {
        "response": "What are you looking for?",
        "made_tool_call_this_turn": False,
        "last_search_results": None,
        "last_search_input": None,
    }

    response = client.post("/chat", json={"thread_id": "t-1b", "message": "hi"})

    assert response.json() == {
        "type": "message",
        "text": "What are you looking for?",
        "products": None,
        "view": None,
    }


@patch("server._graph")
def test_chat_eval_interrupt_response(mock_graph, client):
    server._sessions["t-2"] = GraphContext(client=MagicMock(), model_config=ModelConfig(provider="google", model="m"))
    interrupt = MagicMock()
    interrupt.value = {"query": "laptop", "optimize_for": "price", "recommendation": "Here are the top laptops..."}
    mock_graph.invoke.return_value = {
        "__interrupt__": [interrupt],
        "made_tool_call_this_turn": True,
        "last_search_results": {"products": [{"title": "Laptop"}]},
        "last_search_input": {"query": "laptop", "optimize_for": "price", "max_results": 5, "view": "table"},
    }

    response = client.post("/chat", json={"thread_id": "t-2", "message": "find me a laptop"})

    assert response.json() == {
        "type": "eval_request",
        "query": "laptop",
        "optimize_for": "price",
        "recommendation": "Here are the top laptops...",
        "products": [{"title": "Laptop"}],
        "view": "table",
    }


@patch("server._graph")
def test_chat_graph_exception_returns_error_payload(mock_graph, client):
    server._sessions["t-3"] = GraphContext(client=MagicMock(), model_config=ModelConfig(provider="google", model="m"))
    mock_graph.invoke.side_effect = RuntimeError("SerpAPI down")

    response = client.post("/chat", json={"thread_id": "t-3", "message": "find me a laptop"})

    assert response.status_code == 200
    assert response.json() == {"type": "error", "message": "SerpAPI down"}


def test_resume_unknown_thread_returns_404(client):
    response = client.post("/resume", json={"thread_id": "nope", "score": 5})

    assert response.status_code == 404


@patch("server._graph")
def test_resume_posts_score_and_resumes_graph(mock_graph, client):
    server._sessions["t-4"] = GraphContext(client=MagicMock(), model_config=ModelConfig(provider="google", model="m"))
    mock_graph.invoke.return_value = {"response": "Here are the top laptops..."}

    response = client.post("/resume", json={"thread_id": "t-4", "score": 5, "note": "great"})

    assert response.status_code == 200
    assert response.json() == {"type": "message", "text": "Thanks for the rating!", "products": None, "view": None}
    args, kwargs = mock_graph.invoke.call_args
    resumed_command = args[0]
    assert resumed_command.resume.overall == 5
    assert resumed_command.resume.note == "great"


@patch("server._graph")
def test_resume_graph_exception_returns_error_payload(mock_graph, client):
    server._sessions["t-5"] = GraphContext(client=MagicMock(), model_config=ModelConfig(provider="google", model="m"))
    mock_graph.invoke.side_effect = RuntimeError("boom")

    response = client.post("/resume", json={"thread_id": "t-5", "score": 3})

    assert response.json() == {"type": "error", "message": "boom"}
