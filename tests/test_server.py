import pytest
import time
import threading
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

import server
from llm import ModelConfig


@pytest.fixture(autouse=True)
def isolated_session_store(monkeypatch):
    """In-memory fake so these unit tests don't need a real Postgres."""
    fake_sessions: dict[str, ModelConfig] = {}
    monkeypatch.setattr("server.session_store.create_session", lambda tid, cfg: fake_sessions.__setitem__(tid, cfg))

    def _get(tid):
        if tid not in fake_sessions:
            from tools.session_store import SessionNotFound
            raise SessionNotFound(tid)
        return fake_sessions[tid]

    monkeypatch.setattr("server.session_store.get_session", _get)
    return fake_sessions


@pytest.fixture
def client():
    return TestClient(server.app)


@patch.dict("os.environ", {"SERPAPI_KEY": "fake", "GOOGLE_API_KEY": "fake"})
@patch("server.create_client")
def test_create_session_success(mock_create_client, client, isolated_session_store):
    mock_create_client.return_value = MagicMock()

    response = client.post("/session", json={"provider": "google", "model": "gemini-flash-lite-latest"})

    assert response.status_code == 200
    thread_id = response.json()["thread_id"]
    assert thread_id in isolated_session_store


@patch.dict("os.environ", {}, clear=True)
def test_create_session_missing_credentials(client):
    response = client.post("/session", json={"provider": "google", "model": "gemini-flash-lite-latest"})

    assert response.status_code == 400
    assert "SERPAPI_KEY" in response.json()["detail"]


def test_chat_unknown_thread_returns_404(client):
    response = client.post("/chat", json={"thread_id": "nope", "message": "hi"})

    assert response.status_code == 404


@patch("server.create_client")
@patch("server._graph")
def test_chat_message_response(mock_graph, mock_create_client, client, isolated_session_store):
    mock_create_client.return_value = MagicMock()
    isolated_session_store["t-1"] = ModelConfig(provider="google", model="m")
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


@patch("server.create_client")
@patch("server._graph")
def test_chat_message_omits_products_when_no_search_this_turn(mock_graph, mock_create_client, client, isolated_session_store):
    mock_create_client.return_value = MagicMock()
    isolated_session_store["t-1b"] = ModelConfig(provider="google", model="m")
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


@patch("server.create_client")
@patch("server._graph")
def test_chat_eval_interrupt_response(mock_graph, mock_create_client, client, isolated_session_store):
    mock_create_client.return_value = MagicMock()
    isolated_session_store["t-2"] = ModelConfig(provider="google", model="m")
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


@patch("server.create_client")
@patch("server._graph")
def test_chat_graph_exception_returns_error_payload(mock_graph, mock_create_client, client, isolated_session_store):
    mock_create_client.return_value = MagicMock()
    isolated_session_store["t-3"] = ModelConfig(provider="google", model="m")
    mock_graph.invoke.side_effect = RuntimeError("SerpAPI down")

    response = client.post("/chat", json={"thread_id": "t-3", "message": "find me a laptop"})

    assert response.status_code == 200
    assert response.json() == {"type": "error", "message": "SerpAPI down"}


def test_resume_unknown_thread_returns_404(client):
    response = client.post("/resume", json={"thread_id": "nope", "score": 5})

    assert response.status_code == 404


@patch("server.create_client")
@patch("server._graph")
def test_resume_posts_score_and_resumes_graph(mock_graph, mock_create_client, client, isolated_session_store):
    mock_create_client.return_value = MagicMock()
    isolated_session_store["t-4"] = ModelConfig(provider="google", model="m")
    mock_graph.invoke.return_value = {"response": "Here are the top laptops..."}

    response = client.post("/resume", json={"thread_id": "t-4", "score": 5, "note": "great"})

    assert response.status_code == 200
    assert response.json() == {"type": "message", "text": "Thanks for the rating!", "products": None, "view": None}
    args, kwargs = mock_graph.invoke.call_args
    resumed_command = args[0]
    assert resumed_command.resume.overall == 5
    assert resumed_command.resume.note == "great"


@patch("server.create_client")
@patch("server._graph")
def test_resume_graph_exception_returns_error_payload(mock_graph, mock_create_client, client, isolated_session_store):
    mock_create_client.return_value = MagicMock()
    isolated_session_store["t-5"] = ModelConfig(provider="google", model="m")
    mock_graph.invoke.side_effect = RuntimeError("boom")

    response = client.post("/resume", json={"thread_id": "t-5", "score": 3})

    assert response.json() == {"type": "error", "message": "boom"}


@patch("server._graph")
def test_chat_checkpointer_error_surfaces_as_explicit_error_not_silent_reset(mock_graph, client, isolated_session_store):
    isolated_session_store["t-checkpointer-down"] = ModelConfig(provider="google", model="m")
    mock_graph.invoke.side_effect = OSError("could not connect to Postgres")

    with patch("server.create_client", return_value=MagicMock()):
        response = client.post("/chat", json={"thread_id": "t-checkpointer-down", "message": "hi"})

    # Not a fabricated 200 success, and not a fake empty-history response —
    # an explicit, visible error the client can show to the user.
    assert response.status_code == 200
    assert response.json() == {"type": "error", "message": "could not connect to Postgres"}


def test_healthz_returns_replica_id(client, monkeypatch):
    monkeypatch.setenv("REPLICA_ID", "app-1")
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"replica": "app-1"}


def test_healthz_defaults_to_unknown_without_replica_id(client, monkeypatch):
    monkeypatch.delenv("REPLICA_ID", raising=False)
    response = client.get("/healthz")
    assert response.json() == {"replica": "unknown"}


@patch("server._graph")
def test_chat_requests_do_not_serialize_on_slow_graph_invoke(mock_graph, client, isolated_session_store):
    isolated_session_store["t-slow-1"] = ModelConfig(provider="google", model="m")
    isolated_session_store["t-slow-2"] = ModelConfig(provider="google", model="m")

    def slow_invoke(*args, **kwargs):
        time.sleep(0.3)
        return {"response": "done", "made_tool_call_this_turn": False, "last_search_results": None, "last_search_input": None}

    mock_graph.invoke.side_effect = slow_invoke

    results = []

    def _call(thread_id):
        with patch("server.create_client", return_value=MagicMock()):
            r = client.post("/chat", json={"thread_id": thread_id, "message": "hi"})
            results.append(r.status_code)

    start = time.monotonic()
    t1 = threading.Thread(target=_call, args=("t-slow-1",))
    t2 = threading.Thread(target=_call, args=("t-slow-2",))
    t1.start(); t2.start()
    t1.join(); t2.join()
    elapsed = time.monotonic() - start

    assert results == [200, 200]
    # Serialized would take ~0.6s; concurrent should take ~0.3s. Generous bound for CI jitter.
    assert elapsed < 0.5
