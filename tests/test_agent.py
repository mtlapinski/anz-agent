import json
import os
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
def test_run_tool_creates_langfuse_span_with_trace_id(mock_lf):
    from agent import run_tool
    mock_client = MagicMock(spec=["create_trace_id", "start_observation", "create_score", "flush"])
    mock_span = MagicMock()
    mock_client.start_observation.return_value = mock_span
    mock_lf.return_value = mock_client

    with patch("agent.search_amazon") as mock_search:
        mock_search.return_value = {"products": []}
        run_tool("search_amazon", {"query": "laptop", "optimize_for": "price", "max_results": 5},
                  trace_id="trace-123")

    mock_client.start_observation.assert_called_once_with(
        trace_context={"trace_id": "trace-123"},
        name="search_amazon",
        as_type="tool",
        input={"query": "laptop", "optimize_for": "price", "max_results": 5},
    )
    mock_span.update.assert_called_once_with(output={"products": []})
    mock_span.end.assert_called_once()


@patch("agent._get_langfuse")
def test_run_tool_no_span_when_trace_id_none(mock_lf):
    from agent import run_tool
    mock_client = MagicMock(spec=["create_trace_id", "start_observation", "create_score", "flush"])
    mock_lf.return_value = mock_client

    with patch("agent.search_amazon") as mock_search:
        mock_search.return_value = {"products": []}
        run_tool("search_amazon", {"query": "laptop", "optimize_for": "price", "max_results": 5})

    mock_client.start_observation.assert_not_called()


@patch("agent._get_langfuse")
def test_run_tool_continues_when_langfuse_raises(mock_lf):
    from agent import run_tool
    mock_lf.side_effect = Exception("langfuse down")

    with patch("agent.search_amazon") as mock_search:
        mock_search.return_value = {"products": []}
        result = run_tool("search_amazon", {"query": "laptop", "optimize_for": "price", "max_results": 5},
                            trace_id="trace-123")

    assert json.loads(result) == {"products": []}


def test_eval_score_defaults():
    from agent import EvalScore
    score = EvalScore(overall=4, note="good pick")
    assert score.overall == 4
    assert score.note == "good pick"
    assert score.criteria is None


@patch("agent._get_langfuse")
def test_record_score_writes_jsonl(mock_lf, tmp_path, monkeypatch):
    from agent import record_score, EvalScore
    monkeypatch.chdir(tmp_path)
    score = EvalScore(overall=4, note="good pick")
    context = {"query": "laptop", "optimize_for": "price", "recommendation": "Here are 3 laptops..."}

    record_score("trace-123", context, score)

    jsonl_path = tmp_path / "evals" / "scores.jsonl"
    assert jsonl_path.exists()
    lines = jsonl_path.read_text().strip().split("\n")
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert row["query"] == "laptop"
    assert row["optimize_for"] == "price"
    assert row["recommendation"] == "Here are 3 laptops..."
    assert row["overall"] == 4
    assert row["note"] == "good pick"
    assert "timestamp" in row


@patch("agent._get_langfuse")
def test_record_score_calls_langfuse_create_score(mock_lf, tmp_path, monkeypatch):
    from agent import record_score, EvalScore
    monkeypatch.chdir(tmp_path)
    mock_client = MagicMock()
    mock_lf.return_value = mock_client
    score = EvalScore(overall=5, note=None)
    context = {"query": "mouse", "optimize_for": "quality", "recommendation": "The X mouse..."}

    record_score("trace-abc", context, score)

    mock_client.create_score.assert_called_once_with(
        trace_id="trace-abc", name="usefulness", value=5, comment=None
    )


@patch("agent._get_langfuse")
def test_record_score_skips_when_score_is_none(mock_lf, tmp_path, monkeypatch):
    from agent import record_score
    monkeypatch.chdir(tmp_path)
    mock_client = MagicMock()
    mock_lf.return_value = mock_client
    context = {"query": "mouse", "optimize_for": "quality", "recommendation": "The X mouse..."}

    record_score("trace-abc", context, None)

    mock_client.create_score.assert_not_called()
    assert not (tmp_path / "evals" / "scores.jsonl").exists()


@patch("agent._get_langfuse")
def test_record_score_skips_langfuse_when_trace_id_none(mock_lf, tmp_path, monkeypatch):
    from agent import record_score, EvalScore
    monkeypatch.chdir(tmp_path)
    mock_client = MagicMock()
    mock_lf.return_value = mock_client
    score = EvalScore(overall=3, note=None)

    record_score(None, {"query": "q", "optimize_for": "price", "recommendation": "r"}, score)

    mock_client.create_score.assert_not_called()
    assert (tmp_path / "evals" / "scores.jsonl").exists()


@patch("agent._get_langfuse")
def test_record_score_continues_when_langfuse_raises(mock_lf, tmp_path, monkeypatch):
    from agent import record_score, EvalScore
    monkeypatch.chdir(tmp_path)
    mock_lf.side_effect = Exception("langfuse down")
    score = EvalScore(overall=2, note=None)

    record_score("trace-x", {"query": "q", "optimize_for": "price", "recommendation": "r"}, score)

    assert (tmp_path / "evals" / "scores.jsonl").exists()


@patch("agent._get_langfuse")
def test_record_score_continues_when_jsonl_write_fails(mock_lf, tmp_path, monkeypatch):
    from agent import record_score, EvalScore
    monkeypatch.chdir(tmp_path)
    (tmp_path / "evals").mkdir()
    (tmp_path / "evals" / "scores.jsonl").mkdir()  # a directory where the file should go -> open() fails
    score = EvalScore(overall=2, note=None)

    record_score(None, {"query": "q", "optimize_for": "price", "recommendation": "r"}, score)  # should not raise
