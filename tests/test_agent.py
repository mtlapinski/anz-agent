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
    mock_search.assert_called_once_with(query="laptop", optimize_for="price", max_results=5, trace_id=None)


def test_run_tool_forwards_trace_id_to_search_amazon():
    from agent import run_tool
    with patch("agent.search_amazon") as mock_search, patch("agent._get_langfuse"):
        mock_search.return_value = {"products": []}
        run_tool("search_amazon", {"query": "laptop", "optimize_for": "price", "max_results": 5},
                  trace_id="trace-123")
    mock_search.assert_called_once_with(query="laptop", optimize_for="price", max_results=5, trace_id="trace-123")


def test_run_tool_unknown_raises():
    from agent import run_tool
    with pytest.raises(ValueError, match="Unknown tool"):
        run_tool("nonexistent_tool", {})


def test_strip_image_for_llm_removes_image_field():
    from agent import strip_image_for_llm
    products = [
        {"title": "Laptop", "price": 999.0, "image": "https://example.com/thumb1.jpg"},
        {"title": "Mouse", "price": 19.99, "image": "https://example.com/thumb2.jpg"},
    ]
    result = strip_image_for_llm(products)
    assert result == [
        {"title": "Laptop", "price": 999.0},
        {"title": "Mouse", "price": 19.99},
    ]


def test_strip_image_for_llm_handles_missing_image_field():
    from agent import strip_image_for_llm
    products = [{"title": "Laptop", "price": 999.0}]
    result = strip_image_for_llm(products)
    assert result == [{"title": "Laptop", "price": 999.0}]


def test_strip_image_for_llm_handles_empty_list():
    from agent import strip_image_for_llm
    assert strip_image_for_llm([]) == []


def test_strip_image_for_llm_does_not_mutate_input():
    from agent import strip_image_for_llm
    original = {"title": "Laptop", "image": "https://example.com/thumb1.jpg"}
    products = [original]
    strip_image_for_llm(products)
    assert original == {"title": "Laptop", "image": "https://example.com/thumb1.jpg"}


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


@patch("agent._get_langfuse")
def test_record_score_writes_judge_fields_to_jsonl(mock_lf, tmp_path, monkeypatch):
    from agent import record_score, EvalScore, JudgeScore
    monkeypatch.chdir(tmp_path)
    human = EvalScore(overall=4, note="good pick")
    judge = JudgeScore(relevance=5, fit=4, quality=5, overall=4.7, note="great match")
    context = {"query": "laptop", "optimize_for": "price", "recommendation": "Here are 3 laptops..."}

    record_score("trace-123", context, human, judge)

    row = json.loads((tmp_path / "evals" / "scores.jsonl").read_text().strip())
    assert row["judge_relevance"] == 5
    assert row["judge_fit"] == 4
    assert row["judge_quality"] == 5
    assert row["judge_overall"] == 4.7
    assert row["judge_note"] == "great match"


@patch("agent._get_langfuse")
def test_record_score_writes_none_judge_fields_when_judge_score_is_none(mock_lf, tmp_path, monkeypatch):
    from agent import record_score, EvalScore
    monkeypatch.chdir(tmp_path)
    human = EvalScore(overall=4, note="good pick")
    context = {"query": "laptop", "optimize_for": "price", "recommendation": "..."}

    record_score("trace-123", context, human, None)

    row = json.loads((tmp_path / "evals" / "scores.jsonl").read_text().strip())
    assert row["judge_relevance"] is None
    assert row["judge_overall"] is None
    assert row["judge_note"] is None


@patch("agent._get_langfuse")
def test_record_score_calls_langfuse_for_judge_scores(mock_lf, tmp_path, monkeypatch):
    from agent import record_score, EvalScore, JudgeScore
    monkeypatch.chdir(tmp_path)
    mock_client = MagicMock()
    mock_lf.return_value = mock_client
    human = EvalScore(overall=4, note=None)
    judge = JudgeScore(relevance=5, fit=4, quality=3, overall=4.0, note="solid")
    context = {"query": "q", "optimize_for": "price", "recommendation": "r"}

    record_score("trace-abc", context, human, judge)

    mock_client.create_score.assert_any_call(trace_id="trace-abc", name="judge_relevance", value=5, comment=None)
    mock_client.create_score.assert_any_call(trace_id="trace-abc", name="judge_fit", value=4, comment=None)
    mock_client.create_score.assert_any_call(trace_id="trace-abc", name="judge_quality", value=3, comment=None)
    mock_client.create_score.assert_any_call(trace_id="trace-abc", name="judge_overall", value=4.0, comment="solid")


@patch("agent._get_langfuse")
def test_record_score_writes_row_when_human_none_but_judge_present(mock_lf, tmp_path, monkeypatch):
    from agent import record_score, JudgeScore
    monkeypatch.chdir(tmp_path)
    judge = JudgeScore(relevance=5, fit=4, quality=5, overall=4.7, note="great")
    context = {"query": "q", "optimize_for": "price", "recommendation": "r"}

    record_score("trace-1", context, None, judge)

    assert (tmp_path / "evals" / "scores.jsonl").exists()


@patch("agent._get_langfuse")
def test_record_score_skips_entirely_when_both_scores_none(mock_lf, tmp_path, monkeypatch):
    from agent import record_score
    monkeypatch.chdir(tmp_path)
    context = {"query": "q", "optimize_for": "price", "recommendation": "r"}

    record_score("trace-1", context, None, None)

    assert not (tmp_path / "evals" / "scores.jsonl").exists()


@patch("agent.llm")
def test_judge_recommendation_success(mock_llm):
    from agent import judge_recommendation, JudgeScore
    mock_llm.complete.return_value = LLMResponse(
        text='{"relevance": 5, "fit": 4, "quality": 5, "note": "great match"}',
        tool_calls=None, input_tokens=50, output_tokens=20,
    )
    context = {"query": "laptop", "optimize_for": "price", "recommendation": "Here are 3 laptops..."}

    result = judge_recommendation(MagicMock(), default_config(), context)

    assert result == JudgeScore(relevance=5, fit=4, quality=5, overall=4.7, note="great match")


@patch("agent.llm")
def test_judge_recommendation_retries_once_on_malformed_json(mock_llm):
    from agent import judge_recommendation
    mock_llm.complete.side_effect = [
        LLMResponse(text="not json", tool_calls=None, input_tokens=10, output_tokens=5),
        LLMResponse(text='{"relevance": 3, "fit": 3, "quality": 3, "note": "ok"}',
                    tool_calls=None, input_tokens=10, output_tokens=5),
    ]
    context = {"query": "q", "optimize_for": "price", "recommendation": "r"}

    result = judge_recommendation(MagicMock(), default_config(), context)

    assert result.overall == 3.0
    assert mock_llm.complete.call_count == 2


@patch("agent.llm")
def test_judge_recommendation_returns_none_after_two_failures(mock_llm):
    from agent import judge_recommendation
    mock_llm.complete.side_effect = Exception("rate limited")
    context = {"query": "q", "optimize_for": "price", "recommendation": "r"}

    result = judge_recommendation(MagicMock(), default_config(), context)

    assert result is None
    assert mock_llm.complete.call_count == 2


@patch("agent.llm")
def test_judge_recommendation_strips_markdown_code_fence(mock_llm):
    from agent import judge_recommendation
    mock_llm.complete.return_value = LLMResponse(
        text='```json\n{"relevance": 4, "fit": 4, "quality": 4, "note": "solid"}\n```',
        tool_calls=None, input_tokens=10, output_tokens=5,
    )
    context = {"query": "q", "optimize_for": "price", "recommendation": "r"}

    result = judge_recommendation(MagicMock(), default_config(), context)

    assert result is not None
    assert result.overall == 4.0
    assert mock_llm.complete.call_count == 1


def test_search_amazon_tool_schema_has_view_param():
    from agent import TOOLS
    schema = TOOLS[0]["input_schema"]
    assert schema["properties"]["view"]["enum"] == ["cards", "table", "chart"]
    assert "view" not in schema["required"]
