import json
from unittest.mock import MagicMock, patch
from llm import ModelConfig, LLMResponse


def default_config():
    return ModelConfig(provider="anthropic", model="claude-haiku-4-5-20251001")


def empty_state(**overrides):
    base = {
        "history": [],
        "new_message": None,
        "made_tool_call_this_turn": False,
        "pending_tool_calls": None,
        "last_search_input": None,
        "last_search_results": None,
        "trace_id": None,
        "response": None,
    }
    base.update(overrides)
    return base


class FakeRuntime:
    def __init__(self, context):
        self.context = context


@patch("graph.agent._get_langfuse")
@patch("graph.llm")
def test_agent_node_text_response_new_turn(mock_llm, mock_lf):
    from graph import agent_node, GraphContext
    mock_llm.complete.return_value = LLMResponse(
        text="What are you looking for?", tool_calls=None, input_tokens=10, output_tokens=5
    )
    mock_lf.return_value.create_trace_id.return_value = "trace-1"
    runtime = FakeRuntime(GraphContext(client=MagicMock(), model_config=default_config()))

    result = agent_node(empty_state(new_message="Hello"), runtime)

    assert result["response"] == "What are you looking for?"
    assert result["new_message"] is None
    assert result["made_tool_call_this_turn"] is False
    assert result["pending_tool_calls"] is None
    assert result["trace_id"] == "trace-1"
    assert result["history"][0] == {"role": "user", "content": "Hello"}
    assert result["history"][1]["role"] == "assistant"
    assert result["history"][1]["content"] == [{"type": "text", "text": "What are you looking for?"}]


@patch("graph.agent._get_langfuse")
@patch("graph.llm")
def test_agent_node_tool_call_response(mock_llm, mock_lf):
    from graph import agent_node, GraphContext
    mock_llm.complete.return_value = LLMResponse(
        text=None,
        tool_calls=[{"name": "search_amazon", "id": "tu_1",
                      "input": {"query": "laptop", "optimize_for": "price", "max_results": 5}}],
        input_tokens=80, output_tokens=20,
    )
    mock_lf.return_value.create_trace_id.return_value = "trace-2"
    runtime = FakeRuntime(GraphContext(client=MagicMock(), model_config=default_config()))

    result = agent_node(empty_state(new_message="Find me a laptop"), runtime)

    assert result["made_tool_call_this_turn"] is True
    assert result["pending_tool_calls"] == [{"name": "search_amazon", "id": "tu_1",
        "input": {"query": "laptop", "optimize_for": "price", "max_results": 5}}]
    assert result["last_search_input"] == {"query": "laptop", "optimize_for": "price", "max_results": 5}
    assert result["history"][-1]["content"][0]["type"] == "tool_use"
    assert "response" not in result


@patch("graph.agent._get_langfuse")
@patch("graph.llm")
def test_agent_node_continues_turn_without_new_message(mock_llm, mock_lf):
    from graph import agent_node, GraphContext
    mock_llm.complete.return_value = LLMResponse(
        text="Here are the top laptops...", tool_calls=None, input_tokens=200, output_tokens=50
    )
    runtime = FakeRuntime(GraphContext(client=MagicMock(), model_config=default_config()))
    prior_history = [
        {"role": "user", "content": "Find me a laptop"},
        {"role": "assistant", "content": [{"type": "tool_use", "id": "tu_1", "name": "search_amazon",
                                            "input": {"query": "laptop", "optimize_for": "price", "max_results": 5}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "tu_1", "content": "{}"}]},
    ]

    result = agent_node(
        empty_state(history=prior_history, new_message=None, made_tool_call_this_turn=True, trace_id="trace-2"),
        runtime,
    )

    assert result["response"] == "Here are the top laptops..."
    assert result["made_tool_call_this_turn"] is True  # preserved — this text response follows a tool call earlier in the same turn, must route to eval
    assert result["pending_tool_calls"] is None
    assert result["trace_id"] == "trace-2"  # reused, not regenerated
    mock_lf.return_value.create_trace_id.assert_not_called()


@patch("graph.agent._get_langfuse")
@patch("graph.llm")
def test_agent_node_resets_stale_flags_on_new_turn(mock_llm, mock_lf):
    from graph import agent_node, GraphContext
    mock_llm.complete.return_value = LLMResponse(
        text="Sure, anything else?", tool_calls=None, input_tokens=10, output_tokens=5
    )
    mock_lf.return_value.create_trace_id.return_value = "trace-3"
    runtime = FakeRuntime(GraphContext(client=MagicMock(), model_config=default_config()))

    # Simulate state left over from a PRIOR turn that made a search and produced a recommendation.
    stale_state = empty_state(
        new_message="thanks, what about accessories?",
        made_tool_call_this_turn=True,
        last_search_input={"query": "laptop", "optimize_for": "price", "max_results": 5},
        trace_id="trace-2",
    )

    result = agent_node(stale_state, runtime)

    # This turn made no tool call — both flags must reset, not carry over from the prior turn.
    assert result["made_tool_call_this_turn"] is False
    assert result["last_search_input"] is None


@patch("graph.agent._get_langfuse")
@patch("graph.llm")
def test_agent_node_logs_token_usage(mock_llm, mock_lf, capsys):
    from graph import agent_node, GraphContext
    mock_llm.complete.return_value = LLMResponse(
        text="hi", tool_calls=None, input_tokens=100, output_tokens=50
    )
    runtime = FakeRuntime(GraphContext(client=MagicMock(), model_config=default_config()))

    agent_node(empty_state(new_message="hi"), runtime)

    captured = capsys.readouterr()
    assert "[tokens: 100 in / 50 out]" in captured.out


@patch("graph.agent.run_tool")
def test_tools_node_executes_and_appends_results(mock_run_tool):
    from graph import tools_node
    mock_run_tool.return_value = json.dumps({"products": []})
    state = empty_state(
        pending_tool_calls=[{"name": "search_amazon", "id": "tu_1",
                               "input": {"query": "laptop", "optimize_for": "price", "max_results": 5}}],
        trace_id="trace-1",
    )

    result = tools_node(state)

    mock_run_tool.assert_called_once_with(
        "search_amazon", {"query": "laptop", "optimize_for": "price", "max_results": 5}, trace_id="trace-1"
    )
    assert result["pending_tool_calls"] is None
    tool_result_msg = result["history"][0]
    assert tool_result_msg["role"] == "user"
    assert tool_result_msg["content"][0] == {
        "type": "tool_result", "tool_use_id": "tu_1", "content": json.dumps({"products": []})
    }


@patch("graph.agent.run_tool")
def test_tools_node_captures_last_search_results(mock_run_tool):
    from graph import tools_node
    mock_run_tool.return_value = json.dumps({"products": [{"title": "Laptop"}]})
    state = empty_state(
        pending_tool_calls=[{"name": "search_amazon", "id": "tu_1",
                               "input": {"query": "laptop", "optimize_for": "price", "max_results": 5}}],
        trace_id="trace-1",
    )

    result = tools_node(state)

    assert result["last_search_results"] == {"products": [{"title": "Laptop"}]}


@patch("graph.agent.run_tool")
def test_tools_node_missing_data_when_no_search_call(mock_run_tool):
    from graph import tools_node
    mock_run_tool.return_value = json.dumps({"result": "some_other_tool_result"})
    state = empty_state(
        pending_tool_calls=[{"name": "some_other_tool", "id": "tu_1",
                               "input": {"param": "value"}}],
        trace_id="trace-1",
    )

    result = tools_node(state)

    assert result["last_search_results"] is None


def test_route_after_agent_to_tools_when_pending():
    from graph import route_after_agent
    state = empty_state(pending_tool_calls=[{"name": "search_amazon", "id": "tu_1", "input": {}}])
    assert route_after_agent(state) == "tools"


def test_route_after_agent_to_eval_when_recommendation_made():
    from graph import route_after_agent
    state = empty_state(pending_tool_calls=None, made_tool_call_this_turn=True, response="Here are...")
    assert route_after_agent(state) == "eval"


def test_route_after_agent_to_end_when_no_tool_call():
    from langgraph.graph import END
    from graph import route_after_agent
    state = empty_state(pending_tool_calls=None, made_tool_call_this_turn=False, response="What are you looking for?")
    assert route_after_agent(state) == END


@patch("graph.agent.record_score")
def test_eval_node_interrupts_then_records_score(mock_record_score):
    from graph import eval_node
    from unittest.mock import patch as mock_patch

    state = empty_state(
        last_search_input={"query": "laptop", "optimize_for": "price", "max_results": 5},
        response="Here are the top laptops...",
        trace_id="trace-1",
    )

    with mock_patch("graph.interrupt", return_value="fake-score") as mock_interrupt:
        result = eval_node(state)

    mock_interrupt.assert_called_once_with({
        "query": "laptop", "optimize_for": "price", "recommendation": "Here are the top laptops...",
    })
    mock_record_score.assert_called_once_with(
        "trace-1",
        {"query": "laptop", "optimize_for": "price", "recommendation": "Here are the top laptops..."},
        "fake-score",
    )
    assert result == {}


@patch("graph.agent._get_langfuse")
@patch("graph.llm")
def test_build_graph_full_flow_with_recommendation(mock_llm, mock_lf):
    from graph import build_graph, GraphContext
    from langgraph.types import Command

    mock_llm.complete.side_effect = [
        LLMResponse(text=None, tool_calls=[{"name": "search_amazon", "id": "tu_1",
            "input": {"query": "laptop", "optimize_for": "price", "max_results": 5}}],
            input_tokens=80, output_tokens=20),
        LLMResponse(text="Here are the top laptops...", tool_calls=None, input_tokens=200, output_tokens=50),
    ]
    mock_lf.return_value.create_trace_id.return_value = "trace-1"

    graph = build_graph()
    config = {"configurable": {"thread_id": "t-full-flow"}}
    context = GraphContext(client=MagicMock(), model_config=default_config())

    with patch("graph.agent.run_tool") as mock_run_tool, patch("graph.agent.record_score") as mock_record_score:
        mock_run_tool.return_value = json.dumps({"products": []})
        result = graph.invoke({"new_message": "Find me a laptop"}, config=config, context=context)

        assert "__interrupt__" in result
        interrupt_ctx = result["__interrupt__"][0].value
        assert interrupt_ctx["query"] == "laptop"
        assert interrupt_ctx["recommendation"] == "Here are the top laptops..."

        from agent import EvalScore
        score = EvalScore(overall=5, note="great")
        final = graph.invoke(Command(resume=score), config=config, context=context)

    assert final["response"] == "Here are the top laptops..."
    mock_record_score.assert_called_once()
    assert mock_record_score.call_args.args[2].overall == 5


@patch("graph.agent._get_langfuse")
@patch("graph.llm")
def test_build_graph_skips_eval_when_no_tool_call(mock_llm, mock_lf):
    from graph import build_graph, GraphContext

    mock_llm.complete.return_value = LLMResponse(
        text="What are you looking for?", tool_calls=None, input_tokens=10, output_tokens=5
    )
    mock_lf.return_value.create_trace_id.return_value = "trace-2"

    graph = build_graph()
    config = {"configurable": {"thread_id": "t-no-tool-call"}}
    context = GraphContext(client=MagicMock(), model_config=default_config())

    result = graph.invoke({"new_message": "Find me something"}, config=config, context=context)

    assert "__interrupt__" not in result
    assert result["response"] == "What are you looking for?"
