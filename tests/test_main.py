import builtins
from unittest.mock import MagicMock, patch
import pytest
from llm import ModelConfig


def default_config():
    return ModelConfig(provider="anthropic", model="claude-haiku-4-5-20251001")


def test_prompt_for_score_valid_input(monkeypatch):
    from main import prompt_for_score
    inputs = iter(["4", "worked well"])
    monkeypatch.setattr(builtins, "input", lambda: next(inputs))

    score = prompt_for_score()

    assert score.overall == 4
    assert score.note == "worked well"


def test_prompt_for_score_reprompts_on_invalid(monkeypatch, capsys):
    from main import prompt_for_score
    inputs = iter(["9", "not a number", "3", ""])
    monkeypatch.setattr(builtins, "input", lambda: next(inputs))

    score = prompt_for_score()

    assert score.overall == 3
    assert score.note is None
    assert "Please enter a number from 1 to 5." in capsys.readouterr().out


def test_prompt_for_score_ctrl_c_returns_none(monkeypatch):
    from main import prompt_for_score

    def raise_interrupt():
        raise KeyboardInterrupt()
    monkeypatch.setattr(builtins, "input", lambda: raise_interrupt())

    assert prompt_for_score() is None


@patch.dict("os.environ", {"ANTHROPIC_API_KEY": "fake"})
def test_handle_model_command_direct_args_no_history_no_prompt(monkeypatch):
    from main import handle_model_command
    mock_graph = MagicMock()
    mock_graph.get_state.return_value.values = {}

    config, thread_id = handle_model_command("anthropic claude-haiku-4-5-20251001", default_config(),
                                               mock_graph, "thread-1")

    assert config.provider == "anthropic"
    assert config.model == "claude-haiku-4-5-20251001"
    assert thread_id == "thread-1"  # no history -> no "start fresh" prompt, thread unchanged


@patch.dict("os.environ", {"ANTHROPIC_API_KEY": "fake"})
def test_handle_model_command_start_fresh_mints_new_thread_id(monkeypatch):
    from main import handle_model_command
    mock_graph = MagicMock()
    mock_graph.get_state.return_value.values = {"history": [{"role": "user", "content": "hi"}]}
    monkeypatch.setattr(builtins, "input", lambda: "y")

    config, thread_id = handle_model_command("anthropic claude-haiku-4-5-20251001", default_config(),
                                               mock_graph, "thread-1")

    assert thread_id != "thread-1"


@patch.dict("os.environ", {"ANTHROPIC_API_KEY": "fake"})
def test_handle_model_command_keep_history_same_thread(monkeypatch):
    from main import handle_model_command
    mock_graph = MagicMock()
    mock_graph.get_state.return_value.values = {"history": [{"role": "user", "content": "hi"}]}
    monkeypatch.setattr(builtins, "input", lambda: "n")

    config, thread_id = handle_model_command("anthropic claude-haiku-4-5-20251001", default_config(),
                                               mock_graph, "thread-1")

    assert thread_id == "thread-1"


def test_handle_model_command_missing_key_keeps_current(monkeypatch):
    from main import handle_model_command
    mock_graph = MagicMock()
    with patch.dict("os.environ", {}, clear=True):
        config, thread_id = handle_model_command("anthropic some-model", default_config(),
                                                   mock_graph, "thread-1")

    assert config == default_config()
    assert thread_id == "thread-1"
