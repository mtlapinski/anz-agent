from scripts.eval_report import load_rows


def test_load_rows_missing_file(tmp_path):
    path = tmp_path / "missing.jsonl"
    assert load_rows(path) == []


def test_load_rows_valid_file(tmp_path):
    path = tmp_path / "scores.jsonl"
    path.write_text('{"a": 1}\n{"a": 2}\n')
    assert load_rows(path) == [{"a": 1}, {"a": 2}]


def test_load_rows_skips_malformed_line(tmp_path, capsys):
    path = tmp_path / "scores.jsonl"
    path.write_text('{"a": 1}\nnot json\n{"a": 2}\n')
    rows = load_rows(path)
    assert rows == [{"a": 1}, {"a": 2}]
    captured = capsys.readouterr()
    assert "line 2" in captured.err
