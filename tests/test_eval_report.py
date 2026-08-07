from scripts.eval_report import load_rows, ReportStats, compute_stats


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


def test_compute_stats_empty():
    stats = compute_stats([])
    assert stats == ReportStats(
        total=0,
        human_avg=None,
        judge_avg=None,
        agreement_avg=None,
        missing_human=0,
        missing_judge=0,
    )


def test_compute_stats_all_human_no_judge():
    rows = [
        {"overall": 4, "judge_overall": None},
        {"overall": 2, "judge_overall": None},
    ]
    stats = compute_stats(rows)
    assert stats.total == 2
    assert stats.human_avg == 3.0
    assert stats.judge_avg is None
    assert stats.agreement_avg is None
    assert stats.missing_human == 0
    assert stats.missing_judge == 2


def test_compute_stats_all_judge_no_human():
    rows = [
        {"overall": None, "judge_overall": 3.5},
        {"overall": None, "judge_overall": 4.5},
    ]
    stats = compute_stats(rows)
    assert stats.human_avg is None
    assert stats.judge_avg == 4.0
    assert stats.agreement_avg is None
    assert stats.missing_human == 2
    assert stats.missing_judge == 0


def test_compute_stats_mixed_rows_and_agreement():
    rows = [
        {"overall": 5, "judge_overall": 3.0},
        {"overall": 2, "judge_overall": 2.0},
        {"overall": None, "judge_overall": None},
    ]
    stats = compute_stats(rows)
    assert stats.total == 3
    assert stats.human_avg == 3.5
    assert stats.judge_avg == 2.5
    assert stats.agreement_avg == 1.0  # mean of |5-3|=2 and |2-2|=0
    assert stats.missing_human == 1
    assert stats.missing_judge == 1
