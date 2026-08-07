# Eval Report Dashboard — Design Spec

**Date:** 2026-08-06
**Status:** Approved

## Overview

Add a standalone script, `scripts/eval_report.py`, that turns `evals/scores.jsonl`
(populated by the human `interrupt()` score and the LLM judge, see
[2026-07-26-llm-judge-design.md](2026-07-26-llm-judge-design.md)) into a
self-contained HTML dashboard. Today that file only accumulates rows with no way
to see trends, spot agreement/disagreement between the human and judge scores, or
find the worst-scoring recommendations without reading raw JSON lines by hand.

Run as `python scripts/eval_report.py`. It reads the JSONL, computes summary
stats, writes `evals/report.html`, and opens it in the default browser. No new
dependencies, no server, no Claude session required to regenerate — matches the
project's existing CLI-first, offline-friendly style (`main.py`).

## Architecture

```
scripts/eval_report.py
  load_rows(path) -> list[dict]          # parse JSONL, skip malformed lines w/ stderr warning
  compute_stats(rows) -> ReportStats     # pure function, unit-testable
  render_html(rows, stats) -> str        # embeds rows/stats as JSON in a <script> tag,
                                          # inline CSS/JS, inline SVG line chart — no CDN
  main()                                  # load -> compute -> render -> write evals/report.html
                                          # -> webbrowser.open()
```

`load_rows` and `compute_stats` are separated from `render_html` specifically so
they can be unit tested without touching the filesystem beyond a `tmp_path`
fixture, and without any HTML/browser concerns.

## Data model

`ReportStats` (a `dataclass`, mirroring the `EvalScore`/`JudgeScore` pattern in
`agent.py`):

```python
@dataclass
class ReportStats:
    total: int
    human_avg: float | None       # None if no rows have a human score
    judge_avg: float | None       # None if no rows have a judge score
    agreement_avg: float | None   # mean of |human_overall - judge_overall| over rows with both
    missing_human: int            # rows with human overall == None
    missing_judge: int            # rows with judge_overall == None
```

Both human and judge scores share the same 1–5 direction (1 = poor, 5 =
excellent), confirmed in the existing judge design and the human prompt text in
`main.py`, so no scale inversion is needed anywhere in the report.

## Report contents

- **Header stats bar**: total rated turns, avg human score, avg judge score, avg
  agreement gap, count missing human / missing judge.
- **Full table**, newest first, sortable by clicking a column header: timestamp,
  query, optimize_for, human overall, judge relevance/fit/quality/overall, human
  note, judge note. Missing values render as `—` and are excluded from that
  column's average.
- **Lowest-scoring section**: the 10 rows with the lowest `min(human_overall,
  judge_overall)` (falling back to whichever score exists if only one is
  present), so bad recommendations surface without scrolling the full table.
- **Scores-over-time chart**: a simple inline SVG line chart plotting human
  overall and judge overall per row in timestamp order — no charting library,
  consistent with the Artifact-style "self-contained, no CDN" constraint even
  though this isn't published as an Artifact.

## Error handling

| Scenario | Behavior |
|---|---|
| `evals/scores.jsonl` doesn't exist | Print a clear message ("no eval data yet — rate a recommendation first") and exit 0, no crash |
| A line is malformed JSON | Skip it, print a warning to stderr with the line number, continue processing the rest |
| A row has `null` human or judge `overall` | Included in the table as `—`; excluded only from that column's average, not from `total` |
| `evals/` dir is otherwise empty/missing | Same as "doesn't exist" case above |

## Testing

New `tests/test_eval_report.py`, following the existing `tests/test_*.py`
pattern (plain `pytest`, no mocking needed since this is pure data logic):

- `load_rows`: empty file, one malformed line among valid ones (warns, skips,
  keeps the rest), all-valid file.
- `compute_stats`: empty list → all `None`/zero; all-human-no-judge rows;
  all-judge-no-human rows; mixed rows → correct averages and agreement gap.
- No tests for `render_html`'s exact HTML output or for `main()`'s
  browser-opening — those are thin glue, verified manually by running the
  script once against real `evals/scores.jsonl` data.

## Out of scope (backlog)

- Pulling data from Langfuse instead of / in addition to `evals/scores.jsonl`
  (the two already contain the same judge scores per row; JSONL is the simpler
  single source for this report).
- Automatic regeneration (e.g. on every new eval row, or via a file watcher) —
  this is a manually-run script for now, per user preference in brainstorming.
- Using judge scores to change agent behavior at runtime (e.g. auto-retry a
  low-scoring recommendation) — flagged for later, see
  [[project-anz-agent-backlog]].
