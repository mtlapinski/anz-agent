# Eval Report Dashboard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `scripts/eval_report.py`, a standalone script that reads `evals/scores.jsonl` and generates a self-contained HTML dashboard (`evals/report.html`) showing human vs. judge score trends, opened automatically in the browser.

**Architecture:** Pure data functions (`load_rows`, `compute_stats`) separated from HTML rendering (`render_html`) and CLI wiring (`main`), so the data logic is unit-testable without touching the filesystem beyond `tmp_path`, and without any browser/HTML concerns. See [2026-08-06-eval-report-design.md](../specs/2026-08-06-eval-report-design.md) for full design rationale.

**Tech Stack:** Python stdlib only (`json`, `dataclasses`, `pathlib`, `webbrowser`) — no new dependencies. Vanilla inline JS/CSS/SVG in the generated HTML — no CDN, no external assets.

## Global Constraints

- No new entries in `requirements.txt` — stdlib only.
- Human and judge `overall` scores share the same 1–5 direction: 1 = poor, 5 = excellent. Never invert this anywhere in stats or display.
- `scripts/eval_report.py` must be runnable as `python scripts/eval_report.py` from the repo root, matching how `main.py` and `server.py` are already run.
- Generated HTML must be fully self-contained (inline `<style>`/`<script>`, no external `<link>`/`<script src>`), so it opens correctly offline.
- Follow the existing test style: plain `pytest` functions, no test classes, `tmp_path`/`capsys` fixtures where needed (see `tests/test_cache.py`, `tests/test_agent.py`).

---

### Task 1: `load_rows()` — parse `evals/scores.jsonl` safely

**Files:**
- Create: `scripts/__init__.py` (empty, makes `scripts` a package — mirrors `tools/__init__.py`)
- Create: `scripts/eval_report.py`
- Test: `tests/test_eval_report.py`

**Interfaces:**
- Produces: `load_rows(path: Path) -> list[dict]` — reads a JSONL file, returns one dict per valid line in file order. Missing file returns `[]`. Malformed lines are skipped with a warning to stderr (`Warning: skipping malformed JSON on line N of <path>`) and do not raise.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_eval_report.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_eval_report.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.eval_report'` (or similar import error), since neither file exists yet.

- [ ] **Step 3: Create the package marker and implement `load_rows`**

Create `scripts/__init__.py` (empty file).

Create `scripts/eval_report.py`:

```python
from __future__ import annotations

import json
import sys
from pathlib import Path


def load_rows(path: Path) -> list[dict]:
    """Read a JSONL file of eval score rows.

    Returns an empty list if the file doesn't exist. Skips and warns (to
    stderr) on any line that isn't valid JSON, rather than aborting.
    """
    if not path.exists():
        return []
    rows = []
    with path.open() as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                print(
                    f"Warning: skipping malformed JSON on line {lineno} of {path}",
                    file=sys.stderr,
                )
    return rows
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_eval_report.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add scripts/__init__.py scripts/eval_report.py tests/test_eval_report.py
git commit -m "feat: add load_rows() for eval report script"
```

---

### Task 2: `compute_stats()` — summary statistics over rows

**Files:**
- Modify: `scripts/eval_report.py`
- Test: `tests/test_eval_report.py`

**Interfaces:**
- Consumes: nothing from Task 1 directly (operates on plain `list[dict]`, same shape `load_rows` returns).
- Produces:
  ```python
  @dataclass
  class ReportStats:
      total: int
      human_avg: float | None
      judge_avg: float | None
      agreement_avg: float | None
      missing_human: int
      missing_judge: int

  def compute_stats(rows: list[dict]) -> ReportStats: ...
  ```
  Reads `row["overall"]` (human score) and `row["judge_overall"]` (judge score) — the exact field names already written by `agent.record_score()` in `evals/scores.jsonl`. `agreement_avg` is the mean of `abs(overall - judge_overall)` over rows where both are present (not `None`); `None` if no row has both. Averages are rounded to 2 decimals; `None` when the underlying list is empty.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_eval_report.py`:

```python
from scripts.eval_report import ReportStats, compute_stats


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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_eval_report.py -v`
Expected: FAIL — `ImportError: cannot import name 'ReportStats'` (new tests only; Task 1's 3 tests still pass).

- [ ] **Step 3: Implement `ReportStats` and `compute_stats`**

Add to `scripts/eval_report.py` (below `load_rows`):

```python
from dataclasses import dataclass


@dataclass
class ReportStats:
    total: int
    human_avg: float | None
    judge_avg: float | None
    agreement_avg: float | None
    missing_human: int
    missing_judge: int


def compute_stats(rows: list[dict]) -> ReportStats:
    total = len(rows)
    human_scores = [r["overall"] for r in rows if r.get("overall") is not None]
    judge_scores = [
        r["judge_overall"] for r in rows if r.get("judge_overall") is not None
    ]
    agreement_gaps = [
        abs(r["overall"] - r["judge_overall"])
        for r in rows
        if r.get("overall") is not None and r.get("judge_overall") is not None
    ]

    def avg(values: list[float]) -> float | None:
        return round(sum(values) / len(values), 2) if values else None

    return ReportStats(
        total=total,
        human_avg=avg(human_scores),
        judge_avg=avg(judge_scores),
        agreement_avg=avg(agreement_gaps),
        missing_human=total - len(human_scores),
        missing_judge=total - len(judge_scores),
    )
```

Move the `from dataclasses import dataclass` import to the top of the file alongside the existing `import json` / `import sys` / `from pathlib import Path` imports rather than inline.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_eval_report.py -v`
Expected: PASS (7 passed)

- [ ] **Step 5: Commit**

```bash
git add scripts/eval_report.py tests/test_eval_report.py
git commit -m "feat: add compute_stats() for eval report script"
```

---

### Task 3: `render_html()` + `main()` — dashboard generation and CLI entry point

**Files:**
- Modify: `scripts/eval_report.py`
- Modify: `README.md` (document the new script)

**Interfaces:**
- Consumes: `load_rows(path: Path) -> list[dict]` and `compute_stats(rows: list[dict]) -> ReportStats` from Tasks 1–2; row dicts use the field names `timestamp`, `query`, `optimize_for`, `overall`, `note`, `judge_relevance`, `judge_fit`, `judge_quality`, `judge_overall`, `judge_note` (per `agent.record_score()`).
- Produces: `render_html(rows: list[dict], stats: ReportStats) -> str` and `main() -> None`. No automated tests for these per the design spec (thin glue over already-tested logic + a browser open side effect) — verified manually in Step 3 below.

- [ ] **Step 1: Implement `render_html`**

Add to `scripts/eval_report.py` (below `compute_stats`; add `import dataclasses` and `import webbrowser` to the top-of-file imports alongside the existing ones):

```python
def render_html(rows: list[dict], stats: ReportStats) -> str:
    data_json = json.dumps(rows)
    stats_json = json.dumps(dataclasses.asdict(stats))
    return f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Eval Report</title>
<style>
  :root {{ color-scheme: light dark; }}
  body {{ font-family: system-ui, sans-serif; margin: 2rem; background: #fff; color: #111; }}
  @media (prefers-color-scheme: dark) {{
    body {{ background: #1a1a1a; color: #eee; }}
    th, td {{ border-color: #444 !important; }}
  }}
  h1 {{ font-size: 1.4rem; }}
  .stats {{ display: flex; gap: 2rem; flex-wrap: wrap; margin-bottom: 1.5rem; }}
  .stat {{ border: 1px solid #888; border-radius: 8px; padding: 0.75rem 1rem; min-width: 120px; }}
  .stat .value {{ font-size: 1.5rem; font-weight: 600; }}
  .stat .label {{ font-size: 0.8rem; opacity: 0.7; }}
  table {{ border-collapse: collapse; width: 100%; margin-bottom: 2rem; font-size: 0.85rem; }}
  th, td {{ border: 1px solid #ccc; padding: 0.4rem 0.6rem; text-align: left; vertical-align: top; }}
  th {{ cursor: pointer; user-select: none; background: rgba(128,128,128,0.15); }}
  th:hover {{ background: rgba(128,128,128,0.3); }}
  .note {{ max-width: 320px; white-space: pre-wrap; }}
  svg {{ border: 1px solid #888; border-radius: 8px; }}
</style>
</head>
<body>
<h1>Eval Report</h1>
<div class="stats" id="stats"></div>
<h2>Scores over time</h2>
<svg id="chart" viewBox="0 0 600 200" width="600" height="200"></svg>
<h2>Lowest-scoring recommendations</h2>
<table id="worst-table"><thead></thead><tbody></tbody></table>
<h2>All rated turns</h2>
<table id="main-table"><thead></thead><tbody></tbody></table>
<script>
const ROWS = {data_json};
const STATS = {stats_json};

const COLUMNS = [
  {{ key: 'timestamp', label: 'Timestamp' }},
  {{ key: 'query', label: 'Query' }},
  {{ key: 'optimize_for', label: 'Optimize for' }},
  {{ key: 'overall', label: 'Human' }},
  {{ key: 'judge_relevance', label: 'J.Rel' }},
  {{ key: 'judge_fit', label: 'J.Fit' }},
  {{ key: 'judge_quality', label: 'J.Qual' }},
  {{ key: 'judge_overall', label: 'J.Overall' }},
  {{ key: 'note', label: 'Human note' }},
  {{ key: 'judge_note', label: 'Judge note' }},
];

function renderStats() {{
  const el = document.getElementById('stats');
  const items = [
    ['Total rated turns', STATS.total],
    ['Avg human score', STATS.human_avg ?? '—'],
    ['Avg judge score', STATS.judge_avg ?? '—'],
    ['Avg agreement gap', STATS.agreement_avg ?? '—'],
    ['Missing human', STATS.missing_human],
    ['Missing judge', STATS.missing_judge],
  ];
  el.innerHTML = items.map(([label, value]) =>
    `<div class="stat"><div class="value">${{value}}</div><div class="label">${{label}}</div></div>`
  ).join('');
}}

function cellValue(row, key) {{
  const v = row[key];
  return (v === null || v === undefined || v === '') ? '—' : v;
}}

function buildTable(tableEl, rows, sortable) {{
  const thead = tableEl.querySelector('thead');
  const tbody = tableEl.querySelector('tbody');
  let sortKey = null;
  let sortDir = 1;

  function renderBody(currentRows) {{
    tbody.innerHTML = currentRows.map(row => {{
      const cells = COLUMNS.map(col => {{
        const cls = (col.key === 'note' || col.key === 'judge_note') ? ' class="note"' : '';
        return `<td${{cls}}>${{cellValue(row, col.key)}}</td>`;
      }}).join('');
      return `<tr>${{cells}}</tr>`;
    }}).join('');
  }}

  function sortAndRender() {{
    let sorted = rows.slice();
    if (sortKey) {{
      sorted.sort((a, b) => {{
        const av = a[sortKey], bv = b[sortKey];
        if (av === null || av === undefined) return 1;
        if (bv === null || bv === undefined) return -1;
        if (av < bv) return -1 * sortDir;
        if (av > bv) return 1 * sortDir;
        return 0;
      }});
    }}
    renderBody(sorted);
  }}

  thead.innerHTML = '<tr>' + COLUMNS.map(col =>
    `<th data-key="${{col.key}}">${{col.label}}</th>`
  ).join('') + '</tr>';

  if (sortable) {{
    thead.querySelectorAll('th').forEach(th => {{
      th.addEventListener('click', () => {{
        const key = th.dataset.key;
        sortDir = (sortKey === key) ? -sortDir : 1;
        sortKey = key;
        sortAndRender();
      }});
    }});
  }}

  sortAndRender();
}}

function effectiveScore(row) {{
  const h = row.overall, j = row.judge_overall;
  if (h === null || h === undefined) return j;
  if (j === null || j === undefined) return h;
  return Math.min(h, j);
}}

function renderWorst() {{
  const scored = ROWS.filter(r => effectiveScore(r) !== null && effectiveScore(r) !== undefined);
  scored.sort((a, b) => effectiveScore(a) - effectiveScore(b));
  buildTable(document.getElementById('worst-table'), scored.slice(0, 10), false);
}}

function renderChart() {{
  const svg = document.getElementById('chart');
  const w = 600, h = 200, pad = 30;
  const n = ROWS.length;
  if (n === 0) {{ svg.innerHTML = '<text x="10" y="20">No data</text>'; return; }}
  const xFor = i => pad + (i / Math.max(n - 1, 1)) * (w - 2 * pad);
  const yFor = v => h - pad - ((v - 1) / 4) * (h - 2 * pad);

  function pathFor(key, color) {{
    const pts = ROWS.map((r, i) => [i, r[key]]).filter(([, v]) => v !== null && v !== undefined);
    if (pts.length === 0) return '';
    const d = pts.map(([i, v], idx) => `${{idx === 0 ? 'M' : 'L'}} ${{xFor(i)}} ${{yFor(v)}}`).join(' ');
    return `<path d="${{d}}" fill="none" stroke="${{color}}" stroke-width="2" />`;
  }}

  svg.innerHTML =
    `<line x1="${{pad}}" y1="${{h - pad}}" x2="${{w - pad}}" y2="${{h - pad}}" stroke="currentColor" opacity="0.3" />` +
    pathFor('overall', '#2b8cbe') +
    pathFor('judge_overall', '#de6b35') +
    `<text x="${{pad}}" y="16" fill="#2b8cbe" font-size="12">— human</text>` +
    `<text x="${{pad + 80}}" y="16" fill="#de6b35" font-size="12">— judge</text>`;
}}

renderStats();
renderWorst();
buildTable(document.getElementById('main-table'), ROWS.slice().reverse(), true);
renderChart();
</script>
</body>
</html>"""
```

Note: `ROWS` is in file order (chronological, oldest first) — that's what `renderChart()` needs for a left-to-right timeline. The main table reverses it once at render time so it starts newest-first, matching the spec.

- [ ] **Step 2: Implement `main()` and the CLI entry point**

Add to `scripts/eval_report.py` (below `render_html`):

```python
SCORES_PATH = Path("evals/scores.jsonl")
REPORT_PATH = Path("evals/report.html")


def main() -> None:
    rows = load_rows(SCORES_PATH)
    if not rows:
        print(f"No eval data found at {SCORES_PATH} — rate a recommendation first.")
        return
    stats = compute_stats(rows)
    html = render_html(rows, stats)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(html)
    print(f"Wrote {REPORT_PATH}")
    webbrowser.open(REPORT_PATH.resolve().as_uri())


if __name__ == "__main__":
    main()
```

Move `SCORES_PATH = Path("evals/scores.jsonl")` next to the top-of-file imports is not necessary — module-level constants defined just above `main()` is fine and keeps them next to their only user.

- [ ] **Step 3: Manually verify against real data**

Run:

```bash
python scripts/eval_report.py
```

Expected: prints `Wrote evals/report.html` and opens the file in your default browser. Confirm:
- The stats bar shows non-zero `Total rated turns` (your repo already has real rows in `evals/scores.jsonl`).
- The "Lowest-scoring recommendations" table shows the "drum auger drain snake" row (human `overall: 2`, judge `judge_overall: 2.7`) near the top.
- Clicking a column header in "All rated turns" re-sorts the table.
- The chart renders two lines (blue human, orange judge) without console errors — check the browser console.

Then verify the empty-state path:

```bash
mv evals/scores.jsonl /tmp/scores.jsonl.bak
python scripts/eval_report.py
```

Expected: prints `No eval data found at evals/scores.jsonl — rate a recommendation first.` and does not crash or open a browser.

Restore the file:

```bash
mv /tmp/scores.jsonl.bak evals/scores.jsonl
```

- [ ] **Step 4: Run the full test suite to confirm nothing broke**

Run: `pytest -v`
Expected: all tests pass, including the 7 from Tasks 1–2.

- [ ] **Step 5: Document the script in the README**

Add a new section to `README.md` after the "## Search cache" section (around line 123, before "## Models"):

```markdown
## Eval report

`evals/scores.jsonl` accumulates one row per rated recommendation (human
`overall` score plus the LLM judge's `relevance`/`fit`/`quality`/`overall`
scores — see [docs/superpowers/specs/2026-07-26-llm-judge-design.md](docs/superpowers/specs/2026-07-26-llm-judge-design.md)).
To see it as a dashboard instead of raw JSON lines:

```bash
python scripts/eval_report.py
```

This writes `evals/report.html` and opens it in your browser: summary stats,
a sortable table of every rated turn, the 10 lowest-scoring recommendations,
and a human-vs-judge score chart over time. Re-run it any time to pick up new
rows — it's a static snapshot, not a live-updating page.
```

- [ ] **Step 6: Commit**

```bash
git add scripts/eval_report.py README.md
git commit -m "feat: add render_html() and main() for eval report script"
```

---

## Self-Review Notes

- **Spec coverage:** header stats ✅ (Task 3 Step 1 `renderStats`), sortable full table ✅ (`buildTable` with `sortable=true`), lowest-10 section ✅ (`renderWorst`), scores-over-time chart ✅ (`renderChart`), missing-value handling ✅ (`cellValue`, `compute_stats`'s `missing_human`/`missing_judge`), malformed-JSON handling ✅ (`load_rows`), missing-file handling ✅ (`load_rows` + `main`'s early return), no new dependencies ✅ (stdlib only), self-contained HTML ✅ (inline `<style>`/`<script>`, no CDN), tests for pure logic only ✅ (Tasks 1–2), manual verification for `render_html`/`main` ✅ (Task 3 Step 3), README documentation ✅ (Task 3 Step 5).
- **Placeholder scan:** none — every step has runnable code, exact file paths, and exact commands.
- **Type consistency:** `ReportStats` fields (`total`, `human_avg`, `judge_avg`, `agreement_avg`, `missing_human`, `missing_judge`) are identical across Task 2's definition, tests, and Task 3's `dataclasses.asdict(stats)` usage. `load_rows(path: Path) -> list[dict]` signature matches its use in `main()`. Row field names (`overall`, `judge_overall`, `judge_relevance`, `judge_fit`, `judge_quality`, `note`, `judge_note`, `timestamp`, `query`, `optimize_for`) match `agent.record_score()`'s actual JSONL output verbatim.
