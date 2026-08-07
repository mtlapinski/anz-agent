from __future__ import annotations

import dataclasses
import json
import sys
import webbrowser
from dataclasses import dataclass
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


DISPLAY_KEYS = [
    "timestamp", "query", "optimize_for", "overall", "note",
    "judge_relevance", "judge_fit", "judge_quality", "judge_overall", "judge_note",
]


def _project(rows: list[dict]) -> list[dict]:
    return [{k: row.get(k) for k in DISPLAY_KEYS} for row in rows]


def _embed_json(obj) -> str:
    """json.dumps, but safe to embed inside an HTML <script> block."""
    return (
        json.dumps(obj)
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
    )


def render_html(rows: list[dict], stats: ReportStats) -> str:
    data_json = _embed_json(_project(rows))
    stats_json = _embed_json(dataclasses.asdict(stats))
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

function escapeHtml(str) {{
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');
}}

function cellValue(row, key) {{
  const v = row[key];
  return (v === null || v === undefined || v === '') ? '—' : escapeHtml(v);
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
