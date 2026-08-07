from __future__ import annotations

import json
import sys
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
