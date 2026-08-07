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
