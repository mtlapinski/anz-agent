from __future__ import annotations
import os
import httpx
from tools.cache_judge import CacheMatch, find_match as _find_match_in_process

_client: httpx.Client | None = None


def _get_client() -> httpx.Client:
    global _client
    if _client is None:
        _client = httpx.Client(timeout=15.0)
    return _client


def find_match(query: str, candidates: list[str], trace_id: str | None = None) -> CacheMatch:
    judge_url = os.environ.get("CACHE_JUDGE_URL")
    if not judge_url:
        return _find_match_in_process(query, candidates, trace_id=trace_id)

    try:
        response = _get_client().post(
            judge_url,
            json={"query": query, "candidates": candidates, "trace_id": trace_id},
        )
        response.raise_for_status()
        data = response.json()
        return CacheMatch(matched_query=data["matched_query"], outcome=data["outcome"])
    except Exception:
        return CacheMatch(None, "error")
