from __future__ import annotations
from dataclasses import dataclass
import llm
from tracing import get_langfuse

JUDGE_MODEL_CONFIG = llm.ModelConfig(provider="google", model="gemini-flash-lite-latest")

SYSTEM_PROMPT = (
    "You match a new Amazon product search query against a list of previously cached "
    "search queries, to decide whether a prior search's results can be reused instead "
    "of running a fresh search.\n\n"
    "Two queries match if they describe the same or closely related search intent, "
    "regardless of word order, extra descriptive words (e.g. color, brand, size), or "
    "minor rewording. For example, \"balance beam\" matches \"purple balance beam\", "
    "and \"purple balance beam\" matches \"balance beam purple\".\n\n"
    "Respond with ONLY the exact text of the single best matching cached query, copied "
    "verbatim from the list below. If none of the cached queries are a good match, "
    "respond with exactly: NONE\n"
    "Do not include any other text, explanation, or punctuation in your response."
)

_client = None


def _get_client():
    global _client
    if _client is None:
        _client = llm.create_client(JUDGE_MODEL_CONFIG)
    return _client


@dataclass
class CacheMatch:
    matched_query: str | None   # cached query text if judge found a match, else None
    outcome: str                 # "matched" | "no_match" | "error"


def find_match(query: str, candidates: list[str], trace_id: str | None = None) -> CacheMatch:
    if not candidates:
        return CacheMatch(None, "no_match")

    span = None
    if trace_id:
        try:
            span = get_langfuse().start_observation(
                trace_context={"trace_id": trace_id},
                name="cache_judge",
                as_type="generation",
                input={"query": query, "candidates": candidates},
                model=f"{JUDGE_MODEL_CONFIG.provider}/{JUDGE_MODEL_CONFIG.model}",
            )
        except Exception:
            span = None

    result, attempts_used = _run_attempts(query, candidates)

    if span:
        try:
            span.update(output={
                "matched_query": result.matched_query,
                "outcome": result.outcome,
                "attempts": attempts_used,
            })
        except Exception:
            pass
        try:
            span.end()
        except Exception:
            pass

    return result


def _run_attempts(query: str, candidates: list[str]) -> tuple[CacheMatch, int]:
    candidate_list = "\n".join(f"- {c}" for c in candidates)
    user_message = f"New query: {query}\n\nCached queries:\n{candidate_list}"

    attempts_used = 0
    for attempt in range(2):
        attempts_used = attempt + 1
        try:
            response = llm.complete(
                _get_client(),
                JUDGE_MODEL_CONFIG,
                SYSTEM_PROMPT,
                [],
                [{"role": "user", "content": user_message}],
            )
            answer = (response.text or "").strip()
        except Exception:
            continue

        if answer == "NONE":
            return CacheMatch(None, "no_match"), attempts_used
        if answer in candidates:
            return CacheMatch(answer, "matched"), attempts_used
        # malformed/hallucinated answer — falls through, retried if attempts remain

    return CacheMatch(None, "error"), attempts_used
