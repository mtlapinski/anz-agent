from __future__ import annotations
import llm

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


def find_match(query: str, candidates: list[str]) -> str | None:
    if not candidates:
        return None

    try:
        candidate_list = "\n".join(f"- {c}" for c in candidates)
        user_message = f"New query: {query}\n\nCached queries:\n{candidate_list}"
        response = llm.complete(
            _get_client(),
            JUDGE_MODEL_CONFIG,
            SYSTEM_PROMPT,
            [],
            [{"role": "user", "content": user_message}],
        )
        answer = (response.text or "").strip()
    except Exception:
        return None

    if answer not in candidates:
        return None
    return answer
