# Backlog

Deferred items — not scheduled, but should be picked up by whoever (human or model) is next looking for follow-on work in this area.

- **2026-07-20 — Judge candidate pre-filtering.** The local search cache's judge subagent (see [docs/superpowers/specs/2026-07-20-search-cache-design.md](superpowers/specs/2026-07-20-search-cache-design.md)) currently receives *every* distinct cached query on each lookup, since cache entries never expire. Fine at small scale, but the judge prompt (and its cost/latency) grows unboundedly with usage. Needs a local pre-filter (e.g. shared-word heuristic) to shortlist candidates before invoking the judge. Lives in `tools/cache.py` / `tools/cache_judge.py`.
