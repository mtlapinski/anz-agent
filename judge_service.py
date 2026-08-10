from fastapi import FastAPI
from pydantic import BaseModel

from tools import cache_judge

app = FastAPI()


class MatchRequest(BaseModel):
    query: str
    candidates: list[str]
    trace_id: str | None = None


class MatchResponse(BaseModel):
    matched_query: str | None
    outcome: str


@app.post("/match", response_model=MatchResponse)
def match(request: MatchRequest) -> MatchResponse:
    result = cache_judge.find_match(request.query, request.candidates, trace_id=request.trace_id)
    return MatchResponse(matched_query=result.matched_query, outcome=result.outcome)
