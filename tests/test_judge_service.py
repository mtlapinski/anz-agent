from unittest.mock import patch
from fastapi.testclient import TestClient

import judge_service
from tools.cache_judge import CacheMatch


def client():
    return TestClient(judge_service.app)


def test_match_endpoint_returns_matched_result():
    with patch("judge_service.cache_judge.find_match", return_value=CacheMatch("yoga mat", "matched")) as mock_find:
        response = client().post(
            "/match", json={"query": "kettlebell", "candidates": ["yoga mat"], "trace_id": "trace-1"}
        )

    assert response.status_code == 200
    assert response.json() == {"matched_query": "yoga mat", "outcome": "matched"}
    mock_find.assert_called_once_with("kettlebell", ["yoga mat"], trace_id="trace-1")


def test_match_endpoint_returns_no_match_result():
    with patch("judge_service.cache_judge.find_match", return_value=CacheMatch(None, "no_match")):
        response = client().post("/match", json={"query": "kettlebell", "candidates": ["yoga mat"]})

    assert response.status_code == 200
    assert response.json() == {"matched_query": None, "outcome": "no_match"}


def test_match_endpoint_defaults_trace_id_to_none():
    with patch("judge_service.cache_judge.find_match", return_value=CacheMatch(None, "no_match")) as mock_find:
        client().post("/match", json={"query": "kettlebell", "candidates": []})

    mock_find.assert_called_once_with("kettlebell", [], trace_id=None)


def test_match_endpoint_returns_error_result():
    with patch("judge_service.cache_judge.find_match", return_value=CacheMatch(None, "error")):
        response = client().post("/match", json={"query": "kettlebell", "candidates": ["yoga mat"]})

    assert response.status_code == 200
    assert response.json() == {"matched_query": None, "outcome": "error"}


def test_match_endpoint_rejects_missing_query():
    response = client().post("/match", json={"candidates": ["yoga mat"]})
    assert response.status_code == 422
