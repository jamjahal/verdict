"""Tests for the FastAPI web API wrapper.

Verifies that the web layer correctly wraps the portable src/ library,
exposing eval pipeline functionality over HTTP for Vercel deployment.
"""

from __future__ import annotations

import os
from unittest.mock import patch, MagicMock

import pytest
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _set_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure API key env var is set for all tests."""
    monkeypatch.setenv("EVAL_HARNESS_API_KEY", "test-key-123")


@pytest.fixture()
def client() -> TestClient:
    """Create a test client for the FastAPI app."""
    from web.app import app
    return TestClient(app)


@pytest.fixture()
def auth_headers() -> dict[str, str]:
    """Valid auth headers."""
    return {"X-API-Key": "test-key-123"}


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

class TestHealthEndpoint:
    """GET /api/health — no auth required."""

    def test_health_returns_ok(self, client: TestClient) -> None:
        resp = client.get("/api/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert "version" in body


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

class TestAuth:
    """All /api/eval* endpoints require X-API-Key."""

    def test_missing_api_key_returns_401(self, client: TestClient) -> None:
        resp = client.post("/api/eval/heuristic", json={"output": "hello"})
        assert resp.status_code == 401

    def test_wrong_api_key_returns_401(self, client: TestClient) -> None:
        resp = client.post(
            "/api/eval/heuristic",
            json={"output": "hello"},
            headers={"X-API-Key": "wrong-key"},
        )
        assert resp.status_code == 401

    def test_valid_api_key_passes(self, client: TestClient, auth_headers: dict) -> None:
        resp = client.post(
            "/api/eval/heuristic",
            json={"output": "This is a safe test string."},
            headers=auth_headers,
        )
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# POST /api/eval/heuristic
# ---------------------------------------------------------------------------

class TestHeuristicEndpoint:
    """POST /api/eval/heuristic — Layer 1 only."""

    def test_clean_output_passes(self, client: TestClient, auth_headers: dict) -> None:
        resp = client.post(
            "/api/eval/heuristic",
            json={"output": "This is a perfectly safe output."},
            headers=auth_headers,
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["passed"] is True
        assert body["failed_checks"] == []

    def test_pii_detected(self, client: TestClient, auth_headers: dict) -> None:
        resp = client.post(
            "/api/eval/heuristic",
            json={"output": "Contact me at user@example.com for details."},
            headers=auth_headers,
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["passed"] is False
        assert "pii" in body["failed_checks"]

    def test_secrets_detected(self, client: TestClient, auth_headers: dict) -> None:
        resp = client.post(
            "/api/eval/heuristic",
            json={"output": "Use api_key = AKIA1234567890ABCDEF to authenticate."},
            headers=auth_headers,
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["passed"] is False
        assert "secrets" in body["failed_checks"]

    def test_missing_output_returns_422(self, client: TestClient, auth_headers: dict) -> None:
        resp = client.post(
            "/api/eval/heuristic",
            json={},
            headers=auth_headers,
        )
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# POST /api/eval/judge
# ---------------------------------------------------------------------------

class TestJudgeEndpoint:
    """POST /api/eval/judge — Layer 2 only (LLM call mocked)."""

    MOCK_JUDGE_RESPONSE = """EVAL_RESULT
task_completion: 5
factual_groundedness: 4
coherence: 5
relevance: 5
safety: 5
weighted_score: 4.75
verdict: PASS
critique: |
  Output is excellent.
END_EVAL_RESULT"""

    @patch("web.app.call_judge")
    def test_judge_pass(
        self, mock_judge: MagicMock, client: TestClient, auth_headers: dict
    ) -> None:
        mock_judge.return_value = self.MOCK_JUDGE_RESPONSE
        resp = client.post(
            "/api/eval/judge",
            json={
                "task_prompt": "Write a summary.",
                "agent_output": "Here is the summary.",
            },
            headers=auth_headers,
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["passed"] is True
        assert body["scores"]["task_completion"] == 5
        assert body["scores"]["weighted_score"] == 4.75

    @patch("web.app.call_judge")
    def test_judge_fail_includes_critique(
        self, mock_judge: MagicMock, client: TestClient, auth_headers: dict
    ) -> None:
        mock_judge.return_value = """EVAL_RESULT
task_completion: 2
factual_groundedness: 2
coherence: 3
relevance: 2
safety: 5
weighted_score: 2.40
verdict: FAIL
critique: |
  The response missed the core question and contained unsupported claims.
END_EVAL_RESULT"""
        resp = client.post(
            "/api/eval/judge",
            json={
                "task_prompt": "Write a summary.",
                "agent_output": "Unrelated content.",
            },
            headers=auth_headers,
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["passed"] is False
        assert body["critique"] is not None
        assert "missed" in body["critique"]

    def test_missing_fields_returns_422(self, client: TestClient, auth_headers: dict) -> None:
        resp = client.post(
            "/api/eval/judge",
            json={"task_prompt": "Write a summary."},
            headers=auth_headers,
        )
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# POST /api/eval — full pipeline
# ---------------------------------------------------------------------------

class TestFullEvalEndpoint:
    """POST /api/eval — both layers."""

    @patch("web.app.call_judge")
    def test_full_eval_clean_pass(
        self, mock_judge: MagicMock, client: TestClient, auth_headers: dict
    ) -> None:
        mock_judge.return_value = """EVAL_RESULT
task_completion: 4
factual_groundedness: 4
coherence: 4
relevance: 4
safety: 5
weighted_score: 4.10
verdict: PASS
critique: |
  Solid output.
END_EVAL_RESULT"""
        resp = client.post(
            "/api/eval",
            json={
                "task_prompt": "Summarise Q3 results.",
                "agent_output": "Q3 revenue grew 12% YoY driven by new product launches.",
            },
            headers=auth_headers,
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["passed"] is True
        assert body["heuristic"]["passed"] is True
        assert body["judge"]["passed"] is True

    def test_full_eval_heuristic_blocks(
        self, client: TestClient, auth_headers: dict
    ) -> None:
        """If heuristic guard fails, judge should not be called."""
        resp = client.post(
            "/api/eval",
            json={
                "task_prompt": "Summarise Q3 results.",
                "agent_output": "Contact user@example.com for the full report.",
            },
            headers=auth_headers,
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["passed"] is False
        assert body["heuristic"]["passed"] is False
        assert body["judge"] is None

    def test_missing_fields_returns_422(self, client: TestClient, auth_headers: dict) -> None:
        resp = client.post(
            "/api/eval",
            json={"agent_output": "something"},
            headers=auth_headers,
        )
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# GET /api/config
# ---------------------------------------------------------------------------

class TestConfigEndpoint:
    """GET /api/config — returns current rubric config."""

    def test_returns_config(self, client: TestClient, auth_headers: dict) -> None:
        resp = client.get("/api/config", headers=auth_headers)
        assert resp.status_code == 200
        body = resp.json()
        assert "max_retries" in body
        assert "weights" in body
        assert "overall_pass_threshold" in body
        assert body["max_retries"] == 3
