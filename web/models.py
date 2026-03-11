"""Pydantic request/response models for the eval-harness web API.

Defines the HTTP contract independently of the transport layer, so the
same models work for both Vercel serverless functions and local uvicorn.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Requests
# ---------------------------------------------------------------------------

class HeuristicRequest(BaseModel):
    """Request body for the heuristic-only eval endpoint."""

    output: str = Field(..., description="Agent output text to evaluate.")


class JudgeRequest(BaseModel):
    """Request body for the LLM judge-only endpoint."""

    task_prompt: str = Field(..., description="Original task prompt.")
    agent_output: str = Field(..., description="Agent output to evaluate.")


class FullEvalRequest(BaseModel):
    """Request body for the full two-layer eval endpoint."""

    task_prompt: str = Field(..., description="Original task prompt.")
    agent_output: str = Field(..., description="Agent output to evaluate.")


# ---------------------------------------------------------------------------
# Responses
# ---------------------------------------------------------------------------

class HeuristicResponse(BaseModel):
    """Response from the heuristic guard evaluation."""

    passed: bool
    failed_checks: list[str]
    details: str


class ScoresResponse(BaseModel):
    """Rubric dimension scores from the judge."""

    task_completion: int
    factual_groundedness: int
    coherence: int
    relevance: int
    safety: int
    weighted_score: float


class JudgeResponse(BaseModel):
    """Response from the LLM judge evaluation."""

    passed: bool
    scores: ScoresResponse
    critique: Optional[str] = None


class FullEvalResponse(BaseModel):
    """Response from the full two-layer eval pipeline."""

    passed: bool
    heuristic: HeuristicResponse
    judge: Optional[JudgeResponse] = None


class ConfigResponse(BaseModel):
    """Current rubric configuration."""

    max_retries: int
    overall_pass_threshold: float
    weights: dict[str, float]
    min_pass_scores: dict[str, int]
    judge_model: str
    check_pii: bool
    check_secrets: bool
    check_profanity: bool


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    version: str
