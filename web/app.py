"""FastAPI application for the eval-harness web API.

Wraps the portable src/ library as HTTP endpoints, deployable on Vercel
(Python serverless) or standalone via uvicorn. The Claude Code plugin
and this web API share the same core evaluation logic.

Endpoints:
    GET  /api/health           — liveness check (no auth)
    POST /api/eval/heuristic   — Layer 1 heuristic guard only
    POST /api/eval/judge       — Layer 2 LLM-as-Judge only
    POST /api/eval             — Full two-layer pipeline
    GET  /api/config           — Current rubric configuration
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi import Depends, FastAPI

from src import __version__
from src.api_judge import call_judge
from src.config import EvalConfig, load_config
from src.heuristic_guard import HeuristicGuard
from src.judge import JudgeOutputParser

from web.auth import require_api_key
from web.models import (
    ConfigResponse,
    FullEvalRequest,
    FullEvalResponse,
    HealthResponse,
    HeuristicRequest,
    HeuristicResponse,
    JudgeRequest,
    JudgeResponse,
    ScoresResponse,
)


def _resolve_config() -> EvalConfig:
    """Load config from the bundled rubric.yaml, falling back to defaults."""
    config_path = os.path.join(os.path.dirname(__file__), "..", "config", "rubric.yaml")
    return load_config(config_path)


app = FastAPI(
    title="eval-harness",
    description="Two-layer LLM evaluation pipeline for AI agent outputs.",
    version=__version__,
)


# ---------------------------------------------------------------------------
# GET /api/health
# ---------------------------------------------------------------------------

@app.get("/api/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Liveness check. No authentication required."""
    return HealthResponse(status="ok", version=__version__)


# ---------------------------------------------------------------------------
# POST /api/eval/heuristic
# ---------------------------------------------------------------------------

@app.post(
    "/api/eval/heuristic",
    response_model=HeuristicResponse,
    dependencies=[Depends(require_api_key)],
)
async def eval_heuristic(body: HeuristicRequest) -> HeuristicResponse:
    """Run Layer 1 heuristic guard on agent output."""
    config = _resolve_config()
    guard = config.build_guard()
    result = guard.evaluate(body.output)
    return HeuristicResponse(
        passed=result.passed,
        failed_checks=result.failed_checks,
        details=result.details,
    )


# ---------------------------------------------------------------------------
# POST /api/eval/judge
# ---------------------------------------------------------------------------

@app.post(
    "/api/eval/judge",
    response_model=JudgeResponse,
    dependencies=[Depends(require_api_key)],
)
async def eval_judge(body: JudgeRequest) -> JudgeResponse:
    """Run Layer 2 LLM-as-Judge on agent output."""
    config = _resolve_config()
    raw = call_judge(config, body.task_prompt, body.agent_output)
    parser = config.build_parser()
    verdict = parser.parse(raw)
    return JudgeResponse(
        passed=verdict.passed,
        scores=ScoresResponse(
            task_completion=verdict.scores.task_completion,
            factual_groundedness=verdict.scores.factual_groundedness,
            coherence=verdict.scores.coherence,
            relevance=verdict.scores.relevance,
            safety=verdict.scores.safety,
            weighted_score=verdict.scores.weighted_score,
        ),
        critique=verdict.critique,
    )


# ---------------------------------------------------------------------------
# POST /api/eval
# ---------------------------------------------------------------------------

@app.post(
    "/api/eval",
    response_model=FullEvalResponse,
    dependencies=[Depends(require_api_key)],
)
async def eval_full(body: FullEvalRequest) -> FullEvalResponse:
    """Run the full two-layer eval pipeline.

    Layer 1 (heuristic guard) runs first. If it fails, the judge is
    skipped and the response indicates which heuristic checks failed.
    """
    config = _resolve_config()
    guard = config.build_guard()
    guard_result = guard.evaluate(body.agent_output)

    heuristic_resp = HeuristicResponse(
        passed=guard_result.passed,
        failed_checks=guard_result.failed_checks,
        details=guard_result.details,
    )

    if not guard_result.passed:
        return FullEvalResponse(
            passed=False,
            heuristic=heuristic_resp,
            judge=None,
        )

    raw = call_judge(config, body.task_prompt, body.agent_output)
    parser = config.build_parser()
    verdict = parser.parse(raw)

    judge_resp = JudgeResponse(
        passed=verdict.passed,
        scores=ScoresResponse(
            task_completion=verdict.scores.task_completion,
            factual_groundedness=verdict.scores.factual_groundedness,
            coherence=verdict.scores.coherence,
            relevance=verdict.scores.relevance,
            safety=verdict.scores.safety,
            weighted_score=verdict.scores.weighted_score,
        ),
        critique=verdict.critique,
    )

    return FullEvalResponse(
        passed=verdict.passed,
        heuristic=heuristic_resp,
        judge=judge_resp,
    )


# ---------------------------------------------------------------------------
# GET /api/config
# ---------------------------------------------------------------------------

@app.get(
    "/api/config",
    response_model=ConfigResponse,
    dependencies=[Depends(require_api_key)],
)
async def get_config() -> ConfigResponse:
    """Return the current rubric configuration."""
    config = _resolve_config()
    return ConfigResponse(
        max_retries=config.max_retries,
        overall_pass_threshold=config.overall_pass_threshold,
        weights=config.weights,
        min_pass_scores=config.min_pass_scores,
        judge_model=config.judge_model,
        check_pii=config.check_pii,
        check_secrets=config.check_secrets,
        check_profanity=config.check_profanity,
    )
