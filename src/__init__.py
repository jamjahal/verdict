"""eval-harness: Portable two-layer LLM evaluation pipeline for AI agent outputs.

Layers:
    1. HeuristicGuard  — deterministic checks (PII, secrets, profanity, format)
    2. JudgeAgent      — LLM-as-Judge semantic evaluation (G-Eval rubric)

Retry loop follows the Self-Refine pattern (Madaan et al. 2023):
critique is injected into the primary agent explicitly, up to max_retries.
"""

from .heuristic_guard import HeuristicGuard
from .judge import JudgeVerdict, RubricScore
from .retry_loop import RetryLoop

__all__ = ["HeuristicGuard", "JudgeVerdict", "RubricScore", "RetryLoop"]
__version__ = "0.1.0"
