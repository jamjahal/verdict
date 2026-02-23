"""eval-harness: Portable two-layer LLM evaluation pipeline for AI agent outputs.

Layers:
    1. HeuristicGuard  — deterministic checks (PII, secrets, profanity, format)
    2. JudgeAgent      — LLM-as-Judge semantic evaluation (G-Eval rubric)

Retry loop follows the Self-Refine pattern (Madaan et al. 2023):
critique is injected into the primary agent explicitly, up to max_retries.
"""

from .config import EvalConfig, load_config
from .heuristic_guard import HeuristicGuard
from .judge import JudgeOutputParser, JudgeVerdict, RubricScore
from .logger import EvalLogger, EvalEvent
from .retry_loop import RetryLoop

__all__ = [
    "EvalConfig",
    "EvalEvent",
    "EvalLogger",
    "HeuristicGuard",
    "JudgeOutputParser",
    "JudgeVerdict",
    "load_config",
    "RetryLoop",
    "RubricScore",
]
__version__ = "0.1.0"
