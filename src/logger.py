"""Structured eval event logger.

Writes one JSON object per line (JSONL) to a log file. Each entry records
the full eval pipeline result: task prompt, pass/fail, attempt count, rubric
scores, critique, and blocking status.

Designed to be consumed by the ``/eval-harness:eval-report`` command for trend analysis.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Any, Optional

from .judge import JudgeVerdict
from .retry_loop import LoopResult


# ---------------------------------------------------------------------------
# Eval event
# ---------------------------------------------------------------------------

@dataclass
class EvalEvent:
    """Single evaluation event to be persisted.

    Attributes:
        timestamp: ISO-8601 UTC timestamp of when the event was recorded.
        task_prompt: The original task prompt that was evaluated.
        passed: Whether the eval pipeline ultimately passed.
        attempts: Number of agent attempts (1 = no retries).
        blocked: True if max_retries exhausted without a PASS.
        failure_reason: Human-readable summary if blocked.
        scores: Dict of rubric dimension scores from the *final* verdict.
        critique: Final critique text (None if passed on first attempt).
        final_output: The last agent output produced.
    """

    timestamp: str
    task_prompt: str
    passed: bool
    attempts: int
    blocked: bool
    failure_reason: Optional[str]
    scores: dict[str, Any]
    critique: Optional[str]
    final_output: str

    @classmethod
    def from_loop_result(cls, task_prompt: str, result: LoopResult) -> "EvalEvent":
        """Create an EvalEvent from a completed LoopResult.

        Args:
            task_prompt: The original task prompt.
            result: The LoopResult from the retry loop.

        Returns:
            Populated EvalEvent ready for logging.
        """
        # Extract scores from the last verdict (if any).
        if result.verdicts:
            last_verdict = result.verdicts[-1]
            scores = {
                "task_completion": last_verdict.scores.task_completion,
                "factual_groundedness": last_verdict.scores.factual_groundedness,
                "coherence": last_verdict.scores.coherence,
                "relevance": last_verdict.scores.relevance,
                "safety": last_verdict.scores.safety,
                "weighted_score": last_verdict.scores.weighted_score,
            }
            critique = last_verdict.critique
        else:
            scores = {}
            critique = None

        return cls(
            timestamp=datetime.now(timezone.utc).isoformat(),
            task_prompt=task_prompt,
            passed=result.passed,
            attempts=result.attempts,
            blocked=result.blocked,
            failure_reason=result.failure_reason,
            scores=scores,
            critique=critique,
            final_output=result.final_output,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialise the event to a JSON-compatible dict.

        Returns:
            Dict suitable for ``json.dumps``.
        """
        return asdict(self)


# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------

class EvalLogger:
    """Append-only JSONL logger for eval events.

    Args:
        log_path: Absolute or relative path to the JSONL log file.
                  Parent directories are created if they don't exist.
    """

    def __init__(self, log_path: str) -> None:
        self.log_path = log_path

    def log(self, task_prompt: str, result: LoopResult) -> EvalEvent:
        """Record a single eval event to the log file.

        Args:
            task_prompt: The original task prompt.
            result: The completed LoopResult.

        Returns:
            The EvalEvent that was written.
        """
        event = EvalEvent.from_loop_result(task_prompt=task_prompt, result=result)

        # Ensure parent directory exists.
        parent = os.path.dirname(self.log_path)
        if parent:
            os.makedirs(parent, exist_ok=True)

        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event.to_dict()) + "\n")

        return event

    def read_events(self) -> list[dict[str, Any]]:
        """Read all events from the log file.

        Returns:
            List of dicts, one per logged event. Empty list if no log exists.
        """
        if not os.path.exists(self.log_path):
            return []

        events: list[dict[str, Any]] = []
        with open(self.log_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    events.append(json.loads(line))
        return events
