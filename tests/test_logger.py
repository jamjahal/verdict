"""Tests for logger.py — Structured eval event logger.

TDD: These tests define the interface and expected behaviour.
All tests will FAIL until src/logger.py is implemented (Red phase).
"""

import json
import os
import tempfile

import pytest

from src.logger import EvalLogger, EvalEvent
from src.judge import JudgeVerdict, RubricScore
from src.retry_loop import LoopResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_verdict(passed: bool, critique: str = "Needs work.") -> JudgeVerdict:
    """Build a minimal JudgeVerdict for testing."""
    scores = RubricScore(
        task_completion=4 if passed else 2,
        factual_groundedness=4 if passed else 2,
        coherence=4 if passed else 3,
        relevance=4 if passed else 2,
        safety=5,
        weighted_score=4.0 if passed else 2.5,
    )
    return JudgeVerdict(
        scores=scores,
        passed=passed,
        critique=None if passed else critique,
        raw_output="EVAL_RESULT\n...\nEND_EVAL_RESULT",
    )


def _make_loop_result(passed: bool, attempts: int = 1) -> LoopResult:
    """Build a minimal LoopResult for testing."""
    verdicts = [_make_verdict(passed=(i == attempts - 1 and passed)) for i in range(attempts)]
    return LoopResult(
        final_output="Some agent output.",
        passed=passed,
        attempts=attempts,
        verdicts=verdicts,
        blocked=not passed,
        failure_reason=None if passed else "Max retries exhausted.",
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_log_path(tmp_path):
    """Return a temporary JSONL log file path."""
    return str(tmp_path / "eval_log.jsonl")


@pytest.fixture
def logger(tmp_log_path) -> EvalLogger:
    """EvalLogger pointing at a temporary file."""
    return EvalLogger(log_path=tmp_log_path)


# ---------------------------------------------------------------------------
# EvalEvent contract
# ---------------------------------------------------------------------------

class TestEvalEventContract:

    def test_from_loop_result_creates_event(self) -> None:
        result = _make_loop_result(passed=True)
        event = EvalEvent.from_loop_result(
            task_prompt="Explain transformers.",
            result=result,
        )
        assert isinstance(event, EvalEvent)
        assert event.passed is True
        assert event.attempts == 1

    def test_event_has_timestamp(self) -> None:
        result = _make_loop_result(passed=True)
        event = EvalEvent.from_loop_result(
            task_prompt="Explain transformers.",
            result=result,
        )
        assert event.timestamp is not None
        assert len(event.timestamp) > 0

    def test_event_to_dict_is_json_serialisable(self) -> None:
        result = _make_loop_result(passed=False, attempts=2)
        event = EvalEvent.from_loop_result(
            task_prompt="Explain transformers.",
            result=result,
        )
        d = event.to_dict()
        # Must be JSON-serialisable without error
        serialised = json.dumps(d)
        assert isinstance(serialised, str)

    def test_event_captures_task_prompt(self) -> None:
        result = _make_loop_result(passed=True)
        event = EvalEvent.from_loop_result(
            task_prompt="Write a haiku.",
            result=result,
        )
        assert event.task_prompt == "Write a haiku."

    def test_event_captures_scores(self) -> None:
        result = _make_loop_result(passed=True)
        event = EvalEvent.from_loop_result(
            task_prompt="Task.",
            result=result,
        )
        d = event.to_dict()
        assert "scores" in d
        assert d["scores"]["task_completion"] == 4

    def test_event_captures_failure_reason_when_blocked(self) -> None:
        result = _make_loop_result(passed=False, attempts=3)
        event = EvalEvent.from_loop_result(
            task_prompt="Task.",
            result=result,
        )
        d = event.to_dict()
        assert d["blocked"] is True
        assert d["failure_reason"] is not None


# ---------------------------------------------------------------------------
# Logging to JSONL
# ---------------------------------------------------------------------------

class TestJSONLLogging:

    def test_log_creates_file(self, logger: EvalLogger, tmp_log_path: str) -> None:
        result = _make_loop_result(passed=True)
        logger.log(task_prompt="Task.", result=result)
        assert os.path.exists(tmp_log_path)

    def test_log_writes_one_line_per_event(self, logger: EvalLogger, tmp_log_path: str) -> None:
        for i in range(3):
            result = _make_loop_result(passed=True)
            logger.log(task_prompt=f"Task {i}.", result=result)

        with open(tmp_log_path) as f:
            lines = f.readlines()
        assert len(lines) == 3

    def test_each_line_is_valid_json(self, logger: EvalLogger, tmp_log_path: str) -> None:
        result = _make_loop_result(passed=False, attempts=2)
        logger.log(task_prompt="Task.", result=result)

        with open(tmp_log_path) as f:
            line = f.readline().strip()
        parsed = json.loads(line)
        assert parsed["passed"] is False
        assert parsed["attempts"] == 2

    def test_log_appends_not_overwrites(self, logger: EvalLogger, tmp_log_path: str) -> None:
        logger.log(task_prompt="First.", result=_make_loop_result(passed=True))
        logger.log(task_prompt="Second.", result=_make_loop_result(passed=False, attempts=2))

        with open(tmp_log_path) as f:
            lines = f.readlines()
        assert len(lines) == 2
        first = json.loads(lines[0])
        second = json.loads(lines[1])
        assert first["task_prompt"] == "First."
        assert second["task_prompt"] == "Second."


# ---------------------------------------------------------------------------
# Reading log
# ---------------------------------------------------------------------------

class TestReadLog:

    def test_read_events_returns_list(self, logger: EvalLogger, tmp_log_path: str) -> None:
        logger.log(task_prompt="Task.", result=_make_loop_result(passed=True))
        events = logger.read_events()
        assert isinstance(events, list)
        assert len(events) == 1

    def test_read_events_empty_file(self, tmp_log_path: str) -> None:
        logger = EvalLogger(log_path=tmp_log_path)
        events = logger.read_events()
        assert events == []

    def test_read_events_preserves_order(self, logger: EvalLogger) -> None:
        logger.log(task_prompt="A.", result=_make_loop_result(passed=True))
        logger.log(task_prompt="B.", result=_make_loop_result(passed=False, attempts=2))
        events = logger.read_events()
        assert events[0]["task_prompt"] == "A."
        assert events[1]["task_prompt"] == "B."
