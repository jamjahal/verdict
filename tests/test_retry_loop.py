"""Tests for retry_loop.py — Self-Refine orchestrator.

TDD: These tests define the interface and expected behaviour.
All tests will FAIL until src/retry_loop.py is implemented (Red phase).
"""

import pytest
from unittest.mock import MagicMock, call
from src.retry_loop import RetryLoop, LoopResult
from src.judge import JudgeVerdict, RubricScore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_verdict(passed: bool, critique: str = "Needs improvement.") -> JudgeVerdict:
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


PASS_JUDGE_OUTPUT = """
EVAL_RESULT
task_completion: 4
factual_groundedness: 4
coherence: 4
relevance: 4
safety: 5
weighted_score: 4.05
verdict: PASS
critique: |
  Good response.
END_EVAL_RESULT
"""

FAIL_JUDGE_OUTPUT = """
EVAL_RESULT
task_completion: 2
factual_groundedness: 2
coherence: 3
relevance: 2
safety: 5
weighted_score: 2.35
verdict: FAIL
critique: |
  The response missed the core task. On retry, answer the question directly.
END_EVAL_RESULT
"""


# ---------------------------------------------------------------------------
# Happy path: passes on first attempt
# ---------------------------------------------------------------------------

class TestFirstAttemptPass:

    def test_passes_on_first_attempt(self) -> None:
        agent_fn = MagicMock(return_value="A high quality response.")
        judge_fn = MagicMock(return_value=PASS_JUDGE_OUTPUT)

        loop = RetryLoop(agent_fn=agent_fn, judge_fn=judge_fn)
        result = loop.run("What is the capital of France?")

        assert isinstance(result, LoopResult)
        assert result.passed is True
        assert result.attempts == 1
        assert result.blocked is False
        assert agent_fn.call_count == 1

    def test_first_attempt_critique_is_none(self) -> None:
        agent_fn = MagicMock(return_value="Good output.")
        judge_fn = MagicMock(return_value=PASS_JUDGE_OUTPUT)

        loop = RetryLoop(agent_fn=agent_fn, judge_fn=judge_fn)
        loop.run("task prompt")

        # First call should have critique=None
        args, kwargs = agent_fn.call_args_list[0]
        prompt, critique = args[0], args[1]
        assert critique is None


# ---------------------------------------------------------------------------
# Retry loop behaviour
# ---------------------------------------------------------------------------

class TestRetryBehaviour:

    def test_retries_on_fail_then_passes(self) -> None:
        agent_fn = MagicMock(side_effect=["Bad output.", "Good output."])
        judge_fn = MagicMock(side_effect=[FAIL_JUDGE_OUTPUT, PASS_JUDGE_OUTPUT])

        loop = RetryLoop(agent_fn=agent_fn, judge_fn=judge_fn)
        result = loop.run("task prompt")

        assert result.passed is True
        assert result.attempts == 2
        assert len(result.verdicts) == 2

    def test_critique_injected_on_retry(self) -> None:
        agent_fn = MagicMock(side_effect=["Bad.", "Better."])
        judge_fn = MagicMock(side_effect=[FAIL_JUDGE_OUTPUT, PASS_JUDGE_OUTPUT])

        loop = RetryLoop(agent_fn=agent_fn, judge_fn=judge_fn)
        loop.run("task prompt")

        # Second call must include critique context
        second_call_args = agent_fn.call_args_list[1]
        critique_arg = second_call_args[0][1]
        assert critique_arg is not None
        assert len(critique_arg) > 0

    def test_retry_prompt_includes_original_task(self) -> None:
        loop = RetryLoop(agent_fn=MagicMock(), judge_fn=MagicMock())
        task = "Explain transformer architecture."
        critique = "Response was too vague."

        retry_prompt = loop._build_retry_prompt(task, critique, attempt=2)

        assert task in retry_prompt
        assert critique in retry_prompt


# ---------------------------------------------------------------------------
# Max retries exhausted
# ---------------------------------------------------------------------------

class TestMaxRetriesExhausted:

    def test_blocks_after_max_retries(self) -> None:
        agent_fn = MagicMock(return_value="Still bad.")
        judge_fn = MagicMock(return_value=FAIL_JUDGE_OUTPUT)

        loop = RetryLoop(agent_fn=agent_fn, judge_fn=judge_fn, max_retries=3)
        result = loop.run("task prompt")

        assert result.passed is False
        assert result.blocked is True
        assert result.attempts == 3

    def test_all_verdicts_recorded_when_blocked(self) -> None:
        agent_fn = MagicMock(return_value="Bad.")
        judge_fn = MagicMock(return_value=FAIL_JUDGE_OUTPUT)

        loop = RetryLoop(agent_fn=agent_fn, judge_fn=judge_fn, max_retries=3)
        result = loop.run("task prompt")

        assert len(result.verdicts) == 3

    def test_failure_reason_set_when_blocked(self) -> None:
        agent_fn = MagicMock(return_value="Bad.")
        judge_fn = MagicMock(return_value=FAIL_JUDGE_OUTPUT)

        loop = RetryLoop(agent_fn=agent_fn, judge_fn=judge_fn, max_retries=2)
        result = loop.run("task prompt")

        assert result.failure_reason is not None
        assert len(result.failure_reason) > 0

    def test_custom_max_retries_respected(self) -> None:
        agent_fn = MagicMock(return_value="Bad.")
        judge_fn = MagicMock(return_value=FAIL_JUDGE_OUTPUT)

        loop = RetryLoop(agent_fn=agent_fn, judge_fn=judge_fn, max_retries=1)
        result = loop.run("task prompt")

        assert result.attempts == 1
        assert result.blocked is True


# ---------------------------------------------------------------------------
# Heuristic guard integration
# ---------------------------------------------------------------------------

class TestHeuristicGuardIntegration:

    def test_heuristic_fail_triggers_retry_without_judge(self) -> None:
        """When heuristic guard fails, judge should NOT be called."""
        agent_fn = MagicMock(side_effect=[
            "Email user@example.com",   # fails heuristic
            "Safe clean response.",     # passes
        ])
        judge_fn = MagicMock(return_value=PASS_JUDGE_OUTPUT)

        loop = RetryLoop(agent_fn=agent_fn, judge_fn=judge_fn, max_retries=3)
        result = loop.run("task prompt")

        # Judge only called for the clean second response
        assert judge_fn.call_count == 1

    def test_heuristic_fail_blocks_after_max_retries(self) -> None:
        agent_fn = MagicMock(return_value="Email: user@example.com")
        judge_fn = MagicMock()

        loop = RetryLoop(agent_fn=agent_fn, judge_fn=judge_fn, max_retries=2)
        result = loop.run("task prompt")

        assert result.blocked is True
        # Judge never called — heuristic fails every time
        judge_fn.assert_not_called()
